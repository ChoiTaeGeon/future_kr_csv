"""
Dynamic Tick Resampling Engine
- Resamples raw ticks into N-Tick OHLCV bars.
- Supports single tick size and multi-tick range sweeps (e.g. 1000 ~ 10000 ticks).
- Vectorized computation for high speed.
"""
from typing import List, Dict, Union, Optional, Any
from pathlib import Path
import hashlib
import time
import pandas as pd
import numpy as np
import duckdb
from engine.data_engine import DataEngine, get_data_engine
from config import CACHE_DIR

class Resampler:
    def __init__(self, data_engine: Optional[DataEngine] = None, db_path: Optional[str] = None):
        if data_engine:
            self.data_engine = data_engine
        elif db_path:
            self.data_engine = get_data_engine(db_path)
        else:
            self.data_engine = get_data_engine()

    def _get_cache_path(self, resample_type: str, size: int, symbol: Optional[str], 
                         start_time: Optional[str], end_time: Optional[str], 
                         filter_outliers: bool, min_price: float = 50.0, max_price: float = 2000.0) -> Path:
        """Generate safe, unique Parquet cache path for resampled candle parameters."""
        sym = symbol or "ALL"
        st = (start_time or "START").replace(":", "-").replace(" ", "_")
        et = (end_time or "END").replace(":", "-").replace(" ", "_")
        fo = f"flt1_{int(min_price)}_{int(max_price)}" if filter_outliers else "flt0"
        raw_key = f"{resample_type}_{size}_{sym}_{st}_{et}_{fo}"
        h = hashlib.md5(raw_key.encode('utf-8')).hexdigest()[:12]
        filename = f"{resample_type}_{size}_{sym}_{st[:10]}_{et[:10]}_{h}.parquet"
        return CACHE_DIR / filename

    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load resampled bars DataFrame directly from Parquet cache file with sub-50ms speed."""
        if not cache_path.exists():
            return None
        try:
            with duckdb.connect() as con:
                df = con.execute(f"SELECT * FROM read_parquet('{cache_path.as_posix()}')").fetchdf()
            if not df.empty:
                if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                if 'open_time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['open_time']):
                    df['open_time'] = pd.to_datetime(df['open_time'])
                if 'close_time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['close_time']):
                    df['close_time'] = pd.to_datetime(df['close_time'])
                return df
        except Exception as e:
            try:
                cache_path.unlink()
            except Exception:
                pass
        return None

    def _save_to_cache(self, cache_path: Path, df: pd.DataFrame):
        """Save resampled bars DataFrame directly to Parquet cache file using DuckDB native writer."""
        if df is None or df.empty:
            return
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with duckdb.connect() as con:
                con.execute(f"COPY df TO '{cache_path.as_posix()}' (FORMAT PARQUET)")
        except Exception as e:
            pass

    def resample_from_df(self, df: pd.DataFrame, tick_size: int, 
                          filter_outliers: bool = True, min_price: float = 50.0, max_price: float = 2000.0) -> pd.DataFrame:
        """
        Resample raw tick DataFrame into N-Tick OHLCV bars with intraday daily reset and outlier filtering.
        df must contain ['timestamp', 'price', 'volume'] (and optionally 'bid', 'ask').
        """
        if df.empty or len(df) < tick_size:
            return pd.DataFrame()

        # Filter abnormal prices and negative volumes
        if filter_outliers:
            df = df[(df['price'] >= min_price) & (df['price'] <= max_price) & (df['volume'] >= 0)].copy()
            if df.empty or len(df) < tick_size:
                return pd.DataFrame()

        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. Intraday Daily Session Grouping (일별 누적 틱 주기 리셋)
        ts_series = pd.to_datetime(df['timestamp'])
        trade_dates = ts_series.dt.strftime('%Y-%m-%d')
        df['_trade_date'] = trade_dates
        
        # Calculate group IDs per trading day
        daily_tick_idx = df.groupby('_trade_date').cumcount() // tick_size
        df['_bar_id'] = trade_dates + '_' + daily_tick_idx.astype(str)
        
        # Calculate typical price * volume for VWAP
        df['_pv'] = df['price'] * df['volume']

        # Aggregation
        agg_dict = {
            'timestamp': ['first', 'last'],
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum',
            '_pv': 'sum'
        }
        
        if 'bid' in df.columns:
            agg_dict['bid'] = 'last'
        if 'ask' in df.columns:
            agg_dict['ask'] = 'last'

        grouped = df.groupby('_bar_id', sort=False).agg(agg_dict)
        
        # Flatten multi-level columns
        bars = pd.DataFrame()
        bars['open_time'] = grouped['timestamp']['first']
        bars['close_time'] = grouped['timestamp']['last']
        bars['timestamp'] = grouped['timestamp']['first']  # Open time matching Kiwoom HTS convention
        bars['open'] = grouped['price']['first']
        bars['high'] = grouped['price']['max']
        bars['low'] = grouped['price']['min']
        bars['close'] = grouped['price']['last']
        bars['volume'] = grouped['volume']['sum']
        bars['tick_count'] = df.groupby('_bar_id', sort=False).size().values
        
        # Compute VWAP
        vol_sum = bars['volume'].replace(0, np.nan)
        bars['vwap'] = grouped['_pv']['sum'] / vol_sum
        bars['vwap'] = bars['vwap'].fillna(bars['close'])

        if 'bid' in df.columns:
            bars['bid'] = grouped['bid']['last']
        if 'ask' in df.columns:
            bars['ask'] = grouped['ask']['last']

        # Clean temporary columns
        df.drop(columns=['_trade_date', '_bar_id', '_pv'], inplace=True, errors='ignore')
        
        bars.sort_values('open_time', inplace=True)
        bars.reset_index(drop=True, inplace=True)
        return bars

    def resample_from_db(self, tick_size: int, symbol: Optional[str] = None, 
                         start_time: Optional[str] = None, end_time: Optional[str] = None,
                         filter_outliers: bool = True, min_price: float = 50.0, max_price: float = 2000.0,
                         use_cache: bool = True) -> pd.DataFrame:
        """
        Resample directly from DuckDB for massive datasets using intraday partitioned window functions
        with automatic outlier filtering and Kiwoom HTS compliant intraday reset.
        Uses persistent Parquet cache for instantaneous repeated queries.
        """
        cache_path = None
        if use_cache:
            cache_path = self._get_cache_path("TICK", tick_size, symbol, start_time, end_time, filter_outliers, min_price, max_price)
            cached_df = self._load_from_cache(cache_path)
            if cached_df is not None:
                return cached_df

        if not self.data_engine:
            self.data_engine = get_data_engine()

        conditions = []
        params = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        # DuckDB high-speed SQL aggregation partitioned by trading day (일별 누적 틱 주기 리셋 - 키움 HTS 100% 동일 규격)
        if filter_outliers:
            query = f"""
                WITH base_ticks AS (
                    SELECT rowid, timestamp, symbol, price, volume, bid, ask, CAST(timestamp AS DATE) AS trade_date
                    FROM ticks
                    {where_clause}
                ),
                daily_stats AS (
                    SELECT 
                        trade_date,
                        MEDIAN(price) AS day_median
                    FROM base_ticks
                    WHERE price > 0 AND volume >= 0
                    GROUP BY trade_date
                ),
                moving_daily AS (
                    SELECT 
                        trade_date,
                        day_median,
                        AVG(day_median) OVER (
                            ORDER BY trade_date 
                            ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
                            EXCLUDE CURRENT ROW
                        ) AS neighbor_avg
                    FROM daily_stats
                ),
                ranked_ticks AS (
                    SELECT 
                        t.rowid,
                        t.timestamp,
                        t.symbol,
                        t.price,
                        t.volume,
                        t.bid,
                        t.ask,
                        t.trade_date,
                        ((ROW_NUMBER() OVER (PARTITION BY t.trade_date ORDER BY t.timestamp ASC, t.rowid ASC) - 1) // {tick_size}) AS bar_idx
                    FROM base_ticks t
                    JOIN moving_daily m ON t.trade_date = m.trade_date
                    WHERE t.price > 0 AND t.volume >= 0
                      AND ABS(t.price - m.day_median) <= m.day_median * 0.15
                      AND (m.neighbor_avg IS NULL OR ABS(m.day_median - m.neighbor_avg) <= m.neighbor_avg * 0.15)
                )
                SELECT 
                    MIN(timestamp) AS open_time,
                    MAX(timestamp) AS close_time,
                    MIN(timestamp) AS timestamp,
                    FIRST(price ORDER BY timestamp ASC, rowid ASC) AS open,
                    MAX(price) AS high,
                    MIN(price) AS low,
                    LAST(price ORDER BY timestamp ASC, rowid ASC) AS close,
                    SUM(volume) AS volume,
                    COUNT(*) AS tick_count,
                    CASE WHEN SUM(volume) > 0 THEN SUM(price * volume) / SUM(volume) ELSE LAST(price ORDER BY timestamp ASC, rowid ASC) END AS vwap,
                    LAST(bid ORDER BY timestamp ASC, rowid ASC) AS bid,
                    LAST(ask ORDER BY timestamp ASC, rowid ASC) AS ask
                FROM ranked_ticks
                GROUP BY trade_date, bar_idx
                ORDER BY open_time ASC
            """
        else:
            query = f"""
                WITH ranked_ticks AS (
                    SELECT 
                        rowid,
                        timestamp,
                        symbol,
                        price,
                        volume,
                        bid,
                        ask,
                        CAST(timestamp AS DATE) AS trade_date,
                        ((ROW_NUMBER() OVER (PARTITION BY CAST(timestamp AS DATE) ORDER BY timestamp ASC, rowid ASC) - 1) // {tick_size}) AS bar_idx
                    FROM ticks
                    {where_clause}
                )
                SELECT 
                    MIN(timestamp) AS open_time,
                    MAX(timestamp) AS close_time,
                    MIN(timestamp) AS timestamp,
                    FIRST(price ORDER BY timestamp ASC, rowid ASC) AS open,
                    MAX(price) AS high,
                    MIN(price) AS low,
                    LAST(price ORDER BY timestamp ASC, rowid ASC) AS close,
                    SUM(volume) AS volume,
                    COUNT(*) AS tick_count,
                    CASE WHEN SUM(volume) > 0 THEN SUM(price * volume) / SUM(volume) ELSE LAST(price ORDER BY timestamp ASC, rowid ASC) END AS vwap,
                    LAST(bid ORDER BY timestamp ASC, rowid ASC) AS bid,
                    LAST(ask ORDER BY timestamp ASC, rowid ASC) AS ask
                FROM ranked_ticks
                GROUP BY trade_date, bar_idx
                ORDER BY open_time ASC
            """
        with self.data_engine.lock:
            bars_df = self.data_engine.conn.execute(query, params).fetchdf()

        if use_cache and cache_path and not bars_df.empty:
            self._save_to_cache(cache_path, bars_df)

        return bars_df

    def resample_multiple(self, tick_sizes: List[int], 
                          data_source: Union[pd.DataFrame, str] = "db", 
                          symbol: Optional[str] = None,
                          start_time: Optional[str] = None, 
                          end_time: Optional[str] = None) -> Dict[int, pd.DataFrame]:
        """
        Generate multiple tick bar DataFrames for a list of tick sizes.
        Returns a dictionary: {tick_size: ohlcv_df}
        """
        results = {}
        for ts in tick_sizes:
            if isinstance(data_source, pd.DataFrame):
                df_filtered = data_source
                if start_time:
                    df_filtered = df_filtered[df_filtered['timestamp'] >= start_time]
                if end_time:
                    df_filtered = df_filtered[df_filtered['timestamp'] <= end_time]
                bars = self.resample_from_df(df_filtered, tick_size=ts)
            else:
                bars = self.resample_from_db(tick_size=ts, symbol=symbol, start_time=start_time, end_time=end_time)
            results[ts] = bars
        return results

    def resample_range(self, start_tick: int = 1000, end_tick: int = 10000, step_tick: int = 1000,
                       data_source: Union[pd.DataFrame, str] = "db",
                       symbol: Optional[str] = None,
                       start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[int, pd.DataFrame]:
        """
        Generate multiple tick bar DataFrames across a range [start_tick, end_tick] with step_tick.
        Safely handles start_tick == end_tick, zero step_tick, and reversed ranges.
        """
        start_tick = max(10, int(start_tick))
        end_tick = max(10, int(end_tick))
        step_tick = int(step_tick or 1000)

        if start_tick > end_tick:
            start_tick, end_tick = end_tick, start_tick

        if start_tick == end_tick or step_tick <= 0:
            tick_sizes = [start_tick]
        else:
            tick_sizes = list(range(start_tick, end_tick + 1, step_tick))
            if not tick_sizes:
                tick_sizes = [start_tick]
            elif tick_sizes[-1] < end_tick:
                tick_sizes.append(end_tick)

        return self.resample_multiple(tick_sizes=tick_sizes, data_source=data_source, symbol=symbol, start_time=start_time, end_time=end_time)

    # -------------------------------------------------------------
    # Volume Bar Resampling Engine (거래량 주기 기준 리샘플링)
    # -------------------------------------------------------------
    def resample_volume_from_df(self, df: pd.DataFrame, volume_size: int,
                                filter_outliers: bool = True, min_price: float = 50.0, max_price: float = 2000.0) -> pd.DataFrame:
        """
        Resample raw tick DataFrame into N-Volume (Contracts) OHLCV bars with intraday daily reset and outlier filtering.
        df must contain ['timestamp', 'price', 'volume'] (and optionally 'bid', 'ask').
        """
        if df.empty or 'volume' not in df.columns or df['volume'].sum() < volume_size:
            return pd.DataFrame()

        if filter_outliers:
            df = df[(df['price'] >= min_price) & (df['price'] <= max_price) & (df['volume'] >= 0)].copy()
            if df.empty or df['volume'].sum() < volume_size:
                return pd.DataFrame()

        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. Intraday Daily Session Grouping (일별 누적 거래량 주기 리셋)
        ts_series = pd.to_datetime(df['timestamp'])
        trade_dates = ts_series.dt.strftime('%Y-%m-%d')
        df['_trade_date'] = trade_dates
        
        # Calculate group IDs based on daily cumulative contract volume
        cum_vol = df.groupby('_trade_date')['volume'].cumsum()
        daily_vol_idx = (cum_vol - 1) // volume_size
        
        df['_bar_id'] = trade_dates + '_' + daily_vol_idx.astype(str)
        df['_pv'] = df['price'] * df['volume']

        agg_dict = {
            'timestamp': ['first', 'last'],
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum',
            '_pv': 'sum'
        }
        
        if 'bid' in df.columns:
            agg_dict['bid'] = 'last'
        if 'ask' in df.columns:
            agg_dict['ask'] = 'last'

        grouped = df.groupby('_bar_id', sort=False).agg(agg_dict)
        
        bars = pd.DataFrame()
        bars['open_time'] = grouped['timestamp']['first']
        bars['close_time'] = grouped['timestamp']['last']
        bars['timestamp'] = grouped['timestamp']['first']  # Open time matching Kiwoom HTS convention
        bars['open'] = grouped['price']['first']
        bars['high'] = grouped['price']['max']
        bars['low'] = grouped['price']['min']
        bars['close'] = grouped['price']['last']
        bars['volume'] = grouped['volume']['sum']
        bars['tick_count'] = df.groupby('_bar_id', sort=False).size().values
        
        vol_sum = bars['volume'].replace(0, np.nan)
        bars['vwap'] = grouped['_pv']['sum'] / vol_sum
        bars['vwap'] = bars['vwap'].fillna(bars['close'])

        if 'bid' in df.columns:
            bars['bid'] = grouped['bid']['last']
        if 'ask' in df.columns:
            bars['ask'] = grouped['ask']['last']

        df.drop(columns=['_trade_date', '_bar_id', '_pv'], inplace=True, errors='ignore')
        bars.sort_values('open_time', inplace=True)
        bars.reset_index(drop=True, inplace=True)
        return bars

    def resample_volume_from_db(self, volume_size: int, symbol: Optional[str] = None, 
                                start_time: Optional[str] = None, end_time: Optional[str] = None,
                                filter_outliers: bool = True, min_price: float = 50.0, max_price: float = 2000.0,
                                use_cache: bool = True) -> pd.DataFrame:
        """
        Resample into volume bars directly from DuckDB using intraday partitioned window cumulative volume
        with outlier filtering and Kiwoom HTS compliant intraday reset.
        Uses persistent Parquet cache for instantaneous repeated queries.
        """
        cache_path = None
        if use_cache:
            cache_path = self._get_cache_path("VOLUME", volume_size, symbol, start_time, end_time, filter_outliers, min_price, max_price)
            cached_df = self._load_from_cache(cache_path)
            if cached_df is not None:
                return cached_df

        if not self.data_engine:
            self.data_engine = get_data_engine()

        conditions = []
        params = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        # DuckDB high-speed SQL aggregation partitioned by trading day (일별 누적 거래량 주기 리셋 - 키움 HTS 100% 동일 규격)
        if filter_outliers:
            query = f"""
                WITH base_ticks AS (
                    SELECT rowid, timestamp, symbol, price, volume, bid, ask, CAST(timestamp AS DATE) AS trade_date
                    FROM ticks
                    {where_clause}
                ),
                daily_stats AS (
                    SELECT 
                        trade_date,
                        MEDIAN(price) AS day_median
                    FROM base_ticks
                    WHERE price > 0 AND volume >= 0
                    GROUP BY trade_date
                ),
                moving_daily AS (
                    SELECT 
                        trade_date,
                        day_median,
                        AVG(day_median) OVER (
                            ORDER BY trade_date 
                            ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
                            EXCLUDE CURRENT ROW
                        ) AS neighbor_avg
                    FROM daily_stats
                ),
                ranked_ticks AS (
                    SELECT 
                        t.rowid,
                        t.timestamp,
                        t.symbol,
                        t.price,
                        t.volume,
                        t.bid,
                        t.ask,
                        t.trade_date,
                        ((CAST(SUM(t.volume) OVER (PARTITION BY t.trade_date ORDER BY t.timestamp ASC, t.rowid ASC) AS BIGINT) - 1) // {volume_size}) AS bar_idx
                    FROM base_ticks t
                    JOIN moving_daily m ON t.trade_date = m.trade_date
                    WHERE t.price > 0 AND t.volume >= 0
                      AND ABS(t.price - m.day_median) <= m.day_median * 0.15
                      AND (m.neighbor_avg IS NULL OR ABS(m.day_median - m.neighbor_avg) <= m.neighbor_avg * 0.15)
                )
                SELECT 
                    MIN(timestamp) AS open_time,
                    MAX(timestamp) AS close_time,
                    MIN(timestamp) AS timestamp,
                    FIRST(price ORDER BY timestamp ASC, rowid ASC) AS open,
                    MAX(price) AS high,
                    MIN(price) AS low,
                    LAST(price ORDER BY timestamp ASC, rowid ASC) AS close,
                    SUM(volume) AS volume,
                    COUNT(*) AS tick_count,
                    CASE WHEN SUM(volume) > 0 THEN SUM(price * volume) / SUM(volume) ELSE LAST(price ORDER BY timestamp ASC, rowid ASC) END AS vwap,
                    LAST(bid ORDER BY timestamp ASC, rowid ASC) AS bid,
                    LAST(ask ORDER BY timestamp ASC, rowid ASC) AS ask
                FROM ranked_ticks
                GROUP BY trade_date, bar_idx
                ORDER BY open_time ASC
            """
        else:
            query = f"""
                WITH ranked_ticks AS (
                    SELECT 
                        rowid,
                        timestamp,
                        symbol,
                        price,
                        volume,
                        bid,
                        ask,
                        CAST(timestamp AS DATE) AS trade_date,
                        ((CAST(SUM(volume) OVER (PARTITION BY CAST(timestamp AS DATE) ORDER BY timestamp ASC, rowid ASC) AS BIGINT) - 1) // {volume_size}) AS bar_idx
                    FROM ticks
                    {where_clause}
                )
                SELECT 
                    MIN(timestamp) AS open_time,
                    MAX(timestamp) AS close_time,
                    MIN(timestamp) AS timestamp,
                    FIRST(price ORDER BY timestamp ASC, rowid ASC) AS open,
                    MAX(price) AS high,
                    MIN(price) AS low,
                    LAST(price ORDER BY timestamp ASC, rowid ASC) AS close,
                    SUM(volume) AS volume,
                    COUNT(*) AS tick_count,
                    CASE WHEN SUM(volume) > 0 THEN SUM(price * volume) / SUM(volume) ELSE LAST(price ORDER BY timestamp ASC, rowid ASC) END AS vwap,
                    LAST(bid ORDER BY timestamp ASC, rowid ASC) AS bid,
                    LAST(ask ORDER BY timestamp ASC, rowid ASC) AS ask
                FROM ranked_ticks
                GROUP BY trade_date, bar_idx
                ORDER BY open_time ASC
            """
        with self.data_engine.lock:
            bars_df = self.data_engine.conn.execute(query, params).fetchdf()

        if use_cache and cache_path and not bars_df.empty:
            self._save_to_cache(cache_path, bars_df)

        return bars_df

    def resample_volume_multiple(self, volume_sizes: List[int], 
                                 data_source: Union[pd.DataFrame, str] = "db", 
                                 symbol: Optional[str] = None,
                                 start_time: Optional[str] = None, 
                                 end_time: Optional[str] = None) -> Dict[int, pd.DataFrame]:
        """
        Generate multiple Volume bar DataFrames for a list of contract volume thresholds.
        Returns a dictionary: {volume_size: ohlcv_df}
        """
        results = {}
        for vs in volume_sizes:
            if isinstance(data_source, pd.DataFrame):
                df_filtered = data_source
                if start_time:
                    df_filtered = df_filtered[df_filtered['timestamp'] >= start_time]
                if end_time:
                    df_filtered = df_filtered[df_filtered['timestamp'] <= end_time]
                bars = self.resample_volume_from_df(df_filtered, volume_size=vs)
            else:
                bars = self.resample_volume_from_db(volume_size=vs, symbol=symbol, start_time=start_time, end_time=end_time)
            results[vs] = bars
        return results

    def resample_volume_range(self, start_vol: int = 500, end_vol: int = 2500, step_vol: int = 500,
                              data_source: Union[pd.DataFrame, str] = "db",
                              symbol: Optional[str] = None,
                              start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[int, pd.DataFrame]:
        """
        Generate multiple Volume bar DataFrames across a range [start_vol, end_vol] with step_vol.
        """
        start_vol = max(10, int(start_vol))
        end_vol = max(10, int(end_vol))
        step_vol = int(step_vol or 500)

        if start_vol > end_vol:
            start_vol, end_vol = end_vol, start_vol

        if start_vol == end_vol or step_vol <= 0:
            volume_sizes = [start_vol]
        else:
            volume_sizes = list(range(start_vol, end_vol + 1, step_vol))
            if not volume_sizes:
                volume_sizes = [start_vol]
            elif volume_sizes[-1] < end_vol:
                volume_sizes.append(end_vol)

        return self.resample_volume_multiple(volume_sizes=volume_sizes, data_source=data_source, symbol=symbol, start_time=start_time, end_time=end_time)

    def count_outliers(self, symbol: Optional[str] = None, 
                       start_time: Optional[str] = None, end_time: Optional[str] = None,
                       min_price: float = 50.0, max_price: float = 2000.0) -> int:
        """Count total outlier ticks in the given timeframe using KRX price limits & dynamic consistency."""
        if not self.data_engine:
            self.data_engine = get_data_engine()

        conditions = []
        params = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            WITH base_ticks AS (
                SELECT timestamp, price, volume, CAST(timestamp AS DATE) AS trade_date
                FROM ticks
                {where_clause}
            ),
            daily_stats AS (
                SELECT 
                    trade_date,
                    MEDIAN(price) AS day_median
                FROM base_ticks
                WHERE price > 0 AND volume >= 0
                GROUP BY trade_date
            ),
            moving_daily AS (
                SELECT 
                    trade_date,
                    day_median,
                    AVG(day_median) OVER (
                        ORDER BY trade_date 
                        ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
                        EXCLUDE CURRENT ROW
                    ) AS neighbor_avg
                FROM daily_stats
            )
            SELECT COUNT(*) AS outlier_count
            FROM base_ticks t
            LEFT JOIN moving_daily m ON t.trade_date = m.trade_date
            WHERE t.price <= 0 
               OR t.volume < 0 
               OR t.price IS NULL 
               OR m.day_median IS NULL
               OR ABS(t.price - m.day_median) > m.day_median * 0.15
               OR (m.neighbor_avg IS NOT NULL AND ABS(m.day_median - m.neighbor_avg) > m.neighbor_avg * 0.15)
        """
        try:
            with self.data_engine.lock:
                res = self.data_engine.conn.execute(query, params).fetchone()
                return int(res[0]) if res else 0
        except Exception:
            return 0

    @staticmethod
    def compute_chart_indicators(bars_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute technical indicators for Kiwoom chart overlay."""
        if bars_df.empty:
            return {}

        close = pd.to_numeric(bars_df['close'], errors='coerce')
        indicators = {}

        def _clean_list(series: pd.Series) -> List[Any]:
            return series.where(series.notnull(), None).tolist()

        # 1. Moving Averages (5, 10, 20, 60, 120)
        for p in [5, 10, 20, 60, 120]:
            ma = close.rolling(window=p, min_periods=1).mean().round(2)
            indicators[f'sma_{p}'] = _clean_list(ma)

        # 2. Bollinger Bands (20, 2.0)
        bb_mid = close.rolling(window=20, min_periods=1).mean().round(2)
        bb_std = close.rolling(window=20, min_periods=1).std().fillna(0)
        bb_upper = (bb_mid + 2.0 * bb_std).round(2)
        bb_lower = (bb_mid - 2.0 * bb_std).round(2)
        indicators['bb_upper'] = _clean_list(bb_upper)
        indicators['bb_mid'] = _clean_list(bb_mid)
        indicators['bb_lower'] = _clean_list(bb_lower)

        # 3. RSI (14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.fillna(50.0).round(2)
        indicators['rsi_14'] = _clean_list(rsi)

        # 4. MACD (12, 26, 9)
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = (ema_fast - ema_slow).round(2)
        macd_signal = macd_line.ewm(span=9, adjust=False).mean().round(2)
        macd_hist = (macd_line - macd_signal).round(2)
        indicators['macd_line'] = _clean_list(macd_line)
        indicators['macd_signal'] = _clean_list(macd_signal)
        indicators['macd_hist'] = _clean_list(macd_hist)

        return indicators


