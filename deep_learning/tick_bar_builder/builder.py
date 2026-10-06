"""
High-Performance Vectorized Tick Bar Builder
- Groups ticks into N-Tick bars
- Calculates OHLCV + VWAP, Buy/Sell Ratio, Bar Duration (seconds)
- Supports Parquet caching for fast multi-window reloads
"""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def build_tick_bars(df_ticks: pd.DataFrame, tick_size: int) -> pd.DataFrame:
    """
    Constructs tick bars of size `tick_size` from chronological tick data.
    Input df_ticks columns: ['datetime', 'price', 'volume', 'side']
    Returns DataFrame:
      ['datetime', 'open', 'high', 'low', 'close', 'volume',
       'vwap', 'buy_ratio', 'bar_duration_sec', 'tick_count']
    """
    if df_ticks is None or len(df_ticks) < tick_size:
        return pd.DataFrame()

    total_ticks = len(df_ticks)
    # Assign bar group ID by integer division
    bar_ids = np.arange(total_ticks) // tick_size
    # Truncate incomplete last bar if necessary
    valid_len = (total_ticks // tick_size) * tick_size
    if valid_len == 0:
        return pd.DataFrame()

    df_sub = df_ticks.iloc[:valid_len].copy()
    df_sub['bar_id'] = bar_ids[:valid_len]

    # Precalculate dollar volume for VWAP
    df_sub['dollar_vol'] = df_sub['price'] * df_sub['volume']
    df_sub['buy_vol'] = np.where(df_sub['side'] > 0, df_sub['volume'], 0.0)

    # Fast Groupby Aggregation
    grouped = df_sub.groupby('bar_id')
    agg_df = grouped.agg(
        datetime=('datetime', 'last'),
        open_time=('datetime', 'first'),
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('volume', 'sum'),
        dollar_vol=('dollar_vol', 'sum'),
        buy_vol=('buy_vol', 'sum'),
        tick_count=('price', 'count')
    ).reset_index(drop=True)

    # Derived bar metrics
    # VWAP
    vol_safe = np.where(agg_df['volume'] > 0, agg_df['volume'], 1.0)
    agg_df['vwap'] = agg_df['dollar_vol'] / vol_safe

    # Buy Ratio (Order Flow Imbalance proxy)
    agg_df['buy_ratio'] = agg_df['buy_vol'] / vol_safe

    # Duration in seconds
    agg_df['bar_duration_sec'] = (agg_df['datetime'] - agg_df['open_time']).dt.total_seconds().clip(lower=0.1)

    drop_cols = ['dollar_vol', 'buy_vol', 'open_time']
    agg_df.drop(columns=[c for c in drop_cols if c in agg_df.columns], inplace=True)

    return agg_df


def cache_tick_bars_parquet(
    tick_bars: pd.DataFrame,
    tick_size: int,
    cache_dir: str | Path,
    symbol: str = "KOSPI_F"
) -> Path:
    """Save processed tick bars to Parquet for near-instant reloading."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    out_file = cache_path / f"{symbol}_tickbar_{tick_size}.parquet"
    tick_bars.to_parquet(out_file, index=False, compression="snappy")
    return out_file


def load_cached_tick_bars(
    tick_size: int,
    cache_dir: str | Path,
    symbol: str = "KOSPI_F"
) -> Optional[pd.DataFrame]:
    """Load cached tick bars from Parquet if exists."""
    cache_file = Path(cache_dir) / f"{symbol}_tickbar_{tick_size}.parquet"
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            return None
    return None
