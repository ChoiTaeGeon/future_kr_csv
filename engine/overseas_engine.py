"""
Overseas Futures CDT Engine
- Parses Kiwoom Global / Youngwoom .cdt binary files (64-byte, 80-byte, dynamic row formats, CSV fallback).
- Supports single file, multi-file lists, and folder batch ingestion.
- High-speed timestamp-based deduplication and chronologic sorting.
- Contract specification presets (NQ, MNQ, CL, GC, ES, MES, 6E, Custom).
- Quantitative backtesting integration using Backtester & SignalGenerator.
"""
import struct
import io
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime
import pandas as pd
import numpy as np

from config import BacktestConfig
from engine.strategy import SignalGenerator, StrategyCategory
from engine.backtester import Backtester, BacktestResult
from engine.resampler import Resampler


OVERSEAS_SYMBOLS = {
    "NQ": {"name": "E-mini Nasdaq 100", "tick_value": 0.25, "multiplier": 20.0, "currency": "USD"},
    "MNQ": {"name": "Micro E-mini Nasdaq 100", "tick_value": 0.25, "multiplier": 2.0, "currency": "USD"},
    "CL": {"name": "Crude Oil (WTI)", "tick_value": 0.01, "multiplier": 1000.0, "currency": "USD"},
    "GC": {"name": "Gold", "tick_value": 0.10, "multiplier": 100.0, "currency": "USD"},
    "ES": {"name": "E-mini S&P 500", "tick_value": 0.25, "multiplier": 50.0, "currency": "USD"},
    "MES": {"name": "Micro E-mini S&P 500", "tick_value": 0.25, "multiplier": 5.0, "currency": "USD"},
    "6E": {"name": "Euro FX", "tick_value": 0.00005, "multiplier": 125000.0, "currency": "USD"},
    "CUSTOM": {"name": "사용자 직접입력", "tick_value": 0.25, "multiplier": 20.0, "currency": "USD"}
}


def parse_cdt_bytes(file_content: bytes) -> pd.DataFrame:
    """
    Parses Kiwoom .cdt binary bytes into standard OHLCV DataFrame.
    Supports 64-byte (<8d>, h1=1), 80-byte (<10d>, h1=3), and dynamic row formats.
    """
    if len(file_content) >= 8:
        try:
            h1, h2 = struct.unpack('<2i', file_content[:8])
            data_len = len(file_content) - 8

            record_size = None
            fmt = None
            num_rows = 0

            # Mode 1: Known header type h1 == 1 (64 bytes / 8 doubles)
            if h1 == 1 and data_len % 64 == 0:
                record_size = 64
                fmt = '<8d'
                num_rows = data_len // 64
            # Mode 2: Known header type h1 == 3 (80 bytes / 10 doubles)
            elif h1 == 3 and data_len % 80 == 0:
                record_size = 80
                fmt = '<10d'
                num_rows = data_len // 80
            # Mode 3: Match with declared row count h2
            elif h2 > 0 and data_len % h2 == 0:
                calc_size = data_len // h2
                if calc_size in (64, 80) or (calc_size % 8 == 0):
                    record_size = calc_size
                    fmt = f'<{calc_size // 8}d'
                    num_rows = h2
            # Mode 4: Divisibility fallback
            elif data_len % 64 == 0:
                record_size = 64
                fmt = '<8d'
                num_rows = data_len // 64
            elif data_len % 80 == 0:
                record_size = 80
                fmt = '<10d'
                num_rows = data_len // 80

            if record_size and fmt and num_rows > 0:
                data = []
                for i in range(num_rows):
                    offset = 8 + i * record_size
                    unpacked = struct.unpack(fmt, file_content[offset : offset + record_size])

                    date_val = unpacked[0]
                    date_str = str(int(date_val))

                    # Normalize timestamp to standard YYYY-MM-DD HH:MM:SS
                    if len(date_str) == 14:
                        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {date_str[8:10]}:{date_str[10:12]}:{date_str[12:14]}"
                    elif len(date_str) == 8:
                        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 00:00:00"
                    else:
                        formatted_date = date_str

                    open_val = float(unpacked[1])
                    high_val = float(unpacked[2])
                    low_val = float(unpacked[3])
                    close_val = float(unpacked[4])
                    volume_val = float(unpacked[5]) if len(unpacked) > 5 else 0.0
                    trading_val = float(unpacked[6]) if len(unpacked) > 6 else 0.0
                    open_int_val = float(unpacked[7]) if len(unpacked) > 7 else 0.0

                    data.append({
                        'timestamp': formatted_date,
                        'open': open_val,
                        'high': high_val,
                        'low': low_val,
                        'close': close_val,
                        'volume': volume_val,
                        'trading_value': trading_val,
                        'open_interest': open_int_val
                    })

                df = pd.DataFrame(data)
                if not df.empty:
                    return df
        except Exception as e:
            print(f"[Overseas CDT Parser] Binary unpack failed: {e}. Trying text CSV...")

    # Text CSV fallback
    try:
        text = file_content.decode('utf-8', errors='replace')
        try:
            df_csv = pd.read_csv(io.StringIO(text))
            col_map = {}
            for c in df_csv.columns:
                cl = str(c).strip().replace(' ', '').replace('/', '')
                if cl in ['일자시간', '일자', '시간', 'timestamp', 'date', 'datetime']:
                    col_map[c] = 'timestamp'
                elif cl in ['시가', 'open']:
                    col_map[c] = 'open'
                elif cl in ['고가', 'high']:
                    col_map[c] = 'high'
                elif cl in ['저가', 'low']:
                    col_map[c] = 'low'
                elif cl in ['종가', 'close', '현재가']:
                    col_map[c] = 'close'
                elif cl in ['거래량', 'volume', 'qty']:
                    col_map[c] = 'volume'
                elif cl in ['거래대금', 'trading_value']:
                    col_map[c] = 'trading_value'
                elif cl in ['미결제약정', 'open_interest']:
                    col_map[c] = 'open_interest'

            df_csv = df_csv.rename(columns=col_map)
            req = ['timestamp', 'open', 'high', 'low', 'close']
            if all(r in df_csv.columns for r in req):
                if 'volume' not in df_csv.columns:
                    df_csv['volume'] = 1.0
                if 'trading_value' not in df_csv.columns:
                    df_csv['trading_value'] = 0.0
                if 'open_interest' not in df_csv.columns:
                    df_csv['open_interest'] = 0.0
                df_csv['timestamp'] = df_csv['timestamp'].astype(str).str.replace('/', '-').str.strip()
                return df_csv[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trading_value', 'open_interest']]
        except Exception:
            pass

        # Headerless CSV fallback (e.g. csv_nq with [Date(YYYYMMDD), Time(HHMMSS), Open, High, Low, Close, Volume])
        df_raw = pd.read_csv(io.StringIO(text), header=None)
        if df_raw.shape[1] >= 7:
            d_str = df_raw[0].astype(str).str.strip()
            t_str = df_raw[1].astype(str).str.strip().str.zfill(6)
            if len(d_str.iloc[0]) == 8 and d_str.iloc[0].isdigit():
                formatted_ts = (
                    d_str.str[:4] + '-' + d_str.str[4:6] + '-' + d_str.str[6:8] + ' ' +
                    t_str.str[:2] + ':' + t_str.str[2:4] + ':' + t_str.str[4:6]
                )
            else:
                formatted_ts = d_str + ' ' + t_str
            df_out = pd.DataFrame({
                'timestamp': formatted_ts,
                'open': df_raw[2].astype(float),
                'high': df_raw[3].astype(float),
                'low': df_raw[4].astype(float),
                'close': df_raw[5].astype(float),
                'volume': df_raw[6].astype(float),
                'trading_value': 0.0,
                'open_interest': 0.0
            })
            return df_out
        elif df_raw.shape[1] == 6:
            ts = df_raw[0].astype(str).str.replace('/', '-').str.strip()
            df_out = pd.DataFrame({
                'timestamp': ts,
                'open': df_raw[1].astype(float),
                'high': df_raw[2].astype(float),
                'low': df_raw[3].astype(float),
                'close': df_raw[4].astype(float),
                'volume': df_raw[5].astype(float),
                'trading_value': 0.0,
                'open_interest': 0.0
            })
            return df_out
    except Exception as e:
        print(f"[Overseas CDT/CSV Parser] CSV parse fallback failed: {e}")

    return pd.DataFrame()


class OverseasDataManager:
    """Singleton manager for loaded Overseas Futures CDT datasets."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OverseasDataManager, cls).__new__(cls)
            cls._instance._current_df = pd.DataFrame()
            cls._instance._metadata = {}
            cls._instance._loaded_sources = []
        return cls._instance

    @property
    def current_df(self) -> pd.DataFrame:
        return self._current_df

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def load_sources(self, target: Union[str, Path, List[Union[str, Path]]]) -> Dict[str, Any]:
        """
        Loads .cdt (or .csv) files from a folder, list of file paths, or semicolon-delimited string.
        Performs binary parsing, merges, deduplicates by timestamp, and caches the dataset.
        """
        file_paths: List[Path] = []

        if isinstance(target, (list, tuple)):
            for item in target:
                p = Path(str(item).strip())
                if p.is_file():
                    file_paths.append(p)
                elif p.is_dir():
                    file_paths.extend(sorted(list(p.glob("*.[cC][dD][tT]")) + list(p.glob("*.[cC][sS][vV]"))))
        else:
            t_str = str(target).strip()
            if ';' in t_str:
                parts = [p.strip() for p in t_str.split(';') if p.strip()]
                for part in parts:
                    p = Path(part)
                    if p.is_file():
                        file_paths.append(p)
            else:
                p = Path(t_str)
                if p.is_file():
                    file_paths.append(p)
                elif p.is_dir():
                    file_paths.extend(sorted(list(p.glob("*.[cC][dD][tT]")) + list(p.glob("*.[cC][sS][vV]"))))

        if not file_paths:
            return {
                "status": "error",
                "message": f"선택된 경로에서 .cdt 또는 .csv 파일을 찾을 수 없습니다: {target}"
            }

        dfs = []
        loaded_names = []
        for fp in file_paths:
            try:
                with open(fp, 'rb') as f:
                    content = f.read()
                df = parse_cdt_bytes(content)
                if not df.empty and len(df) > 0:
                    dfs.append(df)
                    loaded_names.append(fp.name)
            except Exception as e:
                print(f"[OverseasDataManager] Error loading {fp.name}: {e}")

        if not dfs:
            return {
                "status": "error",
                "message": "유효한 캔들 데이터를 파싱하지 못했습니다. 파일 손상 여부를 확인하세요."
            }

        # Sort multiple source DataFrames chronologically by their first timestamp
        dfs.sort(key=lambda d: str(d['timestamp'].iloc[0]) if not d.empty else '')

        # Merge and preserve natural candle sequence without scrambling tick timestamps
        merged = pd.concat(dfs, ignore_index=True)
        merged['timestamp'] = merged['timestamp'].astype(str).str.replace('/', '-').str.strip()
        # Deduplicate only true duplicate rows across overlapping file boundaries (matching timestamp AND OHLCV)
        merged = merged.drop_duplicates(subset=['timestamp', 'open', 'high', 'low', 'close', 'volume'], keep='first').reset_index(drop=True)

        # Ensure numeric columns
        for c in ['open', 'high', 'low', 'close', 'volume', 'trading_value', 'open_interest']:
            merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0.0)

        # Synthesize VWAP if volume exists
        if 'vwap' not in merged.columns:
            vol_sum = merged['volume'].replace(0, np.nan)
            merged['vwap'] = np.where(merged['volume'] > 0, (merged['close'] * merged['volume']) / vol_sum, merged['close'])
            merged['vwap'] = merged['vwap'].fillna(merged['close'])

        self._current_df = merged
        self._loaded_sources = [str(fp) for fp in file_paths]

        # Extract available years, months, dates
        ts_str = merged['timestamp'].astype(str)
        years = sorted(list(set(ts_str.str.slice(0, 4).dropna())))
        months = sorted(list(set(ts_str.str.slice(0, 7).dropna())))
        dates = sorted(list(set(ts_str.str.slice(0, 10).dropna())))

        meta = {
            "status": "success",
            "total_bars": len(merged),
            "files_count": len(loaded_names),
            "file_names": loaded_names,
            "date_range": {
                "start": str(merged['timestamp'].iloc[0]),
                "end": str(merged['timestamp'].iloc[-1])
            },
            "available_years": years,
            "available_months": months,
            "available_dates": dates,
            "latest_date": dates[-1] if dates else None,
            "sample_price": round(float(merged['close'].iloc[-1]), 2) if not merged.empty else 0.0
        }
        self._metadata = meta
        return meta

    def filter_period(self, period_mode: str = "ALL", target_date: Optional[str] = None,
                      target_year: Optional[str] = None, target_month: Optional[str] = None,
                      start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Filters the cached DataFrame by user-selected period."""
        if self._current_df.empty:
            return pd.DataFrame()

        df = self._current_df.copy()
        ts_str = df['timestamp'].astype(str)

        mode = str(period_mode).upper()
        if mode == 'YEAR' and target_year and target_year != 'ALL':
            df = df[ts_str.str.startswith(str(target_year))]
        elif mode == 'MONTH' and target_month and target_month != 'ALL':
            df = df[ts_str.str.startswith(str(target_month))]
        elif mode == 'DAY' and target_date and target_date != 'ALL':
            df = df[ts_str.str.startswith(str(target_date)[:10])]
        elif mode == 'CUSTOM':
            if start_date:
                sd = str(start_date)[:10]
                df = df[ts_str >= sd]
            if end_date:
                ed = str(end_date)[:10] + " 23:59:59"
                df = df[ts_str <= ed]
        elif mode == 'ALL' or target_date == 'ALL':
            pass

        return df.reset_index(drop=True)

    def get_chart_data(self, period_mode: str = "ALL", target_date: Optional[str] = None,
                       target_year: Optional[str] = None, target_month: Optional[str] = None,
                       start_date: Optional[str] = None, end_date: Optional[str] = None,
                       filter_outliers: bool = True) -> Dict[str, Any]:
        """Prepares chart data for Plotly with technical indicators."""
        df_bars = self.filter_period(
            period_mode=period_mode, target_date=target_date, target_year=target_year,
            target_month=target_month, start_date=start_date, end_date=end_date
        )

        if df_bars.empty:
            return {
                "status": "error",
                "message": "선택된 기간 조건에 해당하는 캔들 데이터가 없습니다."
            }

        outliers_count = 0
        if filter_outliers and len(df_bars) > 10:
            p_median = df_bars['close'].median()
            valid_mask = (df_bars['close'] > p_median * 0.2) & (df_bars['close'] < p_median * 5.0)
            outliers_count = int((~valid_mask).sum())
            if outliers_count > 0:
                df_bars = df_bars[valid_mask].reset_index(drop=True)

        indicators_data = Resampler.compute_chart_indicators(df_bars)

        max_idx = int(np.argmax(df_bars['high'].values))
        min_idx = int(np.argmin(df_bars['low'].values))

        chart_data = {
            "timestamps": [str(ts) for ts in df_bars['timestamp']],
            "open": df_bars['open'].round(2).tolist(),
            "high": df_bars['high'].round(2).tolist(),
            "low": df_bars['low'].round(2).tolist(),
            "close": df_bars['close'].round(2).tolist(),
            "volume": df_bars['volume'].astype(int).tolist(),
            "vwap": df_bars['vwap'].round(2).tolist() if 'vwap' in df_bars else None,
            "indicators": indicators_data,
            "max_info": {
                "price": round(float(df_bars['high'].iloc[max_idx]), 2),
                "timestamp": str(df_bars['timestamp'].iloc[max_idx]),
                "index": max_idx
            },
            "min_info": {
                "price": round(float(df_bars['low'].iloc[min_idx]), 2),
                "timestamp": str(df_bars['timestamp'].iloc[min_idx]),
                "index": min_idx
            }
        }

        first_ts = str(df_bars['timestamp'].iloc[0])[:10]
        last_ts = str(df_bars['timestamp'].iloc[-1])[:10]
        date_desc = f"{first_ts} ~ {last_ts}" if first_ts != last_ts else first_ts

        return {
            "status": "success",
            "total_bars": len(df_bars),
            "date_desc": date_desc,
            "outliers_excluded": outliers_count,
            "chart_data": chart_data
        }

    def run_backtest(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Executes quantitative backtest across strategies on overseas futures CDT data."""
        if self._current_df.empty:
            return {
                "status": "error",
                "message": "로드된 해외선물 CDT 데이터가 없습니다. 먼저 CDT 파일을 로드해주세요."
            }

        symbol = str(body.get('symbol', 'NQ')).upper()
        sym_info = OVERSEAS_SYMBOLS.get(symbol, OVERSEAS_SYMBOLS["CUSTOM"])

        # Capital handling: Default to 50,000 USD (or 5,000 for Micro symbols).
        # If user passed KRW unit (>= 1,000,000), convert to USD using 1,350 KRW/USD exchange rate.
        default_cap = 5000.0 if symbol in ['MNQ', 'MES'] else 50000.0
        raw_cap = float(body.get('initial_capital', default_cap))
        if raw_cap >= 1_000_000.0:
            init_capital = round(raw_cap / 1350.0, 2)
        else:
            init_capital = raw_cap if raw_cap > 0 else default_cap

        commission_rate = float(body.get('commission_rate', 0.00003))
        slippage_ticks = float(body.get('slippage_ticks', 1.0))
        tick_val = float(body.get('tick_value', sym_info['tick_value']))
        mult = float(body.get('multiplier', sym_info['multiplier']))
        eod_close_raw = str(body.get('eod_close_time', 'NONE')).strip()
        eod_close = 'NONE' if any(k in eod_close_raw.upper() for k in ['NONE', '24H', 'OFF', 'FALSE', '보유', '오버나이트']) else eod_close_raw
        allow_short = bool(body.get('allow_short', True))

        cfg = BacktestConfig(
            initial_capital=init_capital,
            commission_rate=commission_rate,
            slippage_ticks=slippage_ticks,
            tick_value=tick_val,
            multiplier=mult,
            eod_close_time=eod_close,
            allow_short=allow_short
        )

        df_bars = self.filter_period(
            period_mode=body.get('period_mode', 'ALL'),
            target_date=body.get('target_date'),
            target_year=body.get('target_year'),
            target_month=body.get('target_month'),
            start_date=body.get('start_date'),
            end_date=body.get('end_date')
        )

        if df_bars.empty or len(df_bars) < 5:
            return {
                "status": "error",
                "message": "선택한 기간의 캔들 수가 부족하여 백테스트를 수행할 수 없습니다 (최소 5봉 필요)."
            }

        cat_str = str(body.get('category', 'ALL')).upper()
        cat_map = {
            'TREND': StrategyCategory.TREND,
            'MOMENTUM': StrategyCategory.MOMENTUM,
            'VOLATILITY': StrategyCategory.VOLATILITY,
            'CONTRARIAN': StrategyCategory.CONTRARIAN,
            'ALL': StrategyCategory.ALL
        }
        selected_category = cat_map.get(cat_str, StrategyCategory.ALL)

        selected_strategies = body.get('selected_strategies')
        if not selected_strategies and body.get('strategy'):
            selected_strategies = [body.get('strategy')]
        elif not selected_strategies and body.get('strategies'):
            selected_strategies = body.get('strategies')

        backtester = Backtester(config=cfg)
        summary_df, detailed_results = backtester.run_multi_tick_backtest(
            tick_bars_dict={1: df_bars},
            category=selected_category,
            selected_strategies=selected_strategies
        )

        if summary_df.empty:
            return {
                "status": "error",
                "message": "선택한 전략 및 기간에서 거래 신호가 발생하지 않았습니다."
            }

        summary_df['Tick Size'] = symbol

        all_results_dict = {}
        strat_map = detailed_results.get(1, {})
        pnl_map = {}
        for s_name, res_obj in strat_map.items():
            key = f"{symbol}_{s_name}"
            net_pnl_cash = float(res_obj.equity_curve['net_pnl_cash'].sum()) if not res_obj.equity_curve.empty else 0.0
            net_pnl_krw = net_pnl_cash * 1350.0
            pnl_krw_str = f"약 {net_pnl_krw/100000000:.2f}억 원" if abs(net_pnl_krw) >= 100000000 else f"약 {int(net_pnl_krw/10000):,}만 원"
            pnl_formatted = f"+${net_pnl_cash:,.0f}" if net_pnl_cash >= 0 else f"-${abs(net_pnl_cash):,.0f}"
            pnl_map[s_name] = pnl_formatted

            all_results_dict[key] = {
                "strategy_name": res_obj.strategy_name,
                "tick_size": symbol,
                "total_return_pct": res_obj.total_return_pct,
                "net_profit_cash": round(net_pnl_cash, 2),
                "net_profit_krw": round(net_pnl_krw, 0),
                "net_profit_formatted": pnl_formatted,
                "net_profit_krw_str": pnl_krw_str,
                "currency": "USD",
                "initial_capital": init_capital,
                "sharpe_ratio": res_obj.sharpe_ratio,
                "sortino_ratio": res_obj.sortino_ratio,
                "max_drawdown_pct": res_obj.max_drawdown_pct,
                "win_rate": res_obj.win_rate,
                "profit_factor": res_obj.profit_factor,
                "total_trades": res_obj.total_trades,
                "constituent_strategies": res_obj.constituent_strategies,
                "yearly_breakdown": res_obj.yearly_breakdown.to_dict(orient='records') if not res_obj.yearly_breakdown.empty else [],
                "monthly_breakdown": res_obj.monthly_breakdown.to_dict(orient='records') if not res_obj.monthly_breakdown.empty else [],
                "trades": res_obj.trade_log.to_dict(orient='records') if not res_obj.trade_log.empty else [],
                "chart_data": {
                    "timestamps": [str(ts) for ts in df_bars['timestamp']],
                    "open": df_bars['open'].round(2).tolist(),
                    "high": df_bars['high'].round(2).tolist(),
                    "low": df_bars['low'].round(2).tolist(),
                    "close": df_bars['close'].round(2).tolist(),
                    "volume": df_bars['volume'].astype(int).tolist(),
                    "vwap": df_bars['vwap'].round(2).tolist() if 'vwap' in df_bars else None,
                    "buy_signals": {
                        "x": [str(r['entry_time']) for _, r in res_obj.trade_log[res_obj.trade_log['side'] == 'LONG'].iterrows()] if not res_obj.trade_log.empty else [],
                        "y": [float(r['entry_price']) for _, r in res_obj.trade_log[res_obj.trade_log['side'] == 'LONG'].iterrows()] if not res_obj.trade_log.empty else []
                    },
                    "sell_signals": {
                        "x": [str(r['entry_time']) for _, r in res_obj.trade_log[res_obj.trade_log['side'] == 'SHORT'].iterrows()] if not res_obj.trade_log.empty else [],
                        "y": [float(r['entry_price']) for _, r in res_obj.trade_log[res_obj.trade_log['side'] == 'SHORT'].iterrows()] if not res_obj.trade_log.empty else []
                    },
                    "equity_curve": res_obj.equity_curve['equity_pct'].tolist() if not res_obj.equity_curve.empty else []
                }
            }

        summary_df['Net Profit ($)'] = summary_df['Strategy'].map(pnl_map).fillna('$0')

        top_strat_name = str(summary_df.iloc[0]['Strategy'])
        top_res_obj = strat_map.get(top_strat_name, list(strat_map.values())[0])
        top_pnl_cash = float(top_res_obj.equity_curve['net_pnl_cash'].sum()) if not top_res_obj.equity_curve.empty else 0.0
        top_pnl_krw = top_pnl_cash * 1350.0
        top_krw_str = f"약 {top_pnl_krw/100000000:.2f}억 원" if abs(top_pnl_krw) >= 100000000 else f"약 {int(top_pnl_krw/10000):,}만 원"
        top_pnl_fmt = f"+${top_pnl_cash:,.0f}" if top_pnl_cash >= 0 else f"-${abs(top_pnl_cash):,.0f}"

        # Auto-save trade log to output/ directory
        try:
            out_dir = Path("output")
            out_dir.mkdir(parents=True, exist_ok=True)
            if not top_res_obj.trade_log.empty:
                csv_path = out_dir / f"trade_log_{symbol}_{top_strat_name}.csv"
                top_res_obj.trade_log.to_csv(csv_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"[OverseasEngine] Could not auto-save trade log CSV: {e}")

        # Auto-generate Overseas PDF Report
        try:
            from config import REPORTS_DIR
            from engine.reporter import ReportGenerator
            pdf_path = REPORTS_DIR / "Quant_Overseas_Backtest_Report.pdf"
            top_3 = []
            for i in range(min(3, len(summary_df))):
                r = summary_df.iloc[i]
                sname = str(r['Strategy'])
                if sname in strat_map:
                    top_3.append(strat_map[sname])

            sym_desc = sym_info.get('name', symbol)
            ReportGenerator.create_pdf_report(
                summary_df=summary_df,
                best_results=top_3,
                output_pdf_path=str(pdf_path),
                report_title=f"Overseas Futures ({symbol} - {sym_desc}) Backtest Report",
                market_desc=f"Overseas Futures: {sym_desc} ({symbol}) | Initial Capital: ${init_capital:,.0f} USD | Total: {len(df_bars):,} Bars",
                currency="USD"
            )
        except Exception as e:
            print(f"[OverseasEngine] Could not auto-generate PDF report: {e}")

        top_result = {
            "strategy_name": top_res_obj.strategy_name,
            "tick_size": symbol,
            "total_return_pct": top_res_obj.total_return_pct,
            "net_profit_cash": round(top_pnl_cash, 2),
            "net_profit_krw": round(top_pnl_krw, 0),
            "net_profit_formatted": top_pnl_fmt,
            "net_profit_krw_str": top_krw_str,
            "currency": "USD",
            "initial_capital": init_capital,
            "sharpe_ratio": top_res_obj.sharpe_ratio,
            "sortino_ratio": top_res_obj.sortino_ratio,
            "max_drawdown_pct": top_res_obj.max_drawdown_pct,
            "win_rate": top_res_obj.win_rate,
            "profit_factor": top_res_obj.profit_factor,
            "total_trades": top_res_obj.total_trades,
            "constituent_strategies": top_res_obj.constituent_strategies,
            "yearly_breakdown": top_res_obj.yearly_breakdown.to_dict(orient='records') if not top_res_obj.yearly_breakdown.empty else [],
            "monthly_breakdown": top_res_obj.monthly_breakdown.to_dict(orient='records') if not top_res_obj.monthly_breakdown.empty else [],
            "trades": top_res_obj.trade_log.to_dict(orient='records') if not top_res_obj.trade_log.empty else []
        }

        top_chart = all_results_dict.get(f"{symbol}_{top_strat_name}", {}).get("chart_data", {})

        return {
            "status": "success",
            "symbol": symbol,
            "currency": "USD",
            "initial_capital": init_capital,
            "resample_type": "CDT_BARS",
            "unit_label": "CDT Bars",
            "top_result": top_result,
            "summary": summary_df.to_dict(orient='records'),
            "all_results": all_results_dict,
            "chart_data": top_chart
        }


def get_overseas_manager() -> OverseasDataManager:
    """Returns the singleton OverseasDataManager instance."""
    return OverseasDataManager()
