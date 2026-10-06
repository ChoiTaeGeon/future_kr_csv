"""
Main Pipeline Runner for High-Performance Tick Data Pipeline & Multi-Tick Backtest
- Automated CSV ingestion into DuckDB
- Dynamic N-Tick bar resampling
- 30 Indicator calculations & Strategy backtesting
- Interactive Plotly visualization & Automated ReportLab PDF reporting
"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from config import RAW_CSV_DIR, DB_PATH, REPORTS_DIR, CHARTS_DIR, BacktestConfig, ResampleConfig
from engine.data_engine import DataEngine
from engine.resampler import Resampler
from engine.strategy import StrategyCategory, SignalGenerator
from engine.backtester import Backtester
from engine.reporter import Visualizer, ReportGenerator
from engine.tracker import tracker
from engine.server import start_server


def run_pipeline(
    csv_folder: str = str(RAW_CSV_DIR),
    db_path: str = str(DB_PATH),
    start_tick: int = 1000,
    end_tick: int = 5000,
    step_tick: int = 1000,
    symbol: str = "KOSPI_F",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    print("=" * 75)
    print(" [1/5] Initializing Data Pipeline & DuckDB Storage Layer")
    print("=" * 75)
    
    data_engine = DataEngine(db_path=db_path)
    sync_res = data_engine.scan_and_sync(csv_folder=csv_folder, symbol_default=symbol)
    print(f"[+] Sync Status: {sync_res['status']}")
    print(f"[+] Synced Files: {sync_res['synced_files']}, Rows Added: {sync_res['total_rows_added']:,}")
    print(f"[+] Total Ticks in DuckDB: {data_engine.get_total_ticks():,}")
    
    min_ts, max_ts = data_engine.get_date_range(symbol=symbol)
    print(f"[+] Total Database Time Range: {min_ts} ~ {max_ts}")
    if start_date or end_date:
        print(f"[+] Active Backtest Period Filter: {start_date or 'Beginning'} ~ {end_date or 'Latest'}\n")
    else:
        print("")

    print("=" * 75)
    print(f" [2/5] Dynamic Tick Resampling: {start_tick} to {end_tick} Ticks (Step: {step_tick})")
    print("=" * 75)
    resampler = Resampler(data_engine=data_engine)
    tick_bars = resampler.resample_range(
        data_source="db",
        start_tick=start_tick,
        end_tick=end_tick,
        step_tick=step_tick,
        symbol=symbol,
        start_time=start_date,
        end_time=end_date
    )
    for ts, bars in tick_bars.items():
        print(f"[+] Resampled {ts:5d}-Tick Bars: {len(bars):,d} bars generated (OHLCV + VWAP)")

    print("\n" + "=" * 75)
    print(" [3/5] Computing 30 Indicators & Executing Vectorized Backtest")
    print("=" * 75)
    config = BacktestConfig(
        initial_capital=100_000_000.0,
        commission_rate=0.00003,
        slippage_ticks=1.0,
        tick_value=0.05,
        multiplier=250_000.0,
        eod_close_time="15:35:00"
    )
    backtester = Backtester(config=config)
    
    print("[*] Running multi-tick grid test across 30 strategies and category ensembles...")
    summary_df, detailed_results = backtester.run_multi_tick_backtest(
        tick_bars_dict=tick_bars,
        category=StrategyCategory.ALL
    )

    print("\n" + "-" * 75)
    print(" [Top 10 Backtest Results by Sharpe Ratio]")
    print("-" * 75)
    top_cols = ['Tick Size', 'Strategy', 'Cumulative Return (%)', 'Sharpe Ratio', 'MDD (%)', 'Win Rate (%)', 'Profit Factor', 'Trades']
    print(summary_df[top_cols].head(10).to_string(index=False))

    print("\n" + "=" * 75)
    print(" [4/5] Interactive Plotly Chart Generation")
    print("=" * 75)
    # Pick top strategy result
    best_row = summary_df.iloc[0]
    best_tick = int(best_row['Tick Size'])
    best_strat = str(best_row['Strategy'])
    best_result = detailed_results[best_tick][best_strat]
    best_bars = tick_bars[best_tick]

    html_path = CHARTS_DIR / f"interactive_chart_{best_strat}_{best_tick}T.html"
    Visualizer.plot_strategy_backtest(best_bars, best_result, output_html_path=str(html_path))
    print(f"[+] Saved Interactive Plotly Chart -> {html_path}")

    print("\n" + "=" * 75)
    print(" [5/5] Generating Executive PDF Backtest Report")
    print("=" * 75)
    pdf_path = REPORTS_DIR / "Quant_DayTrading_Backtest_Report.pdf"
    
    # Collect top 3 best results
    top_results = []
    for idx in range(min(3, len(summary_df))):
        row = summary_df.iloc[idx]
        t = int(row['Tick Size'])
        s = str(row['Strategy'])
        top_results.append(detailed_results[t][s])

    ReportGenerator.create_pdf_report(
        summary_df=summary_df,
        best_results=top_results,
        output_pdf_path=str(pdf_path),
        report_title="High-Frequency Tick Data Strategy Backtest Report"
    )
    print(f"[+] Saved PDF Report -> {pdf_path}")

    print("\n" + "=" * 75)
    print(" [COMPLETE] All Data Ingestion, Resampling, Backtests & Reports Done!")
    print("=" * 75)
    return summary_df, detailed_results


if __name__ == "__main__":
    csv_folder = str(RAW_CSV_DIR)
    port = 5000
    start_date = None
    end_date = None

    for arg in sys.argv:
        if arg.startswith("--csv-folder="):
            csv_folder = arg.split("=", 1)[1]
        elif arg.startswith("-f="):
            csv_folder = arg.split("=", 1)[1]
        elif arg.startswith("--year="):
            yr = arg.split("=", 1)[1]
            start_date = f"{yr}-01-01 00:00:00"
            end_date = f"{yr}-12-31 23:59:59"
        elif arg.startswith("--month="):
            mo = arg.split("=", 1)[1]
            start_date = f"{mo}-01 00:00:00"
            end_date = f"{mo}-31 23:59:59"
        elif arg.startswith("--start-date="):
            start_date = arg.split("=", 1)[1]
        elif arg.startswith("--end-date="):
            end_date = arg.split("=", 1)[1]
        elif arg.startswith("--port="):
            port = int(arg.split("=")[1])

    if "--cli" in sys.argv:
        run_pipeline(csv_folder=csv_folder, start_date=start_date, end_date=end_date)
    else:
        start_server(port=port, auto_open=True)



