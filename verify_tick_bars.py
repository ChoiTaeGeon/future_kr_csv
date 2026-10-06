"""
Automated Verification Script for Tick Bar Conversion Integrity
Verifies the 5 Core Invariants between Raw Ticks and Resampled Tick Bars:
1. Tick Count Conservation (N_bars == floor(total_ticks / tick_size))
2. OHLC Logical Integrity (Low <= Open <= High, Low <= Close <= High, Max/Min match)
3. Volume Conservation (Sum of Raw Volume == Sum of Bar Volume)
4. VWAP Physical Bound (Low <= VWAP <= High across all bars)
5. Strict Temporal Order & Duration (Non-decreasing timestamps, duration >= 0)
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Root path configuration
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from deep_learning.data_loader import TickDataScanner
from deep_learning.tick_bar_builder import build_tick_bars


def verify_tick_bars_integrity(csv_file: str | Path, tick_size: int = 300) -> bool:
    target_path = Path(csv_file)
    if not target_path.exists():
        print(f"[!] Error: File not found: {target_path}")
        return False

    print("=" * 80)
    print(f" [*] KOSPI 200 Futures Tick Bar Conversion Integrity Verification")
    print(f"     Target File: {target_path.name}")
    print(f"     Tick Resolution Window: {tick_size} Ticks / Bar")
    print("=" * 80)

    # 1. Load and sanitize raw ticks
    scanner = TickDataScanner(data_dir=target_path.parent)
    date_str = scanner.extract_date_from_name(target_path.name) or "2026-01-01"
    enc, col_map = scanner.infer_columns_and_encoding(target_path)
    df_clean, stats = scanner.clean_daily_ticks(target_path, date_str, enc, col_map)

    raw_ticks_count = len(df_clean)
    if raw_ticks_count < tick_size:
        print(f"[!] Insufficient ticks: {raw_ticks_count} (Need at least {tick_size})")
        return False

    print(f"\n[Raw Tick Data Stats]")
    print(f" - Cleaned Ticks Count: {raw_ticks_count:,}")
    print(f" - Price Range: {df_clean['price'].min():.2f} ~ {df_clean['price'].max():.2f}")
    print(f" - Total Raw Volume: {df_clean['volume'].sum():,}")
    print(f" - First Tick: {df_clean['datetime'].iloc[0]} (Price: {df_clean['price'].iloc[0]})")
    print(f" - Last Tick:  {df_clean['datetime'].iloc[-1]} (Price: {df_clean['price'].iloc[-1]})")

    # 2. Build tick bars
    df_bars = build_tick_bars(df_clean, tick_size=tick_size)
    bars_count = len(df_bars)

    print(f"\n[Generated Tick Bars Stats]")
    print(f" - Total Bars Created: {bars_count:,}")

    # =========================================================================
    # 5 CORE INVARIANTS VERIFICATION
    # =========================================================================
    all_passed = True
    print("\n" + "-" * 80)
    print(" [Executing 5 Core Invariant Tests]")
    print("-" * 80)

    # INVARIANT 1: Tick Count Conservation
    expected_bars = raw_ticks_count // tick_size
    discarded_ticks = raw_ticks_count % tick_size
    inv1_passed = (bars_count == expected_bars)
    all_passed &= inv1_passed
    print(f" 1. [Tick Count Conservation]")
    print(f"    Expected: {expected_bars:,} bars | Actual: {bars_count:,} bars (Discarded remainder: {discarded_ticks} ticks)")
    print(f"    -> Status: {'[PASS]' if inv1_passed else '[FAIL]'}")

    # INVARIANT 2: OHLC Mathematical Consistency
    # Low <= Open <= High and Low <= Close <= High
    cond_ohlc = (df_bars['low'] <= df_bars['open']) & (df_bars['open'] <= df_bars['high']) & \
                (df_bars['low'] <= df_bars['close']) & (df_bars['close'] <= df_bars['high'])
    inv2_passed = bool(cond_ohlc.all())
    all_passed &= inv2_passed
    print(f" 2. [OHLC Mathematical Consistency]")
    print(f"    Low <= Open, Close <= High invariant satisfied across {cond_ohlc.sum():,} / {bars_count:,} bars")
    print(f"    -> Status: {'[PASS]' if inv2_passed else '[FAIL]'}")

    # INVARIANT 3: Volume Conservation
    valid_ticks_limit = expected_bars * tick_size
    raw_vol_sum = df_clean['volume'].iloc[:valid_ticks_limit].sum()
    bars_vol_sum = df_bars['volume'].sum()
    vol_diff = abs(raw_vol_sum - bars_vol_sum)
    inv3_passed = (vol_diff < 1e-4)
    all_passed &= inv3_passed
    print(f" 3. [Volume Conservation]")
    print(f"    Raw Volume Sum: {raw_vol_sum:,.1f} | Bar Volume Sum: {bars_vol_sum:,.1f} (Diff: {vol_diff})")
    print(f"    -> Status: {'[PASS]' if inv3_passed else '[FAIL]'}")

    # INVARIANT 4: VWAP Physical Bound (Low <= VWAP <= High)
    cond_vwap = (df_bars['low'] <= df_bars['vwap'] + 1e-4) & (df_bars['vwap'] <= df_bars['high'] + 1e-4)
    inv4_passed = bool(cond_vwap.all())
    all_passed &= inv4_passed
    print(f" 4. [VWAP Physical Bound Verification]")
    print(f"    Low <= VWAP <= High satisfied across {cond_vwap.sum():,} / {bars_count:,} bars")
    print(f"    -> Status: {'[PASS]' if inv4_passed else '[FAIL]'}")

    # INVARIANT 5: Temporal Order & Duration
    time_sorted = df_bars['datetime'].is_monotonic_increasing
    cond_dur = (df_bars['bar_duration_sec'] >= 0.0)
    inv5_passed = time_sorted and bool(cond_dur.all())
    all_passed &= inv5_passed
    avg_dur = df_bars['bar_duration_sec'].mean()
    print(f" 5. [Temporal Monotonicity & Duration]")
    print(f"    Chronological Monotonic: {time_sorted} | Duration >= 0: {cond_dur.all()} | Avg Bar Duration: {avg_dur:.2f}s")
    print(f"    -> Status: {'[PASS]' if inv5_passed else '[FAIL]'}")

    # =========================================================================
    # SIDE-BY-SIDE 1:1 SAMPLE AUDIT (First Bar Inspection)
    # =========================================================================
    print("\n" + "=" * 80)
    print(f" [Side-by-Side 1:1 Sample Audit: First Bar ({tick_size} Ticks Block)]")
    print("=" * 80)
    sample_ticks = df_clean.iloc[:tick_size]
    sample_bar = df_bars.iloc[0]

    raw_o = sample_ticks['price'].iloc[0]
    raw_h = sample_ticks['price'].max()
    raw_l = sample_ticks['price'].min()
    raw_c = sample_ticks['price'].iloc[-1]
    raw_v = sample_ticks['volume'].sum()
    raw_vwap = (sample_ticks['price'] * sample_ticks['volume']).sum() / max(1.0, raw_v)

    audit_table = [
        {"Field": "Open", "Raw Ticks Aggregation": f"{raw_o:.2f}", "Generated Bar": f"{sample_bar['open']:.2f}", "Match": raw_o == sample_bar['open']},
        {"Field": "High", "Raw Ticks Aggregation": f"{raw_h:.2f}", "Generated Bar": f"{sample_bar['high']:.2f}", "Match": raw_h == sample_bar['high']},
        {"Field": "Low", "Raw Ticks Aggregation": f"{raw_l:.2f}", "Generated Bar": f"{sample_bar['low']:.2f}", "Match": raw_l == sample_bar['low']},
        {"Field": "Close", "Raw Ticks Aggregation": f"{raw_c:.2f}", "Generated Bar": f"{sample_bar['close']:.2f}", "Match": raw_c == sample_bar['close']},
        {"Field": "Volume", "Raw Ticks Aggregation": f"{raw_v:,.0f}", "Generated Bar": f"{sample_bar['volume']:,.0f}", "Match": raw_v == sample_bar['volume']},
        {"Field": "VWAP", "Raw Ticks Aggregation": f"{raw_vwap:.3f}", "Generated Bar": f"{sample_bar['vwap']:.3f}", "Match": abs(raw_vwap - sample_bar['vwap']) < 1e-4},
        {"Field": "Tick Count", "Raw Ticks Aggregation": f"{tick_size}", "Generated Bar": f"{int(sample_bar['tick_count'])}", "Match": tick_size == sample_bar['tick_count']},
    ]

    df_audit = pd.DataFrame(audit_table)
    print(df_audit.to_string(index=False))

    print("\n" + "=" * 80)
    if all_passed:
        print(f" [PASS] 100% PERFECT! {tick_size}-Tick Bar conversion is mathematically & logically valid.")
    else:
        print(f" [FAIL] Verification detected discrepancies. Please review failed invariants above.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Tick Bar Conversion Integrity")
    parser.add_argument("--csv", type=str, default="csv/170207.csv", help="Path to sample CSV file")
    parser.add_argument("--tick-size", type=int, default=300, help="Tick size window to test (default: 300)")
    args = parser.parse_args()

    verify_tick_bars_integrity(csv_file=args.csv, tick_size=args.tick_size)
