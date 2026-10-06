"""
Labeling Engine: Triple-Barrier Method & Forward Returns
- Triple-Barrier Method (Upper Take-Profit / Lower Stop-Loss / Time Limit)
  Dynamic barriers scaled by ATR multiplier
- N-bar simple forward return regression label
"""
from typing import Tuple
import numpy as np
import pandas as pd


def apply_triple_barrier_labeling(
    df_bars: pd.DataFrame,
    atr_tp_mult: float = 2.0,
    atr_sl_mult: float = 1.5,
    max_holding_bars: int = 15,
    task_type: str = "multiclass"
) -> Tuple[pd.Series, pd.Series]:
    """
    Applies Lopez de Prado's Triple Barrier Method.
    Upper Barrier: price + (atr * atr_tp_mult) -> Label: +1 (Buy Profit)
    Lower Barrier: price - (atr * atr_sl_mult) -> Label: -1 (Sell Profit / Stop Hit)
    Vertical Barrier: max_holding_bars elapsed -> Label: 0 (Neutral)

    Returns:
      labels: Series (-1, 0, 1 for multiclass, or forward return for regression)
      ret_at_exit: Series (Actual return achieved at exit barrier)
    """
    close = df_bars['close'].values
    high = df_bars['high'].values
    low = df_bars['low'].values
    n = len(close)

    # Calculate ATR (14)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, high[0] - low[0])
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    labels = np.zeros(n, dtype=int)
    ret_at_exit = np.zeros(n, dtype=float)

    for i in range(n - 1):
        curr_price = close[i]
        curr_atr = atr[i]
        tp_price = curr_price + (curr_atr * atr_tp_mult)
        sl_price = curr_price - (curr_atr * atr_sl_mult)

        end_idx = min(n, i + max_holding_bars + 1)
        hit = 0
        exit_p = close[min(n - 1, i + max_holding_bars)]

        for j in range(i + 1, end_idx):
            # Check upper barrier hit first
            if high[j] >= tp_price:
                hit = 1
                exit_p = tp_price
                break
            # Check lower barrier hit
            elif low[j] <= sl_price:
                hit = -1
                exit_p = sl_price
                break

        labels[i] = hit
        ret_at_exit[i] = (exit_p - curr_price) / curr_price

    if task_type == "regression":
        target = pd.Series(ret_at_exit, index=df_bars.index)
    else:
        target = pd.Series(labels, index=df_bars.index)

    return target, pd.Series(ret_at_exit, index=df_bars.index)


def compute_n_bar_forward_returns(df_bars: pd.DataFrame, n_bars: int = 5) -> pd.Series:
    """Alternative simple n-bar forward return for benchmarking."""
    close = df_bars['close']
    return (close.shift(-n_bars) - close) / close
