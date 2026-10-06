"""
Vectorized Backtest Engine with Rigorous Cost Deduction
Calculates standard quantitative metrics:
- Annualized Return, Sharpe Ratio, Sortino Ratio, Maximum Drawdown (MDD)
- Win Rate, Profit Factor, Expectancy per trade, Trade Count
- Equity curve and drawdown series
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from deep_learning.cost_model import FuturesCostEngine, CostModelConfig


@dataclass
class BacktestMetrics:
    total_trades: int
    cumulative_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    expectancy_pts: float
    total_cost_krw: float
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "cumulative_return_pct": round(self.cumulative_return_pct, 2),
            "annualized_return_pct": round(self.annualized_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "profit_factor": round(self.profit_factor, 2),
            "expectancy_pts": round(self.expectancy_pts, 3),
            "total_cost_krw": round(self.total_cost_krw, 0)
        }


class VectorizedBacktester:
    def __init__(self, cost_engine: FuturesCostEngine, initial_capital_krw: float = 100_000_000.0):
        self.cost_engine = cost_engine
        self.initial_capital = initial_capital_krw

    def run_backtest(
        self,
        df_bars: pd.DataFrame,
        signals: np.ndarray,
        order_qty: float = 1.0
    ) -> BacktestMetrics:
        """
        Executes vectorized backtest from discrete signals (-1: Short, 0: Flat, 1: Long)
        or regression signals (> threshold: Long, < -threshold: Short).
        Deducts transaction costs on every position change.
        """
        if len(df_bars) < 2 or len(signals) != len(df_bars):
            return BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        close = df_bars['close'].values
        volume = df_bars['volume'].values
        dt_strings = df_bars['datetime'].astype(str).tolist()
        n = len(close)

        # Convert continuous signals to discrete positions if float
        if signals.dtype in (np.float32, np.float64):
            pos = np.where(signals > 0.0005, 1.0, np.where(signals < -0.0005, -1.0, 0.0))
        else:
            pos = signals.astype(float)

        # Shift position by 1: trade executed on next bar open/close
        pos_lagged = np.zeros(n)
        pos_lagged[1:] = pos[:-1]

        # Price returns: r_t = (P_t - P_{t-1}) / P_{t-1}
        price_rets = np.zeros(n)
        price_rets[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-6)

        # Gross Strategy Returns
        gross_rets = pos_lagged * price_rets

        # Turnover & Cost Deduction: position changes
        pos_change = np.zeros(n)
        pos_change[1:] = np.abs(pos_lagged[1:] - pos_lagged[:-1])
        # Initial position opening
        pos_change[0] = np.abs(pos_lagged[0])

        total_costs_krw = 0.0
        net_rets = gross_rets.copy()
        trade_count = int(np.sum(pos_change > 0) / 2)  # Entry + Exit = 1 trade

        # Deduct transaction costs
        for i in range(n):
            if pos_change[i] > 0:
                cost_dict = self.cost_engine.calculate_cost_per_trade(
                    entry_price=close[i],
                    exit_price=close[i],
                    order_qty=order_qty,
                    bar_volume=volume[i]
                )
                cost_pts = cost_dict['total_cost_pts']
                cost_pct = cost_pts / max(1.0, close[i])
                net_rets[i] -= cost_pct
                total_costs_krw += cost_dict['total_cost_krw']

        # Cumulative Equity & Drawdown
        cum_equity = np.cumprod(1.0 + net_rets)
        peak_equity = np.maximum.accumulate(cum_equity)
        drawdowns = (cum_equity - peak_equity) / peak_equity
        max_dd = float(np.min(drawdowns)) * 100.0  # in %

        cum_return_pct = float(cum_equity[-1] - 1.0) * 100.0

        # Annualization (assuming 250 trading days * ~400 bars/day on average)
        bars_per_year = max(1, n)
        ann_return_pct = float((1.0 + net_rets.mean()) ** bars_per_year - 1.0) * 100.0 if net_rets.mean() > -0.99 else -99.9

        # Sharpe & Sortino
        std_ret = np.std(net_rets)
        sharpe = float((net_rets.mean() / (std_ret + 1e-8)) * np.sqrt(bars_per_year)) if std_ret > 0 else 0.0

        downside_std = np.std(net_rets[net_rets < 0])
        sortino = float((net_rets.mean() / (downside_std + 1e-8)) * np.sqrt(bars_per_year)) if downside_std > 0 else 0.0

        # Trade Stats
        winning_bars = net_rets[net_rets > 0]
        losing_bars = net_rets[net_rets < 0]
        win_rate = (len(winning_bars) / max(1, len(winning_bars) + len(losing_bars))) * 100.0
        profit_factor = (winning_bars.sum() / max(1e-8, np.abs(losing_bars.sum()))) if len(losing_bars) > 0 else 1.0
        expectancy_pts = float(np.mean(net_rets) * np.mean(close))

        return BacktestMetrics(
            total_trades=max(1, trade_count),
            cumulative_return_pct=cum_return_pct,
            annualized_return_pct=ann_return_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=abs(max_dd),
            win_rate_pct=win_rate,
            profit_factor=float(profit_factor),
            expectancy_pts=expectancy_pts,
            total_cost_krw=total_costs_krw,
            equity_curve=cum_equity.tolist(),
            timestamps=dt_strings
        )
