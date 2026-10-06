"""
KOSPI 200 Futures Transaction Cost & Slippage Engine
- Commission: Round-trip realistic brokerage rate (default ~0.003% x 2)
- Slippage Modes:
  1. Fixed Tick Slippage (e.g., 1 tick = 0.05pt)
  2. Volume Impact Slippage (order size / market volume)
  3. Combined Fixed + Volume Impact
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class CostModelConfig:
    commission_rate: float = 0.00003     # Per-side brokerage rate
    slippage_mode: str = "fixed_and_volume"  # 'fixed', 'volume', 'fixed_and_volume'
    fixed_slippage_tick: float = 0.05    # 1 tick = 0.05pt in KOSPI200 Futures
    tick_value_krw: float = 250000.0     # 250,000 KRW per point
    volume_impact_coef: float = 0.00001


class FuturesCostEngine:
    def __init__(self, config: CostModelConfig):
        self.cfg = config

    def calculate_cost_per_trade(
        self,
        entry_price: float,
        exit_price: float,
        order_qty: float = 1.0,
        bar_volume: float = 1000.0
    ) -> Dict[str, float]:
        """
        Calculates round-trip costs in points (pt) and KRW.
        """
        # Commission (Round-trip)
        comm_pts = (entry_price + exit_price) * self.cfg.commission_rate

        # Slippage calculation
        slippage_pts = 0.0
        mode = self.cfg.slippage_mode

        if mode in ("fixed", "fixed_and_volume"):
            # Fixed 1-tick slippage for entry + exit
            slippage_pts += self.cfg.fixed_slippage_tick * 2.0

        if mode in ("volume", "fixed_and_volume"):
            # Price impact proportional to trading volume
            impact = (order_qty / max(1.0, bar_volume)) * self.cfg.volume_impact_coef * entry_price
            slippage_pts += impact * 2.0

        total_pts = comm_pts + slippage_pts
        total_krw = total_pts * self.cfg.tick_value_krw * order_qty

        return {
            "commission_pts": comm_pts,
            "slippage_pts": slippage_pts,
            "total_cost_pts": total_pts,
            "total_cost_krw": total_krw
        }

    def deduct_costs_from_returns(
        self,
        gross_return: float,
        price: float,
        order_qty: float = 1.0,
        bar_volume: float = 1000.0
    ) -> float:
        """
        Deducts transaction costs directly from trade return percentage.
        """
        costs = self.calculate_cost_per_trade(price, price * (1 + gross_return), order_qty, bar_volume)
        cost_pct = costs["total_cost_pts"] / max(1.0, price)
        return gross_return - cost_pct
