"""
High-Performance Vectorized & Event-Aware Backtesting Engine
- Supports 30 indicators, category ensembles, slippage, commission, and EOD intraday liquidation.
- Multi-tick size batch backtesting and performance matrix generation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from config import BacktestConfig
from engine.strategy import SignalGenerator, StrategyCategory


@dataclass
class BacktestResult:
    """Encapsulates backtest metrics and timeseries data."""
    strategy_name: str
    tick_size: int
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    annualized_return_pct: float
    annualized_vol_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    avg_trade_pct: float
    payoff_ratio: float
    equity_curve: pd.DataFrame = field(repr=False)
    trade_log: pd.DataFrame = field(repr=False)
    yearly_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    monthly_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    constituent_strategies: List[str] = field(default_factory=list)


class Backtester:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run_single(self, df_bars: pd.DataFrame, signal: pd.Series, strategy_name: str = "Strategy", 
                   tick_size: int = 1000, constituent_strategies: Optional[List[str]] = None) -> BacktestResult:
        """
        Execute backtest for a single bar series and signal series.
        """
        if df_bars.empty or len(df_bars) < 2:
            return self._empty_result(strategy_name, tick_size)

        df = df_bars.copy().reset_index(drop=True)
        signal = signal.copy().reset_index(drop=True)
        
        # Datetime handling for EOD liquidation
        ts_series = pd.to_datetime(df['timestamp'])
        df['time_str'] = ts_series.dt.strftime("%H:%M:%S")
        df['date_str'] = ts_series.dt.strftime("%Y-%m-%d")
        df['year_str'] = ts_series.dt.strftime("%Y")
        df['month_str'] = ts_series.dt.strftime("%Y-%m")

        # 1. Position tracking (Next-bar execution to prevent lookahead bias)
        # Position at bar t is based on signal generated at bar t-1
        raw_position = signal.shift(1).fillna(0).astype(int)
        
        if not self.config.allow_short:
            raw_position = raw_position.clip(lower=0)

        # 2. Intraday EOD Forced Liquidation (Optional: disabled when eod_close_time is 'NONE', '24H', or 'OFF')
        eod_cfg = str(self.config.eod_close_time or '').upper().strip()
        if eod_cfg and eod_cfg not in ["NONE", "24H", "OFF", "FALSE", ""]:
            is_eod = (df['time_str'] >= self.config.eod_close_time) | (df['date_str'] != df['date_str'].shift(-1))
            position = np.where(is_eod, 0, raw_position)
        else:
            position = raw_position
        position = pd.Series(position, index=df.index)

        # 3. Trade detection and transaction cost calculation
        pos_diff = position.diff().fillna(position.iloc[0]).abs()
        slippage_cost = pos_diff * (self.config.slippage_ticks * self.config.tick_value)
        price_diff = df['close'].diff().fillna(0)
        gross_pnl_pts = position.shift(1).fillna(0) * price_diff
        trade_value = pos_diff * df['close']
        commission_pts = trade_value * self.config.commission_rate
        
        # Net PnL
        net_pnl_pts = gross_pnl_pts - slippage_cost - commission_pts
        net_pnl_cash = net_pnl_pts * self.config.multiplier
        
        # Equity Curve
        cum_pnl_cash = net_pnl_cash.cumsum()
        equity = self.config.initial_capital + cum_pnl_cash
        equity_pct = (equity / self.config.initial_capital - 1.0) * 100.0
        
        # Drawdown calculation
        peak = equity.cummax()
        drawdown_cash = equity - peak
        drawdown_pct = (drawdown_cash / peak) * 100.0
        max_drawdown_pct = abs(drawdown_pct.min()) if not drawdown_pct.empty else 0.0

        bar_ret = net_pnl_cash / equity.shift(1).fillna(self.config.initial_capital)

        # Trades reconstruction for trade statistics
        trades = self._extract_trades(df, position, net_pnl_cash)

        # Calculate summary metrics
        total_trades = len(trades)
        if total_trades > 0:
            wins = trades[trades['pnl'] > 0]
            losses = trades[trades['pnl'] <= 0]
            win_trades = len(wins)
            loss_trades = len(losses)
            win_rate = (win_trades / total_trades) * 100.0
            
            gross_win = wins['pnl'].sum() if not wins.empty else 0.0
            gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 0.0
            profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
            
            avg_win = wins['pnl'].mean() if not wins.empty else 0.0
            avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0.0
            payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
            avg_trade_pct = trades['ret_pct'].mean()
        else:
            win_trades, loss_trades = 0, 0
            win_rate, profit_factor, payoff_ratio, avg_trade_pct = 0.0, 0.0, 0.0, 0.0

        total_return_pct = equity_pct.iloc[-1] if not equity_pct.empty else 0.0
        
        # Fix RuntimeWarning: invalid value encountered in scalar power
        base_factor = 1.0 + total_return_pct / 100.0
        n_days = max(1, (ts_series.max() - ts_series.min()).days)
        if base_factor > 0 and n_days > 0:
            annualized_return_pct = (base_factor ** (365.0 / n_days) - 1.0) * 100.0
        else:
            annualized_return_pct = total_return_pct

        daily_ret = pd.DataFrame({'date': df['date_str'], 'bar_ret': bar_ret}).groupby('date')['bar_ret'].sum()
        ann_vol = daily_ret.std() * np.sqrt(252) * 100.0 if len(daily_ret) > 1 else 0.0
        ann_ret_dec = (daily_ret.mean() * 252) if len(daily_ret) > 0 else 0.0
        
        sharpe_ratio = (ann_ret_dec / (daily_ret.std() * np.sqrt(252))) if daily_ret.std() > 0 else 0.0
        
        downside_std = daily_ret[daily_ret < 0].std() * np.sqrt(252)
        sortino_ratio = (ann_ret_dec / downside_std) if downside_std > 0 else 0.0

        equity_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'close': df['close'],
            'position': position,
            'net_pnl_cash': net_pnl_cash,
            'equity': equity,
            'equity_pct': equity_pct,
            'drawdown_pct': drawdown_pct,
            'year': df['year_str'],
            'month': df['month_str']
        })

        # Yearly & Monthly Breakdown
        yearly_rows = []
        for yr, g in equity_df.groupby('year'):
            yr_pnl = g['net_pnl_cash'].sum()
            yr_ret = (yr_pnl / self.config.initial_capital) * 100.0
            yr_pk = g['equity'].cummax()
            yr_dd = abs(((g['equity'] - yr_pk) / yr_pk).min() * 100.0) if not g.empty else 0.0
            yr_trades = trades[trades['entry_time'].astype(str).str.startswith(yr)] if not trades.empty else pd.DataFrame()
            yt_cnt = len(yr_trades)
            yw_cnt = len(yr_trades[yr_trades['pnl'] > 0]) if yt_cnt > 0 else 0
            yearly_rows.append({
                'Period': yr,
                'Return (%)': round(yr_ret, 2),
                'Net PnL': round(yr_pnl, 2),
                'MDD (%)': round(yr_dd, 2),
                'Trades': yt_cnt,
                'Win Rate (%)': round(yw_cnt / yt_cnt * 100.0, 1) if yt_cnt > 0 else 0.0
            })
        yearly_breakdown = pd.DataFrame(yearly_rows)

        monthly_rows = []
        for mo, g in equity_df.groupby('month'):
            mo_pnl = g['net_pnl_cash'].sum()
            mo_ret = (mo_pnl / self.config.initial_capital) * 100.0
            mo_pk = g['equity'].cummax()
            mo_dd = abs(((g['equity'] - mo_pk) / mo_pk).min() * 100.0) if not g.empty else 0.0
            mo_trades = trades[trades['entry_time'].astype(str).str.startswith(mo)] if not trades.empty else pd.DataFrame()
            mt_cnt = len(mo_trades)
            mw_cnt = len(mo_trades[mo_trades['pnl'] > 0]) if mt_cnt > 0 else 0
            monthly_rows.append({
                'Period': mo,
                'Return (%)': round(mo_ret, 2),
                'Net PnL': round(mo_pnl, 2),
                'MDD (%)': round(mo_dd, 2),
                'Trades': mt_cnt,
                'Win Rate (%)': round(mw_cnt / mt_cnt * 100.0, 1) if mt_cnt > 0 else 0.0
            })
        monthly_breakdown = pd.DataFrame(monthly_rows)

        return BacktestResult(
            strategy_name=strategy_name,
            tick_size=tick_size,
            total_trades=total_trades,
            win_trades=win_trades,
            loss_trades=loss_trades,
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            total_return_pct=round(total_return_pct, 2),
            annualized_return_pct=round(annualized_return_pct, 2),
            annualized_vol_pct=round(ann_vol, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            sortino_ratio=round(sortino_ratio, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            avg_trade_pct=round(avg_trade_pct, 3),
            payoff_ratio=round(payoff_ratio, 2),
            equity_curve=equity_df,
            trade_log=trades,
            yearly_breakdown=yearly_breakdown,
            monthly_breakdown=monthly_breakdown,
            constituent_strategies=constituent_strategies or []
        )

    def _extract_trades(self, df: pd.DataFrame, position: pd.Series, net_pnl_cash: pd.Series) -> pd.DataFrame:
        """Extract individual trade entry/exit details."""
        trades = []
        in_trade = False
        entry_idx = 0
        curr_pos = 0

        pos_vals = position.values
        ts_vals = df['timestamp'].values
        close_vals = df['close'].values
        pnl_vals = net_pnl_cash.values

        for i in range(len(pos_vals)):
            p = pos_vals[i]
            if not in_trade:
                if p != 0:
                    in_trade = True
                    curr_pos = p
                    entry_idx = i
            else:
                if p == 0 or p != curr_pos or i == len(pos_vals) - 1:
                    trade_pnl = float(np.sum(pnl_vals[entry_idx : i + 1]))
                    ret_pct = float(((close_vals[i] / close_vals[entry_idx] - 1.0) * 100.0) * curr_pos)
                    pts_pnl = float((close_vals[i] - close_vals[entry_idx]) * curr_pos)

                    e_ts = str(ts_vals[entry_idx])[:19]
                    x_ts = str(ts_vals[i])[:19]

                    trades.append({
                        'entry_time': e_ts,
                        'exit_time': x_ts,
                        'side': 'LONG' if curr_pos > 0 else 'SHORT',
                        'entry_price': round(float(close_vals[entry_idx]), 2),
                        'exit_price': round(float(close_vals[i]), 2),
                        'pts_pnl': round(pts_pnl, 2),
                        'pnl': round(trade_pnl, 2),
                        'ret_pct': round(ret_pct, 2),
                        'bars_held': int(i - entry_idx + 1)
                    })
                    if p != 0 and p != curr_pos:
                        in_trade = True
                        curr_pos = p
                        entry_idx = i
                    else:
                        in_trade = False
                        curr_pos = 0

        if trades:
            tdf = pd.DataFrame(trades)
            tdf.insert(0, 'trade_no', range(1, len(tdf) + 1))
            tdf['cum_pnl'] = tdf['pnl'].cumsum().round(2)
            return tdf
        else:
            return pd.DataFrame(columns=[
                'trade_no', 'entry_time', 'exit_time', 'side', 'entry_price', 'exit_price',
                'pts_pnl', 'pnl', 'ret_pct', 'bars_held', 'cum_pnl'
            ])

    def _empty_result(self, strategy_name: str, tick_size: int) -> BacktestResult:
        return BacktestResult(
            strategy_name=strategy_name,
            tick_size=tick_size,
            total_trades=0,
            win_trades=0,
            loss_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            annualized_vol_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            avg_trade_pct=0.0,
            payoff_ratio=0.0,
            equity_curve=pd.DataFrame(),
            trade_log=pd.DataFrame(),
            constituent_strategies=[]
        )

    def run_multi_tick_backtest(self, tick_bars_dict: Dict[int, pd.DataFrame], 
                                category: StrategyCategory = StrategyCategory.ALL,
                                selected_strategies: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[int, Dict[str, BacktestResult]]]:
        """
        Runs concurrent vectorized backtests across multiple tick resolutions and strategies.
        Reports real-time progress and estimated completion time to ProgressTracker.
        """
        from concurrent.futures import ThreadPoolExecutor
        from engine.tracker import tracker
        from engine.indicators import calculate_all_indicators

        summary_rows = []
        detailed_results = {}
        indicators = SignalGenerator.get_indicator_list(category)

        # If specific strategies are requested, filter to them
        if selected_strategies and len(selected_strategies) > 0:
            canon_map = {name.lower(): name for name in SignalGenerator.get_indicator_list(StrategyCategory.ALL)}
            normalized_selected = [canon_map.get(s.strip().lower(), s.strip()) for s in selected_strategies if s.strip()]
            if normalized_selected:
                indicators = normalized_selected

        if category == StrategyCategory.ALL and not (selected_strategies and len(selected_strategies) < 5):
            ensemble_cats = [StrategyCategory.TREND, StrategyCategory.MOMENTUM, StrategyCategory.VOLATILITY, StrategyCategory.CONTRARIAN, StrategyCategory.ALL]
        elif selected_strategies and len(selected_strategies) < 3:
            ensemble_cats = []
        else:
            ensemble_cats = [category] if category != StrategyCategory.ALL else []
        
        valid_ticks = {ts: df for ts, df in tick_bars_dict.items() if not df.empty and len(df) >= 20}
        dynamic_ensemble_count = 3 if len(indicators) >= 3 else 0
        total_tasks = len(valid_ticks) * (len(indicators) + len(ensemble_cats) + dynamic_ensemble_count)

        
        if total_tasks > 0:
            tracker.start("BACKTEST", total_tasks, f"Starting parallel backtest on {len(valid_ticks)} resolutions...")

        for tick_size, df_bars in valid_ticks.items():
            detailed_results[tick_size] = {}
            single_results = {}
            
            # 1. Compute all indicators on this resolution
            df_ind = calculate_all_indicators(df_bars.copy())

            # 2. Parallel single strategy evaluations
            def _eval_single(name: str):
                sig = SignalGenerator.generate_single_signal(df_ind, name, tick_size=tick_size)
                res = self.run_single(df_bars, sig, strategy_name=name, tick_size=tick_size, constituent_strategies=[name])
                return name, res


            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_eval_single, ind_name) for ind_name in indicators]
                for fut in futures:
                    name, res = fut.result()
                    detailed_results[tick_size][name] = res
                    single_results[name] = res
                    summary_rows.append(self._result_to_summary_dict(res, category="Single"))
                    tracker.step(1, message=f"Backtesting {name} @ {tick_size}T", details=f"Tick: {tick_size}T, Strategy: {name}")

            # 3. Category Ensembles
            for cat in ensemble_cats:
                ens_name = f"{cat.value}_Ensemble"
                cat_indicators = SignalGenerator.get_indicator_list(cat)
                sig = SignalGenerator.generate_category_ensemble_signal(df_ind, cat)
                res = self.run_single(df_bars, sig, strategy_name=ens_name, tick_size=tick_size, constituent_strategies=cat_indicators)
                detailed_results[tick_size][ens_name] = res
                summary_rows.append(self._result_to_summary_dict(res, category="Ensemble"))
                tracker.step(1, message=f"Backtesting {ens_name} @ {tick_size}T", details=f"Tick: {tick_size}T, Ensemble: {ens_name}")

            # 4. Dynamic Top-N Ensembles (Top 3 by Return, Top 5 by Return, Top 3 by Win Rate)
            if len(single_results) >= 3:
                sorted_by_ret = sorted(single_results.items(), key=lambda x: (x[1].total_return_pct, x[1].sharpe_ratio), reverse=True)
                top3_ret_names = [x[0] for x in sorted_by_ret[:3]]
                top5_ret_names = [x[0] for x in sorted_by_ret[:min(5, len(sorted_by_ret))]]

                sorted_by_win = sorted(single_results.items(), key=lambda x: (x[1].win_rate, x[1].total_return_pct), reverse=True)
                top3_win_names = [x[0] for x in sorted_by_win[:3]]

                # 4.1 Top3_Ensemble (Top 3 by Return)
                top3_sig = SignalGenerator.generate_custom_ensemble_signal(df_ind, top3_ret_names, name="sig_Top3_Ensemble")
                top3_res = self.run_single(df_bars, top3_sig, strategy_name="Top3_Ensemble", tick_size=tick_size, constituent_strategies=top3_ret_names)
                detailed_results[tick_size]["Top3_Ensemble"] = top3_res
                summary_rows.append(self._result_to_summary_dict(top3_res, category="Ensemble"))
                tracker.step(1, message=f"Backtesting Top3_Ensemble ({', '.join(top3_ret_names)}) @ {tick_size}T", details=f"Top 3: {', '.join(top3_ret_names)}")

                # 4.2 Top5_Ensemble (Top 5 by Return)
                top5_sig = SignalGenerator.generate_custom_ensemble_signal(df_ind, top5_ret_names, name="sig_Top5_Ensemble")
                top5_res = self.run_single(df_bars, top5_sig, strategy_name="Top5_Ensemble", tick_size=tick_size, constituent_strategies=top5_ret_names)
                detailed_results[tick_size]["Top5_Ensemble"] = top5_res
                summary_rows.append(self._result_to_summary_dict(top5_res, category="Ensemble"))
                tracker.step(1, message=f"Backtesting Top5_Ensemble @ {tick_size}T", details=f"Top 5: {', '.join(top5_ret_names)}")

                # 4.3 Top3_WinRate_Ensemble (Top 3 by Win Rate)
                top3_win_sig = SignalGenerator.generate_custom_ensemble_signal(df_ind, top3_win_names, name="sig_Top3_WinRate_Ensemble")
                top3_win_res = self.run_single(df_bars, top3_win_sig, strategy_name="Top3_WinRate_Ensemble", tick_size=tick_size, constituent_strategies=top3_win_names)
                detailed_results[tick_size]["Top3_WinRate_Ensemble"] = top3_win_res
                summary_rows.append(self._result_to_summary_dict(top3_win_res, category="Ensemble"))
                tracker.step(1, message=f"Backtesting Top3_WinRate_Ensemble ({', '.join(top3_win_names)}) @ {tick_size}T", details=f"Top WinRate: {', '.join(top3_win_names)}")

        tracker.finish("Backtest matrix completed!")
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(by=['Sharpe Ratio', 'Cumulative Return (%)'], ascending=[False, False]).reset_index(drop=True)

        return summary_df, detailed_results

    def _result_to_summary_dict(self, res: BacktestResult, category: str) -> Dict[str, Any]:
        return {
            'Tick Size': res.tick_size,
            'Strategy': res.strategy_name,
            'Type': category,
            'Cumulative Return (%)': res.total_return_pct,
            'Annualized Return (%)': res.annualized_return_pct,
            'Sharpe Ratio': res.sharpe_ratio,
            'Sortino Ratio': res.sortino_ratio,
            'MDD (%)': res.max_drawdown_pct,
            'Win Rate (%)': res.win_rate,
            'Profit Factor': res.profit_factor,
            'Trades': res.total_trades,
            'Payoff Ratio': res.payoff_ratio,
            'Constituent Strategies': res.constituent_strategies
        }

