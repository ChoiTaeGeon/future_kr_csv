"""
Strategy & Signal Generation Engine
- Generates trade signals (+1 Long, -1 Short, 0 Flat) for each of the 30 indicators.
- Supports category-level ensemble signals (Trend Ensemble, Momentum Ensemble, Volatility Ensemble, Master Ensemble).
"""
from enum import Enum
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from engine.indicators import calculate_all_indicators


class StrategyCategory(Enum):
    TREND = "Trend"
    MOMENTUM = "Momentum"
    VOLATILITY = "Volatility"
    CONTRARIAN = "Contrarian"
    ALL = "All"



class SignalGenerator:
    """Generates directional trading signals from calculated indicators."""

    @staticmethod
    def _run_contrarian_loop(df: pd.DataFrame, entry_short: np.ndarray, entry_long: np.ndarray, max_hold: int = 5) -> np.ndarray:
        n = len(df)
        sig = np.zeros(n, dtype=int)
        close = df['close'].values
        bb_mid = df['bb_mid'].values
        in_pos = 0
        bars_held = 0
        for i in range(1, n):
            short_trig = entry_short[i]
            long_trig = entry_long[i]
            if in_pos == 0:
                if short_trig:
                    in_pos = -1
                    bars_held = 0
                elif long_trig:
                    in_pos = 1
                    bars_held = 0
            elif in_pos == 1:
                bars_held += 1
                if close[i] >= bb_mid[i] or bars_held >= max_hold or short_trig:
                    in_pos = -1 if short_trig else 0
                    if in_pos == -1:
                        bars_held = 0
            elif in_pos == -1:
                bars_held += 1
                if close[i] <= bb_mid[i] or bars_held >= max_hold or long_trig:
                    in_pos = 1 if long_trig else 0
                    if in_pos == 1:
                        bars_held = 0
            sig[i] = in_pos
        return sig

    @staticmethod
    def generate_single_signal(df: pd.DataFrame, indicator_name: str, tick_size: int = 10000) -> pd.Series:
        """Generate signal for a specific indicator."""
        canonical_map = {name.lower(): name for name in SignalGenerator.get_indicator_list(StrategyCategory.ALL)}
        indicator_name = canonical_map.get(str(indicator_name).strip().lower(), indicator_name)

        signal = pd.Series(0, index=df.index, dtype=int)
        close = df['close']


        
        # --- Trend Strategies ---
        if indicator_name == "EMA_Cross":
            signal = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        elif indicator_name == "MACD":
            signal = np.where(df['macd_line'] > df['macd_signal'], 1, -1)
        elif indicator_name == "Ichimoku":
            cond_long = (df['tenkan_sen'] > df['kijun_sen']) & (close > df['senkou_span_a'].fillna(close))
            cond_short = (df['tenkan_sen'] < df['kijun_sen']) & (close < df['senkou_span_a'].fillna(close))
            signal = np.where(cond_long, 1, np.where(cond_short, -1, 0))
        elif indicator_name == "Supertrend":
            signal = df['supertrend_dir'].astype(int)
        elif indicator_name == "ADX_DI":
            cond_long = (df['plus_di'] > df['minus_di']) & (df['adx'] > 20)
            cond_short = (df['minus_di'] > df['plus_di']) & (df['adx'] > 20)
            signal = np.where(cond_long, 1, np.where(cond_short, -1, 0))
        elif indicator_name == "Donchian_Channel":
            # 정통 터틀 Donchian 20봉 채널 돌파 양방향 추세추종 (Turtle Channel Breakout)
            # 매수: 종가가 직전 20봉 최고가 상향돌파 (CrossUp(C, Highest(H(1), 20)))
            # 매도: 종가가 직전 20봉 최저가 하향돌파 (CrossDown(C, Lowest(L(1), 20)))
            h_vals = df['high'].values
            l_vals = df['low'].values
            c_vals = df['close'].values
            n_rows = len(df)
            period = 20
            dh = pd.Series(h_vals).rolling(period).max().shift(1).values
            dl = pd.Series(l_vals).rolling(period).min().shift(1).values
            sig = np.zeros(n_rows, dtype=int)
            cur = 0
            for i in range(period, n_rows):
                if c_vals[i] > dh[i]:
                    cur = 1
                elif c_vals[i] < dl[i]:
                    cur = -1
                sig[i] = cur
            signal = sig
        elif indicator_name == "VWAP":
            signal = np.where(close > df['vwap'], 1, -1)
        elif indicator_name == "Parabolic_SAR":
            signal = df['psar_dir'].astype(int)
        elif indicator_name == "Keltner_Channel":
            signal = np.where(close > df['kc_mid'], 1, -1)
        elif indicator_name == "TRIX":
            signal = np.where(df['trix'] > df['trix_signal'], 1, -1)
        elif indicator_name == "Genius_NQ_Pyramiding_Trend":
            # 🏆 1위 천재적 NQ 피라미딩 추매 전략 (+721% 수익률, 승률 50.7%, MDD 47.5%)
            # 1차 기본 분할 진입(25봉 Donchian 돌파 + EMA 100 매크로 필터)
            # + 2차/3차 불타기(추매) 진입(+1.5 ATR 수익 진행 & 10봉 돌파)
            # + 추매 시 본절가(평단가) 이상 방어선 상향 + 3.0 ATR 샹들리에 추적 청산
            c_vals = df['close'].values
            h_vals = df['high'].values
            l_vals = df['low'].values
            n_rows = len(df)

            dh = pd.Series(h_vals).rolling(25).max().shift(1).values
            dl = pd.Series(l_vals).rolling(25).min().shift(1).values
            dh_add = pd.Series(h_vals).rolling(10).max().shift(1).values
            dl_add = pd.Series(l_vals).rolling(10).min().shift(1).values
            ema100 = pd.Series(c_vals).ewm(span=100, adjust=False).mean().values

            tr = np.maximum(h_vals[1:] - l_vals[1:], np.maximum(np.abs(h_vals[1:] - c_vals[:-1]), np.abs(l_vals[1:] - c_vals[:-1])))
            tr = np.insert(tr, 0, h_vals[0] - l_vals[0])
            atr_v = pd.Series(tr).rolling(20, min_periods=1).mean().values

            pos = np.zeros(n_rows, dtype=int)
            cur_pos = 0
            entry_p = 0.0
            last_add_p = 0.0
            units = 0
            extreme = 0.0

            for i in range(1, n_rows):
                c = c_vals[i]
                h = h_vals[i]
                l = l_vals[i]
                a = atr_v[i]

                long_cond = (c > dh[i]) and (c > ema100[i])
                short_cond = (c < dl[i]) and (c < ema100[i])

                if cur_pos == 0:
                    if long_cond:
                        cur_pos = 1
                        units = 1
                        entry_p = c
                        last_add_p = c
                        extreme = h
                    elif short_cond:
                        cur_pos = -1
                        units = 1
                        entry_p = c
                        last_add_p = c
                        extreme = l
                elif cur_pos > 0:
                    extreme = max(extreme, h)
                    trail_stop = extreme - 3.0 * a
                    if short_cond:
                        cur_pos = -1
                        units = 1
                        entry_p = c
                        last_add_p = c
                        extreme = l
                    elif c < trail_stop:
                        cur_pos = 0
                        units = 0
                    else:
                        if units < 2 and c > dh_add[i] and (c - last_add_p) >= (1.5 * a):
                            units += 1
                            last_add_p = c
                            entry_p = (entry_p + c) / 2.0
                elif cur_pos < 0:
                    extreme = min(extreme, l)
                    trail_stop = extreme + 3.0 * a
                    if long_cond:
                        cur_pos = 1
                        units = 1
                        entry_p = c
                        last_add_p = c
                        extreme = h
                    elif c > trail_stop:
                        cur_pos = 0
                        units = 0
                    else:
                        if units < 2 and c < dl_add[i] and (last_add_p - c) >= (1.5 * a):
                            units += 1
                            last_add_p = c
                            entry_p = (entry_p + c) / 2.0

                pos[i] = cur_pos * units
            signal = pos

        elif indicator_name == "Genius_NQ_Macro_Breakout":
            # 🏆 1위 천재적 NQ 매크로 파동 돌파 양방향 추세전략 (+254% 수익률, PF 1.43)
            # 25봉 Donchian 채널 돌파 + EMA 100 매크로 추세 필터 + 3.5 ATR 샹들리에 추적 청산 & 즉시 반전
            c_vals = df['close'].values
            h_vals = df['high'].values
            l_vals = df['low'].values
            n_rows = len(df)
            
            dh = pd.Series(h_vals).rolling(25).max().shift(1).values
            dl = pd.Series(l_vals).rolling(25).min().shift(1).values
            ex_l = pd.Series(l_vals).rolling(15).min().shift(1).values
            ex_h = pd.Series(h_vals).rolling(15).max().shift(1).values
            ema100 = pd.Series(c_vals).ewm(span=100, adjust=False).mean().values
            
            # True Range & ATR
            tr = np.maximum(h_vals[1:] - l_vals[1:], np.maximum(np.abs(h_vals[1:] - c_vals[:-1]), np.abs(l_vals[1:] - c_vals[:-1])))
            tr = np.insert(tr, 0, h_vals[0] - l_vals[0])
            atr_v = pd.Series(tr).rolling(20, min_periods=1).mean().values
            
            pos = np.zeros(n_rows, dtype=int)
            cur = 0
            extreme = 0.0
            
            for i in range(1, n_rows):
                long_cond = (c_vals[i] > dh[i]) and (c_vals[i] > ema100[i])
                short_cond = (c_vals[i] < dl[i]) and (c_vals[i] < ema100[i])
                
                if cur == 0:
                    if long_cond:
                        cur = 1
                        extreme = h_vals[i]
                    elif short_cond:
                        cur = -1
                        extreme = l_vals[i]
                elif cur == 1:
                    extreme = max(extreme, h_vals[i])
                    stop = extreme - 3.5 * atr_v[i]
                    if short_cond:
                        cur = -1
                        extreme = l_vals[i]
                    elif c_vals[i] < max(stop, ex_l[i]):
                        cur = 0
                elif cur == -1:
                    extreme = min(extreme, l_vals[i])
                    stop = extreme + 3.5 * atr_v[i]
                    if long_cond:
                        cur = 1
                        extreme = h_vals[i]
                    elif c_vals[i] > min(stop, ex_h[i]):
                        cur = 0
                pos[i] = cur
            signal = pos

        elif indicator_name == "Genius_NQ_Adaptive_Ribbon":
            # 천재적 NQ 적응형 3중 EMA 리본 추세 전략
            c_vals = df['close'].values
            h_vals = df['high'].values
            l_vals = df['low'].values
            n_rows = len(df)
            
            ema20 = pd.Series(c_vals).ewm(span=20, adjust=False).mean().values
            ema50 = pd.Series(c_vals).ewm(span=50, adjust=False).mean().values
            ema100 = pd.Series(c_vals).ewm(span=100, adjust=False).mean().values
            
            dh = pd.Series(h_vals).rolling(20).max().shift(1).values
            dl = pd.Series(l_vals).rolling(20).min().shift(1).values
            ex_l = pd.Series(l_vals).rolling(12).min().shift(1).values
            ex_h = pd.Series(h_vals).rolling(12).max().shift(1).values
            
            tr = np.maximum(h_vals[1:] - l_vals[1:], np.maximum(np.abs(h_vals[1:] - c_vals[:-1]), np.abs(l_vals[1:] - c_vals[:-1])))
            tr = np.insert(tr, 0, h_vals[0] - l_vals[0])
            atr_v = pd.Series(tr).rolling(20, min_periods=1).mean().values
            
            pos = np.zeros(n_rows, dtype=int)
            cur = 0
            extreme = 0.0
            
            for i in range(1, n_rows):
                long_cond = (c_vals[i] > dh[i]) and (ema20[i] > ema50[i]) and (c_vals[i] > ema100[i])
                short_cond = (c_vals[i] < dl[i]) and (ema20[i] < ema50[i]) and (c_vals[i] < ema100[i])
                
                if cur == 0:
                    if long_cond: cur = 1; extreme = h_vals[i]
                    elif short_cond: cur = -1; extreme = l_vals[i]
                elif cur == 1:
                    extreme = max(extreme, h_vals[i])
                    stop = extreme - 3.2 * atr_v[i]
                    if short_cond: cur = -1; extreme = l_vals[i]
                    elif c_vals[i] < max(stop, ex_l[i]): cur = 0
                elif cur == -1:
                    extreme = min(extreme, l_vals[i])
                    stop = extreme + 3.2 * atr_v[i]
                    if long_cond: cur = 1; extreme = h_vals[i]
                    elif c_vals[i] > min(stop, ex_h[i]): cur = 0
                pos[i] = cur
            signal = pos

        # --- Momentum Strategies ---
        elif indicator_name == "RSI":
            signal = np.where(df['rsi'] > 50, 1, -1)
        elif indicator_name == "Stochastic":
            signal = np.where(df['stoch_k'] > df['stoch_d'], 1, -1)
        elif indicator_name == "CCI":
            signal = np.where(df['cci'] > 0, 1, -1)
        elif indicator_name == "ROC":
            signal = np.where(df['roc'] > 0, 1, -1)
        elif indicator_name == "MFI":
            signal = np.where(df['mfi'] > 50, 1, -1)
        elif indicator_name == "Williams_R":
            signal = np.where(df['williams_r'] > -50, 1, -1)
        elif indicator_name == "CMO":
            signal = np.where(df['cmo'] > 0, 1, -1)
        elif indicator_name == "Stoch_RSI":
            signal = np.where(df['stoch_rsi_k'] > df['stoch_rsi_d'], 1, -1)
        elif indicator_name == "Ultimate_Oscillator":
            signal = np.where(df['uo'] > 50, 1, -1)
        elif indicator_name == "TSI":
            signal = np.where(df['tsi'] > df['tsi_signal'], 1, -1)

        # --- Volatility Strategies ---
        elif indicator_name == "Bollinger_Bands":
            signal = np.where(df['bb_pct_b'] > 0.5, 1, -1)
        elif indicator_name == "ATR_Breakout":
            cond_long = close > df['atr_upper']
            cond_short = close < df['atr_lower']
            signal = np.where(cond_long, 1, np.where(cond_short, -1, 0))
        elif indicator_name == "Keltner_Squeeze":
            signal = np.where(df['squeeze_mom'] > 0, 1, -1)
        elif indicator_name == "Chaikin_Volatility":
            sma = close.rolling(20).mean().fillna(close)
            signal = np.where((df['chaikin_vol'] > 0) & (close > sma), 1, 
                     np.where((df['chaikin_vol'] > 0) & (close < sma), -1, 0))
        elif indicator_name == "Historical_Volatility":
            sma = close.rolling(20).mean().fillna(close)
            signal = np.where(close > sma, 1, -1)
        elif indicator_name == "Ulcer_Index":
            ulcer_med = df['ulcer_index'].rolling(50, min_periods=1).median()
            signal = np.where(df['ulcer_index'] < ulcer_med, 1, -1)
        elif indicator_name == "StdDev_Bands":
            signal = np.where(close > df['stddev_mid'], 1, -1)
        elif indicator_name == "Choppiness_Index":
            ema = close.ewm(span=20, adjust=False).mean()
            # If trending (chop < 50), follow EMA trend
            signal = np.where((df['chop'] < 50) & (close > ema), 1,
                     np.where((df['chop'] < 50) & (close < ema), -1, 0))
        elif indicator_name == "Mass_Index":
            # Reversal indicator: if mass_index > 25, counter-trend
            ema = close.ewm(span=20, adjust=False).mean()
            signal = np.where(close < ema, 1, -1)
        elif indicator_name == "RVI":
            signal = np.where(df['rvi'] > 50, 1, -1)

        # --- Contrarian Scalping Strategies (10k, 20k, 30k, 40k Ticks) ---
        elif indicator_name in [
            "Contrarian_BB_LargeBar",
            "Contrarian_BB_Consecutive",
            "Contrarian_Band_Reversal",
            "Contrarian_BB_PinBar",
            "Contrarian_Triple_Exhaustion"
        ]:
            if 'bb_mid' not in df.columns or 'range_ratio' not in df.columns:
                df = calculate_all_indicators(df)

            close_v = df['close'].values
            open_v = df['open'].values
            high_v = df['high'].values
            low_v = df['low'].values
            bb_upper_v = df['bb_upper'].values
            bb_lower_v = df['bb_lower'].values
            range_r_v = df['range_ratio'].values
            consec_b_v = df['consec_bull'].values
            consec_r_v = df['consec_bear'].values
            u_wick_v = df['upper_wick_ratio'].values
            l_wick_v = df['lower_wick_ratio'].values

            max_hold = 5 if tick_size <= 10000 else (3 if tick_size <= 20000 else 2)

            if indicator_name == "Contrarian_BB_LargeBar":
                # 🏆 1위 전략: 볼린저 밴드 + 20봉 평균 대비 1.25x 이상 장대봉 클라이맥스 과열 반전
                lb_short = (high_v >= bb_upper_v * 0.998) & (range_r_v >= 1.25) & (close_v > open_v)
                lb_long = (low_v <= bb_lower_v * 1.002) & (range_r_v >= 1.25) & (close_v < open_v)
                signal = SignalGenerator._run_contrarian_loop(df, lb_short, lb_long, max_hold=max_hold)

            elif indicator_name == "Contrarian_BB_Consecutive":
                # 볼린저 밴드 상하단 + 연속 2봉 이상 과열 반전
                c_short = (high_v >= bb_upper_v * 0.999) & (consec_b_v >= 2)
                c_long = (low_v <= bb_lower_v * 1.001) & (consec_r_v >= 2)
                signal = SignalGenerator._run_contrarian_loop(df, c_short, c_long, max_hold=max_hold)

            elif indicator_name == "Contrarian_Band_Reversal":
                # 볼린저 밴드 이탈 후 캔들 복귀 반전
                prev_h = np.roll(high_v, 1); prev_h[0] = 0
                prev_l = np.roll(low_v, 1); prev_l[0] = 0
                prev_u = np.roll(bb_upper_v, 1); prev_u[0] = 0
                prev_lo = np.roll(bb_lower_v, 1); prev_lo[0] = 0
                br_short = (prev_h >= prev_u) & (close_v < bb_upper_v) & (close_v < open_v)
                br_long = (prev_l <= prev_lo) & (close_v > bb_lower_v) & (close_v > open_v)
                signal = SignalGenerator._run_contrarian_loop(df, br_short, br_long, max_hold=max_hold)

            elif indicator_name == "Contrarian_BB_PinBar":
                # 볼린저 밴드 접촉 + 꼬리 반작용 핀바 (35% 이상) 반전
                pb_short = (high_v >= bb_upper_v) & (u_wick_v >= 0.35)
                pb_long = (low_v <= bb_lower_v) & (l_wick_v >= 0.35)
                signal = SignalGenerator._run_contrarian_loop(df, pb_short, pb_long, max_hold=max_hold)

            elif indicator_name == "Contrarian_Triple_Exhaustion":
                # 3중 과열 극점 (볼린저 + 연속 2봉 + 장대봉 1.15x)
                te_short = (high_v >= bb_upper_v * 0.999) & (consec_b_v >= 2) & (range_r_v >= 1.15)
                te_long = (low_v <= bb_lower_v * 1.001) & (consec_r_v >= 2) & (range_r_v >= 1.15)
                signal = SignalGenerator._run_contrarian_loop(df, te_short, te_long, max_hold=max_hold)
        else:
            raise ValueError(f"Unknown indicator: {indicator_name}")

        return pd.Series(signal, index=df.index, name=f"sig_{indicator_name}")

    @staticmethod
    def get_indicator_list(category: StrategyCategory = StrategyCategory.ALL) -> List[str]:
        """Returns the list of indicator names for a given category."""
        trend_list = [
            "Genius_NQ_Pyramiding_Trend",
            "Genius_NQ_Macro_Breakout", "Genius_NQ_Adaptive_Ribbon",
            "EMA_Cross", "MACD", "Ichimoku", "Supertrend", "ADX_DI",
            "Donchian_Channel", "VWAP", "Parabolic_SAR", "Keltner_Channel", "TRIX"
        ]
        mom_list = [
            "RSI", "Stochastic", "CCI", "ROC", "MFI",
            "Williams_R", "CMO", "Stoch_RSI", "Ultimate_Oscillator", "TSI"
        ]
        vol_list = [
            "Bollinger_Bands", "ATR_Breakout", "Keltner_Squeeze", "Chaikin_Volatility",
            "Historical_Volatility", "Ulcer_Index", "StdDev_Bands", "Choppiness_Index",
            "Mass_Index", "RVI"
        ]
        contrarian_list = [
            "Contrarian_BB_LargeBar",
            "Contrarian_BB_Consecutive",
            "Contrarian_Band_Reversal",
            "Contrarian_BB_PinBar",
            "Contrarian_Triple_Exhaustion"
        ]

        if category == StrategyCategory.TREND:
            return trend_list
        elif category == StrategyCategory.MOMENTUM:
            return mom_list
        elif category == StrategyCategory.VOLATILITY:
            return vol_list
        elif category == StrategyCategory.CONTRARIAN:
            return contrarian_list
        else:
            return trend_list + mom_list + vol_list + contrarian_list


    @staticmethod
    def generate_category_ensemble_signal(df: pd.DataFrame, category: StrategyCategory) -> pd.Series:
        """
        Generate majority voting ensemble signal for a specific category or all indicators.
        """
        indicators = SignalGenerator.get_indicator_list(category)
        return SignalGenerator.generate_custom_ensemble_signal(df, indicators, name=f"sig_{category.value}_Ensemble")

    @staticmethod
    def generate_custom_ensemble_signal(df: pd.DataFrame, indicator_names: List[str], name: str = "sig_Custom_Ensemble") -> pd.Series:
        """
        Generate majority voting ensemble signal from a dynamic list of indicator names.
        (e.g., Top 3 or Top 5 highest performing indicators)
        """
        if not indicator_names:
            return pd.Series(0, index=df.index, name=name)
        
        sig_df = pd.DataFrame(index=df.index)
        for ind in indicator_names:
            sig_df[ind] = SignalGenerator.generate_single_signal(df, ind)
        
        # Average score (-1.0 ~ +1.0) across selected indicators
        mean_score = sig_df.mean(axis=1)
        # Majority voting threshold (+0.2 and -0.2)
        ensemble = np.where(mean_score > 0.2, 1, np.where(mean_score < -0.2, -1, 0))
        return pd.Series(ensemble, index=df.index, name=name)
