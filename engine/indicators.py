"""
Technical Indicators Engine
- 30 Core Indicators across 3 Strategy Categories (10 Trend, 10 Momentum, 10 Volatility)
- Fully vectorized using Pandas & Numpy for maximum performance and zero C-dependency issues.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


class TrendIndicators:
    """Category A: Top 10 Trend Indicators"""

    @staticmethod
    def ema_cross(df: pd.DataFrame, fast_span: int = 9, slow_span: int = 21) -> pd.DataFrame:
        """1. EMA Cross: Fast EMA vs Slow EMA"""
        out = pd.DataFrame(index=df.index)
        out['ema_fast'] = df['close'].ewm(span=fast_span, adjust=False).mean()
        out['ema_slow'] = df['close'].ewm(span=slow_span, adjust=False).mean()
        out['ema_diff'] = out['ema_fast'] - out['ema_slow']
        return out

    @staticmethod
    def macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """2. MACD: Moving Average Convergence Divergence"""
        out = pd.DataFrame(index=df.index)
        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
        out['macd_line'] = fast_ema - slow_ema
        out['macd_signal'] = out['macd_line'].ewm(span=signal_period, adjust=False).mean()
        out['macd_hist'] = out['macd_line'] - out['macd_signal']
        return out

    @staticmethod
    def ichimoku(df: pd.DataFrame, conversion_period: int = 9, base_period: int = 26, 
                 span_b_period: int = 52, displacement: int = 26) -> pd.DataFrame:
        """3. Ichimoku Kinko Hyo (일목균형표)"""
        out = pd.DataFrame(index=df.index)
        # Tenkan-sen (Conversion Line)
        out['tenkan_sen'] = (df['high'].rolling(conversion_period).max() + df['low'].rolling(conversion_period).min()) / 2
        # Kijun-sen (Base Line)
        out['kijun_sen'] = (df['high'].rolling(base_period).max() + df['low'].rolling(base_period).min()) / 2
        # Senkou Span A (Leading Span A)
        out['senkou_span_a'] = ((out['tenkan_sen'] + out['kijun_sen']) / 2).shift(displacement)
        # Senkou Span B (Leading Span B)
        out['senkou_span_b'] = ((df['high'].rolling(span_b_period).max() + df['low'].rolling(span_b_period).min()) / 2).shift(displacement)
        # Chikou Span (Lagging Span)
        out['chikou_span'] = df['close'].shift(-displacement)
        return out

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """4. Supertrend Indicator"""
        out = pd.DataFrame(index=df.index)
        high, low, close = df['high'].values, df['low'].values, df['close'].values
        n = len(df)
        
        # Calculate ATR
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0], tr3[0] = 0, 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values

        hl2 = (high + low) / 2
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        
        supertrend = np.zeros(n)
        direction = np.ones(n)  # 1 = Bullish, -1 = Bearish

        for i in range(1, n):
            if close[i] > upperband[i-1]:
                direction[i] = 1
            elif close[i] < lowerband[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                    lowerband[i] = lowerband[i-1]
                if direction[i] == -1 and upperband[i] > upperband[i-1]:
                    upperband[i] = upperband[i-1]

            supertrend[i] = lowerband[i] if direction[i] == 1 else upperband[i]

        out['supertrend'] = supertrend
        out['supertrend_dir'] = direction
        out['supertrend_upper'] = upperband
        out['supertrend_lower'] = lowerband
        return out

    @staticmethod
    def adx_di(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """5. ADX (Average Directional Index) + DI"""
        out = pd.DataFrame(index=df.index)
        high, low, close = df['high'], df['low'], df['close']
        
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        out['plus_di'] = plus_di
        out['minus_di'] = minus_di
        out['adx'] = adx
        return out

    @staticmethod
    def donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """6. Donchian Channel"""
        out = pd.DataFrame(index=df.index)
        out['donchian_upper'] = df['high'].rolling(period).max()
        out['donchian_lower'] = df['low'].rolling(period).min()
        out['donchian_mid'] = (out['donchian_upper'] + out['donchian_lower']) / 2
        return out

    @staticmethod
    def vwap_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """7. VWAP (Volume Weighted Average Price) & Standard Deviation Bands"""
        out = pd.DataFrame(index=df.index)
        pv = df['close'] * df['volume']
        cum_pv = pv.rolling(period, min_periods=1).sum()
        cum_vol = df['volume'].rolling(period, min_periods=1).sum().replace(0, np.nan)
        out['vwap'] = (cum_pv / cum_vol).fillna(df['close'])
        
        # Rolling std of price around VWAP
        rolling_std = (df['close'] - out['vwap']).rolling(period, min_periods=1).std().fillna(0)
        out['vwap_upper'] = out['vwap'] + num_std * rolling_std
        out['vwap_lower'] = out['vwap'] - num_std * rolling_std
        return out

    @staticmethod
    def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
        """8. Parabolic SAR"""
        out = pd.DataFrame(index=df.index)
        high, low, close = df['high'].values, df['low'].values, df['close'].values
        n = len(df)
        sar = np.zeros(n)
        direction = np.ones(n)  # 1 = Up, -1 = Down
        af = step
        ep = high[0]
        sar[0] = low[0]

        for i in range(1, n):
            prev_sar = sar[i-1]
            prev_dir = direction[i-1]
            
            if prev_dir == 1:
                cur_sar = prev_sar + af * (ep - prev_sar)
                cur_sar = min(cur_sar, low[i-1], low[i-2] if i > 1 else low[i-1])
                if low[i] < cur_sar:
                    cur_dir = -1
                    cur_sar = ep
                    af = step
                    ep = low[i]
                else:
                    cur_dir = 1
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + step, max_step)
            else:
                cur_sar = prev_sar + af * (ep - prev_sar)
                cur_sar = max(cur_sar, high[i-1], high[i-2] if i > 1 else high[i-1])
                if high[i] > cur_sar:
                    cur_dir = 1
                    cur_sar = ep
                    af = step
                    ep = high[i]
                else:
                    cur_dir = -1
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + step, max_step)

            sar[i] = cur_sar
            direction[i] = cur_dir

        out['psar'] = sar
        out['psar_dir'] = direction
        return out

    @staticmethod
    def keltner_channel(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
        """9. Keltner Channel"""
        out = pd.DataFrame(index=df.index)
        out['kc_mid'] = df['close'].ewm(span=ema_period, adjust=False).mean()
        
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=atr_period, adjust=False).mean()
        
        out['kc_upper'] = out['kc_mid'] + multiplier * atr
        out['kc_lower'] = out['kc_mid'] - multiplier * atr
        return out

    @staticmethod
    def trix(df: pd.DataFrame, period: int = 14, signal_period: int = 9) -> pd.DataFrame:
        """10. TRIX (Triple Exponential Average)"""
        out = pd.DataFrame(index=df.index)
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        
        out['trix'] = ema3.pct_change() * 100
        out['trix_signal'] = out['trix'].ewm(span=signal_period, adjust=False).mean()
        out['trix_hist'] = out['trix'] - out['trix_signal']
        return out


class MomentumIndicators:
    """Category B: Top 10 Momentum Indicators"""

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """1. RSI (Relative Strength Index)"""
        out = pd.DataFrame(index=df.index)
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        avg_gain = pd.Series(gain, index=df.index).ewm(alpha=1/period, adjust=False).mean()
        avg_loss = pd.Series(loss, index=df.index).ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out['rsi'] = 100 - (100 / (1 + rs))
        out['rsi'] = out['rsi'].fillna(50.0)
        return out

    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> pd.DataFrame:
        """2. Stochastic Oscillator (%K, %D)"""
        out = pd.DataFrame(index=df.index)
        lowest_low = df['low'].rolling(k_period).min()
        highest_high = df['high'].rolling(k_period).max()
        
        fast_k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan))
        out['stoch_k'] = fast_k.rolling(smooth_k).mean()
        out['stoch_d'] = out['stoch_k'].rolling(d_period).mean()
        return out

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """3. CCI (Commodity Channel Index)"""
        out = pd.DataFrame(index=df.index)
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).replace(0, np.nan)
        out['cci'] = (tp - sma_tp) / (0.015 * mad)
        return out

    @staticmethod
    def roc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """4. ROC (Rate of Change)"""
        out = pd.DataFrame(index=df.index)
        out['roc'] = df['close'].pct_change(period) * 100
        return out

    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """5. MFI (Money Flow Index)"""
        out = pd.DataFrame(index=df.index)
        tp = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = tp * df['volume']
        
        pos_flow = np.where(tp > tp.shift(1), raw_money_flow, 0.0)
        neg_flow = np.where(tp < tp.shift(1), raw_money_flow, 0.0)
        
        pos_mf = pd.Series(pos_flow, index=df.index).rolling(period).sum()
        neg_mf = pd.Series(neg_flow, index=df.index).rolling(period).sum().replace(0, np.nan)
        
        mfr = pos_mf / neg_mf
        out['mfi'] = 100 - (100 / (1 + mfr))
        out['mfi'] = out['mfi'].fillna(50.0)
        return out

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """6. Williams %R"""
        out = pd.DataFrame(index=df.index)
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()
        out['williams_r'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low).replace(0, np.nan))
        return out

    @staticmethod
    def cmo(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """7. CMO (Chande Momentum Oscillator)"""
        out = pd.DataFrame(index=df.index)
        delta = df['close'].diff()
        up_sum = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index).rolling(period).sum()
        down_sum = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index).rolling(period).sum()
        
        denom = (up_sum + down_sum).replace(0, np.nan)
        out['cmo'] = 100 * ((up_sum - down_sum) / denom)
        return out

    @staticmethod
    def stoch_rsi(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
        """8. Stochastic RSI"""
        out = pd.DataFrame(index=df.index)
        rsi_df = MomentumIndicators.rsi(df, period=rsi_period)
        rsi = rsi_df['rsi']
        
        rsi_min = rsi.rolling(stoch_period).min()
        rsi_max = rsi.rolling(stoch_period).max()
        stoch = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
        
        out['stoch_rsi_k'] = (stoch * 100).rolling(k_period).mean()
        out['stoch_rsi_d'] = out['stoch_rsi_k'].rolling(d_period).mean()
        return out

    @staticmethod
    def ultimate_oscillator(df: pd.DataFrame, period1: int = 7, period2: int = 14, period3: int = 28) -> pd.DataFrame:
        """9. Ultimate Oscillator (UO)"""
        out = pd.DataFrame(index=df.index)
        prev_close = df['close'].shift(1)
        bp = df['close'] - np.minimum(df['low'], prev_close)
        tr = np.maximum(df['high'], prev_close) - np.minimum(df['low'], prev_close)
        
        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum().replace(0, np.nan)
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum().replace(0, np.nan)
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum().replace(0, np.nan)
        
        out['uo'] = 100 * ((4 * avg1 + 2 * avg2 + avg3) / 7)
        return out

    @staticmethod
    def tsi(df: pd.DataFrame, long_period: int = 25, short_period: int = 13, signal_period: int = 7) -> pd.DataFrame:
        """10. TSI (True Strength Index)"""
        out = pd.DataFrame(index=df.index)
        diff = df['close'].diff()
        abs_diff = diff.abs()
        
        # Double smoothed momentum
        smooth1 = diff.ewm(span=long_period, adjust=False).mean()
        smooth2 = smooth1.ewm(span=short_period, adjust=False).mean()
        
        # Double smoothed absolute momentum
        abs_smooth1 = abs_diff.ewm(span=long_period, adjust=False).mean()
        abs_smooth2 = abs_smooth1.ewm(span=short_period, adjust=False).mean().replace(0, np.nan)
        
        out['tsi'] = 100 * (smooth2 / abs_smooth2)
        out['tsi_signal'] = out['tsi'].ewm(span=signal_period, adjust=False).mean()
        return out


class VolatilityIndicators:
    """Category C: Top 10 Volatility Indicators"""

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """1. Bollinger Bands"""
        out = pd.DataFrame(index=df.index)
        out['bb_mid'] = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        out['bb_upper'] = out['bb_mid'] + (num_std * std)
        out['bb_lower'] = out['bb_mid'] - (num_std * std)
        out['bb_width'] = (out['bb_upper'] - out['bb_lower']) / out['bb_mid'].replace(0, np.nan)
        out['bb_pct_b'] = (df['close'] - out['bb_lower']) / (out['bb_upper'] - out['bb_lower']).replace(0, np.nan)
        return out

    @staticmethod
    def atr_breakout(df: pd.DataFrame, period: int = 14, multiplier: float = 2.0) -> pd.DataFrame:
        """2. ATR Breakout Bands"""
        out = pd.DataFrame(index=df.index)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        out['atr'] = tr.ewm(alpha=1/period, adjust=False).mean()
        sma = df['close'].rolling(period).mean()
        out['atr_upper'] = sma + (multiplier * out['atr'])
        out['atr_lower'] = sma - (multiplier * out['atr'])
        return out

    @staticmethod
    def keltner_squeeze(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0, 
                        kc_period: int = 20, kc_mult: float = 1.5) -> pd.DataFrame:
        """3. Keltner-Bollinger Volatility Squeeze (TTM Squeeze)"""
        out = pd.DataFrame(index=df.index)
        bb = VolatilityIndicators.bollinger_bands(df, period=bb_period, num_std=bb_std)
        kc = TrendIndicators.keltner_channel(df, ema_period=kc_period, atr_period=kc_period, multiplier=kc_mult)
        
        # Squeeze condition: Bollinger bands inside Keltner channel
        out['squeeze_on'] = (bb['bb_lower'] > kc['kc_lower']) & (bb['bb_upper'] < kc['kc_upper'])
        
        # Squeeze Momentum (Linear regression of close vs SMA)
        mid_line = (kc['kc_upper'] + kc['kc_lower']) / 2
        delta = df['close'] - mid_line
        out['squeeze_mom'] = delta.rolling(bb_period).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x)-1) + np.polyfit(np.arange(len(x)), x, 1)[1] if len(x)==bb_period else 0,
            raw=True
        )
        return out

    @staticmethod
    def chaikin_volatility(df: pd.DataFrame, ema_period: int = 10, roc_period: int = 10) -> pd.DataFrame:
        """4. Chaikin Volatility"""
        out = pd.DataFrame(index=df.index)
        hl_diff = df['high'] - df['low']
        hl_ema = hl_diff.ewm(span=ema_period, adjust=False).mean()
        out['chaikin_vol'] = hl_ema.pct_change(roc_period) * 100
        return out

    @staticmethod
    def historical_volatility(df: pd.DataFrame, period: int = 20, trading_periods: int = 252) -> pd.DataFrame:
        """5. Historical Volatility (HV) annualized"""
        out = pd.DataFrame(index=df.index)
        log_ret = np.log(df['close'] / df['close'].shift(1))
        out['hv'] = log_ret.rolling(period).std() * np.sqrt(trading_periods) * 100
        return out

    @staticmethod
    def ulcer_index(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """6. Ulcer Index (Downside risk metric)"""
        out = pd.DataFrame(index=df.index)
        max_high = df['close'].rolling(period).max()
        pct_drawdown = 100 * ((df['close'] - max_high) / max_high)
        squared_dd = pct_drawdown ** 2
        out['ulcer_index'] = np.sqrt(squared_dd.rolling(period).mean())
        return out

    @staticmethod
    def stddev_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """7. Standard Deviation Bands"""
        out = pd.DataFrame(index=df.index)
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        out['stddev_mid'] = sma
        out['stddev_upper'] = sma + (num_std * std)
        out['stddev_lower'] = sma - (num_std * std)
        out['stddev_val'] = std
        return out

    @staticmethod
    def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """8. Choppiness Index (0-100: <38.2 Trending, >61.8 Choppy/Consolidation)"""
        out = pd.DataFrame(index=df.index)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        sum_tr = tr.rolling(period).sum()
        max_hi = df['high'].rolling(period).max()
        min_lo = df['low'].rolling(period).min()
        
        denom = (max_hi - min_lo).replace(0, np.nan)
        out['chop'] = 100 * (np.log10(sum_tr / denom) / np.log10(period))
        return out

    @staticmethod
    def mass_index(df: pd.DataFrame, fast_period: int = 9, sum_period: int = 25) -> pd.DataFrame:
        """9. Mass Index"""
        out = pd.DataFrame(index=df.index)
        hl_diff = df['high'] - df['low']
        ema1 = hl_diff.ewm(span=fast_period, adjust=False).mean()
        ema2 = ema1.ewm(span=fast_period, adjust=False).mean().replace(0, np.nan)
        ratio = ema1 / ema2
        out['mass_index'] = ratio.rolling(sum_period).sum()
        return out

    @staticmethod
    def rvi(df: pd.DataFrame, period: int = 14, std_period: int = 10) -> pd.DataFrame:
        """10. RVI (Relative Volatility Index)"""
        out = pd.DataFrame(index=df.index)
        std = df['close'].rolling(std_period).std()
        delta = df['close'].diff()
        
        up_std = np.where(delta > 0, std, 0.0)
        down_std = np.where(delta < 0, std, 0.0)
        
        avg_up = pd.Series(up_std, index=df.index).ewm(alpha=1/period, adjust=False).mean()
        avg_down = pd.Series(down_std, index=df.index).ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
        
        rvi = 100 * (avg_up / (avg_up + avg_down))
        out['rvi'] = rvi.fillna(50.0)
        return out


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all 30 indicators and joins them with the original OHLCV DataFrame.
    """
    dfs = [
        TrendIndicators.ema_cross(df),
        TrendIndicators.macd(df),
        TrendIndicators.ichimoku(df),
        TrendIndicators.supertrend(df),
        TrendIndicators.adx_di(df),
        TrendIndicators.donchian_channel(df),
        TrendIndicators.vwap_bands(df),
        TrendIndicators.parabolic_sar(df),
        TrendIndicators.keltner_channel(df),
        TrendIndicators.trix(df),
        MomentumIndicators.rsi(df),
        MomentumIndicators.stochastic(df),
        MomentumIndicators.cci(df),
        MomentumIndicators.roc(df),
        MomentumIndicators.mfi(df),
        MomentumIndicators.williams_r(df),
        MomentumIndicators.cmo(df),
        MomentumIndicators.stoch_rsi(df),
        MomentumIndicators.ultimate_oscillator(df),
        MomentumIndicators.tsi(df),
        VolatilityIndicators.bollinger_bands(df),
        VolatilityIndicators.atr_breakout(df),
        VolatilityIndicators.keltner_squeeze(df),
        VolatilityIndicators.chaikin_volatility(df),
        VolatilityIndicators.historical_volatility(df),
        VolatilityIndicators.ulcer_index(df),
        VolatilityIndicators.stddev_bands(df),
        VolatilityIndicators.choppiness_index(df),
        VolatilityIndicators.mass_index(df),
        VolatilityIndicators.rvi(df)
    ]
    
    result = df.copy()
    for ind_df in dfs:
        for col in ind_df.columns:
            result[col] = ind_df[col]

    # --- Price Action & Candle Structure for Contrarian Scalping ---
    high = result['high'].values
    low = result['low'].values
    close = result['close'].values
    open_p = result['open'].values
    n = len(result)

    bar_range = high - low
    bar_body = np.abs(close - open_p)
    avg_range_20 = pd.Series(bar_range, index=result.index).rolling(20, min_periods=1).mean().values
    avg_body_20 = pd.Series(bar_body, index=result.index).rolling(20, min_periods=1).mean().values

    safe_avg_range = np.where(avg_range_20 == 0, np.nan, avg_range_20)
    safe_avg_body = np.where(avg_body_20 == 0, np.nan, avg_body_20)
    safe_bar_range = np.where(bar_range == 0, np.nan, bar_range)

    result['bar_range'] = bar_range
    result['bar_body'] = bar_body
    result['range_ratio'] = np.nan_to_num(bar_range / safe_avg_range, nan=1.0)
    result['body_ratio'] = np.nan_to_num(bar_body / safe_avg_body, nan=1.0)

    upper_wick = high - np.maximum(open_p, close)
    lower_wick = np.minimum(open_p, close) - low
    result['upper_wick_ratio'] = np.nan_to_num(upper_wick / safe_bar_range, nan=0.0)
    result['lower_wick_ratio'] = np.nan_to_num(lower_wick / safe_bar_range, nan=0.0)

    # Consecutive Bull / Bear bar counters
    is_bull = (close > open_p)
    is_bear = (close < open_p)
    consec_bull = np.zeros(n, dtype=int)
    consec_bear = np.zeros(n, dtype=int)
    for i in range(1, n):
        if is_bull[i]:
            consec_bull[i] = consec_bull[i - 1] + 1
        if is_bear[i]:
            consec_bear[i] = consec_bear[i - 1] + 1
    result['consec_bull'] = consec_bull
    result['consec_bear'] = consec_bear
            
    return result


