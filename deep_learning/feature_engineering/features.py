"""
Feature Engineering Engine for Tick Bars
Extracts 4 Feature Families with Strict Lag (Shift = 1) to Prevent Lookahead Bias:
1. Price & Volatility: Returns, ATR, EMAs, RSI, MACD, Bollinger Bands
2. Volume & Order Flow: Buy/Sell Imbalance, Volume Surge, VWAP Disparity
3. Time Features: Session Progress (Open/Close Proximity), Day of Week
4. Tick Bar Specific: Bar Generation Velocity (Ticks/Sec = Market Activity Proxy)
"""
from typing import List, Tuple
import numpy as np
import pandas as pd


def compute_features(df_bars: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Computes lookahead-free features for tick bars.
    All indicator values are computed and then shifted by 1 so that features
    at time t only use information strictly up to t-1.
    """
    if len(df_bars) < 50:
        return pd.DataFrame(), []

    df = df_bars.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    vwap = df['vwap']
    duration = df['bar_duration_sec'].clip(lower=0.1)

    feat = pd.DataFrame(index=df.index)
    feat['datetime'] = df['datetime']
    feat['close_raw'] = close

    # -------------------------------------------------------------
    # 1. Price & Volatility Indicators
    # -------------------------------------------------------------
    # Returns across multiple horizons
    feat['ret_1'] = close.pct_change(1)
    feat['ret_3'] = close.pct_change(3)
    feat['ret_5'] = close.pct_change(5)
    feat['ret_10'] = close.pct_change(10)

    # True Range & ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()
    feat['atr_14'] = atr_14
    feat['atr_pct'] = atr_14 / (close + 1e-6)

    # Normalized Distance from Moving Averages
    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_30 = close.ewm(span=30, adjust=False).mean()
    feat['ema_diff_10'] = (close - ema_10) / (atr_14 + 1e-6)
    feat['ema_diff_30'] = (close - ema_30) / (atr_14 + 1e-6)
    feat['ema_ratio_10_30'] = (ema_10 - ema_30) / (atr_14 + 1e-6)

    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    feat['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feat['macd_hist'] = (macd_line - macd_signal) / (atr_14 + 1e-6)

    # Bollinger Bands
    ma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    feat['bb_pct_b'] = (close - (ma_20 - 2 * std_20)) / (4 * std_20 + 1e-8)
    feat['bb_bandwidth'] = (4 * std_20) / (ma_20 + 1e-8)

    # -------------------------------------------------------------
    # 2. Volume & Order Flow Indicators
    # -------------------------------------------------------------
    vol_ma_20 = volume.rolling(20).mean()
    feat['vol_surge_ratio'] = volume / (vol_ma_20 + 1e-6)
    feat['vwap_disparity'] = (close - vwap) / (atr_14 + 1e-6)
    feat['buy_ratio'] = df['buy_ratio']
    feat['buy_ratio_ma5'] = df['buy_ratio'].rolling(5).mean()

    # -------------------------------------------------------------
    # 3. Time-Based Indicators
    # -------------------------------------------------------------
    dt = pd.to_datetime(df['datetime'])
    # Minute of trading day (from 09:00 -> 0 to 405 mins at 15:45)
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute - 540  # 9 * 60 = 540
    feat['session_progress'] = minute_of_day.clip(lower=0, upper=405) / 405.0
    feat['is_open_hour'] = np.where(minute_of_day <= 60, 1.0, 0.0)
    feat['is_close_hour'] = np.where(minute_of_day >= 360, 1.0, 0.0)
    feat['day_of_week'] = dt.dt.dayofweek / 4.0

    # -------------------------------------------------------------
    # 4. Tick Bar Specific Velocity (Activity Proxy)
    # -------------------------------------------------------------
    ticks_count = df['tick_count'] if 'tick_count' in df.columns else 100
    feat['ticks_per_sec'] = ticks_count / duration
    feat['ticks_per_sec_log'] = np.log1p(feat['ticks_per_sec'])

    # -------------------------------------------------------------
    # STRICT SHIFT (Shift = 1) for all predictive features
    # -------------------------------------------------------------
    feature_cols = [c for c in feat.columns if c not in ['datetime', 'close_raw']]
    for c in feature_cols:
        feat[c] = feat[c].shift(1)

    # Drop NaNs created by rolling and shifting
    feat.dropna(inplace=True)
    return feat, feature_cols
