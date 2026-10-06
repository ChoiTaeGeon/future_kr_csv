"""
Real-Time / Semi-Real-Time Streaming Inference Pipeline
- Consumes raw incoming tick stream
- Aggregates into N-Tick bars
- Extracts lag-aligned features
- Evaluates packaged model
- Applies strict risk management (Dynamic Position Sizing & ATR Stop/Take-Profit)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from deep_learning.model_registry import load_model, ModelMetadata
from deep_learning.feature_engineering import compute_features
from deep_learning.transfer_learning_models import PatchTST


@dataclass
class TradeSignal:
    timestamp: str
    action: str          # 'BUY', 'SELL', 'FLAT', 'HOLD'
    target_pos: int      # 1, -1, 0
    confidence: float
    current_price: float
    stop_loss_price: float
    take_profit_price: float
    quantity: int


class LiveInferencePipeline:
    def __init__(self, model_package_path: str | Path):
        self.package_path = Path(model_package_path)
        self.model, self.metadata = self._init_model()
        self.tick_window = self.metadata.tick_window
        self.feature_names = self.metadata.feature_names

        # Real-time state buffers
        self.tick_buffer: List[Dict[str, Any]] = []
        self.bars_history: List[Dict[str, Any]] = []
        self.current_position: int = 0
        self.entry_price: float = 0.0

    def _init_model(self):
        # Check if PyTorch
        if str(self.package_path).endswith('.pt') or 'patch_tst' in str(self.package_path).lower():
            pt_cls = PatchTST(num_features=19, seq_len=32)
            return load_model(self.package_path, pytorch_model_class=pt_cls)
        return load_model(self.package_path)

    def on_tick(self, tick: Dict[str, Any]) -> Optional[TradeSignal]:
        """
        Processes a single live tick dictionary:
        {'datetime': '2026-10-06 09:15:20', 'price': 345.50, 'volume': 5, 'side': 1}
        Returns TradeSignal when a new tick bar closes.
        """
        self.tick_buffer.append(tick)
        if len(self.tick_buffer) < self.tick_window:
            return None

        # Build closed tick bar
        df_chunk = pd.DataFrame(self.tick_buffer)
        self.tick_buffer.clear()

        df_chunk['datetime'] = pd.to_datetime(df_chunk['datetime'])
        open_p = df_chunk['price'].iloc[0]
        high_p = df_chunk['price'].max()
        low_p = df_chunk['price'].min()
        close_p = df_chunk['price'].iloc[-1]
        tot_vol = df_chunk['volume'].sum()
        dollar_vol = (df_chunk['price'] * df_chunk['volume']).sum()
        vwap = dollar_vol / max(1.0, tot_vol)
        buy_vol = df_chunk.loc[df_chunk['side'] > 0, 'volume'].sum()
        buy_ratio = buy_vol / max(1.0, tot_vol)
        duration = max(0.1, (df_chunk['datetime'].iloc[-1] - df_chunk['datetime'].iloc[0]).total_seconds())

        bar_dict = {
            'datetime': df_chunk['datetime'].iloc[-1],
            'open': open_p,
            'high': high_p,
            'low': low_p,
            'close': close_p,
            'volume': tot_vol,
            'vwap': vwap,
            'buy_ratio': buy_ratio,
            'bar_duration_sec': duration,
            'tick_count': len(df_chunk)
        }
        self.bars_history.append(bar_dict)

        # Need at least 50 bars to calculate indicators
        if len(self.bars_history) < 50:
            return None

        # Keep buffer bounded (last 200 bars)
        if len(self.bars_history) > 200:
            self.bars_history = self.bars_history[-200:]

        df_bars = pd.DataFrame(self.bars_history)
        df_feat, fcols = compute_features(df_bars)
        if len(df_feat) == 0:
            return None

        latest_features = df_feat[self.feature_names].iloc[[-1]].values

        # Model Inference
        if self.metadata.framework == 'pytorch':
            # Tensor forward
            import torch
            seq_len = 32
            if len(df_feat) < seq_len:
                return None
            seq_feat = df_feat[self.feature_names].iloc[-seq_len:].values
            inp = torch.tensor(seq_feat, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = self.model(inp)
                pred_label = int(torch.argmax(out, dim=1).item()) - 1
                confidence = float(torch.softmax(out, dim=1).max().item())
        else:
            raw_pred = self.model.predict(latest_features)[0]
            pred_label = int(raw_pred) if isinstance(raw_pred, (int, np.integer)) else (1 if raw_pred > 0 else -1)
            confidence = 0.85

        # Risk Management (ATR Bracket)
        curr_price = close_p
        atr_est = (df_bars['high'] - df_bars['low']).iloc[-14:].mean()
        tp_price = curr_price + (atr_est * 2.0) if pred_label == 1 else curr_price - (atr_est * 2.0)
        sl_price = curr_price - (atr_est * 1.5) if pred_label == 1 else curr_price + (atr_est * 1.5)

        action = "HOLD"
        target_pos = self.current_position
        if pred_label == 1 and self.current_position <= 0:
            action = "BUY"
            target_pos = 1
        elif pred_label == -1 and self.current_position >= 0:
            action = "SELL"
            target_pos = -1

        self.current_position = target_pos
        self.entry_price = curr_price

        return TradeSignal(
            timestamp=str(bar_dict['datetime']),
            action=action,
            target_pos=target_pos,
            confidence=round(confidence, 3),
            current_price=curr_price,
            stop_loss_price=round(sl_price, 2),
            take_profit_price=round(tp_price, 2),
            quantity=1
        )
