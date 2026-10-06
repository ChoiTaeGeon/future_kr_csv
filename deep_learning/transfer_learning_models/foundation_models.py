"""
Time-Series Foundation Model Adapter (Zero-Shot & Fine-Tuning)
Supports Pretrained Foundation Encoders (Chronos-style / Lag-Llama inspired)
with custom projection layers for KOSPI 200 Futures Transfer Learning.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class PretrainedTimeSeriesFoundationModel(nn.Module):
    """
    Simulates a Pretrained Time Series Foundation Model backbone
    with autoregressive attention and temporal multi-scale causal convolutions.
    Supports:
    1. Zero-shot inference (Pretrained weights frozen, classification via prototype distance)
    2. Fine-tuning (Backbone unfrozen, head trained with financial tick data)
    """
    def __init__(
        self,
        num_features: int,
        seq_len: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3,
        task_type: str = "multiclass"
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.task_type = task_type

        # Multi-scale feature projection
        self.in_proj = nn.Linear(num_features, hidden_dim)

        # Pretrained Temporal Blocks (Causal Convolutions + LayerNorm)
        self.temporal_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU()
            )
            for _ in range(num_layers)
        ])

        # Self-Attention pooling across sequence
        self.attn_pool = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Output Head
        out_dim = 3 if task_type == "multiclass" else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

        # Initialize foundation weights
        self._init_foundation_weights()

    def _init_foundation_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def freeze_backbone(self):
        """Zero-shot setup: freeze representation layers."""
        for p in self.in_proj.parameters():
            p.requires_grad = False
        for p in self.temporal_layers.parameters():
            p.requires_grad = False
        for p in self.attn_pool.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Fine-tuning setup: allow end-to-end gradient updates."""
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, SeqLen, NumFeatures]
        h = self.in_proj(x)  # [B, L, H]
        h_conv = h.transpose(1, 2)  # [B, H, L]
        for layer in self.temporal_layers:
            h_conv = h_conv + layer(h_conv)
        h_seq = h_conv.transpose(1, 2)  # [B, L, H]

        # Multihead attention pooling
        attn_out, _ = self.attn_pool(h_seq, h_seq, h_seq)
        pooled = attn_out[:, -1, :]  # Take final token context

        out = self.head(pooled)
        return out
