"""
Patch-Based Transformer Architecture (PatchTST) for Financial Tick Data
- Segments continuous multivariate tick features into overlapping patches
- Uses Transformer Encoder with self-attention across patches
- Projection head for multiclass signals (-1, 0, 1) or return regression
"""
import torch
import torch.nn as nn
from typing import Optional


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int = 4, stride: int = 2, d_model: int = 64):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Channels, SeqLen]
        # Unfold into patches
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, NumPatches, PatchLen]
        b, c, num_p, p_len = patches.shape
        patches = patches.contiguous().view(b * c, num_p, p_len)
        embeddings = self.proj(patches)  # [B*C, NumPatches, d_model]
        return embeddings, b, c, num_p


class PatchTST(nn.Module):
    def __init__(
        self,
        num_features: int,
        seq_len: int = 32,
        patch_len: int = 4,
        stride: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        task_type: str = "multiclass",
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.task_type = task_type

        self.patch_embed = PatchEmbedding(patch_len=patch_len, stride=stride, d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Head
        num_patches = (seq_len - patch_len) // stride + 1
        flatten_dim = num_features * num_patches * d_model
        out_dim = num_classes if task_type == "multiclass" else 1

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, SeqLen, NumFeatures] -> permute to [Batch, NumFeatures, SeqLen]
        x = x.transpose(1, 2)
        embed, b, c, num_p = self.patch_embed(x)
        encoded = self.transformer(embed)  # [B*C, NumPatches, d_model]
        # Reshape back to [B, C * NumPatches * d_model]
        d_m = encoded.shape[-1]
        out_repr = encoded.contiguous().view(b, c * num_p * d_m)
        out = self.head(out_repr)
        return out
