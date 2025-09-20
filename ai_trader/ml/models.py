from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_features: int
    seq_len: int
    conv_channels: int = 64
    lstm_hidden: int = 128
    lstm_layers: int = 1
    attn_heads: int = 4
    dropout: float = 0.1


class CNNLSTMAttn(nn.Module):
    """
    Input: x [B, T, F]
    - Conv1d over temporal dimension (features as channels after permute)
    - BiLSTM over time
    - Multi-Head Attention over sequence with a learned CLS token to aggregate
    - Regression head -> scalar
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Conv over time: treat features as channels
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.input_features, cfg.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cfg.conv_channels, cfg.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Project conv channels to LSTM input feature size
        self.conv_proj = nn.Linear(cfg.conv_channels, cfg.lstm_hidden)

        self.lstm = nn.LSTM(
            input_size=cfg.lstm_hidden,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        attn_embed = cfg.lstm_hidden * 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, attn_embed))
        self.attn = nn.MultiheadAttention(embed_dim=attn_embed, num_heads=cfg.attn_heads, batch_first=True, dropout=cfg.dropout)
        self.norm = nn.LayerNorm(attn_embed)

        self.head = nn.Sequential(
            nn.Linear(attn_embed, attn_embed // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(attn_embed // 2, 1),
        )

        self._reset()

    def _reset(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        b, t, f = x.shape
        # Conv over time: need [B, C=F, T]
        x_c = x.permute(0, 2, 1)
        h = self.conv(x_c)  # [B, Cc, T]
        h = h.permute(0, 2, 1)  # [B, T, Cc]
        h = self.conv_proj(h)   # [B, T, H]

        h, _ = self.lstm(h)  # [B, T, 2H]
        # prepend CLS token
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, h], dim=1)  # [B, 1+T, 2H]
        # Self-attention with CLS as query position
        out, _ = self.attn(seq, seq, seq)  # [B, 1+T, 2H]
        out = self.norm(out)
        cls_out = out[:, 0]  # [B, 2H]
        y = self.head(cls_out).squeeze(-1)  # [B]
        return y
