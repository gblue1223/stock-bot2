from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_features: int
    seq_len: int
    
    # 옵션 1: 중간 크기 (약 150만 파라미터)
    conv_channels: int = 128      # 64 → 128 (더 많은 특성 추출)
    lstm_hidden: int = 256        # 128 → 256 (더 큰 은닉 상태)
    lstm_layers: int = 2          # 1 → 2 (더 깊은 시간 모델링)
    attn_heads: int = 8           # 4 → 8 (더 세밀한 attention)
    dropout: float = 0.1

    # 분류 지원: num_classes >= 2 이면 분류 logits 출력
    # 기존 회귀와의 하위호환을 위해 기본값은 1
    num_classes: int = 1
    task: str = "regression"  # optional hint
    
    # 옵션 2: 대형 모델 (약 400만 파라미터)
    # conv_channels: int = 256      # 64 → 256
    # lstm_hidden: int = 512        # 128 → 512
    # lstm_layers: int = 3          # 1 → 3
    # attn_heads: int = 16          # 4 → 16
    # dropout: int = 0.15           # 과적합 방지

    # 옵션 3: 초대형 모델 (약 1000만 파라미터)
    # conv_channels: int = 512       # 64 → 512
    # lstm_hidden: int = 1024        # 128 → 1024
    # lstm_layers: int = 4           # 1 → 4
    # attn_heads: int = 32           # 4 → 32
    # dropout: int = 0.2             # 더 강한 정규화


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

        out_dim = 1 if (cfg.num_classes is None or int(cfg.num_classes) <= 1) else int(cfg.num_classes)
        self.head = nn.Sequential(
            nn.Linear(attn_embed, attn_embed // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(attn_embed // 2, out_dim),
        )

        self._reset()

    def _reset(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
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
        return cls_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return compressed state vector [B, D]."""
        return self._forward_features(x)

    @property
    def state_dim(self) -> int:
        return self.cfg.lstm_hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        cls_out = self._forward_features(x)
        y = self.head(cls_out)
        # 회귀(1차원)인 경우 [B]로 squeeze, 분류는 [B, C] 유지
        if y.dim() == 2 and y.size(-1) == 1:
            y = y.squeeze(-1)
        return y
