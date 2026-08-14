"""models/tcn_encoder.py

共享编码器 + HI/RUL 双头 (plan §5.5)。

编码器可选:
  tcn  — 残差因果膨胀卷积 (主), 适合长程退化动力学
  lstm / gru — 基线对照 (--encoder lstm/gru)

输入 X ∈ R^(B, L, F): B batch, L 滑窗长, F 特征维。
输出 (hi, rul, z):
  hi  ∈ [0,1]   健康指标 (退化程度, 大=严重)
  rul ∈ R+      剩余寿命 (softplus 保证非负)
  z   ∈ R^D     latent (迁移阶段域对齐用)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """残差因果膨胀卷积块 (两 conv + dropout + residual)。

    因果性: 用左 padding=(k-1)*dilation 后裁掉右侧, 保证不窥探未来。
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):               # x: (B, C, L)
        L = x.size(2)
        out = F.relu(self.conv1(x)[:, :, :L])   # 裁右 -> 因果
        out = self.drop(out)
        out = F.relu(self.conv2(out)[:, :, :L])
        out = self.drop(out)
        return F.relu(out + self.res(x))


class TCNEncoder(nn.Module):
    def __init__(self, n_features: int, channels: int = 64, kernel_size: int = 5,
                 num_blocks: int = 4, dropout: float = 0.1, latent_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, channels, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(channels, channels, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_blocks)
        ])
        self.out_proj = nn.Linear(channels, latent_dim)

    def forward(self, x):               # x: (B, L, F)
        h = x.transpose(1, 2)           # (B, F, L)
        h = self.input_proj(h)
        for blk in self.blocks:
            h = blk(h)
        h = h.mean(dim=2)               # 全局平均池化 -> (B, channels)
        return self.out_proj(h)         # (B, latent)


class LSTMEncoder(nn.Module):
    def __init__(self, n_features, hidden=64, num_layers=2, dropout=0.1, latent_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, latent_dim)

    def forward(self, x):               # (B, L, F)
        out, _ = self.lstm(x)
        return self.proj(out[:, -1, :])


class GRUEncoder(nn.Module):
    def __init__(self, n_features, hidden=64, num_layers=2, dropout=0.1, latent_dim=64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, latent_dim)

    def forward(self, x):               # (B, L, F)
        out, _ = self.gru(x)
        return self.proj(out[:, -1, :])


class RULModel(nn.Module):
    """编码器 + HI 头 + RUL 头。"""

    def __init__(self, encoder_type: str = "tcn", n_features: int = 12, input_len: int = 64,
                 channels: int = 64, kernel_size: int = 5, num_blocks: int = 4,
                 dropout: float = 0.1, latent_dim: int = 64,
                 lstm_hidden: int = 64, lstm_layers: int = 2):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "tcn":
            self.encoder = TCNEncoder(n_features, channels, kernel_size, num_blocks,
                                      dropout, latent_dim)
        elif encoder_type == "lstm":
            self.encoder = LSTMEncoder(n_features, lstm_hidden, lstm_layers, dropout, latent_dim)
        elif encoder_type == "gru":
            self.encoder = GRUEncoder(n_features, lstm_hidden, lstm_layers, dropout, latent_dim)
        else:
            raise ValueError(f"未知 encoder: {encoder_type} (可选 tcn|lstm|gru)")
        self.hi_head = nn.Linear(latent_dim, 1)
        self.rul_head = nn.Linear(latent_dim, 1)

    def forward(self, x):               # x: (B, L, F)
        z = self.encoder(x)             # (B, latent)
        hi = torch.sigmoid(self.hi_head(z)).squeeze(-1)
        rul = F.softplus(self.rul_head(z)).squeeze(-1)
        return hi, rul, z

    def encode(self, x):
        return self.encoder(x)
