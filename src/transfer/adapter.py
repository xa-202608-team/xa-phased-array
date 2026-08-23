"""transfer/adapter.py — 迁移模型。

两套并列接口 (CLAUDE.md 约束 2: 迁移落在 HI 动力学层):

[旧 · 观测层迁移] 飞轮仍在用, 保留做相控阵消融基线
  - `Adapter`           g_φ: 目标域 x_T → 源域编码器输入维度
  - `TransferModel`     adapter(x_T) → 冻结 E_θ → HI/RUL 头 (三阶段 S1/S2/S3)

[新 · HI 动力学层迁移] 相控阵 P7 增益诊断 D2 重构 (见
  docs/开发推进计划/hi_layer_refactor_design.md §3.1)
  - `HISeqEncoder`      HI 序列编码器 (固定 n_features=2: [HI, ΔHI])
  - `HIDynamicsModel`   HISeqEncoder + hi_head + rul_head (无 adapter 瓶颈)
  - `TargetAuxHead`     目标域 12 维 obs → RUL 校正量 (可选末端晚融合)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.tcn_encoder import TCNEncoder, LSTMEncoder, GRUEncoder   # noqa: E402


class Adapter(nn.Module):
    """g_φ: 目标域 x_T (8维) → 投影到源域编码器输入维度 (12维), 送入冻结编码器。"""

    def __init__(self, n_in: int, n_out: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):                 # (B, L, n_in) -> (B, L, n_out)
        return self.net(x)


class TransferModel(nn.Module):
    """adapter g_φ(x_T) → 编码器 E_θ → HI/RUL 头。

    encoder/heads 命名与源域 RULModel 一致, load_pretrained(strict=False) 可加载
    源域权重 (adapter 保持随机初始化)。
    """

    def __init__(self, encoder_type: str = "tcn", n_features: int = 12, n_target: int = 8,
                 input_len: int = 64, channels: int = 64, kernel_size: int = 5,
                 num_blocks: int = 4, dropout: float = 0.1, latent_dim: int = 64,
                 adapter_hidden: int = 64, gru_hidden: int = 64, gru_layers: int = 2,
                 gru_dropout: float | None = None):
        super().__init__()
        self.adapter = Adapter(n_target, n_features, adapter_hidden)
        # F1-A: GRU 架构参数显式化 (旧版吃 GRUEncoder 默认 64/2/0.1, 训练/导出/推理间隐式漂移)
        _gru_dropout = dropout if gru_dropout is None else gru_dropout
        if encoder_type == "tcn":
            self.encoder = TCNEncoder(n_features, channels, kernel_size, num_blocks, dropout, latent_dim)
        elif encoder_type == "lstm":
            self.encoder = LSTMEncoder(n_features, latent_dim=latent_dim)
        elif encoder_type == "gru":
            self.encoder = GRUEncoder(n_features, hidden=gru_hidden, num_layers=gru_layers,
                                      dropout=_gru_dropout, latent_dim=latent_dim)
        else:
            raise ValueError(f"未知 encoder: {encoder_type}")
        self.hi_head = nn.Linear(latent_dim, 1)
        self.rul_head = nn.Linear(latent_dim, 1)

    def forward(self, x):                 # x: (B, L, n_target)
        h = self.adapter(x)
        z = self.encoder(h)
        hi = torch.sigmoid(self.hi_head(z)).squeeze(-1)
        rul = F.softplus(self.rul_head(z)).squeeze(-1)
        return hi, rul, z

    def load_pretrained(self, ckpt_path, device: str = "cpu"):
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        # 只迁移 encoder (退化动力学); heads 任务相关、RUL 量级不同, 重新训练
        sd_enc = {k: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = self.load_state_dict(sd_enc, strict=False)
        return missing, unexpected

    def freeze_encoder(self, freeze: bool = True):
        for p in self.encoder.parameters():
            p.requires_grad_(not freeze)


# ============================================================
# HI 动力学层迁移 (路径 A, 设计 §3.1)
# ============================================================
#
# encoder 只看 [HI, ΔHI] 两维输入 — 源域 (NASA MOSFET) 与目标域 (相控阵)
# 输入语义完全统一 (归一化健康轨迹 + 一阶差分), 不再需要 adapter 弥合异构观测。
# 源域学到的 "给定 HI 轨迹形状 → 距失效多远 / 退化阶段" 是域无关的退化动力学知识。
# ============================================================


class HISeqEncoder(nn.Module):
    """HI 动力学共享编码器。

    输入: x_HI ∈ R^(B, L, 2), 通道维 2 = [HI(t), ΔHI(t)]。
    输出: z ∈ R^(B, latent_dim)。

    参数与现有 TCN/LSTM/GRU 对齐 (channels/kernel_size/num_blocks/dropout/latent_dim),
    仅 n_features 固定为 2 (HI + ΔHI), 这是跨域统一语义的根基。

    内部直接包装 src/models/tcn_encoder.py 的 TCNEncoder/LSTMEncoder/GRUEncoder,
    以 n_features=2 实例化; 不重写 TCN。
    """

    def __init__(self,
                 encoder_type: str = "tcn",      # "tcn" | "lstm" | "gru"
                 input_len: int = 64,            # 保留接口字段 (底层 encoder 当前不使用)
                 channels: int = 64,
                 kernel_size: int = 5,
                 num_blocks: int = 4,
                 dropout: float = 0.1,
                 latent_dim: int = 64,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 2):
        super().__init__()
        self.encoder_type = encoder_type
        self.input_len = input_len
        # n_features 固定为 2 (HI + ΔHI), 跨域统一语义
        if encoder_type == "tcn":
            self.encoder = TCNEncoder(
                n_features=2, channels=channels, kernel_size=kernel_size,
                num_blocks=num_blocks, dropout=dropout, latent_dim=latent_dim)
        elif encoder_type == "lstm":
            self.encoder = LSTMEncoder(
                n_features=2, hidden=lstm_hidden, num_layers=lstm_layers,
                dropout=dropout, latent_dim=latent_dim)
        elif encoder_type == "gru":
            self.encoder = GRUEncoder(
                n_features=2, hidden=lstm_hidden, num_layers=lstm_layers,
                dropout=dropout, latent_dim=latent_dim)
        else:
            raise ValueError(f"未知 encoder_type: {encoder_type} (可选 tcn|lstm|gru)")

    def forward(self, x_HI: torch.Tensor) -> torch.Tensor:
        """x_HI: (B, L, 2) [HI, ΔHI] → z: (B, latent_dim)。"""
        return self.encoder(x_HI)


class HIDynamicsModel(nn.Module):
    """HI 动力学迁移模型 = HISeqEncoder + HI_head + RUL_head。

    替代 transfer/adapter.py:TransferModel。无 adapter, 无 n_target/n_features 瓶颈。
    forward 接收已构造好的 x_HI 序列 (B, L, 2), 不接收原始观测。

    源域预训练 (src/train/pretrain.py --hi-layer) 与目标域迁移 (src/transfer/
    train_transfer.py --hi-layer) 共用此模型, 保证 encoder.* keys 完全一致,
    load_pretrained 可用 strict=True 加载。
    """

    def __init__(self,
                 encoder_type: str = "tcn",
                 input_len: int = 64,
                 channels: int = 64,
                 kernel_size: int = 5,
                 num_blocks: int = 4,
                 dropout: float = 0.1,
                 latent_dim: int = 64,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 2):
        super().__init__()
        self.encoder = HISeqEncoder(
            encoder_type=encoder_type, input_len=input_len, channels=channels,
            kernel_size=kernel_size, num_blocks=num_blocks, dropout=dropout,
            latent_dim=latent_dim, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers)
        self.hi_head = nn.Linear(latent_dim, 1)
        self.rul_head = nn.Linear(latent_dim, 1)

    def forward(self, x_HI: torch.Tensor):
        """x_HI: (B, L, 2) → (hi∈[0,1] (B,), rul∈R+ (B,), z (B, latent_dim))。"""
        z = self.encoder(x_HI)
        hi = torch.sigmoid(self.hi_head(z)).squeeze(-1)
        rul = F.softplus(self.rul_head(z)).squeeze(-1)
        return hi, rul, z

    def load_pretrained(self, ckpt_path, device: str = "cpu"):
        """加载源域预训练 HISeqEncoder 权重。

        要求源域预训练 checkpoint 是用 HIDynamicsModel 训出来的
        (见 src/train/pretrain.py --hi-layer), 其 state_dict 包含
        encoder.* / hi_head.* / rul_head.*。

        严格 strict=True 加载 encoder.* 到 self.encoder (HISeqEncoder);
        hi_head / rul_head 不加载 (任务相关、RUL 量级不同, 重新训练)。
        形状完全一致 (F_in=2), strict=True 保证不会静默漏权重。
        """
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        # 过滤 encoder.* 并去掉前缀, 加载到 self.encoder (HISeqEncoder)
        sd_enc = {k[len("encoder."):]: v for k, v in sd.items()
                  if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(sd_enc, strict=True)
        return missing, unexpected

    def freeze_encoder(self, freeze: bool = True):
        """冻结 (S2) / 解冻 (S3) HISeqEncoder 参数; heads 不受影响。"""
        for p in self.encoder.parameters():
            p.requires_grad_(not freeze)


class TargetAuxHead(nn.Module):
    """目标域 12 维 obs → RUL 校正量 (可选末端晚融合, target-only)。

    设计 §2.4: 12 维目标域独有观测 (x_global 6 + x_nodes.mean 6) 不进共享 encoder,
    以免污染 HI 迁移语义; 经此 head 提取一个 residual RUL 校正量, 末端加到共享
    RUL_head 输出:  rul_final = rul_shared + α · rul_aux  (α 从 config 读, 默认 0)。

    默认 alpha=0 时此模块不参与 forward, 只保留接口供后续实验。
    """

    def __init__(self, n_target: int = 12, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_target, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_obs_last: torch.Tensor) -> torch.Tensor:
        """x_obs_last: (B, n_target) 窗末时刻的 obs → rul_correction: (B,)。"""
        return self.net(x_obs_last).squeeze(-1)
