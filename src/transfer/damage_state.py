"""GaN RFALT 与 LEO T/R 共用的三状态损伤转移模型。

只共享 ``transition``：源域与目标域的观测编码器、观测头和 RUL 头均为
域专属模块。因此 checkpoint 加载接口特意不接受完整模型 state_dict。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DamageStateTransition(nn.Module):
    """状态 ``[d_perm, q_trap, r_th]`` 的一步转移。

    永久损伤和热阻退化用非负增量约束；陷阱状态在恢复工况下可下降，且始终
    截断在 [0, 1]。应力向量由各域各自的 adapter 提供，避免共享观测语义。
    """

    def __init__(self, state_dim: int = 3, stress_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        if state_dim != 3:
            raise ValueError("当前物理契约固定三状态 [d_perm, q_trap, r_th]")
        self.state_dim = state_dim
        self.stress_dim = stress_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + stress_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.recovery_rate = nn.Parameter(torch.tensor(0.25))

    def forward(
        self, previous_state: torch.Tensor, stress: torch.Tensor, normalized_dt: torch.Tensor,
        *, recovery: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if previous_state.shape[-1] != self.state_dim or stress.shape[-1] != self.stress_dim:
            raise ValueError("state 或 stress 最后一维不符合 transition 配置")
        dt = torch.clamp(normalized_dt.to(previous_state.dtype), min=0.0)
        if dt.shape[-1:] != (1,):
            raise ValueError("normalized_dt 必须为 (..., 1)")
        raw = self.net(torch.cat([previous_state, stress], dim=-1))
        d_perm = previous_state[..., 0] + F.softplus(raw[..., 0]) * 1e-3 * dt[..., 0]
        r_th = previous_state[..., 2] + F.softplus(raw[..., 2]) * 1e-3 * dt[..., 0]
        q_trap = torch.clamp(previous_state[..., 1] + torch.tanh(raw[..., 1]) * 0.05 * dt[..., 0], 0.0, 1.0)
        if recovery is not None:
            rate = torch.sigmoid(self.recovery_rate)
            recovered = previous_state[..., 1] * torch.pow(1.0 - rate, dt[..., 0])
            q_trap = torch.where(recovery.bool(), recovered, q_trap)
        return torch.stack([d_perm, q_trap, r_th], dim=-1)


class DamageStateModel(nn.Module):
    """域专属观测编码/头 + 可迁移三状态转移模块。"""

    def __init__(
        self, source_input_dim: int, target_input_dim: int, *, state_dim: int = 3,
        stress_dim: int = 5, hidden_dim: int = 64, target_node_input_dim: int | None = None,
    ):
        super().__init__()
        self.transition = DamageStateTransition(state_dim, stress_dim, hidden_dim)
        self.source_encoder = nn.Sequential(nn.Linear(source_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, state_dim))
        self.target_encoder = nn.Sequential(nn.Linear(target_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, state_dim))
        self.source_stress = nn.Sequential(nn.Linear(source_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, stress_dim))
        self.target_stress = nn.Sequential(nn.Linear(target_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, stress_dim))
        self.source_head = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
        self.target_head = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
        self.rul_head = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.target_node_input_dim = target_node_input_dim
        # 节点接口独立于当前全局训练主干；仅在空间阶段显式传入节点维度时创建。
        if target_node_input_dim is not None:
            self.target_node_encoder = nn.Sequential(
                nn.Linear(target_node_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, state_dim))
            self.target_node_stress = nn.Sequential(
                nn.Linear(target_node_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, stress_dim))
            self.target_node_head = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
        else:
            self.target_node_encoder = None
            self.target_node_stress = None
            self.target_node_head = None

    @staticmethod
    def _bounded_state(raw: torch.Tensor) -> torch.Tensor:
        return torch.stack([F.softplus(raw[..., 0]), torch.sigmoid(raw[..., 1]), F.softplus(raw[..., 2])], dim=-1)

    def encode_source(self, observations: torch.Tensor) -> torch.Tensor:
        return self._bounded_state(self.source_encoder(observations))

    def encode_target(self, observations: torch.Tensor) -> torch.Tensor:
        return self._bounded_state(self.target_encoder(observations))

    def encode_target_nodes(self, node_observations: torch.Tensor) -> torch.Tensor:
        """将可观测子阵节点 `(B, N, F_node)` 编码为独立三状态 `(B, N, 3)`。"""
        if self.target_node_encoder is None or self.target_node_input_dim is None:
            raise RuntimeError("未配置 target_node_input_dim，空间节点接口不可用")
        if node_observations.ndim != 3 or node_observations.shape[-1] != self.target_node_input_dim:
            raise ValueError("node_observations 必须为 (B, N, target_node_input_dim)")
        return self._bounded_state(self.target_node_encoder(node_observations))

    def next_target_node_state(self, node_observations: torch.Tensor, normalized_dt: torch.Tensor,
                               *, recovery: torch.Tensor | None = None) -> torch.Tensor:
        """节点路径复用唯一共享 ``transition``，支持 `(B,N,3)` 批量状态转移。"""
        if self.target_node_stress is None:
            raise RuntimeError("未配置 target_node_input_dim，空间节点接口不可用")
        state = self.encode_target_nodes(node_observations)
        if normalized_dt.shape != (*state.shape[:-1], 1):
            raise ValueError("节点 normalized_dt 必须为 (B, N, 1)")
        return self.transition(state, self.target_node_stress(node_observations), normalized_dt, recovery=recovery)

    def next_source_state(self, observations: torch.Tensor, normalized_dt: torch.Tensor,
                          *, recovery: torch.Tensor | None = None) -> torch.Tensor:
        previous_state = self.encode_source(observations)
        return self.transition(previous_state, self.source_stress(observations), normalized_dt, recovery=recovery)

    def next_target_state(self, observations: torch.Tensor, normalized_dt: torch.Tensor,
                          *, recovery: torch.Tensor | None = None) -> torch.Tensor:
        previous_state = self.encode_target(observations)
        return self.transition(previous_state, self.target_stress(observations), normalized_dt, recovery=recovery)

    def observe_source(self, state: torch.Tensor) -> torch.Tensor:
        return self.source_head(state)

    def observe_target(self, state: torch.Tensor) -> torch.Tensor:
        return self.target_head(state)

    def observe_target_nodes(self, node_state: torch.Tensor) -> torch.Tensor:
        """将每个子阵三状态映射为 `(gain, phase, Pout, PAE)` 四通道预测。"""
        if self.target_node_head is None:
            raise RuntimeError("未配置 target_node_input_dim，空间节点接口不可用")
        if node_state.ndim != 3 or node_state.shape[-1] != self.transition.state_dim:
            raise ValueError("node_state 必须为 (B, N, 3)")
        return self.target_node_head(node_state)

    def predict_target_rul(self, state: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.rul_head(state)).squeeze(-1)

    def transition_state_dict(self) -> dict[str, torch.Tensor]:
        """返回仅可跨域加载的状态转移权重副本。"""
        return {key: value.detach().clone() for key, value in self.transition.state_dict().items()}

    def load_transition_state_dict(self, transition_state: Mapping[str, torch.Tensor]) -> bool:
        """仅加载 Fθ；拒绝任何观测头或 encoder 权重。"""
        expected = set(self.transition.state_dict())
        received = set(transition_state)
        if received != expected:
            raise ValueError("只允许加载完整 transition state_dict")
        self.transition.load_state_dict(dict(transition_state), strict=True)
        return True


# 物理 state_scale：d_perm 取 EOL 物理上限 D_EOL_fixed(config)=0.006；
# r_th = 14·d_perm（源仿真 gan_rfalt_sim.integrate_damage_states）→ EOL 0.084；
# q_trap∈[0,1]。与 run_gan_diagnostics._STATE_SCALE_PHYS 同源，固化自 sim/config 而非数据。
_PHYS_STATE_SCALE = (0.006, 1.0, 0.084)


class PhysicalDamageTransition(nn.Module):
    """Gate 1.1 物理 Fθ 的正式版：归一状态 + 物理 u → 归一下一状态。

    与 ``run_gan_diagnostics.PhysicalTransition`` 同构（Gate 1.1 已验证 holdout
    skill +38%）。状态以物理尺度 ``[D_EOL, 1.0, 14·D_EOL]`` 归一；应力
    ``u=[a_T, s, recovery]`` 由调用方从可观测量构造，绕过域专属 stress MLP。
    d_perm 有界 sigmoid 增量；r_th = r + Δd_perm（归一后 Δr̃=Δd̃，源自源仿真
    r_th=14·d_perm）；q_trap tanh 增量 clamp[0,1]，recovery 段经 net 由 u 第 3 维
    隐式驱动释放。与旧 ``DamageStateTransition`` 的区别：输入是物理 u 而非 stress
    MLP 输出，状态在归一空间转移，d/r 耦合。
    """

    def __init__(self, state_dim: int = 3, stress_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        if state_dim != 3:
            raise ValueError("当前物理契约固定三状态 [d_perm, q_trap, r_th]")
        if stress_dim != 3:
            raise ValueError("物理应力固定三维 [a_T, s, recovery]")
        self.state_dim = state_dim
        self.stress_dim = stress_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + stress_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self, z_norm: torch.Tensor, u_norm: torch.Tensor, normalized_dt: torch.Tensor,
    ) -> torch.Tensor:
        if z_norm.shape[-1] != self.state_dim or u_norm.shape[-1] != self.stress_dim:
            raise ValueError("z_norm 或 u_norm 最后一维不符合物理 transition 配置")
        dt = torch.clamp(normalized_dt.to(z_norm.dtype), min=0.0)
        if dt.shape[-1:] != (1,):
            raise ValueError("normalized_dt 必须为 (..., 1)")
        raw = self.net(torch.cat([z_norm, u_norm], dim=-1))
        dd = torch.sigmoid(raw[..., 0]) * 0.2 * dt[..., 0]
        d_next = z_norm[..., 0] + dd
        q_next = torch.clamp(z_norm[..., 1] + torch.tanh(raw[..., 1]) * 0.3 * dt[..., 0], 0.0, 1.0)
        r_next = z_norm[..., 2] + dd  # d/r 耦合（归一后 Δr̃=Δd̃）
        return torch.stack([d_next, q_next, r_next], dim=-1)


class PhysicalDamageStateModel(nn.Module):
    """域专属 encoder/head + 可迁移物理 transition（无域专属 stress MLP）。

    与 ``DamageStateModel`` 的区别：Fθ 输入是物理应力 u（``[a_T, s, recovery]``，
    由调用方从可观测量构造），不再经过 ``source_stress``/``target_stress`` MLP；
    状态在物理尺度归一空间转移（``state_scale`` 为固定物理常数，与源仿真
    ``D_EOL_fixed`` 及 ``r_th=14·d_perm`` 对齐）。旧 ``DamageStateModel`` 保留以守
    §6-7 预注册口径，本类服务 Gate 2 冻结迁移（``source_F_frozen`` /
    ``random_F_frozen`` / ``target_only``）。

    encoder 输出物理量纲 z（``_bounded_state``），由 ``normalize_state`` 折算到归一
    空间喂 transition，``denormalize_state`` 还原物理量纲；u 归一尺度 ``u_scale``
    由源域训练 u 的 std 拟合（``fit_stress_standardizer``），无均值中心化（与 Gate
    1.1 诊断一致）。
    """

    def __init__(
        self, source_input_dim: int, target_input_dim: int, *, state_dim: int = 3,
        stress_dim: int = 3, hidden_dim: int = 64, state_scale: Sequence[float] | None = None,
        target_node_input_dim: int | None = None,
    ):
        super().__init__()
        if state_dim != 3:
            raise ValueError("当前物理契约固定三状态 [d_perm, q_trap, r_th]")
        if stress_dim != 3:
            raise ValueError("物理应力固定三维 [a_T, s, recovery]")
        self.transition = PhysicalDamageTransition(state_dim, stress_dim, 32)
        self.source_encoder = nn.Sequential(nn.Linear(source_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, state_dim))
        self.target_encoder = nn.Sequential(nn.Linear(target_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, state_dim))
        self.source_head = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
        self.target_head = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
        self.rul_head = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        scale = torch.tensor(state_scale if state_scale is not None else _PHYS_STATE_SCALE, dtype=torch.float32)
        if scale.shape[-1] != state_dim:
            raise ValueError("state_scale 长度必须等于 state_dim")
        if not torch.all(scale > 0):
            raise ValueError("state_scale 必须全正")
        self.register_buffer("state_scale", scale)
        self.register_buffer("u_scale", torch.ones(stress_dim, dtype=torch.float32))
        self.target_node_input_dim = target_node_input_dim
        if target_node_input_dim is not None:
            self.target_node_encoder = nn.Sequential(
                nn.Linear(target_node_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, state_dim))
            self.target_node_head = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
        else:
            self.target_node_encoder = None
            self.target_node_head = None

    @staticmethod
    def _bounded_state(raw: torch.Tensor) -> torch.Tensor:
        return torch.stack([F.softplus(raw[..., 0]), torch.sigmoid(raw[..., 1]), F.softplus(raw[..., 2])], dim=-1)

    def encode_source(self, observations: torch.Tensor) -> torch.Tensor:
        return self._bounded_state(self.source_encoder(observations))

    def encode_target(self, observations: torch.Tensor) -> torch.Tensor:
        return self._bounded_state(self.target_encoder(observations))

    def encode_target_nodes(self, node_observations: torch.Tensor) -> torch.Tensor:
        if self.target_node_encoder is None or self.target_node_input_dim is None:
            raise RuntimeError("未配置 target_node_input_dim，空间节点接口不可用")
        if node_observations.ndim != 3 or node_observations.shape[-1] != self.target_node_input_dim:
            raise ValueError("node_observations 必须为 (B, N, target_node_input_dim)")
        return self._bounded_state(self.target_node_encoder(node_observations))

    def normalize_state(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.state_scale

    def denormalize_state(self, z_norm: torch.Tensor) -> torch.Tensor:
        return z_norm * self.state_scale

    def normalize_stress(self, u: torch.Tensor) -> torch.Tensor:
        return u / self.u_scale

    def fit_stress_standardizer(self, u_train: torch.Tensor | np.ndarray) -> None:
        """以源域训练 u 的 std 拟合 u_scale（无均值中心化，与 Gate 1.1 诊断一致）。"""
        u = torch.as_tensor(u_train, dtype=torch.float32, device=self.u_scale.device)
        if u.shape[-1] != self.transition.stress_dim:
            raise ValueError("u_train 最后一维必须等于 stress_dim")
        self.u_scale.copy_((u.std(dim=0) + 1e-6).clamp_min(1e-6))

    def next_source_state(self, observations: torch.Tensor, u_phys: torch.Tensor,
                          normalized_dt: torch.Tensor) -> torch.Tensor:
        z = self.encode_source(observations)
        z_next_norm = self.transition(self.normalize_state(z), self.normalize_stress(u_phys), normalized_dt)
        return self.denormalize_state(z_next_norm)

    def next_target_state(self, observations: torch.Tensor, u_phys: torch.Tensor,
                          normalized_dt: torch.Tensor) -> torch.Tensor:
        z = self.encode_target(observations)
        z_next_norm = self.transition(self.normalize_state(z), self.normalize_stress(u_phys), normalized_dt)
        return self.denormalize_state(z_next_norm)

    def next_target_node_state(self, node_observations: torch.Tensor, node_u_phys: torch.Tensor,
                               normalized_dt: torch.Tensor) -> torch.Tensor:
        """节点路径复用唯一共享物理 transition，支持 `(B,N,3)` 批量转移。"""
        if self.target_node_encoder is None:
            raise RuntimeError("未配置 target_node_input_dim，空间节点接口不可用")
        z = self.encode_target_nodes(node_observations)
        if normalized_dt.shape != (*z.shape[:-1], 1):
            raise ValueError("节点 normalized_dt 必须为 (B, N, 1)")
        z_next_norm = self.transition(self.normalize_state(z), self.normalize_stress(node_u_phys), normalized_dt)
        return self.denormalize_state(z_next_norm)

    def observe_source(self, state: torch.Tensor) -> torch.Tensor:
        return self.source_head(state)

    def observe_target(self, state: torch.Tensor) -> torch.Tensor:
        return self.target_head(state)

    def observe_target_nodes(self, node_state: torch.Tensor) -> torch.Tensor:
        if self.target_node_head is None:
            raise RuntimeError("未配置 target_node_input_dim，空间节点接口不可用")
        if node_state.ndim != 3 or node_state.shape[-1] != self.transition.state_dim:
            raise ValueError("node_state 必须为 (B, N, 3)")
        return self.target_node_head(node_state)

    def predict_target_rul(self, state: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.rul_head(state)).squeeze(-1)

    def transition_state_dict(self) -> dict[str, torch.Tensor]:
        """返回仅可跨域加载的物理转移权重副本（Fθ）。"""
        return {key: value.detach().clone() for key, value in self.transition.state_dict().items()}

    def load_transition_state_dict(self, transition_state: Mapping[str, torch.Tensor]) -> bool:
        """仅加载 Fθ；拒绝任何观测头或 encoder 权重。"""
        expected = set(self.transition.state_dict())
        received = set(transition_state)
        if received != expected:
            raise ValueError("只允许加载完整 transition state_dict")
        self.transition.load_state_dict(dict(transition_state), strict=True)
        return True
