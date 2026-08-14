"""GaN RFALT→LEO 损伤状态迁移路线的诊断脚本（设计文档 §9.4 廉价诊断）。

不修改既有预注册实验口径，只产出机制分析报告。诊断是探索性分析，不进入 IID
主验收、不构成正迁移结论依据；目的只在跑弱监督矩阵（§9.2）前判断负结果的
机制解释是否成立。

诊断 1（源域留出泛化）：源域按器件留出，验证预训练 Fθ 在未见器件上的闭环六步
状态 rollout 是否优于 persistence 与 random transition；顺带为 ``pretrain_source_transition``
补 val 选模（既有预训练固定轮数无源域 val，"源域已学好"缺独立证据）。
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from torch.nn import functional as F

from src.experiments.run_gan_transfer import (
    DomainRows, TrainOnlyStandardizer, _source_loss, _source_rows, make_transition_pairs,
)
from src.sim.physical_stress import (
    PHYSICAL_STRESS_SCHEMA,
    TARGET_PIN_DBM_REFERENCE,
    TARGET_VSWR_REFERENCE,
    build_physical_stress,
)
from src.transfer.damage_state import DamageStateModel, DamageStateTransition
from src.utils import load_config, set_seed

ROLLOUT_STEPS = 6


def _subset(rows: DomainRows, keep_ids) -> DomainRows:
    keep_ids = list(keep_ids)
    mask = np.isin(rows.ids, keep_ids)
    link = None if rows.array_link_margin is None else rows.array_link_margin[mask]
    return DomainRows(
        rows.x[mask], rows.states[mask], rows.obs_labels[mask], rows.rul[mask],
        rows.event[mask], rows.ids[mask], rows.time[mask], rows.transition_time_scale_s, link,
    )


def _replace_x(rows: DomainRows, new_x) -> DomainRows:
    """标准化后重建 DomainRows（仅替换 x，其余字段原样保留）。"""
    return DomainRows(
        np.asarray(new_x, dtype=np.float32), rows.states, rows.obs_labels, rows.rul,
        rows.event, rows.ids, rows.time, rows.transition_time_scale_s, rows.array_link_margin,
    )


def split_devices(ids, *, n_holdout, n_val, seed):
    """按器件随机划分 train/val/holdout，三者互斥；holdout 是诊断专用未见器件集。"""
    uniq = np.array(sorted(np.unique(ids).tolist()))
    if len(uniq) < n_holdout + n_val + 1:
        raise ValueError(f"源域器件数 {len(uniq)} 不足以同时留出 {n_holdout} holdout + {n_val} val + ≥1 train")
    rng = np.random.default_rng(seed)
    perm = uniq.copy()
    rng.shuffle(perm)
    holdout = perm[:n_holdout].tolist()
    val = perm[n_holdout:n_holdout + n_val].tolist()
    train = perm[n_holdout + n_val:].tolist()
    return train, val, holdout


def train_source_with_val(model, train, val, *, epochs, device, patience=5, state_std=None):
    """源域 Fθ 训练 + val early stop（补既有 pretrain 固定轮数、无 val 选模的缺口）。"""
    optimizer = Adam(model.parameters(), lr=2e-3)
    best = None
    best_val = float("inf")
    stale = 0
    done = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = (_source_loss_balanced(model, train, device, state_std) if state_std is not None
                else _source_loss(model, train, device))
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            v = float((_source_loss_balanced(model, val, device, state_std) if state_std is not None
                       else _source_loss(model, val, device)).item())
        done = epoch + 1
        if np.isfinite(v) and v < best_val:
            best_val = v
            stale = 0
            best = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best is not None:
        model.load_state_dict(best)
    return model, best_val, done


def _source_loss_oracle(model, rows, device):
    """oracle-state：transition 直接用真实 z（不经 encoder），隔离 Fθ 资格验证。

    回答"Fθ 本身能否学会动力学"，与"encoder 能否从观测识别 z"解耦。
    """
    pairs = make_transition_pairs(rows)
    z_t = torch.as_tensor(pairs.state_t, dtype=torch.float32, device=device)
    z_next = torch.as_tensor(pairs.state_t_plus_1, dtype=torch.float32, device=device)
    x_t = torch.as_tensor(pairs.x_t, dtype=torch.float32, device=device)
    dt = torch.as_tensor(pairs.normalized_dt, dtype=torch.float32, device=device)
    stress = model.source_stress(x_t)
    pred = model.transition(z_t, stress, dt)
    return F.mse_loss(pred, z_next)


def _source_loss_balanced(model, rows, device, state_std, *, obs_weight=0.1):
    """块③ encoder 友好损失：状态各维度归一（/ train std）+ obs 降权，让 state_anchor 主导。

    修复 _source_loss 的尺度压制——d_perm(0.006)/q_trap(0.5)/r_th(0.04) 量级悬殊致 MSE 被
    q_trap 主导，且 obs(gain~10) MSE 压过 state MSE。归一后 d_perm 维度监督不再被淹没。
    """
    pairs = make_transition_pairs(rows)
    x_t = torch.as_tensor(pairs.x_t, dtype=torch.float32, device=device)
    z_t = torch.as_tensor(pairs.state_t, dtype=torch.float32, device=device)
    z_next = torch.as_tensor(pairs.state_t_plus_1, dtype=torch.float32, device=device)
    obs_next = torch.as_tensor(pairs.obs_t_plus_1, dtype=torch.float32, device=device)
    dt = torch.as_tensor(pairs.normalized_dt, dtype=torch.float32, device=device)
    std = torch.as_tensor(state_std, dtype=torch.float32, device=device)
    state_hat = model.encode_source(x_t)
    stress_b = model.source_stress(x_t)
    pred_next = model.transition(state_hat, stress_b, dt)
    state_loss = F.mse_loss(state_hat / std, z_t / std) + F.mse_loss(pred_next / std, z_next / std)
    obs_loss = F.mse_loss(model.observe_source(pred_next), obs_next)
    return state_loss + obs_weight * obs_loss


def train_source_oracle(model, train, val, *, epochs, device, patience=5):
    """oracle-state Fθ 训练：只更新 transition 参数，encoder/head 不参与。"""
    optimizer = Adam(model.transition.parameters(), lr=2e-3)
    best = None
    best_val = float("inf")
    stale = 0
    done = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = _source_loss_oracle(model, train, device)
        loss.backward()
        clip_grad_norm_(model.transition.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            v = float(_source_loss_oracle(model, val, device).item())
        done = epoch + 1
        if np.isfinite(v) and v < best_val:
            best_val = v
            stale = 0
            best = deepcopy(model.transition.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best is not None:
        model.transition.load_state_dict(best)
    return model, best_val, done


def _rollout_indices(ids):
    """每器件内可作 ROLLOUT_STEPS 起点的全局索引；起点须与后续 6 帧同器件。"""
    starts = []
    for device in np.unique(ids):
        idx = np.flatnonzero(ids == device)
        if len(idx) <= ROLLOUT_STEPS:
            continue
        starts.extend(idx[:-ROLLOUT_STEPS].tolist())
    return np.asarray(starts, dtype=int)


def _seq_index_matrix(ids, starts):
    """每起点的 7 帧（t..t+6）全局索引矩阵 (N, ROLLOUT_STEPS+1)。"""
    seq = np.empty((len(starts), ROLLOUT_STEPS + 1), dtype=int)
    for i, s in enumerate(starts):
        device = ids[s]
        idx = np.flatnonzero(ids == device)
        pos = int(np.flatnonzero(idx == s)[0])
        seq[i] = idx[pos:pos + ROLLOUT_STEPS + 1]
    return seq


@torch.no_grad()
def _rollout6(model, rows, *, device, transition_time_scale_s, transition_state_override=None, init_mode="encode"):
    """闭环六步 rollout：z_{k+1}=Fθ(z_k, stress_k)。

    init_mode='encode'：z_0=encode_source(obs_t)（端到端，含 encoder 误差）；
    init_mode='oracle'：z_0=真实 states[t]（隔离 encoder，只评 Fθ 动力学）。
    dt 用每步真实 Δt=t_{k+1}-t_k（降采样后相邻间隔可能不等），按 transition_time_scale_s 归一。
    transition_state_override 非空时临时替换 transition 权重（用于 random-transition 基线：
    保留 trained encoder 提供有意义 z_0，仅把 transition 换成未训练权重，隔离 Fθ 的转移价值）。
    返回 (pred (N,6,3) ndarray, target (N,6,3) ndarray)。
    """
    saved = None
    if transition_state_override is not None:
        saved = deepcopy(model.transition.state_dict())
        model.transition.load_state_dict(transition_state_override)
    model.eval()
    starts = _rollout_indices(rows.ids)
    if len(starts) == 0:
        raise ValueError("holdout 无足够长的轨迹做六步 rollout")
    seq = _seq_index_matrix(rows.ids, starts)
    obs_seq = torch.as_tensor(rows.x[seq], dtype=torch.float32, device=device)  # (N,7,F)
    target = rows.states[seq[:, 1:]].astype(np.float32)  # (N,6,3)
    dt_steps = (rows.time[seq[:, 1:]] - rows.time[seq[:, :-1]]).astype(np.float64)  # (N,6) 逐步真实 Δt
    dt_norm = torch.as_tensor(dt_steps / float(transition_time_scale_s), dtype=torch.float32, device=device)
    z = model.encode_source(obs_seq[:, 0]) if init_mode == "encode" else torch.as_tensor(
        rows.states[starts], dtype=torch.float32, device=device)
    preds = []
    for k in range(ROLLOUT_STEPS):
        stress = model.source_stress(obs_seq[:, k])
        z = model.transition(z, stress, dt_norm[:, k:k+1])
        preds.append(z)
    pred = torch.stack(preds, dim=1).cpu().numpy()  # (N,6,3)
    if saved is not None:
        model.transition.load_state_dict(saved)
    return pred, target


def _persistence_pred(rows, starts):
    """persistence 基线：pred 恒等于真值初态 states[t]（纯无转移参照）。"""
    seq = _seq_index_matrix(rows.ids, starts)
    init = rows.states[seq[:, 0]].astype(np.float32)  # (N,3)
    target = rows.states[seq[:, 1:]].astype(np.float32)  # (N,6,3)
    pred = np.broadcast_to(init[:, None, :], target.shape)
    return pred.astype(np.float32), target


def _nrmse_per_dim(pred, target, scale):
    """每状态维度 nRMSE（scale 为该维度 std，纯评估侧归一）。"""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64) + 1e-8
    rmse = np.sqrt(np.mean((pred - target) ** 2, axis=(0, 1)))
    return (rmse / scale).tolist()


def _median_step_dt(rows):
    """源域等间隔采样的相邻步 dt（秒），取所有器件内 diff 的全局中位。"""
    diffs = []
    for device in np.unique(rows.ids):
        idx = np.flatnonzero(rows.ids == device)
        if len(idx) >= 2:
            diffs.extend(np.diff(rows.time[idx]).tolist())
    if not diffs:
        raise ValueError("无法估计源域采样步长")
    return float(np.median(diffs))


def diagnostic1(config, *, source_path, n_holdout, n_val, epochs, seed, device, max_points):
    """诊断 1：源域按器件留出，比较 trained-Fθ / random-transition / persistence 的六步 rollout。"""
    source, dynamics = _source_rows(source_path, max_points)
    source.transition_time_scale_s = float(config["transfer"]["transition_time_scale_s"])
    train_ids, val_ids, holdout_ids = split_devices(
        source.ids, n_holdout=n_holdout, n_val=n_val, seed=seed)
    train = _subset(source, train_ids)
    val = _subset(source, val_ids)
    holdout = _subset(source, holdout_ids)
    # P0-1: train-only 标准化（仅在 train 器件拟合；禁止用全源域或 holdout，否则 holdout 分布泄漏）
    source_scaler = TrainOnlyStandardizer().fit(train.x)
    train = _replace_x(train, source_scaler.transform(train.x))
    val = _replace_x(val, source_scaler.transform(val.x))
    holdout = _replace_x(holdout, source_scaler.transform(holdout.x))
    state_names = list(config["damage_state"]["names"])

    state_std_train = train.states.std(axis=0)  # 块③: 状态归一参考（train 器件各维度 std）
    model = DamageStateModel(source.x.shape[1], source.x.shape[1]).to(device)
    model, best_val, epochs_done = train_source_with_val(
        model, train, val, epochs=epochs, device=device, patience=5, state_std=state_std_train)

    median_step_dt = _median_step_dt(source)  # 诊断信息：源域采样步长中位（秒）
    state_scale = holdout.states.std(axis=0)  # 纯评估侧归一，每维度

    # P0-2: _rollout6 内部按 rows.time 算逐步真实 dt
    pred_trained, target = _rollout6(
        model, holdout, device=device, transition_time_scale_s=source.transition_time_scale_s)
    fresh_transition = DamageStateTransition(
        state_dim=model.transition.state_dim, stress_dim=model.transition.stress_dim,
        hidden_dim=int(model.transition.net[0].out_features)).state_dict()  # 同架构未训练 transition
    pred_random, _ = _rollout6(
        model, holdout, device=device, transition_time_scale_s=source.transition_time_scale_s,
        transition_state_override=fresh_transition)
    starts = _rollout_indices(holdout.ids)
    pred_persist, _ = _persistence_pred(holdout, starts)

    # oracle-state Fθ（隔离 encoder：transition 用真实 z 训练，rollout 用真实 z 初态）
    oracle_model = DamageStateModel(source.x.shape[1], source.x.shape[1]).to(device)
    oracle_model, oracle_val, oracle_epochs = train_source_oracle(
        oracle_model, train, val, epochs=epochs, device=device, patience=5)
    pred_oracle, _ = _rollout6(
        oracle_model, holdout, device=device,
        transition_time_scale_s=source.transition_time_scale_s, init_mode="oracle")
    nrmse_oracle = _nrmse_per_dim(pred_oracle, target, state_scale)

    nrmse_trained = _nrmse_per_dim(pred_trained, target, state_scale)
    nrmse_random = _nrmse_per_dim(pred_random, target, state_scale)
    nrmse_persist = _nrmse_per_dim(pred_persist, target, state_scale)

    def mean(xs):
        return float(np.mean(xs))

    return {
        "run_scope": "diagnostic_exploratory_not_in_acceptance",
        "diagnostic": 1,
        "diagnostic_name": "source_holdout_generalization",
        "source_dynamics_id": dynamics,
        "transition_time_scale_s": source.transition_time_scale_s,
        "source_median_step_dt_s": median_step_dt,
        "source_scaler": {"fit_scope": "train_devices_only", "mean": source_scaler.mean_.tolist(), "std": source_scaler.scale_.tolist()},
        "device_split": {"train": len(train_ids), "val": len(val_ids), "holdout": len(holdout_ids)},
        "source_train_device_ids": train_ids,
        "source_val_device_ids": val_ids,
        "source_holdout_device_ids": holdout_ids,
        "rollout_starts": int(len(starts)),
        "rollout_steps": ROLLOUT_STEPS,
        "source_best_val_loss": float(best_val),
        "source_epochs_run": int(epochs_done),
        "state_names": state_names,
        "state_scale_std": state_scale.tolist(),
        "oracle_val_loss": float(oracle_val),
        "oracle_epochs_run": int(oracle_epochs),
        "oracle_init": "true_state_holdout",
        "nrmse_6step": {
            "trained_Ftheta": {"per_dim": nrmse_trained, "mean": mean(nrmse_trained)},
            "random_transition": {"per_dim": nrmse_random, "mean": mean(nrmse_random)},
            "persistence": {"per_dim": nrmse_persist, "mean": mean(nrmse_persist)},
            "oracle_Ftheta": {"per_dim": nrmse_oracle, "mean": mean(nrmse_oracle)},
        },
        "deltas": {
            # 正值 = trained 优于基线（更低的 nRMSE）；负值 = trained 更差
            "trained_minus_persistence_mean": mean(nrmse_persist) - mean(nrmse_trained),
            "trained_minus_random_mean": mean(nrmse_random) - mean(nrmse_trained),
            "oracle_minus_persistence_mean": mean(nrmse_persist) - mean(nrmse_oracle),
        },
    }


def _render_diagnostic1_markdown(result: dict) -> str:
    n = result["nrmse_6step"]
    lines = [
        "# 诊断 1：源域留出泛化（探索性，不进 IID 验收）",
        "",
        f"主问题：在未见器件上，预训练 Fθ 的闭环六步状态 rollout 是否优于 persistence 与 random transition？",
        f"回答\"源域 Fθ 是否学到了可泛化的退化动力学\"，排除既有负结果的\"源域没学好\"解释。",
        "",
        f"- 源域 dynamics_id：`{result['source_dynamics_id']}`；源域采样步长中位 dt={result['source_median_step_dt_s']:.6g} s（rollout 用逐步真实 Δt）",
        f"- 器件划分：train {result['device_split']['train']} / val {result['device_split']['val']} / holdout {result['device_split']['holdout']}",
        f"- holdout rollout 起点数：{result['rollout_starts']}（每起点闭环 {result['rollout_steps']} 步）",
        f"- 源域训练：best val loss={result['source_best_val_loss']:.6g} @ {result['source_epochs_run']} epoch（val early-stop）",
        "",
        "## 六步 rollout nRMSE（按状态维度，holdout std 归一）",
        "",
        "| 组别 | " + " | ".join(result["state_names"]) + " | 算术平均 |",
        "| --- | " + " | ".join(["---:"] * len(result["state_names"])) + " | ---: |",
    ]
    for label, key in (("trained Fθ (encode)", "trained_Ftheta"),
                       ("random transition", "random_transition"),
                       ("persistence", "persistence"),
                       ("oracle Fθ (true z)", "oracle_Ftheta")):
        per = n[key]["per_dim"]
        lines.append(f"| {label} | " + " | ".join(f"{v:.4f}" for v in per) + f" | {n[key]['mean']:.4f} |")
    lines += [
        "",
        "## Δ（正值=trained 优于基线）",
        "",
        f"- trained − persistence（平均 nRMSE）：{result['deltas']['trained_minus_persistence_mean']:+.4f}",
        f"- trained − random transition（平均 nRMSE）：{result['deltas']['trained_minus_random_mean']:+.4f}",
        f"- **oracle − persistence（平均 nRMSE）：{result['deltas']['oracle_minus_persistence_mean']:+.4f}**（隔离 encoder；接近 0 或正 = Fθ 学到动力学）",
        "",
        "## 解释边界",
        "",
        "本诊断是设计文档 §9.4 的第一个廉价诊断，结果仅用于判断是否值得跑 §9.2 弱监督矩阵，",
        "不构成正迁移证据、不进入 IID 主验收。若 trained Fθ 显著优于 persistence 与 random transition，",
        "则\"源域已学到泛化退化动力学\"成立，负结果主因更可能是\"目标监督覆盖源初始化\"；",
        "若 trained 不优于（甚至差于）基线，则\"源域没学好\"成立，需先改进源域预训练再谈迁移。",
    ]
    return "\n".join(lines) + "\n"


# ---- Gate 1.1: 真物理 oracle（diagnostic 2）----
# FEATURE_NAMES 列索引：T_j_C=1, duty_cycle=6, VSWR=8, Pin_dBm=9
_PHYS_TJ_IDX, _PHYS_DUTY_IDX, _PHYS_VSWR_IDX, _PHYS_PIN_IDX = 1, 6, 8, 9
# 固定物理状态尺度（GPT P1-5）：d_perm EOL=0.006, q_trap=1.0, r_th=14·0.006=0.084
_STATE_SCALE_PHYS = np.array([0.006, 1.0, 0.084], dtype=np.float32)


def _physical_stress(x_raw: np.ndarray) -> np.ndarray:
    """从原始（未标准化）观测 x 构造物理应力；公式委托共享 build_physical_stress。"""
    return build_physical_stress(
        tj_c=x_raw[:, _PHYS_TJ_IDX],
        duty=x_raw[:, _PHYS_DUTY_IDX],
        pin_dbm=x_raw[:, _PHYS_PIN_IDX],
        vswr=x_raw[:, _PHYS_VSWR_IDX],
        recovery=(x_raw[:, _PHYS_DUTY_IDX] < 0.10),
    )


def _physical_transition_pairs(rows, u_phys):
    """构造 (z_t, u_t, dt, z_next) 相邻对，按 device 内时间正向。"""
    starts, ends = [], []
    for device in np.unique(rows.ids):
        idx = np.flatnonzero(rows.ids == device)
        if len(idx) < 2:
            continue
        starts.extend(idx[:-1].tolist()); ends.extend(idx[1:].tolist())
    s = np.asarray(starts, dtype=int); e = np.asarray(ends, dtype=int)
    if len(s) == 0:
        raise ValueError("无足够长的轨迹构造物理 transition 对")
    dt = ((rows.time[e] - rows.time[s]) / rows.transition_time_scale_s).astype(np.float32)
    return (rows.states[s].astype(np.float32), u_phys[s].astype(np.float32),
            dt, rows.states[e].astype(np.float32))


class PhysicalTransition(nn.Module):
    """Gate 1.1 物理 oracle Fθ：归一 z + 物理 u + dt → 归一 z_next；d/r 耦合。

    d_perm 有界 sigmoid 增量；r_th = r + 14·Δd_perm（归一后 Δr̃=Δd̃，与源仿真一致）；
    q_trap 捕获-释放（tanh 增量 + clamp[0,1]）。
    """
    def __init__(self, stress_dim: int = 3, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 + stress_dim, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    def forward(self, z_norm, u_norm, dt):
        raw = self.net(torch.cat([z_norm, u_norm], dim=-1))
        dd = torch.sigmoid(raw[..., 0]) * 0.2 * dt[..., 0]
        d_next = z_norm[..., 0] + dd
        q_next = torch.clamp(z_norm[..., 1] + torch.tanh(raw[..., 1]) * 0.3 * dt[..., 0], 0.0, 1.0)
        r_next = z_norm[..., 2] + dd  # d/r 耦合（归一后同增量）
        return torch.stack([d_next, q_next, r_next], dim=-1)


def train_physical_oracle(model, pairs_train, pairs_val, state_scale, u_scale, *, epochs, device, patience=5):
    """物理 oracle 训练：归一状态 + 归一物理应力，Huber 损失（无 encoder、无 source_stress）。"""
    z_t, u_t, dt, z_next = pairs_train
    z_tv, u_tv, dtv, z_nextv = pairs_val
    z_t_n = torch.as_tensor(z_t / state_scale, dtype=torch.float32, device=device)
    z_next_n = torch.as_tensor(z_next / state_scale, dtype=torch.float32, device=device)
    u = torch.as_tensor(u_t / u_scale, dtype=torch.float32, device=device)
    dt_t = torch.as_tensor(dt.reshape(-1, 1), dtype=torch.float32, device=device)
    z_tv_n = torch.as_tensor(z_tv / state_scale, dtype=torch.float32, device=device)
    z_nextv_n = torch.as_tensor(z_nextv / state_scale, dtype=torch.float32, device=device)
    uv = torch.as_tensor(u_tv / u_scale, dtype=torch.float32, device=device)
    dtv = torch.as_tensor(dtv.reshape(-1, 1), dtype=torch.float32, device=device)
    optimizer = Adam(model.parameters(), lr=2e-3)
    best, best_val, stale, done = None, float("inf"), 0, 0
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = F.huber_loss(model(z_t_n, u, dt_t), z_next_n)
        loss.backward(); clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            v = float(F.huber_loss(model(z_tv_n, uv, dtv), z_nextv_n).item())
        done = epoch + 1
        if np.isfinite(v) and v < best_val:
            best_val, stale, best = v, 0, deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best is not None:
        model.load_state_dict(best)
    return model, best_val, done


@torch.no_grad()
def _physical_rollout(model, rows, u_phys, state_scale, u_scale, *, device, horizon=6):
    """闭环 rollout horizon 步（归一空间）；返回 pred/target/persistence（均归一）。"""
    starts = _rollout_indices(rows.ids)
    if len(starts) == 0:
        raise ValueError("holdout 无足够长轨迹")
    seq = _seq_index_matrix(rows.ids, starts)
    horizon = min(horizon, seq.shape[1] - 1)
    u_seq = u_phys[seq[:, :horizon + 1]]
    dt_steps = (rows.time[seq[:, 1:horizon + 1]] - rows.time[seq[:, :horizon]]).astype(np.float64)
    dt_norm = (dt_steps / rows.transition_time_scale_s).astype(np.float32)
    z = torch.as_tensor(rows.states[starts] / state_scale, dtype=torch.float32, device=device)
    u_t = torch.as_tensor(u_seq[:, :horizon] / u_scale, dtype=torch.float32, device=device)
    dt_t = torch.as_tensor(dt_norm, dtype=torch.float32, device=device)
    preds = []
    for k in range(horizon):
        z = model(z, u_t[:, k], dt_t[:, k:k + 1])
        preds.append(z)
    pred = torch.stack(preds, dim=1).cpu().numpy()
    target = (rows.states[seq[:, 1:horizon + 1]] / state_scale).astype(np.float32)
    persist = np.broadcast_to((rows.states[starts] / state_scale)[:, None, :], target.shape).astype(np.float32)
    return pred, target, persist


def diagnostic2(config, *, source_path, n_holdout, n_val, epochs, seed, device, max_points, horizon=6):
    """Gate 1.1 真物理 oracle：transition(真实 z, 物理 u, dt) → z_next，状态归一 + d/r 耦合。"""
    source, dynamics = _source_rows(source_path, max_points)
    source.transition_time_scale_s = float(config["transfer"]["transition_time_scale_s"])
    train_ids, val_ids, holdout_ids = split_devices(
        source.ids, n_holdout=n_holdout, n_val=n_val, seed=seed)
    train = _subset(source, train_ids); val = _subset(source, val_ids); holdout = _subset(source, holdout_ids)
    u_train = _physical_stress(train.x); u_val = _physical_stress(val.x); u_holdout = _physical_stress(holdout.x)
    u_scale = u_train.std(axis=0) + 1e-6
    tr = _physical_transition_pairs(train, u_train)
    va = _physical_transition_pairs(val, u_val)
    model = PhysicalTransition().to(device)
    model, best_val, epochs_done = train_physical_oracle(
        model, tr, va, _STATE_SCALE_PHYS, u_scale, epochs=epochs, device=device, patience=5)
    pred, target, persist = _physical_rollout(
        model, holdout, u_holdout, _STATE_SCALE_PHYS, u_scale, device=device, horizon=horizon)

    def nrmse(p, t):
        return np.sqrt(np.mean((p - t) ** 2, axis=(0, 1))).tolist()
    def skill(p, t):
        mse_f = np.mean((p - t) ** 2, axis=(0, 1))
        mse_p = np.mean((persist - t) ** 2, axis=(0, 1))
        return (1.0 - mse_f / np.maximum(mse_p, 1e-12)).tolist()
    state_names = list(config["damage_state"]["names"])
    nrmse_f, nrmse_p = nrmse(pred, target), nrmse(persist, target)
    sk = skill(pred, target)
    return {
        "run_scope": "diagnostic_exploratory_not_in_acceptance",
        "diagnostic": 2,
        "diagnostic_name": "physical_oracle_true_stress",
        "source_dynamics_id": dynamics,
        "horizon_steps": horizon,
        "device_split": {"train": len(train_ids), "val": len(val_ids), "holdout": len(holdout_ids)},
        "state_scale_phys": _STATE_SCALE_PHYS.tolist(),
        "physical_stress": "[a_T, s, recovery]（绕过 source_stress MLP；从原始 x 构造）",
        "dr_coupling": "r_th = r + 14·Δd_perm（归一后 Δr̃=Δd̃）",
        "source_best_val_loss": float(best_val),
        "source_epochs_run": int(epochs_done),
        "nrmse_norm": {
            "physical_Ftheta": {"per_dim": nrmse_f, "mean": float(np.mean(nrmse_f))},
            "persistence": {"per_dim": nrmse_p, "mean": float(np.mean(nrmse_p))},
        },
        "skill_vs_persistence": {"per_dim": sk, "d_perm": sk[0], "q_trap": sk[1], "r_th": sk[2]},
        "state_names": state_names,
    }


def _render_diagnostic2_markdown(result: dict) -> str:
    sk = result["skill_vs_persistence"]["per_dim"]
    names = result["state_names"]
    nf = result["nrmse_norm"]["physical_Ftheta"]["per_dim"]
    npers = result["nrmse_norm"]["persistence"]["per_dim"]
    lines = [
        "# 诊断 2：真物理 oracle（Gate 1.1，探索性，不进 IID 验收）",
        "",
        f"主问题：用**真实 z + 真实物理应力 [a_T, s, recovery]**（绕过 source_stress MLP）+ **状态归一 + d/r 耦合**训练 Fθ，",
        f"在 holdout 闭环 {result['horizon_steps']} 步是否优于 persistence？修正诊断 1 的 oracle 三缺陷（随机 stress / 未归一 / d-r 解耦）。",
        "",
        f"- 物理 oracle：固定状态尺度 {result['state_scale_phys']}；{result['dr_coupling']}",
        f"- holdout 器件 {result['device_split']['holdout']}；best val loss={result['source_best_val_loss']:.6g} @ {result['source_epochs_run']} ep",
        "",
        f"## 归一 nRMSE（holdout 闭环 {result['horizon_steps']} 步）",
        "",
        "| 状态 | 物理 Fθ | persistence | skill（正值=Fθ 优）|",
        "| --- | ---: | ---: | ---: |",
    ]
    for i, name in enumerate(names):
        lines.append(f"| {name} | {nf[i]:.4f} | {npers[i]:.4f} | {sk[i]:+.4f} |")
    lines += [
        "",
        "## 解释边界",
        "",
        "skill>0 表示物理 Fθ 优于 persistence。d_perm/r_th 处于永久损伤时间尺度、q_trap 处于可恢复时间尺度；",
        f"{result['horizon_steps']} 步对永久状态偏短（待多跨度评价）。本诊断只验证\"真实物理应力下 Fθ 能否学到动力学\"，不构成正迁移证据。",
    ]
    return "\n".join(lines) + "\n"


# ---- u 支持域审计（Gate 2 前置，GPT §6）----
_TARGET_TJ_COL_IDX = 4  # X_GLOBAL_COLS = [G_array_dB, M_link_dB, SLL_dB, theta_err_deg, Tj, I_D_obs]
_U_QUANTILES = (0.0, 0.05, 0.5, 0.95, 1.0)
_U_NAMES = ("a_T", "s", "recovery")


def _target_u_phys(features_path: Path, max_points_per_traj: int) -> np.ndarray:
    """读目标域 feature h5 的 Tj(x_global[idx4]) + physical_duty，构造物理 u（Pin/VSWR 基准）。"""
    u_list = []
    with h5py.File(features_path, "r") as h5:
        for traj in sorted(h5.keys()):
            g = h5[traj]
            if "physical_duty" not in g:
                raise ValueError(f"目标 {traj} 缺 physical_duty；请重建特征 (python -m src.sim.build_array_hi)")
            n = len(g["x_global"])
            ix = np.linspace(0, n - 1, min(n, max_points_per_traj), dtype=int)
            tj = np.asarray(g["x_global"][ix, _TARGET_TJ_COL_IDX], dtype=np.float64)
            duty = np.asarray(g["physical_duty"][ix], dtype=np.float64)
            u_list.append(build_physical_stress(
                tj, duty, pin_dbm=TARGET_PIN_DBM_REFERENCE,
                vswr=TARGET_VSWR_REFERENCE, recovery=(duty < 0.10)))
    return np.concatenate(u_list, axis=0)


def u_support_audit(source_u: np.ndarray, target_u: np.ndarray) -> dict:
    """源/目标 u 支持域统计（GPT §6，Gate 2 前置）。

    区分低激励插值（目标在源支持域内）/ OOD 外推（目标超出源范围）/ 支持域失配，
    避免 Gate 2 失败被错误归因为"u 信号弱"。
    """

    def _stats(u: np.ndarray) -> dict:
        return {name: {q: float(np.quantile(u[:, j], q)) for q in _U_QUANTILES}
                for j, name in enumerate(_U_NAMES)}

    source_u = np.asarray(source_u, dtype=np.float64)
    target_u = np.asarray(target_u, dtype=np.float64)
    if source_u.shape[-1] != 3 or target_u.shape[-1] != 3:
        raise ValueError("u 必须为 (..., 3)")
    coverage = {}
    for j, name in enumerate(_U_NAMES):
        s_lo, s_hi = float(source_u[:, j].min()), float(source_u[:, j].max())
        in_range = (target_u[:, j] >= s_lo) & (target_u[:, j] <= s_hi)
        coverage[name] = {"source_min": s_lo, "source_max": s_hi,
                          "target_in_range_fraction": float(in_range.mean())}
    # 标准化最近邻距离（每维除以源 std），抽样防 O(N·M) 内存
    source_std = source_u.std(axis=0) + 1e-8
    src_n = source_u / source_std
    tgt_n = target_u / source_std
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(tgt_n), size=min(len(tgt_n), 2000), replace=False)
    nn_dist = np.empty(len(sample_idx))
    for i, idx in enumerate(sample_idx):
        nn_dist[i] = float(np.linalg.norm(src_n - tgt_n[idx], axis=1).min())
    return {
        "run_scope": "diagnostic_exploratory_not_in_acceptance",
        "diagnostic": 0,
        "diagnostic_name": "u_support_audit",
        "physical_stress_schema": PHYSICAL_STRESS_SCHEMA,
        "target_pin_dbm_assumption": TARGET_PIN_DBM_REFERENCE,
        "target_vswr_assumption": TARGET_VSWR_REFERENCE,
        "quantiles": list(_U_QUANTILES),
        "state_names": list(_U_NAMES),
        "source_u_stats": _stats(source_u),
        "target_u_stats": _stats(target_u),
        "coverage_in_source_range": coverage,
        "target_nn_distance_normalized": {
            "median": float(np.median(nn_dist)),
            "p95": float(np.quantile(nn_dist, 0.95)),
            "max": float(np.max(nn_dist)),
        },
        "recovery_active_fraction": {
            "source": float((source_u[:, 2] > 0.5).mean()),
            "target": float((target_u[:, 2] > 0.5).mean()),
        },
        "n_source_points": int(len(source_u)),
        "n_target_points": int(len(target_u)),
    }


def diagnostic0(config, *, source_path, target_path, max_points) -> dict:
    """u 支持域审计：源域 _physical_stress + 目标域 Tj/physical_duty -> 支持范围统计。"""
    source, _ = _source_rows(source_path, max_points)
    source_u = _physical_stress(source.x)
    target_u = _target_u_phys(target_path, max_points)
    return u_support_audit(source_u, target_u)


def _render_u_audit_markdown(result: dict) -> str:
    names = result["state_names"]
    lines = [
        "# u 支持域审计（Gate 2 前置，探索性，不进 IID 验收）",
        "",
        "主问题：目标域 u 是否落在源域 u 的支持范围内？区分低激励插值 / OOD 外推 / 支持域失配。",
        "",
        f"- 物理 stress schema: `{result['physical_stress_schema']}`",
        f"- 目标域 Pin/VSWR 基准: {result['target_pin_dbm_assumption']} / {result['target_vswr_assumption']}（预注册，不得事后改）",
        f"- 源域点数 {result['n_source_points']} / 目标域点数 {result['n_target_points']}",
        f"- recovery active 占比: 源 {result['recovery_active_fraction']['source']:.4f} / 目标 {result['recovery_active_fraction']['target']:.4f}",
        "",
        "## u 分位数（源 vs 目标）",
        "",
        "| 维度 | 域 | min | p05 | median | p95 | max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        for label, key in (("源", "source_u_stats"), ("目标", "target_u_stats")):
            s = result[key][name]
            lines.append(f"| {name} | {label} | {s[0.0]:.4f} | {s[0.05]:.4f} | {s[0.5]:.4f} | {s[0.95]:.4f} | {s[1.0]:.4f} |")
    lines += [
        "",
        "## 目标点落在源 [min, max] 的比例",
        "",
        "| 维度 | 源 min | 源 max | 目标 in-range 占比 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in names:
        c = result["coverage_in_source_range"][name]
        lines.append(f"| {name} | {c['source_min']:.4f} | {c['source_max']:.4f} | {c['target_in_range_fraction']:.4f} |")
    nn = result["target_nn_distance_normalized"]
    lines += [
        "",
        f"标准化最近邻距离（目标->源，除以源 std）：median {nn['median']:.4f} / p95 {nn['p95']:.4f} / max {nn['max']:.4f}",
        "",
        "## 解读",
        "",
        "- in-range 占比高 + 最近邻小 = 低激励插值（目标在源支持域内，可评价 Fθ 局部工作点）",
        "- in-range 占比低 = OOD 外推（失败应归因于支持域失配，非 u 信号弱）",
        "- recovery 目标恒 0 = 释放分支在目标域未验证（非错误）",
    ]
    return "\n".join(lines) + "\n"


# ---- Gate 2A: 跨域冻结动力学诊断（GPT §8，diagnostic3）----
_GATE2A_HORIZONS = (6, 28, 120, 360, 720, 1460)
_GATE2A_GROUPS = ("persistence", "constant_source_rate", "random_raw", "random_rate_matched", "source_F_frozen")


def _target_node_rows(features_path: Path, max_points_per_traj: int, transition_time_scale_s: float) -> list[dict]:
    """读目标域节点级真实 z (label_node_*, (T,16,3)) + u_phys (Tj+physical_duty, (T,3)) + dt。"""
    trajectories = []
    with h5py.File(features_path, "r") as h5:
        for traj in sorted(h5.keys()):
            g = h5[traj]
            required = ("x_global", "physical_duty", "time_s", "label_node_d_perm", "label_node_q_trap", "label_node_r_th")
            missing = [n for n in required if n not in g]
            if missing:
                raise ValueError(f"target {traj} missing {missing}; rebuild features (python -m src.sim.build_array_hi)")
            n = len(g["x_global"])
            ix = np.linspace(0, n - 1, min(n, max_points_per_traj), dtype=int)
            tj = np.asarray(g["x_global"][ix, _TARGET_TJ_COL_IDX], dtype=np.float64)
            duty = np.asarray(g["physical_duty"][ix], dtype=np.float64)
            u = build_physical_stress(tj, duty, pin_dbm=TARGET_PIN_DBM_REFERENCE,
                                      vswr=TARGET_VSWR_REFERENCE, recovery=(duty < 0.10))
            z_node = np.stack([g["label_node_d_perm"][ix], g["label_node_q_trap"][ix],
                               g["label_node_r_th"][ix]], axis=-1).astype(np.float32)
            time = np.asarray(g["time_s"][ix], dtype=np.float64)
            dt = (np.diff(time) / transition_time_scale_s).astype(np.float32)
            trajectories.append({"id": traj, "z_node": z_node, "u_phys": u.astype(np.float32),
                                 "dt": dt, "n": len(ix)})
    return trajectories


def _calibrate_rate_matched_bias(random_F, source_F, pairs_train, state_scale, u_scale, device) -> dict:
    """校准 random_F.net[-1].bias[0] 使 E[Δd_random]=E[Δd_source]（源训练对，GPT §3 一阶近似）。"""
    z_t, u_t, dt, _ = pairs_train
    z_n = torch.as_tensor(z_t / state_scale, dtype=torch.float32, device=device)
    u_n = torch.as_tensor(u_t / u_scale, dtype=torch.float32, device=device)
    dt_t = torch.as_tensor(dt.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        source_mean = float((source_F(z_n, u_n, dt_t)[:, 0] - z_n[:, 0]).mean())
        random_mean = float((random_F(z_n, u_n, dt_t)[:, 0] - z_n[:, 0]).mean())
        raw = random_F.net(torch.cat([z_n, u_n], dim=-1))
        sig = torch.sigmoid(raw[:, 0])
        grad_mean = float((sig * (1 - sig) * 0.2 * dt_t[:, 0]).mean())
        offset = (source_mean - random_mean) / (grad_mean + 1e-8)
        random_F.net[-1].bias.data[0] += offset
        after_mean = float((random_F(z_n, u_n, dt_t)[:, 0] - z_n[:, 0]).mean())
    return {"source_dd_mean": source_mean, "random_before": random_mean, "random_after": after_mean}


def _rollout_gate2a(group, traj, state_scale, u_scale, device, horizons, *, source_F, random_F_raw, random_F_matched, source_dd_mean):
    """单轨迹闭环 rollout 各 horizon 终点（归一空间，16 节点并行）。返回 {H: pred(16,3)} + 真实 z_node_norm。"""
    z_node_norm = torch.as_tensor(traj["z_node"] / state_scale, dtype=torch.float32, device=device)
    u_norm = torch.as_tensor(traj["u_phys"] / u_scale, dtype=torch.float32, device=device)
    dt = torch.as_tensor(traj["dt"], dtype=torch.float32, device=device)
    n_nodes = z_node_norm.shape[1]
    z = z_node_norm[0].clone()
    max_H = min(max(horizons), z_node_norm.shape[0] - 1)
    preds = {}
    for k in range(max_H):
        u_k = u_norm[k].unsqueeze(0).expand(n_nodes, -1)
        dt_k = dt[k].reshape(1, 1).expand(n_nodes, -1)
        if group == "persistence":
            z_next = z
        elif group == "constant_source_rate":
            dd = source_dd_mean * dt_k[:, 0]
            z_next = torch.stack([z[:, 0] + dd, z[:, 1], z[:, 2] + dd], dim=-1)
        elif group == "source_F_frozen":
            z_next = source_F(z, u_k, dt_k)
        elif group == "random_raw":
            z_next = random_F_raw(z, u_k, dt_k)
        else:  # random_rate_matched
            z_next = random_F_matched(z, u_k, dt_k)
        z = z_next
        if (k + 1) in horizons:
            preds[k + 1] = z.detach().cpu().numpy()
    return preds, z_node_norm.detach().cpu().numpy()


def diagnostic3(config, *, source_path, target_path, n_holdout, n_val, epochs, seed, device,
                source_max_points, target_max_points, horizons=_GATE2A_HORIZONS) -> dict:
    """Gate 2A: 跨域冻结动力学诊断（五组对照 + 多跨度终点 skill + 节点级主口径，GPT §8）。"""
    time_scale = float(config["transfer"]["transition_time_scale_s"])
    source, dynamics = _source_rows(source_path, source_max_points)
    source.transition_time_scale_s = time_scale
    train_ids, val_ids, _ = split_devices(source.ids, n_holdout=n_holdout, n_val=n_val, seed=seed)
    train = _subset(source, train_ids); val = _subset(source, val_ids)
    u_train = _physical_stress(train.x); u_val = _physical_stress(val.x)
    u_scale = u_train.std(axis=0) + 1e-6
    tr = _physical_transition_pairs(train, u_train); va = _physical_transition_pairs(val, u_val)
    source_F = PhysicalTransition().to(device)
    source_F, source_val, epochs_done = train_physical_oracle(
        source_F, tr, va, _STATE_SCALE_PHYS, u_scale, epochs=epochs, device=device, patience=5)
    z_t, u_t, dt, _ = tr
    z_n = torch.as_tensor(z_t / _STATE_SCALE_PHYS, dtype=torch.float32, device=device)
    u_n = torch.as_tensor(u_t / u_scale, dtype=torch.float32, device=device)
    dt_t = torch.as_tensor(dt.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        source_dd_mean = float((source_F(z_n, u_n, dt_t)[:, 0] - z_n[:, 0]).mean())
    random_F_raw = PhysicalTransition().to(device)
    random_F_matched = deepcopy(random_F_raw)
    rate_match = _calibrate_rate_matched_bias(random_F_matched, source_F, tr, _STATE_SCALE_PHYS, u_scale, device)
    target_traj = _target_node_rows(target_path, target_max_points, time_scale)
    group_preds = {g: {H: [] for H in horizons} for g in _GATE2A_GROUPS}
    targets = {H: [] for H in horizons}
    for traj in target_traj:
        for g in _GATE2A_GROUPS:
            preds, z_node_norm = _rollout_gate2a(
                g, traj, _STATE_SCALE_PHYS, u_scale, device, horizons,
                source_F=source_F, random_F_raw=random_F_raw, random_F_matched=random_F_matched,
                source_dd_mean=source_dd_mean)
            for H in horizons:
                if H in preds:
                    group_preds[g][H].append(preds[H])
        for H in horizons:
            if H < z_node_norm.shape[0]:
                targets[H].append(z_node_norm[H])
    results_by_horizon = {}
    for H in horizons:
        if not targets[H] or not group_preds["persistence"][H]:
            continue
        target_arr = np.stack(targets[H], axis=0)
        persist_arr = np.stack(group_preds["persistence"][H], axis=0)
        mse_persist = float(np.mean((persist_arr - target_arr) ** 2))
        groups_stat = {}
        for g in _GATE2A_GROUPS:
            if not group_preds[g][H]:
                continue
            pred_arr = np.stack(group_preds[g][H], axis=0)
            mse_f = float(np.mean((pred_arr - target_arr) ** 2))
            groups_stat[g] = {"endpoint_mse_norm": mse_f,
                              "endpoint_skill_vs_persistence": float(1.0 - mse_f / max(mse_persist, 1e-12))}
        results_by_horizon[str(H)] = {"persistence_mse_norm": mse_persist, "groups": groups_stat}
    return {
        "run_scope": "diagnostic_exploratory_not_in_acceptance",
        "diagnostic": 3, "diagnostic_name": "gate2a_frozen_dynamics",
        "source_dynamics_id": dynamics, "source_best_val_loss": float(source_val),
        "source_epochs_run": int(epochs_done), "source_dd_mean_normalized": source_dd_mean,
        "rate_match_calibration": rate_match, "horizons": list(horizons),
        "state_scale_phys": _STATE_SCALE_PHYS.tolist(), "n_target_trajectories": int(len(target_traj)),
        "normalized_dt_note": "source ~0.028 vs target ~1.0 (36x); Gate 2A strict freeze expected to fail on clock amplification",
        "results_by_horizon": results_by_horizon,
    }


def _render_diagnostic3_markdown(result: dict) -> str:
    horizons = result["horizons"]
    rm = result["rate_match_calibration"]
    lines = [
        "# Gate 2A: 跨域冻结动力学诊断（探索性，不进 IID 验收）",
        "",
        "主问题：源域 Fθ 严格冻结到目标域（无时钟校准），多跨度终点 skill 是否优于 persistence / rate-matched random / constant-rate？",
        "",
        f"- 源 Fθ best val loss={result['source_best_val_loss']:.6g} @ {result['source_epochs_run']} ep；源平均 Δd̃={result['source_dd_mean_normalized']:.6f}（归一/per normalized_dt）",
        f"- rate-matched 校准：random Δd̃ {rm['random_before']:.6f} -> {rm['random_after']:.6f}（目标 {rm['source_dd_mean']:.6f}）",
        f"- {result['n_target_trajectories']} 条目标轨迹，节点级 16 子阵（全局 u 广播）；{result['normalized_dt_note']}",
        "",
        "## 终点 skill vs persistence（正值=优于 persistence）",
        "",
        "| H 步 | persistence MSE | source_F_frozen | random_rate_matched | random_raw | constant_source_rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for H in horizons:
        r = result["results_by_horizon"].get(str(H))
        if not r:
            continue
        g = r["groups"]
        row = [f"{H}", f"{r['persistence_mse_norm']:.4f}"]
        for name in ("source_F_frozen", "random_rate_matched", "random_raw", "constant_source_rate"):
            if name in g:
                row.append(f"{g[name]['endpoint_skill_vs_persistence']:+.4f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## 解读（GPT §9 通过/停止条件）",
        "",
        "- 情况 A：source_F 长跨度 skill 显著 >0 且优于 rate-matched -> 绝对退化时钟也能迁移（最强）",
        "- 情况 B：source_F 失败但优于 constant-rate -> 动力学形状可迁移，时钟需校准（进 Gate 2B）",
        "- 情况 C：source_F 仅追平 constant-rate -> 源 Fθ 无条件动力学增益，只提供平均速率（停止）",
        "- 情况 D：source_F 差于 constant-rate/persistence -> 无迁移价值（停止路线二）",
        "- 预期：normalized_dt 36 倍 -> source_F 长跨度大概率发散（时钟放大），2B 校准是必经路径",
    ]
    return "\n".join(lines) + "\n"


# ---- Gate 2B: 时钟校准诊断（GPT §8，diagnostic4）----
_GATE2B_HORIZONS = _GATE2A_HORIZONS
_GATE2B_GROUPS = ("persistence", "target_constant_rate", "source_F_clock_calibrated", "source_F_two_clock", "target_oracle_F")


def _split_target_calib_test(traj_ids, n_calib, seed):
    uniq = np.array(sorted(traj_ids))
    rng = np.random.default_rng(seed)
    perm = uniq.copy(); rng.shuffle(perm)
    return perm[:n_calib].tolist(), perm[n_calib:].tolist()


def _calibrate_clock_analytic(source_F, calib_trajs, state_scale, u_scale, device) -> dict:
    """解析时钟校准 κ = Σ(Δtarget·Δsource)/Σ(Δsource²)（through-origin 最小二乘，GPT §2）。"""
    dd_s, dd_t, dq_s, dq_t = [], [], [], []
    for traj in calib_trajs:
        z_norm = torch.as_tensor(traj["z_node"] / state_scale, dtype=torch.float32, device=device)
        u_norm = torch.as_tensor(traj["u_phys"] / u_scale, dtype=torch.float32, device=device)
        dt = torch.as_tensor(traj["dt"], dtype=torch.float32, device=device)
        n_nodes = z_norm.shape[1]
        for t in range(z_norm.shape[0] - 1):
            z_t = z_norm[t]
            u_t = u_norm[t].unsqueeze(0).expand(n_nodes, -1)
            dt_t = dt[t].reshape(1, 1).expand(n_nodes, -1)
            with torch.no_grad():
                out = source_F(z_t, u_t, dt_t)
            dd_s.append((out[:, 0] - z_t[:, 0]).flatten())
            dq_s.append((out[:, 1] - z_t[:, 1]).flatten())
            dd_t.append((z_norm[t + 1, :, 0] - z_t[:, 0]).flatten())
            dq_t.append((z_norm[t + 1, :, 1] - z_t[:, 1]).flatten())
    dd_s = torch.cat(dd_s); dd_t = torch.cat(dd_t); dq_s = torch.cat(dq_s); dq_t = torch.cat(dq_t)
    k_perm = float((dd_t * dd_s).sum() / ((dd_s * dd_s).sum() + 1e-12))
    k_q = float((dq_t * dq_s).sum() / ((dq_s * dq_s).sum() + 1e-12))
    return {"k_perm": max(k_perm, 1e-6), "k_q": max(k_q, 1e-6),
            "source_dd_mean": float(dd_s.mean()), "target_dd_mean": float(dd_t.mean())}


def _target_constant_rate_vd(calib_trajs, state_scale) -> float:
    """目标校准集平均 d 速率（per normalized_dt，归一空间）。"""
    dd_per_dt = []
    for traj in calib_trajs:
        z_norm = traj["z_node"] / state_scale
        dt = traj["dt"]
        dd = (z_norm[1:, :, 0] - z_norm[:-1, :, 0]) / dt[:, None]
        dd_per_dt.append(dd.flatten())
    return float(np.mean(np.concatenate(dd_per_dt)))


def _train_target_oracle(calib_trajs, state_scale, u_scale_target, device, epochs) -> tuple:
    """目标校准集训练 Fθ（上界参照，节点展平为 2D pairs 复用 train_physical_oracle）。"""
    z_t_l, u_t_l, dt_l, z_next_l = [], [], [], []
    for traj in calib_trajs:
        z_norm = traj["z_node"]; u_phys = traj["u_phys"]; dt = traj["dt"]
        n_nodes = z_norm.shape[1]
        for t in range(len(dt)):
            z_t_l.append(z_norm[t])
            u_t_l.append(np.broadcast_to(u_phys[t], (n_nodes, 3)).copy())
            dt_l.append(np.full(n_nodes, dt[t], dtype=np.float32))
            z_next_l.append(z_norm[t + 1])
    z_t = np.vstack(z_t_l); u_t = np.vstack(u_t_l); dt = np.concatenate(dt_l); z_next = np.vstack(z_next_l)
    n = len(z_t); perm = np.random.default_rng(0).permutation(n)
    n_val = max(1, n // 5)
    tr = (z_t[perm[n_val:]], u_t[perm[n_val:]], dt[perm[n_val:]], z_next[perm[n_val:]])
    va = (z_t[perm[:n_val]], u_t[perm[:n_val]], dt[perm[:n_val]], z_next[perm[:n_val]])
    model = PhysicalTransition().to(device)
    model, best_val, done = train_physical_oracle(model, tr, va, state_scale, u_scale_target,
                                                   epochs=epochs, device=device, patience=5)
    return model, float(best_val), int(done)


def _rollout_gate2b(group, traj, state_scale, u_scale, device, horizons, *,
                    source_F, target_oracle_F, k_perm, k_q, v_d_target):
    z_node_norm = torch.as_tensor(traj["z_node"] / state_scale, dtype=torch.float32, device=device)
    u_norm = torch.as_tensor(traj["u_phys"] / u_scale, dtype=torch.float32, device=device)
    dt = torch.as_tensor(traj["dt"], dtype=torch.float32, device=device)
    n_nodes = z_node_norm.shape[1]
    z = z_node_norm[0].clone()
    max_H = min(max(horizons), z_node_norm.shape[0] - 1)
    preds = {}
    for k in range(max_H):
        u_k = u_norm[k].unsqueeze(0).expand(n_nodes, -1)
        dt_k = dt[k].reshape(1, 1).expand(n_nodes, -1)
        if group == "persistence":
            z_next = z
        elif group == "target_constant_rate":
            dd = v_d_target * dt_k[:, 0]
            z_next = torch.stack([z[:, 0] + dd, z[:, 1], z[:, 2] + dd], dim=-1)
        elif group == "target_oracle_F":
            z_next = target_oracle_F(z, u_k, dt_k)
        else:  # source_F_clock_calibrated / source_F_two_clock
            out = source_F(z, u_k, dt_k)
            dd = (out[:, 0] - z[:, 0]) * k_perm
            dq = (out[:, 1] - z[:, 1]) * (k_q if group == "source_F_two_clock" else 1.0)
            dr = (out[:, 2] - z[:, 2]) * k_perm
            z_next = torch.stack([z[:, 0] + dd, z[:, 1] + dq, z[:, 2] + dr], dim=-1)
        z = z_next
        if (k + 1) in horizons:
            preds[k + 1] = z.detach().cpu().numpy()
    return preds, z_node_norm.detach().cpu().numpy()


def diagnostic4(config, *, source_path, target_path, n_holdout, n_val, n_calib, epochs, seed, device,
                source_max_points, target_max_points, horizons=_GATE2B_HORIZONS) -> dict:
    """Gate 2B: 时钟校准诊断（κ_perm[+κ_q]，对照 target_constant_rate / target_oracle_F，GPT §8）。"""
    time_scale = float(config["transfer"]["transition_time_scale_s"])
    source, dynamics = _source_rows(source_path, source_max_points)
    source.transition_time_scale_s = time_scale
    train_ids, val_ids, _ = split_devices(source.ids, n_holdout=n_holdout, n_val=n_val, seed=seed)
    train = _subset(source, train_ids); val = _subset(source, val_ids)
    u_train = _physical_stress(train.x); u_val = _physical_stress(val.x)
    u_scale = u_train.std(axis=0) + 1e-6
    tr = _physical_transition_pairs(train, u_train); va = _physical_transition_pairs(val, u_val)
    source_F = PhysicalTransition().to(device)
    source_F, source_val, epochs_done = train_physical_oracle(
        source_F, tr, va, _STATE_SCALE_PHYS, u_scale, epochs=epochs, device=device, patience=5)
    all_traj = _target_node_rows(target_path, target_max_points, time_scale)
    traj_ids = [t["id"] for t in all_traj]
    calib_ids, test_ids = _split_target_calib_test(traj_ids, n_calib, seed)
    calib_trajs = [t for t in all_traj if t["id"] in calib_ids]
    test_trajs = [t for t in all_traj if t["id"] in test_ids]
    u_target_all = np.concatenate([t["u_phys"] for t in calib_trajs], axis=0)
    u_scale_target = u_target_all.std(axis=0) + 1e-6
    clock = _calibrate_clock_analytic(source_F, calib_trajs, _STATE_SCALE_PHYS, u_scale_target, device)
    v_d_target = _target_constant_rate_vd(calib_trajs, _STATE_SCALE_PHYS)
    target_oracle_F, oracle_val, oracle_done = _train_target_oracle(
        calib_trajs, _STATE_SCALE_PHYS, u_scale_target, device, epochs=epochs)
    group_preds = {g: {H: [] for H in horizons} for g in _GATE2B_GROUPS}
    targets = {H: [] for H in horizons}
    for traj in test_trajs:
        for g in _GATE2B_GROUPS:
            preds, z_node_norm = _rollout_gate2b(
                g, traj, _STATE_SCALE_PHYS, u_scale_target, device, horizons,
                source_F=source_F, target_oracle_F=target_oracle_F,
                k_perm=clock["k_perm"], k_q=clock["k_q"], v_d_target=v_d_target)
            for H in horizons:
                if H in preds:
                    group_preds[g][H].append(preds[H])
        for H in horizons:
            if H < z_node_norm.shape[0]:
                targets[H].append(z_node_norm[H])
    results_by_horizon = {}
    for H in horizons:
        if not targets[H] or not group_preds["persistence"][H]:
            continue
        target_arr = np.stack(targets[H], axis=0)
        persist_arr = np.stack(group_preds["persistence"][H], axis=0)
        mse_persist = float(np.mean((persist_arr - target_arr) ** 2))
        groups_stat = {}
        for g in _GATE2B_GROUPS:
            if not group_preds[g][H]:
                continue
            pred_arr = np.stack(group_preds[g][H], axis=0)
            mse_f = float(np.mean((pred_arr - target_arr) ** 2))
            groups_stat[g] = {"endpoint_mse_norm": mse_f,
                              "endpoint_skill_vs_persistence": float(1.0 - mse_f / max(mse_persist, 1e-12))}
        results_by_horizon[str(H)] = {"persistence_mse_norm": mse_persist, "groups": groups_stat}
    return {
        "run_scope": "diagnostic_exploratory_not_in_acceptance",
        "diagnostic": 4, "diagnostic_name": "gate2b_clock_calibrated",
        "source_dynamics_id": dynamics, "source_best_val_loss": float(source_val),
        "target_oracle_val_loss": float(oracle_val),
        "clock_calibration": clock, "target_v_d_normalized": v_d_target,
        "horizons": list(horizons), "n_calib": int(len(calib_trajs)), "n_test": int(len(test_trajs)),
        "results_by_horizon": results_by_horizon,
    }


def _render_diagnostic4_markdown(result: dict) -> str:
    horizons = result["horizons"]
    ck = result["clock_calibration"]
    lines = [
        "# Gate 2B: 时钟校准诊断（探索性，不进 IID 验收）",
        "",
        "主问题：校准 κ_perm[+κ_q] 后，source_F 是否优于 target_constant_rate？（路线二生死点）",
        "",
        f"- κ_perm={ck['k_perm']:.4f} / κ_q={ck['k_q']:.4f}（解析 through-origin）；源 Δd̃={ck['source_dd_mean']:.6f} -> 目标 Δd̃={ck['target_dd_mean']:.6f}",
        f"- target_constant_rate v_d={result['target_v_d_normalized']:.6f}（目标校准集，per normalized_dt）",
        f"- target_oracle_F val loss={result['target_oracle_val_loss']:.6g}（上界参照）",
        f"- 校准 {result['n_calib']} / 测试 {result['n_test']} 条轨迹",
        "",
        "## 终点 skill vs persistence（正值=优于 persistence）",
        "",
        "| H 步 | persistence MSE | target_constant_rate | source_F_clock_calibrated | source_F_two_clock | target_oracle_F |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for H in horizons:
        r = result["results_by_horizon"].get(str(H))
        if not r:
            continue
        g = r["groups"]
        row = [f"{H}", f"{r['persistence_mse_norm']:.4f}"]
        for name in ("target_constant_rate", "source_F_clock_calibrated", "source_F_two_clock", "target_oracle_F"):
            if name in g:
                row.append(f"{g[name]['endpoint_skill_vs_persistence']:+.4f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## 解读（GPT §9）",
        "",
        "- 情况 B：source_F_calibrated 优于 target_constant_rate -> 动力学形状可迁移，时钟需域适配（路线二成立）",
        "- 情况 C：source_F_calibrated 仅追平 target_constant_rate -> 源 Fθ 无条件动力学增益（停止）",
        "- 情况 D：source_F_calibrated 差于 target_constant_rate -> 无迁移价值（停止路线二）",
        "- target_oracle_F 是目标域自身训练上界，不参与正迁移结论，仅作参照",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phased_array_gan.yaml")
    parser.add_argument("--diagnostic", choices=["0", "1", "2", "3", "4"], default="1")
    parser.add_argument("--n-holdout", type=int, default=10)
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-points", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=6, help="diagnostic 2 闭环步数")
    parser.add_argument("--target-max-points", type=int, default=1500, help="diagnostic 3 target per-traj points")
    parser.add_argument("--n-calib", type=int, default=10, help="diagnostic 4 target calib traj count")
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed, deterministic=True, cudnn_benchmark=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_path = Path(config["source"]["feature_path"])
    target_path = Path(config["target"]["feature_path"])

    if args.diagnostic == "0":
        result = diagnostic0(config, source_path=source_path, target_path=target_path,
                             max_points=args.max_points)
        render = _render_u_audit_markdown
        default_output = "docs/results/data/gan_diag_gate0.json"
        default_report = "docs/results/data/gan_diag_gate0.md"
    elif args.diagnostic == "1":
        result = diagnostic1(
            config, source_path=source_path, n_holdout=args.n_holdout, n_val=args.n_val,
            epochs=args.epochs, seed=args.seed, device=device, max_points=args.max_points)
        render = _render_diagnostic1_markdown
        default_output = "docs/results/data/gan_diag_gate1.json"
        default_report = "docs/results/data/gan_diag_gate1.md"
    elif args.diagnostic == "2":
        result = diagnostic2(
            config, source_path=source_path, n_holdout=args.n_holdout, n_val=args.n_val,
            epochs=args.epochs, seed=args.seed, device=device, max_points=args.max_points,
            horizon=args.horizon)
        render = _render_diagnostic2_markdown
        default_output = "docs/results/data/gan_diag_gate1_1.json"
        default_report = "docs/results/data/gan_diag_gate1_1.md"
    elif args.diagnostic == "3":
        result = diagnostic3(
            config, source_path=source_path, target_path=target_path,
            n_holdout=args.n_holdout, n_val=args.n_val, epochs=args.epochs, seed=args.seed,
            device=device, source_max_points=args.max_points, target_max_points=args.target_max_points)
        render = _render_diagnostic3_markdown
        default_output = "docs/results/data/gan_diag_gate2a.json"
        default_report = "docs/results/data/gan_diag_gate2a.md"
    elif args.diagnostic == "4":
        result = diagnostic4(
            config, source_path=source_path, target_path=target_path,
            n_holdout=args.n_holdout, n_val=args.n_val, n_calib=args.n_calib, epochs=args.epochs,
            seed=args.seed, device=device, source_max_points=args.max_points,
            target_max_points=args.target_max_points)
        render = _render_diagnostic4_markdown
        default_output = "docs/results/data/gan_diag_gate2b.json"
        default_report = "docs/results/data/gan_diag_gate2b.md"
    else:
        raise ValueError(f"诊断 {args.diagnostic} 尚未实现")

    output = Path(args.output or default_output)
    report = Path(args.report or default_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report.write_text(render(result), encoding="utf-8")
    print(f">> 诊断 {args.diagnostic} 完成 -> {output} / {report}")
    if args.diagnostic == "0":
        print(json.dumps({"coverage_in_source_range": result["coverage_in_source_range"],
                          "target_nn_distance_normalized": result["target_nn_distance_normalized"],
                          "recovery_active_fraction": result["recovery_active_fraction"]},
                         ensure_ascii=False, indent=2))
    elif args.diagnostic == "2":
        print(json.dumps({"skill_vs_persistence": result["skill_vs_persistence"]}, ensure_ascii=False, indent=2))
    elif args.diagnostic == "3":
        print(json.dumps({"source_dd_mean_normalized": result["source_dd_mean_normalized"],
                          "rate_match_calibration": result["rate_match_calibration"],
                          "results_by_horizon": result["results_by_horizon"]},
                         ensure_ascii=False, indent=2))
    elif args.diagnostic == "4":
        print(json.dumps({"clock_calibration": result["clock_calibration"],
                          "target_v_d_normalized": result["target_v_d_normalized"],
                          "results_by_horizon": result["results_by_horizon"]},
                         ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"deltas": result["deltas"], "nrmse_mean": {
            k: v["mean"] for k, v in result["nrmse_6step"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
