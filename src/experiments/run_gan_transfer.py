"""GaN RFALT 三状态迁移实验。

主比较仅含 target_only、gan_transition_init、gan_joint_no_align 和
random_source_control。旧 MOSFET 路线在独立历史结果中保留，不会导入本模块。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from src.transfer.damage_state import DamageStateModel
from src.utils import load_config, set_seed


ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("target_only", "gan_transition_init", "gan_joint_no_align", "random_source_control")
_FORBIDDEN_GROUP_TOKENS = ("oracle", "nasa", "mosfet", "legacy")
_RFALT_SOURCE_NAME = "synthetic-GaN-RFALT"
_RFALT_SCHEMA_VERSION = "gan_rfalt_v1"
_RFALT_DYNAMICS_ID = "rfalt_lumped_v1"
_RFALT_FEATURE_NAMES = (
    "T_base_C", "T_j_C", "VDS", "VGS", "ID", "IG", "duty_cycle", "PAPR_dB", "VSWR", "Pin_dBm",
    "Pout_dBm", "gain_dB", "PAE", "AM_AM_dB", "AM_PM_deg", "EVM_pct", "ACPR_dBc", "RDS_dynamic_ohm", "gm_S", "Vth_V",
)
_EVALUATION_HORIZON_STEPS = 6
_TARGET_CHANNEL_NAMES = ("gain_dB", "phase_deg", "Pout_dBm", "PAE")
_TARGET_OOD_CONDITION_SCHEMA = "target_ood_conditions_v1"


class TrainOnlyStandardizer:
    """只用调用者提供的训练行拟合，避免目标验证/测试遥测进入 scaler。"""

    def fit(self, x: np.ndarray) -> "TrainOnlyStandardizer":
        self.mean_ = np.asarray(x, dtype=np.float32).mean(axis=0)
        self.scale_ = np.asarray(x, dtype=np.float32).std(axis=0) + 1e-6
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("standardizer 必须先在训练集 fit")
        return ((np.asarray(x, dtype=np.float32) - self.mean_) / self.scale_).astype(np.float32)


def split_target_trajectories(ids: list[str], seed: int, ratios: tuple[float, float] = (0.7, 0.1),
                              event_by_id: dict[str, bool] | None = None):
    if len(ids) < 3:
        raise ValueError("至少需要三条目标轨迹才能作 train/val/test 隔离")
    rng = np.random.default_rng(seed)
    if event_by_id is not None:
        missing = set(ids) - set(event_by_id)
        if missing:
            raise ValueError(f"event_by_id 缺少轨迹: {sorted(missing)}")
        failed = np.asarray([trajectory for trajectory in ids if event_by_id[trajectory]], dtype=object)
        censored = np.asarray([trajectory for trajectory in ids if not event_by_id[trajectory]], dtype=object)
        if len(failed) < 2 or len(censored) < 2:
            raise ValueError("分层测试至少需要两条失效和两条删失轨迹")

        def split_stratum(stratum: np.ndarray):
            shuffled = stratum[rng.permutation(len(stratum))]
            n_train = int(round(ratios[0] * len(stratum)))
            n_val = int(round(ratios[1] * len(stratum)))
            # 每种事件类型必须至少留一条在测试集；其余保持近似既定比例。
            n_train = min(max(1, n_train), len(stratum) - 1)
            n_val = min(max(0, n_val), len(stratum) - n_train - 1)
            return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]

        f_train, f_val, f_test = split_stratum(failed)
        c_train, c_val, c_test = split_stratum(censored)
        train = rng.permutation(np.concatenate([f_train, c_train])).tolist()
        val = rng.permutation(np.concatenate([f_val, c_val])).tolist()
        test = rng.permutation(np.concatenate([f_test, c_test])).tolist()
        return train, val, test
    perm = np.asarray(ids, dtype=object)[rng.permutation(len(ids))]
    n_train = max(1, int(round(len(ids) * ratios[0])))
    n_val = max(1, int(round(len(ids) * ratios[1])))
    n_val = min(n_val, len(ids) - n_train - 1)
    return perm[:n_train].tolist(), perm[n_train:n_train + n_val].tolist(), perm[n_train + n_val:].tolist()


@dataclass(frozen=True)
class TargetOODSplit:
    """预注册组合 OOD 测试与 IID train/val/test 的轨迹级隔离结果。"""
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    iid_test_ids: tuple[str, ...]
    ood_test_ids: tuple[str, ...]


def _canonical_ood_protocol(protocol: dict) -> dict[str, float | str]:
    required = {"schema_version", "abs_scan_az_deg_min", "duty_cycle_min", "Tj_base_C_min"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"OOD 协议缺少字段: {', '.join(sorted(missing))}")
    if str(protocol["schema_version"]) != _TARGET_OOD_CONDITION_SCHEMA:
        raise ValueError(f"OOD 协议 schema_version 必须是 {_TARGET_OOD_CONDITION_SCHEMA}")
    result = {
        "schema_version": _TARGET_OOD_CONDITION_SCHEMA,
        "abs_scan_az_deg_min": float(protocol["abs_scan_az_deg_min"]),
        "duty_cycle_min": float(protocol["duty_cycle_min"]),
        "Tj_base_C_min": float(protocol["Tj_base_C_min"]),
    }
    if not all(np.isfinite(value) and value > 0.0 for key, value in result.items() if key != "schema_version"):
        raise ValueError("OOD 协议阈值必须为有限正数")
    return result


def is_target_ood_condition(condition: dict[str, float], protocol: dict) -> bool:
    """仅保留扫描、占空比、基准结温同时处于预注册角点的轨迹为 OOD test。"""
    rules = _canonical_ood_protocol(protocol)
    required = {"scan_az_deg", "duty_cycle", "Tj_base_C"}
    missing = required - set(condition)
    if missing:
        raise ValueError(f"目标条件缺少字段: {', '.join(sorted(missing))}")
    scan, duty, temperature = (float(condition[name]) for name in ("scan_az_deg", "duty_cycle", "Tj_base_C"))
    if not all(np.isfinite(value) for value in (scan, duty, temperature)):
        raise ValueError("目标 OOD 条件必须为有限数")
    return (abs(scan) >= rules["abs_scan_az_deg_min"]
            and duty >= rules["duty_cycle_min"] and temperature >= rules["Tj_base_C_min"])


def read_target_ood_conditions(path: Path, protocol: dict) -> dict[str, dict[str, float]]:
    """读取 feature HDF5 的轨迹级任务条件；缺元数据时拒绝降级为随机 IID 切分。"""
    rules = _canonical_ood_protocol(protocol)
    conditions: dict[str, dict[str, float]] = {}
    with h5py.File(path, "r") as h5:
        schema = _h5_attr_text(h5.attrs, "target_condition_schema")
        if schema != _TARGET_OOD_CONDITION_SCHEMA:
            raise ValueError(f"target feature 缺少 OOD 条件 schema={_TARGET_OOD_CONDITION_SCHEMA}；请重建特征")
        for trajectory in sorted(h5.keys()):
            group = h5[trajectory]
            required = ("scan_az_deg", "duty_cycle", "Tj_base_C")
            missing = [name for name in required if name not in group.attrs]
            if missing:
                raise ValueError(f"target {trajectory} 缺少 OOD 条件元数据: {', '.join(missing)}")
            condition = {name: float(group.attrs[name]) for name in required}
            is_target_ood_condition(condition, rules)
            conditions[str(trajectory)] = condition
    if not conditions:
        raise ValueError("target feature 不含轨迹，无法建立 OOD 保持集")
    return conditions


def split_target_iid_ood(ids: list[str], conditions: dict[str, dict[str, float]], protocol: dict, *, seed: int,
                         ratios: tuple[float, float] = (0.7, 0.1),
                         event_by_id: dict[str, bool] | None = None) -> TargetOODSplit:
    """先固定 OOD 组合角点，再从其余 IID 轨迹划 train/val/IID-test；OOD 绝不参与选模。"""
    unique_ids = sorted({str(value) for value in ids})
    if set(unique_ids) != set(conditions):
        raise ValueError("OOD 条件轨迹集合必须与待切分目标轨迹完全一致")
    ood_ids = [trajectory for trajectory in unique_ids if is_target_ood_condition(conditions[trajectory], protocol)]
    iid_ids = [trajectory for trajectory in unique_ids if trajectory not in set(ood_ids)]
    if not ood_ids:
        raise ValueError("预注册 OOD 角点没有轨迹；拒绝将 IID 测试伪装为 OOD")
    if len(iid_ids) < 3:
        raise ValueError("IID 池至少需要三条轨迹以隔离 train/val/IID-test")
    if event_by_id is not None:
        if set(event_by_id) != set(unique_ids):
            raise ValueError("event_by_id 必须覆盖全部 IID/OOD 轨迹")
        train, val, iid_test = split_target_trajectories(
            iid_ids, seed, ratios=ratios, event_by_id={trajectory: event_by_id[trajectory] for trajectory in iid_ids})
    else:
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(iid_ids, dtype=object)[rng.permutation(len(iid_ids))]
        n_train = min(max(1, int(round(ratios[0] * len(shuffled)))), len(shuffled) - 2)
        n_val = min(max(1, int(round(ratios[1] * len(shuffled)))), len(shuffled) - n_train - 1)
        train = shuffled[:n_train].tolist()
        val = shuffled[n_train:n_train + n_val].tolist()
        iid_test = shuffled[n_train + n_val:].tolist()
    split = TargetOODSplit(tuple(str(value) for value in train), tuple(str(value) for value in val),
                           tuple(str(value) for value in iid_test), tuple(ood_ids))
    assigned = set(split.train_ids) | set(split.val_ids) | set(split.iid_test_ids)
    if (not split.train_ids or not split.val_ids or not split.iid_test_ids or assigned & set(split.ood_test_ids)
            or assigned | set(split.ood_test_ids) != set(unique_ids)):
        raise ValueError("OOD/IID 轨迹切分无效或发生泄漏")
    return split


def _ood_fingerprint(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_ood_split_manifest(protocol: dict, conditions: dict[str, dict[str, float]], split: TargetOODSplit) -> dict:
    """为未来 OOD 报告绑定预注册协议、所有条件与四个互斥轨迹集合。"""
    rules = _canonical_ood_protocol(protocol)
    canonical_conditions = {str(key): {name: float(value[name]) for name in ("scan_az_deg", "duty_cycle", "Tj_base_C")}
                            for key, value in sorted(conditions.items())}
    expected_ood = {trajectory for trajectory, condition in canonical_conditions.items()
                    if is_target_ood_condition(condition, rules)}
    actual_ood = set(split.ood_test_ids)
    assigned_iid = set(split.train_ids) | set(split.val_ids) | set(split.iid_test_ids)
    if actual_ood != expected_ood:
        raise ValueError("OOD split manifest ood_test_ids 必须精确匹配预注册角点谓词")
    if actual_ood & assigned_iid or actual_ood | assigned_iid != set(canonical_conditions):
        raise ValueError("OOD split manifest 的 IID/OOD 轨迹集合不完整或发生泄漏")
    return {
        "schema_version": "gan_target_ood_split_manifest_v1", "ood_protocol": rules,
        "condition_fingerprint": _ood_fingerprint(canonical_conditions),
        "all_trajectory_ids": sorted(canonical_conditions),
        "train_ids": list(split.train_ids), "val_ids": list(split.val_ids),
        "iid_test_ids": list(split.iid_test_ids), "ood_test_ids": list(split.ood_test_ids),
        "ood_test_fingerprint": _ood_fingerprint(list(split.ood_test_ids)),
    }


def validate_ood_split_manifest(saved: dict, protocol: dict, conditions: dict[str, dict[str, float]], split: TargetOODSplit) -> None:
    current = build_ood_split_manifest(protocol, conditions, split)
    required = tuple(current)
    for key in required:
        if key not in saved:
            raise ValueError(f"OOD split manifest 缺少 {key}")
        if saved[key] != current[key]:
            raise ValueError(f"OOD split manifest {key} 与当前运行不匹配")


def persist_ood_split_manifest(path: Path, protocol: dict, conditions: dict[str, dict[str, float]], split: TargetOODSplit) -> dict:
    """持久化纯数据切分审计产物；不训练、不产生 IID/OOD 性能结论。"""
    manifest = build_ood_split_manifest(protocol, conditions, split)
    validate_ood_split_manifest(manifest, protocol, conditions, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def target_event_by_trajectory(rows: "DomainRows") -> dict[str, bool]:
    """从扁平目标行恢复轨迹级 event_observed，并拒绝同轨迹内部不一致。"""
    events: dict[str, bool] = {}
    for trajectory in np.unique(rows.ids):
        values = np.unique(rows.event[rows.ids == trajectory])
        if len(values) != 1:
            raise ValueError(f"轨迹 {trajectory} 的 event_observed 不一致")
        events[str(trajectory)] = bool(values[0])
    return events


def validate_test_event_coverage(event_by_id: dict[str, bool], test_ids: list[str]) -> None:
    values = [event_by_id[trajectory] for trajectory in test_ids]
    if not any(values) or all(values):
        raise ValueError("测试集必须同时保留失效与删失轨迹；拒绝全删失/全失效评估")


def validate_domain_separation(source_dynamics_id: str, target_dynamics_id: str) -> None:
    if not source_dynamics_id or not target_dynamics_id or source_dynamics_id == target_dynamics_id:
        raise ValueError("source 与 target 的 dynamics_id 必须存在且不同")


def validate_experiment_groups(groups) -> tuple[str, ...]:
    """拒绝将理论上界或旧 MOSFET 路线放入 damage-state 主比较。"""
    normalized = tuple(str(group) for group in groups)
    for group in normalized:
        if any(token in group.lower() for token in _FORBIDDEN_GROUP_TOKENS):
            raise ValueError(f"禁止将 {group} 作为 damage_state 实验组")
        if group not in GROUPS:
            raise ValueError(f"未知 GaN 组: {group}")
    return normalized


def validate_source_config(source_config: dict) -> None:
    """damage_state 路线仅接受声明为 GaN RFALT 的源域配置。"""
    if source_config.get("name") != _RFALT_SOURCE_NAME:
        raise ValueError("damage_state 源域必须是 synthetic-GaN-RFALT（GaN RFALT）")
    if source_config.get("schema_version") != _RFALT_SCHEMA_VERSION:
        raise ValueError(f"damage_state source schema_version 必须是 {_RFALT_SCHEMA_VERSION}")
    if source_config.get("dynamics_id") != _RFALT_DYNAMICS_ID:
        raise ValueError(f"damage_state source dynamics_id 必须是 {_RFALT_DYNAMICS_ID}")


def _h5_attr_text(attrs, name: str) -> str:
    value = attrs.get(name, "")
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def validate_source_h5_contract(path: Path) -> None:
    """字段读取前校验 RFALT HDF5 元数据，禁止旧源域伪装进入训练。"""
    with h5py.File(path, "r") as h5:
        schema = _h5_attr_text(h5.attrs, "schema_version")
        dynamics = _h5_attr_text(h5.attrs, "dynamics_id")
        if schema != _RFALT_SCHEMA_VERSION:
            raise ValueError(f"RFALT source schema_version 非法: {schema!r}（要求 {_RFALT_SCHEMA_VERSION!r}）")
        if dynamics != _RFALT_DYNAMICS_ID:
            raise ValueError(f"RFALT source dynamics_id 非法: {dynamics!r}（要求 {_RFALT_DYNAMICS_ID!r}）")
        names = [value.decode("utf-8") if isinstance(value, bytes) else str(value)
                 for value in h5.attrs.get("feature_names", [])]
        dim = int(h5.attrs.get("feature_dim", -1))
        if dim != len(_RFALT_FEATURE_NAMES) or tuple(names) != _RFALT_FEATURE_NAMES:
            raise ValueError("RFALT source feature_names/feature_dim 不符合精确 20 列契约")
        if "devices" not in h5 or not h5["devices"].keys():
            raise ValueError("RFALT source 缺少非空 devices 分组")
        required = ("x", "time_s", "rul_lower_bound_s", "latent_d_perm", "latent_q_trap", "latent_r_th")
        for device, group in h5["devices"].items():
            missing = [name for name in required if name not in group]
            if missing:
                raise ValueError(f"RFALT source {device} 缺少必需数据集: {', '.join(missing)}")
            if "event_observed" not in group.attrs:
                raise ValueError(f"RFALT source {device} 缺少 event_observed 属性")
            x = group["x"]
            if x.ndim != 2 or x.shape[1] != dim:
                raise ValueError(f"RFALT source {device}/x 维度必须为 (T, {dim})")
            for name in required[1:]:
                if group[name].ndim != 1 or len(group[name]) != len(x):
                    raise ValueError(f"RFALT source {device}/{name} 长度必须与 x 对齐")
            for name in required:
                if not np.isfinite(group[name][:]).all():
                    raise ValueError(f"RFALT source {device}/{name} 含非有限值")


def make_target_observations(
    x_global: np.ndarray, x_nodes: np.ndarray, _latent_states: np.ndarray | None = None,
    _labels: np.ndarray | None = None, *, preserve_node_axis: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """模型输入仅来自当前可观测遥测；后两个参数显式忽略，便于泄漏测试。"""
    if preserve_node_axis:
        # 空间阶段接口：保留子阵节点，不接触 latent 或 score-only 标签。
        return np.asarray(x_global, np.float32), np.asarray(x_nodes, np.float32)
    return np.concatenate([np.asarray(x_global, np.float32), np.asarray(x_nodes, np.float32).mean(axis=1)], axis=1)


def scramble_source_state_pairs(states: np.ndarray, device_ids: np.ndarray, time: np.ndarray, seed: int):
    """保留每台器件的时间行及应力顺序，仅打乱它对应的状态监督。"""
    result = np.asarray(states, np.float32).copy()
    rng = np.random.default_rng(seed)
    for device in np.unique(device_ids):
        idx = np.flatnonzero(device_ids == device)
        if len(idx) > 1:
            result[idx] = result[idx][rng.permutation(len(idx))]
    return result, np.asarray(time).copy()


@dataclass
class DomainRows:
    x: np.ndarray
    states: np.ndarray
    obs_labels: np.ndarray
    rul: np.ndarray
    event: np.ndarray
    ids: np.ndarray
    time: np.ndarray
    transition_time_scale_s: float = 1.0
    array_link_margin: np.ndarray | None = None


@dataclass
class SpatialDomainRows:
    """空间阶段专用、尚未接入全局训练的节点监督接口。"""
    x_global: np.ndarray
    x_nodes: np.ndarray
    node_states: np.ndarray
    node_labels: np.ndarray
    array_score_truth: dict[str, np.ndarray]
    rul: np.ndarray
    event: np.ndarray
    ids: np.ndarray
    time: np.ndarray


_SPATIAL_SCORE_DATASETS = {
    "G_array_dB": "label_array_G_array_dB",
    "EIRP_norm": "label_array_EIRP_norm",
    "SLL_dB": "label_array_SLL_dB",
    "theta_err_deg": "label_array_theta_err_deg",
    "M_link_dB": "label_array_M_link_dB",
}
_SPATIAL_NODE_STATE_DATASETS = ("label_node_d_perm", "label_node_q_trap", "label_node_r_th")
_SPATIAL_NODE_CHANNEL_DATASETS = (
    "label_channel_gain_dB", "label_channel_phase_deg", "label_channel_Pout_dBm", "label_channel_PAE",
)


@dataclass
class TransitionPairs:
    """同一轨迹内严格相邻的 t→t+1 监督对；state 标签永不作为模型输入。"""
    x_t: np.ndarray
    stress_t: np.ndarray
    state_t: np.ndarray
    state_t_plus_1: np.ndarray
    obs_t_plus_1: np.ndarray
    rul_t_plus_1: np.ndarray
    event_t_plus_1: np.ndarray
    normalized_dt: np.ndarray
    time_t: np.ndarray
    time_t_plus_1: np.ndarray
    ids: np.ndarray


def make_transition_pairs(rows: DomainRows) -> TransitionPairs:
    """构造同轨迹、时间正向的 `(x_t, state_{t+1}, dt)`，禁止跨轨迹拼接。"""
    starts: list[int] = []
    ends: list[int] = []
    for device in np.unique(rows.ids):
        index = np.flatnonzero(rows.ids == device)
        if len(index) < 2:
            continue
        dt = np.diff(rows.time[index])
        if np.any(dt <= 0):
            raise ValueError(f"轨迹 {device} 的 time 必须严格递增")
        starts.extend(index[:-1].tolist())
        ends.extend(index[1:].tolist())
    if not starts:
        raise ValueError("没有长度至少为 2 的完整轨迹可构造 transition 对")
    start = np.asarray(starts, dtype=int)
    end = np.asarray(ends, dtype=int)
    raw_dt = (rows.time[end] - rows.time[start]).astype(np.float32)
    scale = float(rows.transition_time_scale_s)
    if scale <= 0:
        raise ValueError("transition_time_scale_s 必须为正")
    return TransitionPairs(
        x_t=rows.x[start], stress_t=rows.x[start], state_t=rows.states[start],
        state_t_plus_1=rows.states[end], obs_t_plus_1=rows.obs_labels[end],
        rul_t_plus_1=rows.rul[end], event_t_plus_1=rows.event[end],
        normalized_dt=(raw_dt / scale).reshape(-1, 1).astype(np.float32),
        time_t=rows.time[start], time_t_plus_1=rows.time[end], ids=rows.ids[end],
    )


def _source_rows(path: Path, max_points_per_device: int) -> tuple[DomainRows, str]:
    xs: list[np.ndarray] = []; states: list[np.ndarray] = []; labels: list[np.ndarray] = []
    ruls: list[np.ndarray] = []; events: list[np.ndarray] = []; ids: list[np.ndarray] = []; times: list[np.ndarray] = []
    validate_source_h5_contract(path)
    with h5py.File(path, "r") as h5:
        dynamics = str(h5.attrs.get("dynamics_id", ""))
        names = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in h5.attrs["feature_names"]]
        label_idx = [names.index(name) for name in ("gain_dB", "Pout_dBm", "PAE", "RDS_dynamic_ohm")]
        for device in sorted(h5["devices"].keys()):
            g = h5["devices"][device]
            n = len(g["x"]); ix = np.linspace(0, n - 1, min(n, max_points_per_device), dtype=int)
            xs.append(g["x"][ix]); labels.append(g["x"][ix][:, label_idx])
            states.append(np.stack([g["latent_d_perm"][ix], g["latent_q_trap"][ix], g["latent_r_th"][ix]], axis=1))
            ruls.append(g["rul_lower_bound_s"][ix]); events.append(np.full(len(ix), bool(g.attrs["event_observed"])))
            ids.append(np.full(len(ix), str(device))); times.append(g["time_s"][ix])
    return DomainRows(*(np.concatenate(v) for v in (xs, states, labels, ruls, events, ids, times))), dynamics


def _target_rows(path: Path, max_points_per_traj: int) -> tuple[DomainRows, str]:
    xs: list[np.ndarray] = []; states: list[np.ndarray] = []; labels: list[np.ndarray] = []
    ruls: list[np.ndarray] = []; events: list[np.ndarray] = []; ids: list[np.ndarray] = []; times: list[np.ndarray] = []; link_margins: list[np.ndarray] = []
    with h5py.File(path, "r") as h5:
        dynamics = str(h5.attrs.get("dynamics_id", "leo_coupled_v1"))
        for trajectory in sorted(h5.keys()):
            g = h5[trajectory]; n = len(g["x_global"])
            ix = np.linspace(0, n - 1, min(n, max_points_per_traj), dtype=int)
            x = make_target_observations(g["x_global"][ix], g["x_nodes"][ix])
            # 256 元件 latent 聚合到轨迹级状态监督；它们始终不拼入 x。
            state = np.stack([g["latent_d_perm"][ix].mean(axis=1), g["latent_q_trap"][ix].mean(axis=1), g["latent_r_th"][ix].mean(axis=1)], axis=1)
            channel = np.stack([g["label_channel_gain_dB"][ix].mean(axis=1), g["label_channel_phase_deg"][ix].mean(axis=1), g["label_channel_Pout_dBm"][ix].mean(axis=1), g["label_channel_PAE"][ix].mean(axis=1)], axis=1)
            xs.append(x); states.append(state); labels.append(channel); ruls.append(g["rul"][ix])
            link_margins.append(g["x_global"][ix, 1])
            events.append(np.full(len(ix), bool(g.attrs.get("event_observed", 1))))
            if "time_s" not in g:
                raise ValueError(f"target feature {trajectory} 缺少 time_s；请重新运行 build_array_hi")
            ids.append(np.full(len(ix), trajectory)); times.append(g["time_s"][ix])
    return DomainRows(*(np.concatenate(v) for v in (xs, states, labels, ruls, events, ids, times)),
                      array_link_margin=np.concatenate(link_margins).astype(np.float32)), dynamics


def _spatial_target_rows(path: Path, max_points_per_traj: int) -> tuple[SpatialDomainRows, str]:
    """读取空间节点训练标签与 score-only 阵列真值；不接入现有全局训练路径。"""
    global_x: list[np.ndarray] = []; node_x: list[np.ndarray] = []; node_states: list[np.ndarray] = []
    node_labels: list[np.ndarray] = []; ruls: list[np.ndarray] = []; events: list[np.ndarray] = []
    ids: list[np.ndarray] = []; times: list[np.ndarray] = []
    scores: dict[str, list[np.ndarray]] = {name: [] for name in _SPATIAL_SCORE_DATASETS}
    with h5py.File(path, "r") as h5:
        dynamics = str(h5.attrs.get("dynamics_id", "leo_coupled_v1"))
        for trajectory in sorted(h5.keys()):
            g = h5[trajectory]
            required = {"x_global", "x_nodes", "rul", "label_fail", "time_s", *_SPATIAL_NODE_STATE_DATASETS,
                        *_SPATIAL_NODE_CHANNEL_DATASETS, *_SPATIAL_SCORE_DATASETS.values()}
            missing = sorted(required - set(g.keys()))
            if missing:
                raise ValueError(f"空间 target {trajectory} 缺少字段: {', '.join(missing)}")
            n = len(g["x_global"])
            if n < 2 or g["x_global"].shape != (n, 6) or g["x_nodes"].shape != (n, 16, 6):
                raise ValueError(f"空间 target {trajectory} 的观测维度必须为 (T,6)/(T,16,6)")
            if int(g.attrs.get("eol_idx", -1)) != n - 1:
                raise ValueError(f"空间 target {trajectory} 的 EOL 截断未同步到末行")
            event = bool(g.attrs.get("event_observed", 0))
            label_fail = np.asarray(g["label_fail"][:], dtype=bool)
            if bool(label_fail.any()) != event or (event and not bool(label_fail[-1])):
                raise ValueError(f"空间 target {trajectory} 的 event_observed/label_fail 不一致")
            time = np.asarray(g["time_s"][:], dtype=np.float64)
            if len(time) != n or np.any(np.diff(time) <= 0):
                raise ValueError(f"空间 target {trajectory} 的 time_s 必须严格递增并与 EOL 截断同步")
            for name in (*_SPATIAL_NODE_STATE_DATASETS, *_SPATIAL_NODE_CHANNEL_DATASETS):
                if g[name].shape != (n, 16):
                    raise ValueError(f"空间 target {trajectory}/{name} 必须为 (T,16)")
            for score_name, dataset in _SPATIAL_SCORE_DATASETS.items():
                if g[dataset].shape != (n,) or g[dataset].attrs.get("access", "") != "score_only_not_model_input":
                    raise ValueError(f"空间 target {trajectory}/{dataset} 必须是 score-only 的 (T,) 评分真值")
            ix = np.linspace(0, n - 1, min(n, max_points_per_traj), dtype=int)
            global_x.append(np.asarray(g["x_global"][ix], np.float32))
            node_x.append(np.asarray(g["x_nodes"][ix], np.float32))
            node_states.append(np.stack([g[name][ix] for name in _SPATIAL_NODE_STATE_DATASETS], axis=-1).astype(np.float32))
            node_labels.append(np.stack([g[name][ix] for name in _SPATIAL_NODE_CHANNEL_DATASETS], axis=-1).astype(np.float32))
            for score_name, dataset in _SPATIAL_SCORE_DATASETS.items():
                scores[score_name].append(np.asarray(g[dataset][ix], np.float32))
            ruls.append(np.asarray(g["rul"][ix], np.float32)); events.append(np.full(len(ix), event))
            ids.append(np.full(len(ix), trajectory)); times.append(time[ix])
    result = SpatialDomainRows(
        x_global=np.concatenate(global_x), x_nodes=np.concatenate(node_x), node_states=np.concatenate(node_states),
        node_labels=np.concatenate(node_labels), array_score_truth={key: np.concatenate(values) for key, values in scores.items()},
        rul=np.concatenate(ruls), event=np.concatenate(events), ids=np.concatenate(ids), time=np.concatenate(times),
    )
    arrays = [result.x_global, result.x_nodes, result.node_states, result.node_labels, result.rul, result.time,
              *result.array_score_truth.values()]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("空间 target 含非有限观测、标签或评分真值")
    return result, dynamics


def _select_rows(rows: DomainRows, ids: list[str]) -> DomainRows:
    keep = np.isin(rows.ids, ids)
    return DomainRows(rows.x[keep], rows.states[keep], rows.obs_labels[keep], rows.rul[keep],
                      rows.event[keep], rows.ids[keep], rows.time[keep], rows.transition_time_scale_s,
                      None if rows.array_link_margin is None else rows.array_link_margin[keep])


def normalize_target_rul(train_rows: DomainRows, *other_rows: DomainRows):
    """以训练轨迹的最大 RUL 为尺度归一，绝不从验证或测试拟合该尺度。"""
    scale = max(float(np.max(train_rows.rul)), 1.0)

    def scaled(rows: DomainRows) -> DomainRows:
        return DomainRows(rows.x, rows.states, rows.obs_labels, (rows.rul / scale).astype(np.float32),
                          rows.event, rows.ids, rows.time, rows.transition_time_scale_s, rows.array_link_margin)

    return (scaled(train_rows), *(scaled(rows) for rows in other_rows), scale)


def _pair_tensor(pairs: TransitionPairs, device: torch.device):
    return tuple(torch.as_tensor(getattr(pairs, field), device=device) for field in (
        "x_t", "stress_t", "state_t", "state_t_plus_1", "obs_t_plus_1", "rul_t_plus_1", "event_t_plus_1", "normalized_dt",
    ))


def _rul_loss(pred: torch.Tensor, rul: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    observed = event.bool()
    losses = []
    if observed.any():
        losses.append(nn.functional.huber_loss(pred[observed], rul[observed]))
    if (~observed).any():
        losses.append(torch.square(torch.relu(rul[~observed] - pred[~observed])).mean())
    return sum(losses) / max(len(losses), 1)


def state_anchor_loss(predicted_state_t: torch.Tensor, pairs: TransitionPairs) -> torch.Tensor:
    """state_t 仅为 encoder 的监督标签；绝不进入 model.forward。"""
    target = torch.as_tensor(pairs.state_t, dtype=predicted_state_t.dtype, device=predicted_state_t.device)
    return nn.functional.mse_loss(predicted_state_t, target)


def _source_loss(model: DamageStateModel, rows: DomainRows, device: torch.device,
                 *, include_observation_loss: bool = True) -> torch.Tensor:
    pairs = make_transition_pairs(rows)
    x_t, stress_t, _, state_next, obs_next, _, _, dt = _pair_tensor(pairs, device)
    state_hat_t = model.encode_source(x_t)
    pred_next = model.transition(state_hat_t, model.source_stress(stress_t), dt)
    loss = state_anchor_loss(state_hat_t, pairs) + nn.functional.mse_loss(pred_next, state_next)
    if include_observation_loss:
        loss = loss + nn.functional.mse_loss(model.observe_source(pred_next), obs_next)
    return loss


def _target_loss(model: DamageStateModel, rows: DomainRows, device: torch.device) -> torch.Tensor:
    pairs = make_transition_pairs(rows)
    x_t, stress_t, _, state_next, obs_next, rul_next, event_next, dt = _pair_tensor(pairs, device)
    # 目标预测只由 t 时刻可观测量编码的 state_hat_t rollout；真实 state_t 不会输入模型。
    state_hat_t = model.encode_target(x_t)
    pred_next = model.transition(state_hat_t, model.target_stress(stress_t), dt)
    return (state_anchor_loss(state_hat_t, pairs) + nn.functional.mse_loss(pred_next, state_next) + nn.functional.mse_loss(model.observe_target(pred_next), obs_next)
            + _rul_loss(model.predict_target_rul(pred_next), rul_next, event_next))


def pretrain_source_transition(model: DamageStateModel, source_rows: DomainRows, *, epochs: int,
                               device: torch.device, include_observation_loss: bool = True) -> int:
    """只用 RFALT 源域监督预训练 Fθ，不读取目标 train/val/test。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = _source_loss(model, source_rows, device, include_observation_loss=include_observation_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return epochs


def train_with_early_stop(model: DamageStateModel, train_rows: DomainRows, val_rows: DomainRows, *,
                          source_rows: DomainRows | None, epochs: int, patience: int = 5,
                          device: torch.device) -> tuple[DamageStateModel, int]:
    """仅由 val_loss 选择 checkpoint；测试集在此函数完全不可见。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    best_state: dict[str, torch.Tensor] | None = None; best_val = float("inf"); stale_epochs = 0; steps = 0
    for _ in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = _target_loss(model, train_rows, device)
        if source_rows is not None:
            loss = loss + _source_loss(model, source_rows, device)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        steps += 1
        model.eval()
        with torch.no_grad():
            val_loss = float(_target_loss(model, val_rows, device).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, steps


def test_trajectory_fingerprint(rows: DomainRows, trajectory_id: str) -> str:
    """哈希测试轨迹的评分真值，阻止模型组在不同标签或服务事件上比较。"""
    index = np.flatnonzero(rows.ids == trajectory_id)
    payload = {
        "trajectory_id": str(trajectory_id), "n_rows": int(len(index)),
        "time_s": np.asarray(rows.time[index], dtype=float).round(6).tolist(),
        "event": np.asarray(rows.event[index], dtype=int).tolist(),
        "rul": np.asarray(rows.rul[index], dtype=float).round(8).tolist(),
        "channel_labels": np.asarray(rows.obs_labels[index], dtype=float).round(8).tolist(),
        "array_link_margin": (None if rows.array_link_margin is None else
                               np.asarray(rows.array_link_margin[index], dtype=float).round(8).tolist()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_set_fingerprint(rows: DomainRows) -> str:
    payload = [{"trajectory_id": str(trajectory), "fingerprint": test_trajectory_fingerprint(rows, str(trajectory))}
               for trajectory in sorted(np.unique(rows.ids).tolist())]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def fit_link_margin_bridge(train_rows: DomainRows) -> tuple[float, float]:
    """只用目标训练轨迹拟合通道 Pout 到阵列遥测链路余量的线性投影。"""
    if train_rows.array_link_margin is None:
        raise ValueError("目标训练数据缺少 array_link_margin，无法计算阵列层指标")
    pout = np.asarray(train_rows.obs_labels[:, 2], dtype=float)
    design = np.column_stack([np.ones(len(pout)), pout])
    intercept, slope = np.linalg.lstsq(design, np.asarray(train_rows.array_link_margin, dtype=float), rcond=None)[0]
    return float(intercept), float(slope)


def channel_6step_nrmse_arithmetic_mean(channels: dict[str, float]) -> float:
    """正式通道主指标：四个通道六步 nRMSE 的等权算术平均。"""
    if set(channels) != set(_TARGET_CHANNEL_NAMES):
        raise ValueError("通道六步 nRMSE 必须精确包含 gain/phase/Pout/PAE")
    values = np.asarray([channels[name] for name in _TARGET_CHANNEL_NAMES], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("通道六步 nRMSE 含非有限值")
    return float(values.mean())


def _target_transition_records(rows: DomainRows) -> dict[str, tuple[DomainRows, TransitionPairs]]:
    """按轨迹组织 t→t+1 对，供参考实现和批量实现共享同一输入顺序。"""
    return {
        str(trajectory): (trajectory_rows := _select_rows(rows, [str(trajectory)]), make_transition_pairs(trajectory_rows))
        for trajectory in sorted(np.unique(rows.ids).tolist())
    }


def _rollout_target_trajectories_reference(model: DamageStateModel, rows: DomainRows,
                                           device: torch.device, max_steps: int | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """逐轨迹参考 rollout，仅作为批量实现的等价性测试基准。"""
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for trajectory, (_, pairs) in _target_transition_records(rows).items():
            x_t, stress_t, _, _, _, _, _, dt = _pair_tensor(pairs, device)
            state = model.encode_target(x_t[:1])
            predicted_obs: list[torch.Tensor] = []
            predicted_rul: list[torch.Tensor] = []
            for index in range(min(len(x_t), max_steps) if max_steps is not None else len(x_t)):
                state = model.transition(state, model.target_stress(stress_t[index:index + 1]), dt[index:index + 1])
                predicted_obs.append(model.observe_target(state))
                predicted_rul.append(model.predict_target_rul(state))
            result[trajectory] = (torch.cat(predicted_obs, dim=0).cpu().numpy(),
                                  torch.cat(predicted_rul, dim=0).cpu().numpy())
    return result


def _rollout_target_trajectories_batched(model: DamageStateModel, rows: DomainRows,
                                         device: torch.device,
                                         records: dict[str, tuple[DomainRows, TransitionPairs]] | None = None,
                                         max_steps: int | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """跨轨迹按时间步批量 rollout；每条轨迹仍只从自身首时刻状态连续推进。"""
    records = _target_transition_records(rows) if records is None else records
    identifiers = list(records)
    model.eval()
    with torch.no_grad():
        initial_x = torch.as_tensor(np.stack([records[key][1].x_t[0] for key in identifiers]), device=device)
        states = model.encode_target(initial_x)
        predicted_obs: dict[str, list[torch.Tensor]] = {key: [] for key in identifiers}
        predicted_rul: dict[str, list[torch.Tensor]] = {key: [] for key in identifiers}
        available_steps = max(len(records[key][1].x_t) for key in identifiers)
        step_limit = min(available_steps, max_steps) if max_steps is not None else available_steps
        for step in range(step_limit):
            active_positions = [position for position, key in enumerate(identifiers)
                                if step < len(records[key][1].x_t)]
            if not active_positions:
                continue
            active_keys = [identifiers[position] for position in active_positions]
            position_tensor = torch.as_tensor(active_positions, dtype=torch.long, device=device)
            stress = torch.as_tensor(np.stack([records[key][1].stress_t[step] for key in active_keys]), device=device)
            dt = torch.as_tensor(np.stack([records[key][1].normalized_dt[step] for key in active_keys]), device=device)
            next_state = model.transition(states[position_tensor], model.target_stress(stress), dt)
            states[position_tensor] = next_state
            obs = model.observe_target(next_state)
            rul = model.predict_target_rul(next_state)
            for offset, key in enumerate(active_keys):
                predicted_obs[key].append(obs[offset:offset + 1])
                predicted_rul[key].append(rul[offset:offset + 1])
    return {key: (torch.cat(predicted_obs[key], dim=0).cpu().numpy(),
                  torch.cat(predicted_rul[key], dim=0).cpu().numpy())
            for key in identifiers}


def _evaluate(model: DamageStateModel, rows: DomainRows, device: torch.device, *,
              channel_scale: np.ndarray, rul_scale_s: float,
              link_margin_bridge: tuple[float, float] | None) -> dict[str, object]:
    """逐测试轨迹 rollout，持久化通道、阵列、服务三级指标。

    主验收通道指标为从首时刻状态连续 rollout 的多步 nRMSE；阵列层为训练集
    拟合的 Pout→M_link 投影 MAE；服务层为失效轨迹的事件时间（RUL）MAE。
    """
    model.eval()
    channel_scale = np.asarray(channel_scale, dtype=np.float32).reshape(1, -1)
    trajectory_metrics: list[dict] = []
    all_rul_errors: list[float] = []
    records = _target_transition_records(rows)
    if any(len(pairs.x_t) < _EVALUATION_HORIZON_STEPS for _, pairs in records.values()):
        raise ValueError(f"固定 {_EVALUATION_HORIZON_STEPS} 步评估要求每条测试轨迹至少包含 {_EVALUATION_HORIZON_STEPS + 1} 个观测点")
    rollouts = _rollout_target_trajectories_batched(model, rows, device, records, _EVALUATION_HORIZON_STEPS)
    channel_aggregate: dict[str, list[float]] = {name: [] for name in _TARGET_CHANNEL_NAMES}
    censor_violations: list[bool] = []
    for trajectory, (trajectory_rows, pairs) in records.items():
        _, _, _, _, obs_next, rul_next, event_next, _ = _pair_tensor(pairs, device)
        obs_hat, rul_hat = rollouts[trajectory]
        steps = len(obs_hat)
        observed = obs_next.detach().cpu().numpy()[:steps]
        rul_truth = rul_next.detach().cpu().numpy()[:steps]
        event_truth = event_next.detach().cpu().numpy().astype(bool)[:steps]
        channel_nrmse = float(np.sqrt(np.mean(np.square((obs_hat - observed) / channel_scale))))
        channel_by_metric = {name: float(np.sqrt(np.mean(np.square((obs_hat[:, index] - observed[:, index]) / channel_scale[0, index]))))
                             for index, name in enumerate(_TARGET_CHANNEL_NAMES)}
        channel_mean = channel_6step_nrmse_arithmetic_mean(channel_by_metric)
        for name, value in channel_by_metric.items():
            channel_aggregate[name].append(value)
        if link_margin_bridge is None or trajectory_rows.array_link_margin is None:
            link_mae = None
        else:
            link_intercept, link_slope = link_margin_bridge
            link_prediction = link_intercept + link_slope * obs_hat[:, 2]
            link_truth = trajectory_rows.array_link_margin[1:1 + steps]
            link_mae = float(np.mean(np.abs(link_prediction - link_truth)))
        rul_error_s = np.abs((rul_hat - rul_truth) * rul_scale_s)
        event_mask = event_truth
        if event_mask.any():
            service_mae = float(np.mean(rul_error_s[event_mask]))
            all_rul_errors.extend(rul_error_s[event_mask].tolist())
        else:
            service_mae = None
        censor_mask = ~event_mask
        censor_violation = (rul_hat[censor_mask] < rul_truth[censor_mask]) if censor_mask.any() else np.asarray([], dtype=bool)
        censor_violations.extend(censor_violation.tolist())
        trajectory_metrics.append({
            "trajectory_id": str(trajectory),
            "test_trajectory_fingerprint": test_trajectory_fingerprint(trajectory_rows, str(trajectory)),
            "n_test_rows": int(len(trajectory_rows.x)), "n_failed_rows": int(event_mask.sum()),
            "n_censored_rows": int(censor_mask.sum()), "prediction_steps": steps,
            "channel_multistep_nrmse": channel_nrmse,
            "channel_6step_nrmse_by_metric": channel_by_metric,
            "channel_6step_nrmse_mean": channel_mean,
            "array_link_margin_mae_dB": link_mae,
            "service_event_mae_s": service_mae,
            "service_censor_lower_bound_violation_rate": (float(censor_violation.mean()) if censor_violation.size else None),
        })
    rmse = float(np.sqrt(np.mean(np.square(all_rul_errors))) / max(rul_scale_s, 1.0)) if all_rul_errors else float("nan")
    return {
        "rmse": rmse,
        "n_failed_rows": int(sum(item["n_failed_rows"] for item in trajectory_metrics)),
        "n_censored_rows": int(sum(item["n_censored_rows"] for item in trajectory_metrics)),
        "evaluation_horizon_steps": _EVALUATION_HORIZON_STEPS,
        "n_test_rows": int(sum(item["n_test_rows"] for item in trajectory_metrics)),
        "test_fingerprint": test_set_fingerprint(rows),
        "trajectory_metrics": trajectory_metrics,
        "channel_multistep_nrmse": float(np.mean([item["channel_multistep_nrmse"] for item in trajectory_metrics])),
        "channel_6step_nrmse_by_metric": {name: float(np.mean(values)) for name, values in channel_aggregate.items()},
        "channel_6step_nrmse_mean": float(np.mean([item["channel_6step_nrmse_mean"] for item in trajectory_metrics])),
        "array_link_margin_mae_dB": (float(np.mean([item["array_link_margin_mae_dB"] for item in trajectory_metrics]))
                                     if all(item["array_link_margin_mae_dB"] is not None for item in trajectory_metrics) else None),
        "service_event_mae_s": float(np.mean([item["service_event_mae_s"] for item in trajectory_metrics
                                                if item["service_event_mae_s"] is not None])) if all_rul_errors else None,
        "service_censor_lower_bound_violation_rate": (float(np.mean(censor_violations)) if censor_violations else None),
    }


def run_group(group: str, source: DomainRows, target_train: DomainRows, target_val: DomainRows,
              target_test: DomainRows, *, seed: int, epochs: int, patience: int = 5,
              device: torch.device, channel_scale: np.ndarray | None = None,
              rul_scale_s: float = 1.0, link_margin_bridge: tuple[float, float] | None = None) -> dict[str, object]:
    validate_experiment_groups([group])
    set_seed(seed, deterministic=True, cudnn_benchmark=False)
    model = DamageStateModel(source.x.shape[1], target_train.x.shape[1]).to(device)
    source_pretrain_steps = 0
    source_joint_steps = 0
    source_observation_supervision = group == "gan_joint_no_align"
    if group == "gan_transition_init":
        source_model = DamageStateModel(source.x.shape[1], target_train.x.shape[1]).to(device)
        # 使用源域预训练 Fθ；随后仅复制 transition，所有目标模块自然随机初始化。
        source_pretrain_steps = pretrain_source_transition(source_model, source, epochs=epochs, device=device, include_observation_loss=True)
        source_observation_supervision = True
        model.load_transition_state_dict(source_model.transition_state_dict())
        source_rows = None
    elif group == "gan_joint_no_align":
        source_rows = source
    elif group == "random_source_control":
        scrambled, time = scramble_source_state_pairs(source.states, source.ids, source.time, seed)
        scrambled_source = DomainRows(source.x, scrambled, source.obs_labels, source.rul, source.event, source.ids, time, source.transition_time_scale_s)
        # 与 init 组完全同构的源预训练预算；唯一区别是设备内观测—状态配对被置乱。
        source_model = DamageStateModel(source.x.shape[1], target_train.x.shape[1]).to(device)
        source_pretrain_steps = pretrain_source_transition(source_model, scrambled_source, epochs=epochs, device=device, include_observation_loss=False)
        model.load_transition_state_dict(source_model.transition_state_dict())
        source_rows = None
    else:
        source_rows = None
    model, target_steps = train_with_early_stop(
        model, target_train, target_val, source_rows=source_rows, epochs=epochs, patience=patience, device=device)
    if group == "gan_joint_no_align":
        source_joint_steps = target_steps
    if channel_scale is None:
        channel_scale = np.std(target_train.obs_labels, axis=0) + 1e-6
    if link_margin_bridge is None and target_train.array_link_margin is not None:
        link_margin_bridge = fit_link_margin_bridge(target_train)
    metrics = _evaluate(model, target_test, device, channel_scale=channel_scale,
                        rul_scale_s=rul_scale_s, link_margin_bridge=link_margin_bridge)
    metrics.update({
        "source_pretrain_steps": source_pretrain_steps,
        "source_joint_steps": source_joint_steps,
        "source_observation_supervision": source_observation_supervision,
        "target_steps": target_steps,
        "budget_note": (
            "design: target-only 仅有目标更新" if group == "target_only" else
            "design: joint 组每轮并行加入一次源损失，不含独立源预训练" if group == "gan_joint_no_align" else
            "matched: source-pretrain -> transition-only load -> target-train"
        ),
    })
    return metrics


def build_experiment_schedule(target_train_counts, smoke: bool) -> list[int]:
    """正式与 smoke 均覆盖配置中的全部少样本档位；smoke 仅缩短每档训练。"""
    counts = [int(count) for count in target_train_counts]
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("target_train_counts 必须是正整数列表")
    return list(dict.fromkeys(counts))


def persist_metrics(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(entries), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _json_safe(value):
    """将 numpy 标量和非有限数递归转为 JSON 合法值；NaN/Inf 一律持久化为 null。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _exact_signflip_pvalue(deltas: np.ndarray) -> float:
    """轨迹级配对差值的双侧精确 sign-flip p 值；不把 seed 当独立样本。"""
    if len(deltas) == 0:
        return float("nan")
    signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * len(deltas)))).T.reshape(-1, len(deltas))
    null_means = (signs * deltas.reshape(1, -1)).mean(axis=1)
    return float(np.mean(np.abs(null_means) >= abs(float(deltas.mean())) - 1e-12))


def _trajectory_map(entry: dict, reasons: list[str]) -> dict[str, dict]:
    metrics = entry.get("trajectory_metrics")
    if not isinstance(metrics, list) or not metrics:
        _append_reason(reasons, "missing_trajectory_metrics")
        return {}
    mapped: dict[str, dict] = {}
    for item in metrics:
        trajectory_id = str(item.get("trajectory_id", ""))
        if not trajectory_id or trajectory_id in mapped:
            _append_reason(reasons, "invalid_or_duplicate_trajectory_metric")
            continue
        if not np.isfinite(float(item.get("channel_multistep_nrmse", float("nan")))):
            _append_reason(reasons, "nonfinite_channel_multistep_nrmse")
        if int(item.get("prediction_steps", -1)) != _EVALUATION_HORIZON_STEPS:
            _append_reason(reasons, "requires_fixed_6step_trajectory_metrics")
        channels = item.get("channel_6step_nrmse_by_metric", {})
        if (not isinstance(channels, dict) or set(channels) != set(_TARGET_CHANNEL_NAMES)
                or not all(np.isfinite(float(channels[name])) for name in _TARGET_CHANNEL_NAMES)):
            _append_reason(reasons, "missing_or_nonfinite_four_channel_6step_metrics")
        else:
            expected_mean = channel_6step_nrmse_arithmetic_mean(channels)
            if not np.isclose(float(item.get("channel_6step_nrmse_mean", float("nan"))), expected_mean,
                              rtol=1e-7, atol=1e-9):
                _append_reason(reasons, "channel_6step_arithmetic_mean_mismatch")
        mapped[trajectory_id] = item
    return mapped


def summarize_paired_acceptance(entries: list[dict], *, primary_counts: list[int], min_delta_nrmse: float,
                                expected_seeds=range(42, 52), n_bootstrap: int = 10_000,
                                bootstrap_seed: int = 20260728) -> dict[str, dict]:
    """按同一测试轨迹配对 bootstrap 的预注册验收，拒绝任何运行集合或测试集错配。"""
    expected = tuple(int(seed) for seed in expected_seeds)
    if len(set(expected)) != len(expected):
        raise ValueError("expected_seeds 不得重复")
    primary = {int(count) for count in primary_counts}
    summary: dict[str, dict] = {}
    counts = sorted({int(entry["target_train_count"]) for entry in entries})
    for count in counts:
        tier = "primary" if count in primary else "supportive_exploratory"
        reasons: list[str] = []
        group_entries = {group: [entry for entry in entries if int(entry["target_train_count"]) == count
                                 and entry.get("group") == group]
                         for group in ("target_only", "gan_transition_init")}
        by_group: dict[str, dict[int, dict]] = {}
        for group, items in group_entries.items():
            mapped: dict[int, dict] = {}
            seen: set[int] = set()
            for entry in items:
                seed = int(entry.get("seed", -1))
                if seed in seen:
                    _append_reason(reasons, "duplicate_seed")
                seen.add(seed)
                mapped.setdefault(seed, entry)
                if not np.isfinite(float(entry.get("rmse", float("nan")))):
                    _append_reason(reasons, "nonfinite_rmse")
            if set(mapped) - set(expected):
                _append_reason(reasons, "extra_seed")
            if set(expected) - set(mapped):
                _append_reason(reasons, "missing_seed")
            by_group[group] = mapped
        pairs = [(by_group["target_only"][seed], by_group["gan_transition_init"][seed])
                 for seed in expected if seed in by_group["target_only"] and seed in by_group["gan_transition_init"]]
        if len(pairs) != len(expected):
            _append_reason(reasons, "requires_10_paired_seeds")
            _append_reason(reasons, "requires_exact_expected_seeds")
        trajectory_deltas: dict[str, list[float]] = {}
        seed_deltas: list[float] = []
        for target, init in pairs:
            if (int(target.get("evaluation_horizon_steps", -1)) != _EVALUATION_HORIZON_STEPS
                    or int(init.get("evaluation_horizon_steps", -1)) != _EVALUATION_HORIZON_STEPS):
                _append_reason(reasons, "requires_fixed_6step_evaluation")
            if (target.get("metric_scope") != "failed_rows" or init.get("metric_scope") != "failed_rows"
                    or int(target.get("n_failed_rows", 0)) <= 0 or int(init.get("n_failed_rows", 0)) <= 0):
                _append_reason(reasons, "failed_rows")
                _append_reason(reasons, "formal_metrics_require_failed_rows_with_n_failed_rows_gt_0")
            if (target.get("test_fingerprint") != init.get("test_fingerprint")):
                _append_reason(reasons, "test_fingerprint_mismatch")
            if (int(target.get("n_test_rows", -1)) != int(init.get("n_test_rows", -2))
                    or int(target.get("n_failed_rows", -1)) != int(init.get("n_failed_rows", -2))):
                _append_reason(reasons, "test_sample_or_failed_count_mismatch")
            target_metrics, init_metrics = _trajectory_map(target, reasons), _trajectory_map(init, reasons)
            if set(target_metrics) != set(init_metrics):
                _append_reason(reasons, "test_trajectory_set_mismatch")
                continue
            per_seed: list[float] = []
            for trajectory_id in sorted(target_metrics):
                target_item, init_item = target_metrics[trajectory_id], init_metrics[trajectory_id]
                if (target_item.get("test_trajectory_fingerprint") != init_item.get("test_trajectory_fingerprint")
                        or int(target_item.get("n_test_rows", -1)) != int(init_item.get("n_test_rows", -2))
                        or int(target_item.get("n_failed_rows", -1)) != int(init_item.get("n_failed_rows", -2))):
                    _append_reason(reasons, "test_trajectory_fingerprint_or_size_mismatch")
                    continue
                delta = float(target_item["channel_6step_nrmse_mean"]) - float(init_item["channel_6step_nrmse_mean"])
                if not np.isfinite(delta):
                    _append_reason(reasons, "nonfinite_trajectory_delta")
                    continue
                trajectory_deltas.setdefault(trajectory_id, []).append(delta)
                per_seed.append(delta)
            if per_seed:
                seed_deltas.append(float(np.mean(per_seed)))
        trajectory_means = np.asarray([np.mean(values) for _, values in sorted(trajectory_deltas.items())], dtype=float)
        valid = (len(pairs) == len(expected) and len(seed_deltas) == len(expected)
                 and len(trajectory_means) > 0 and not reasons)
        stats = {
            "target_train_count": count, "acceptance_tier": tier, "formal_valid": valid,
            "formal_primary_metric": "channel_6step_nrmse_arithmetic_mean",
            "expected_seeds": list(expected), "paired_seed_count": len(pairs),
            "bootstrap_unit": "test_trajectory", "n_bootstrap_trajectories": int(len(trajectory_means)),
            "bootstrap_samples": n_bootstrap, "bootstrap_seed": bootstrap_seed + count,
            "primary_min_delta_nrmse": float(min_delta_nrmse), "failure_reasons": reasons,
        }
        if len(trajectory_means) > 0:
            rng = np.random.default_rng(bootstrap_seed + count)
            boot = rng.choice(trajectory_means, size=(n_bootstrap, len(trajectory_means)), replace=True).mean(axis=1)
            stats.update(mean_delta=float(trajectory_means.mean()), ci95_lo=float(np.percentile(boot, 2.5)),
                         ci95_hi=float(np.percentile(boot, 97.5)), p_value_raw=_exact_signflip_pvalue(trajectory_means),
                         positive_seed_count=int(np.sum(np.asarray(seed_deltas) > 0.0)))
        else:
            stats.update(mean_delta=float("nan"), ci95_lo=float("nan"), ci95_hi=float("nan"),
                         p_value_raw=None, positive_seed_count=0)
        if not valid:
            stats.update(acceptance_status="invalid", acceptance_pass=False, holm_adjusted_p=None)
        elif tier != "primary":
            stats.update(acceptance_status="supportive_exploratory", acceptance_pass=None, holm_adjusted_p=None)
        else:
            if stats["mean_delta"] < min_delta_nrmse:
                _append_reason(reasons, "mean_delta_below_preregistered_threshold")
            if stats["ci95_lo"] <= 0.0:
                _append_reason(reasons, "bootstrap_ci95_lower_not_above_zero")
            if stats["positive_seed_count"] < 8:
                _append_reason(reasons, "positive_seed_count_below_8_of_10")
            accepted = not reasons
            stats.update(acceptance_status="accepted" if accepted else "rejected", acceptance_pass=accepted,
                         holm_adjusted_p=None)
        summary[str(count)] = stats
    holm_items = [item for item in summary.values()
                  if item["formal_valid"] and item["p_value_raw"] is not None
                  and np.isfinite(float(item["p_value_raw"]))]
    ordered_holm = sorted(holm_items, key=lambda value: value["p_value_raw"])
    for rank, item in enumerate(ordered_holm):
        adjusted = min(1.0, (len(ordered_holm) - rank) * float(item["p_value_raw"]))
        if rank:
            adjusted = max(adjusted, float(ordered_holm[rank - 1]["holm_adjusted_p"]))
        item["holm_adjusted_p"] = adjusted
        if item["acceptance_tier"] == "supportive_exploratory":
            item["acceptance_tier"] = "supportive_holm_exploratory"
            item["acceptance_status"] = "supportive_holm_exploratory"
    return summary


def summarize_full_data_noninferiority(entries: list[dict], *, max_allowed_increase_nrmse: float,
                                       expected_seeds=range(42, 52), n_bootstrap: int = 10_000,
                                       bootstrap_seed: int = 20260728) -> dict:
    """完整 IID 目标数据的独立非劣汇总，不修改或覆盖少样本 IID 主验收。"""
    expected = tuple(int(seed) for seed in expected_seeds)
    result = {
        "scope": "iid_full_data_noninferiority", "candidate_group": "gan_transition_init",
        "reference_group": "target_only", "max_allowed_increase_nrmse": float(max_allowed_increase_nrmse),
        "expected_seeds": list(expected), "bootstrap_unit": "test_trajectory", "status": "invalid",
        "failure_reasons": [], "mean_candidate_minus_target": None, "ci95_lo": None, "ci95_hi": None,
        "paired_seed_count": 0, "n_bootstrap_trajectories": 0,
    }
    if not expected or len(set(expected)) != len(expected):
        result["failure_reasons"].append("invalid_expected_seeds")
        return result
    indexed: dict[tuple[str, int], dict] = {}
    for entry in entries:
        group = str(entry.get("group", ""))
        if group not in {"target_only", "gan_transition_init"}:
            continue
        if str(entry.get("evaluation_scope", "")) != "iid_full_data":
            continue
        key = (group, int(entry.get("seed", -1)))
        if key in indexed:
            result["failure_reasons"].append("duplicate_seed")
        indexed.setdefault(key, entry)
    trajectory_deltas: dict[str, list[float]] = {}
    for seed in expected:
        target, candidate = indexed.get(("target_only", seed)), indexed.get(("gan_transition_init", seed))
        if target is None or candidate is None:
            result["failure_reasons"].append("missing_paired_seed")
            continue
        if target.get("test_fingerprint") != candidate.get("test_fingerprint"):
            result["failure_reasons"].append("test_fingerprint_mismatch")
            continue
        target_map = {str(item.get("trajectory_id")): item for item in target.get("trajectory_metrics", [])}
        candidate_map = {str(item.get("trajectory_id")): item for item in candidate.get("trajectory_metrics", [])}
        if not target_map or set(target_map) != set(candidate_map):
            result["failure_reasons"].append("test_trajectory_set_mismatch")
            continue
        seed_valid = True
        for trajectory in sorted(target_map):
            if target_map[trajectory].get("test_trajectory_fingerprint") != candidate_map[trajectory].get("test_trajectory_fingerprint"):
                seed_valid = False
                break
            try:
                delta = float(candidate_map[trajectory]["channel_6step_nrmse_mean"]
                              - target_map[trajectory]["channel_6step_nrmse_mean"])
            except (KeyError, TypeError):
                seed_valid = False
                break
            if not np.isfinite(delta):
                seed_valid = False
                break
            trajectory_deltas.setdefault(trajectory, []).append(delta)
        if not seed_valid:
            result["failure_reasons"].append("missing_or_nonfinite_trajectory_metric")
            continue
        result["paired_seed_count"] += 1
    if result["paired_seed_count"] != len(expected):
        return result
    deltas = np.asarray([np.mean(values) for _, values in sorted(trajectory_deltas.items())], dtype=float)
    if not len(deltas):
        result["failure_reasons"].append("no_paired_trajectory_metrics")
        return result
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = rng.choice(deltas, size=(n_bootstrap, len(deltas)), replace=True).mean(axis=1)
    result.update(
        mean_candidate_minus_target=float(deltas.mean()), ci95_lo=float(np.percentile(bootstrap, 2.5)),
        ci95_hi=float(np.percentile(bootstrap, 97.5)), n_bootstrap_trajectories=int(len(deltas)),
    )
    result["status"] = "accepted" if result["ci95_hi"] <= result["max_allowed_increase_nrmse"] else "rejected"
    return result


def load_metrics(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload["metrics"] if isinstance(payload, dict) else payload
    # 已有运行已持久化四通道逐轨迹字段时，可无训练复算新的唯一正式主指标。
    for entry in entries:
        trajectory_means: list[float] = []
        for item in entry.get("trajectory_metrics", []):
            channels = item.get("channel_6step_nrmse_by_metric")
            if isinstance(channels, dict) and set(channels) == set(_TARGET_CHANNEL_NAMES):
                try:
                    item["channel_6step_nrmse_mean"] = channel_6step_nrmse_arithmetic_mean(channels)
                except ValueError:
                    continue
                trajectory_means.append(item["channel_6step_nrmse_mean"])
        if trajectory_means:
            entry["channel_6step_nrmse_mean"] = float(np.mean(trajectory_means))
    return entries


def persist_results_with_acceptance(path: Path, entries: list[dict], acceptance: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe({"metrics": entries, "acceptance": acceptance})
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_results(path: Path, results: dict[str, list[dict]], smoke: bool,
                   acceptance: dict[str, dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["# GaN RFALT 损伤状态迁移" + ("（smoke）" if smoke else ""), "", "固定评估口径：每条测试轨迹从首个可观测状态连续 rollout **6 步**；唯一正式通道主指标为 gain/phase/Pout/PAE 四个六步 nRMSE 的**算术平均**。", "", "| 目标标注轨迹数 | 组别 | 正式通道主指标：四通道六步 nRMSE 算术平均 | 四通道六步 nRMSE（gain / phase / Pout / PAE） | 阵列 M_link 投影 MAE (dB，非正式) | 服务事件 MAE (s) | 删失下界违规率 | 失效测试行 | 运行数 |", "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for group, entries in results.items():
        for count in sorted({int(entry.get("target_train_count", 0)) for entry in entries}):
            subset = [entry for entry in entries if int(entry.get("target_train_count", 0)) == count]
            channel = [entry.get("channel_6step_nrmse_mean", float("nan")) for entry in subset]
            link = [entry.get("array_link_margin_mae_dB") for entry in subset]
            service = [entry.get("service_event_mae_s") for entry in subset]
            censor = [entry.get("service_censor_lower_bound_violation_rate") for entry in subset]
            failed = [entry.get("n_failed_rows", 0) for entry in subset]
            link_mean = float(np.mean([value for value in link if value is not None])) if any(value is not None for value in link) else float("nan")
            service_mean = float(np.mean([value for value in service if value is not None])) if any(value is not None for value in service) else float("nan")
            censor_mean = float(np.mean([value for value in censor if value is not None])) if any(value is not None for value in censor) else float("nan")
            channel_detail = {name: float(np.mean([entry.get("channel_6step_nrmse_by_metric", {}).get(name, float("nan")) for entry in subset]))
                              for name in _TARGET_CHANNEL_NAMES}
            detail_text = " / ".join(f"{channel_detail[name]:.6f}" for name in _TARGET_CHANNEL_NAMES)
            rows.append(f"| {count} | {group} | {np.mean(channel):.6f} | {detail_text} | {link_mean:.6f} | {service_mean:.2f} | {censor_mean:.6f} | {np.mean(failed):.0f} | {len(subset)} |")
    all_entries = [entry for entries in results.values() for entry in entries]
    if all_entries and not all("evaluation_horizon_steps" in entry for entry in all_entries):
        rows.extend(["", "> **历史产物无效：** 这些运行未持久化固定六步的逐轨迹评分字段，不能用于当前正式验收。"])
    zero_failed = all_entries and all(int(entry.get("n_failed_rows", 0)) == 0 for entry in all_entries)
    caveat = ("本次 smoke 的 n_failed_rows=0：服务 RUL/失效指标不可解释，RMSE 回退为全测试行管线连通性数值。"
              if zero_failed else "服务 RUL 指标仅在失效测试行上计算；删失行仍按下界约束单独解释。")
    rows.extend(["", "预算说明：init 与随机源对照均执行相同的“源预训练→仅加载 Fθ→目标训练”步骤；target-only 与联合组的设计性预算差异在每条运行记录的 `budget_note` 中保留。", f"说明：{caveat}"])
    if acceptance:
        rows.extend(["", "## 预注册配对验收：Target-only − GaN transition-init", "", "主指标为同一测试轨迹上的通道**六步四指标算术平均 nRMSE**差值（Δ = target-only − transition-init）；95% CI 对测试轨迹重采样，seed 仅用于重复训练方向统计。", "", "| k | 定位 | 配对 seeds | 测试轨迹 | 平均 Δ | 轨迹 bootstrap 95% CI | 正向 seeds | 原始 p | Holm p（四个 k 统一） | 状态 | 失败原因 |", "| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |"])
        for count in sorted(acceptance, key=int):
            item = acceptance[count]
            ci = f"[{item.get('ci95_lo', float('nan')):.6f}, {item.get('ci95_hi', float('nan')):.6f}]"
            reasons = ", ".join(item.get("failure_reasons", [])) or "—"
            raw_p = item.get("p_value_raw")
            holm_p = item.get("holm_adjusted_p")
            raw_text = f"{raw_p:.6f}" if raw_p is not None else "—"
            holm_text = f"{holm_p:.6f}" if holm_p is not None else "—"
            rows.append(f"| {count} | {item['acceptance_tier']} | {item.get('paired_seed_count', 0)} | {item.get('n_bootstrap_trajectories', 0)} | {item.get('mean_delta', float('nan')):.6f} | {ci} | {item.get('positive_seed_count', 0)} | {raw_text} | {holm_text} | {item['acceptance_status']} | {reasons} |")
        thresholds = sorted({item.get("primary_min_delta_nrmse") for item in acceptance.values()
                             if item.get("primary_min_delta_nrmse") is not None})
        threshold_text = thresholds[0] if len(thresholds) == 1 else "配置值"
        rows.append(f"主验收仅适用于 YAML `primary_counts`，门槛为平均 Δ≥{threshold_text}、95% CI 下界>0、至少 8/10 seed 为正。四个 k 的有效原始 p 值统一作 Holm 校正；无有效 p 值不显示 Holm 结论。")
    rows.append("阵列 M_link 指标当前仅是由通道 Pout 训练集线性投影得到的**非正式探索性桥接**，不是确定性阵列孪生输出；本报告不将其称为完整三级验收。")
    rows.append("本表为自建仿真产物；源、目标 dynamics_id 已在运行前校验不同。" + ("当前 smoke 仅验证四档位数据与训练管线，不得据此宣称正迁移。" if smoke else "正式结论仅以以上 failed_rows 的通道层配对验收表为准。"))
    scopes = {str(entry.get("evaluation_scope", "iid_legacy_no_ood_holdout")) for entry in all_entries}
    if "iid_with_pre_registered_ood_holdout" in scopes:
        ood_count = max(int(entry.get("ood_test_trajectory_count", 0)) for entry in all_entries)
        rows.append(f"OOD 保持集：已从 train/val/IID test 隔离 {ood_count} 条预注册扫描—占空比—温度角点轨迹；本报告仅含 IID 评分，未计算 OOD 性能，不得以 IID 结论替代 OOD 外推结论。")
    else:
        rows.append("OOD 保持集：本报告未启用预注册 OOD 切分，未计算 OOD 性能；不得将当前 IID 结论写成未见任务条件的外推验证。")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 GaN 三状态迁移对比")
    parser.add_argument("--config", default="configs/phased_array_gan.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--summarize-only", action="store_true", help="只读取既有 metrics JSON 并生成验收统计/报告")
    parser.add_argument("--ood-holdout", action="store_true", help="训练前固定组合 OOD test；只训练/选模 IID 轨迹")
    parser.add_argument("--prepare-ood-holdout", action="store_true", help="仅写 OOD 切分 manifest，不训练")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result_name = "results_phased_array_gan_smoke.md" if args.smoke else "results_phased_array_gan.md"
    metrics_name = "results_phased_array_gan_smoke.json" if args.smoke else "results_phased_array_gan.json"
    result_path = ROOT / "docs" / result_name
    metrics_path = ROOT / "docs" / metrics_name
    ood_manifest_path = ROOT / "docs" / "gan_target_ood_split_manifest.json"
    acceptance_options = {
        "primary_counts": [int(count) for count in cfg["transfer"]["primary_counts"]],
        "min_delta_nrmse": float(cfg["transfer"]["primary_min_delta_nrmse"]),
        "expected_seeds": range(int(cfg["seed"]), int(cfg["seed"]) + int(cfg["transfer"]["n_seeds"])),
    }
    if args.summarize_only:
        all_metrics = load_metrics(metrics_path)
        acceptance = summarize_paired_acceptance(all_metrics, **acceptance_options)
        grouped: dict[str, list[dict]] = {}
        for entry in all_metrics:
            grouped.setdefault(entry["group"], []).append(entry)
        _write_results(result_path, grouped, args.smoke, acceptance)
        persist_results_with_acceptance(metrics_path, all_metrics, acceptance)
        return
    if args.prepare_ood_holdout:
        protocol = cfg["target"].get("ood_holdout")
        if not isinstance(protocol, dict):
            raise ValueError("配置缺少 target.ood_holdout，无法准备 OOD 保持集")
        conditions = read_target_ood_conditions(ROOT / cfg["target"]["feature_path"], protocol)
        split = split_target_iid_ood(sorted(conditions), conditions, protocol, seed=int(cfg["seed"]))
        persist_ood_split_manifest(ood_manifest_path, protocol, conditions, split)
        print(f">> 已写入 OOD 组合保持集 manifest（未训练） -> {ood_manifest_path}")
        return
    validate_source_config(cfg["source"])
    groups = validate_experiment_groups(cfg["transfer"].get("groups", GROUPS))
    source, source_id = _source_rows(ROOT / cfg["source"]["feature_path"], 48 if args.smoke else 160)
    target, target_id = _target_rows(ROOT / cfg["target"]["feature_path"], 48 if args.smoke else 256)
    validate_domain_separation(source_id, target_id)
    time_scale_s = float(cfg["transfer"]["transition_time_scale_s"])
    source = DomainRows(source.x, source.states, source.obs_labels, source.rul, source.event, source.ids, source.time, time_scale_s)
    target = DomainRows(target.x, target.states, target.obs_labels, target.rul, target.event, target.ids, target.time,
                        time_scale_s, target.array_link_margin)
    event_by_id = target_event_by_trajectory(target)
    ood_manifest = None
    if args.ood_holdout:
        protocol = cfg["target"].get("ood_holdout")
        if not isinstance(protocol, dict):
            raise ValueError("配置缺少 target.ood_holdout，无法启用 OOD 保持集")
        conditions = read_target_ood_conditions(ROOT / cfg["target"]["feature_path"], protocol)
        ood_split = split_target_iid_ood(sorted(np.unique(target.ids).tolist()), conditions, protocol,
                                         seed=int(cfg["seed"]), event_by_id=event_by_id)
        ood_manifest = persist_ood_split_manifest(ood_manifest_path, protocol, conditions, ood_split)
        tr_pool, val_ids, te_ids = list(ood_split.train_ids), list(ood_split.val_ids), list(ood_split.iid_test_ids)
    else:
        tr_pool, val_ids, te_ids = split_target_trajectories(
            sorted(np.unique(target.ids).tolist()), cfg["seed"], event_by_id=event_by_id)
    validate_test_event_coverage(event_by_id, te_ids)
    source_scaler = TrainOnlyStandardizer().fit(source.x)
    source = DomainRows(source_scaler.transform(source.x), source.states, source.obs_labels, source.rul, source.event, source.ids, source.time, source.transition_time_scale_s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_seeds = args.n_seeds or (3 if args.smoke else int(cfg["transfer"]["n_seeds"]))
    epochs = 2 if args.smoke else 20
    patience = int(cfg["transfer"].get("early_stop_patience", 5))
    schedule = build_experiment_schedule(cfg["transfer"]["target_train_counts"], args.smoke)
    results = {group: [] for group in groups}
    all_metrics: list[dict] = []
    for count in schedule:
        if count > len(tr_pool):
            raise ValueError(f"target_train_count={count} 超过可用训练轨迹 {len(tr_pool)}；请增加 --n_traj")
        # 每个 count 独立选择轨迹、fit scaler 和 RUL 尺度，禁止跨档位共享统计量。
        train = _select_rows(target, tr_pool[:count]); val = _select_rows(target, val_ids); test = _select_rows(target, te_ids)
        channel_scale = np.std(train.obs_labels, axis=0).astype(np.float32) + 1e-6
        link_margin_bridge = fit_link_margin_bridge(train)
        target_scaler = TrainOnlyStandardizer().fit(train.x)
        train = DomainRows(target_scaler.transform(train.x), train.states, train.obs_labels, train.rul, train.event, train.ids, train.time, train.transition_time_scale_s, train.array_link_margin)
        val = DomainRows(target_scaler.transform(val.x), val.states, val.obs_labels, val.rul, val.event, val.ids, val.time, val.transition_time_scale_s, val.array_link_margin)
        test = DomainRows(target_scaler.transform(test.x), test.states, test.obs_labels, test.rul, test.event, test.ids, test.time, test.transition_time_scale_s, test.array_link_margin)
        train, val, test, _rul_scale = normalize_target_rul(train, val, test)
        for seed in range(cfg["seed"], cfg["seed"] + n_seeds):
            for group in groups:
                metric = run_group(group, source, train, val, test, seed=seed, epochs=epochs, patience=patience, device=device,
                                   channel_scale=channel_scale, rul_scale_s=_rul_scale, link_margin_bridge=link_margin_bridge)
                metric.update({
                    "seed": seed, "target_train_count": count, "rul_scale_train_only": _rul_scale,
                    "service_rul_interpretable": bool(metric["n_failed_rows"] > 0),
                    "metric_scope": "failed_rows" if metric["n_failed_rows"] > 0 else "all_rows_pipeline_connectivity_only",
                    "evaluation_scope": ("iid_with_pre_registered_ood_holdout" if args.ood_holdout
                                         else "iid_legacy_no_ood_holdout"),
                    "ood_manifest_fingerprint": (ood_manifest["ood_test_fingerprint"] if ood_manifest else None),
                    "ood_test_trajectory_count": (len(ood_manifest["ood_test_ids"]) if ood_manifest else 0),
                })
                results[group].append(metric); all_metrics.append({"group": group, **metric})
                print(f">> count={count} {group} seed={seed} rmse={metric['rmse']:.6f} "
                      f"source_pretrain_steps={metric['source_pretrain_steps']} target_steps={metric['target_steps']}")
    acceptance = summarize_paired_acceptance(all_metrics, **acceptance_options)
    _write_results(result_path, results, args.smoke, acceptance)
    persist_results_with_acceptance(metrics_path, all_metrics, acceptance)


if __name__ == "__main__":
    main()
