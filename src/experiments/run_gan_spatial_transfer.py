"""GaN RFALT→LEO 子阵空间迁移的独立最小 smoke 训练入口。

本入口不修改既有 ``run_gan_transfer`` 的通道实验或其结果。它仅用于验证：
源域预训练的共享损伤转移能加载到子阵节点模型；节点六步 rollout 可经无学习参数
阵列孪生计算四项阵列评分。``--smoke`` 输出明确不是正式统计验收。
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

from src.experiments.run_gan_transfer import (
    DomainRows,
    SpatialDomainRows,
    TargetOODSplit,
    _source_rows,
    _spatial_target_rows,
    pretrain_source_transition,
    read_target_ood_conditions,
    scramble_source_state_pairs,
    validate_ood_split_manifest,
    validate_domain_separation,
    validate_source_config,
)
from src.sim.subarray_array_twin import evaluate_subarray_array
from src.transfer.damage_state import DamageStateModel
from src.utils import load_config, set_seed


ROOT = Path(__file__).resolve().parents[2]
_ARRAY_METRICS = ("EIRP_norm", "SLL_dB", "theta_err_deg", "M_link_dB")
_ROLLOUT_STEPS = 6
_MIN_ROLLOUT_OBSERVATIONS = _ROLLOUT_STEPS + 1
_FEATURE_REBUILD_COMMAND = "python -m src.sim.build_array_hi --config configs/phased_array_gan.yaml"
SPATIAL_GROUPS = ("target_only", "gan_transition_init_spatial", "random_source_control_spatial")
_CHANNEL_METRICS = ("gain_dB", "phase_deg", "Pout_dBm", "PAE")
_FORMAL_SEED_COUNT = 10
_EXACT_SIGNFLIP_MAX_N = 16
_MONTE_CARLO_SIGNFLIP_SAMPLES = 100_000


@dataclass(frozen=True)
class SpatialOODHoldout:
    """空间入口已核验的预注册 OOD 切分；不得由运行期重新抽样。"""

    split: TargetOODSplit
    manifest: dict
    manifest_fingerprint: str

    @property
    def iid_ids(self) -> tuple[str, ...]:
        return self.split.train_ids + self.split.val_ids + self.split.iid_test_ids


def load_spatial_ood_holdout(manifest_path: Path, protocol: dict, feature_path: Path,
                             rows: SpatialDomainRows) -> SpatialOODHoldout:
    """核验 manifest、YAML OOD 协议、HDF5 条件及空间行轨迹集合的四重一致性。"""
    try:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 OOD split manifest: {manifest_path}") from exc
    if not isinstance(saved, dict):
        raise ValueError("OOD split manifest 必须是 JSON object")
    try:
        split = TargetOODSplit(
            tuple(str(value) for value in saved["train_ids"]),
            tuple(str(value) for value in saved["val_ids"]),
            tuple(str(value) for value in saved["iid_test_ids"]),
            tuple(str(value) for value in saved["ood_test_ids"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("OOD split manifest 缺少 IID/OOD 轨迹切分") from exc
    conditions = read_target_ood_conditions(feature_path, protocol)
    validate_ood_split_manifest(saved, protocol, conditions, split)
    manifest_ids = {str(value) for value in saved.get("all_trajectory_ids", [])}
    spatial_ids = {str(value) for value in np.unique(rows.ids).tolist()}
    if spatial_ids != manifest_ids:
        raise ValueError("空间轨迹集合与 OOD split manifest 不一致")
    return SpatialOODHoldout(split=split, manifest=saved, manifest_fingerprint=_fingerprint(saved))


class NodeTrainOnlyStandardizer:
    """只在训练轨迹的全部子阵节点上拟合逐特征尺度。"""

    def fit(self, x_nodes: np.ndarray) -> "NodeTrainOnlyStandardizer":
        values = np.asarray(x_nodes, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError("x_nodes 必须为 (T, N_subarray, F)")
        self.mean_ = values.mean(axis=(0, 1))
        self.scale_ = values.std(axis=(0, 1)) + 1e-6
        return self

    def transform(self, x_nodes: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("NodeTrainOnlyStandardizer 必须先在训练轨迹 fit")
        return ((np.asarray(x_nodes, dtype=np.float32) - self.mean_) / self.scale_).astype(np.float32)


def _select_spatial_rows(rows: SpatialDomainRows, ids: list[str]) -> SpatialDomainRows:
    keep = np.isin(rows.ids, ids)
    if not keep.any():
        raise ValueError("空间轨迹划分为空")
    return SpatialDomainRows(
        x_global=rows.x_global[keep], x_nodes=rows.x_nodes[keep], node_states=rows.node_states[keep],
        node_labels=rows.node_labels[keep], array_score_truth={key: values[keep] for key, values in rows.array_score_truth.items()},
        rul=rows.rul[keep], event=rows.event[keep], ids=rows.ids[keep], time=rows.time[keep],
    )


def split_spatial_rows(rows: SpatialDomainRows, seed: int) -> tuple[SpatialDomainRows, SpatialDomainRows, SpatialDomainRows]:
    """按完整轨迹而非窗口切分，至少为 train/val/test 各保留一条。"""
    ids = np.unique(rows.ids)
    if len(ids) < 3:
        raise ValueError("空间训练至少需要三条完整轨迹")
    shuffled = ids[np.random.default_rng(seed).permutation(len(ids))]
    n_train = min(max(1, int(round(0.6 * len(ids)))), len(ids) - 2)
    n_val = min(max(1, int(round(0.2 * len(ids)))), len(ids) - n_train - 1)
    return (
        _select_spatial_rows(rows, shuffled[:n_train].tolist()),
        _select_spatial_rows(rows, shuffled[n_train:n_train + n_val].tolist()),
        _select_spatial_rows(rows, shuffled[n_train + n_val:].tolist()),
    )


def _standardize_spatial(train: SpatialDomainRows, *others: SpatialDomainRows):
    scaler = NodeTrainOnlyStandardizer().fit(train.x_nodes)

    def transform(rows: SpatialDomainRows) -> SpatialDomainRows:
        return SpatialDomainRows(
            x_global=rows.x_global, x_nodes=scaler.transform(rows.x_nodes), node_states=rows.node_states,
            node_labels=rows.node_labels, array_score_truth=rows.array_score_truth, rul=rows.rul,
            event=rows.event, ids=rows.ids, time=rows.time,
        )

    return (transform(train), *(transform(rows) for rows in others), scaler)


def _spatial_node_pairs(rows: SpatialDomainRows, time_scale_s: float) -> dict[str, np.ndarray]:
    if time_scale_s <= 0:
        raise ValueError("transition_time_scale_s 必须为正")
    starts: list[int] = []; ends: list[int] = []
    for trajectory in np.unique(rows.ids):
        index = np.flatnonzero(rows.ids == trajectory)
        if len(index) < 2 or np.any(np.diff(rows.time[index]) <= 0):
            raise ValueError(f"空间轨迹 {trajectory} 必须至少两点且 time 严格递增")
        starts.extend(index[:-1].tolist()); ends.extend(index[1:].tolist())
    start, end = np.asarray(starts, dtype=int), np.asarray(ends, dtype=int)
    return {
        "x_t": rows.x_nodes[start], "state_t": rows.node_states[start], "state_next": rows.node_states[end],
        "channel_next": rows.node_labels[end],
        "dt": ((rows.time[end] - rows.time[start]) / time_scale_s).reshape(-1, 1, 1).astype(np.float32),
    }


def spatial_node_one_step_loss(model: DamageStateModel, rows: SpatialDomainRows, *, time_scale_s: float,
                               device: torch.device) -> torch.Tensor:
    """节点状态/通道一步监督；score-only 阵列真值绝不参与训练图。"""
    pairs = _spatial_node_pairs(rows, time_scale_s)
    x_t = torch.as_tensor(pairs["x_t"], dtype=torch.float32, device=device)
    state_t = torch.as_tensor(pairs["state_t"], dtype=torch.float32, device=device)
    state_next = torch.as_tensor(pairs["state_next"], dtype=torch.float32, device=device)
    channel_next = torch.as_tensor(pairs["channel_next"], dtype=torch.float32, device=device)
    dt = torch.as_tensor(pairs["dt"], dtype=torch.float32, device=device).expand(-1, x_t.shape[1], -1)
    state_hat_t = model.encode_target_nodes(x_t)
    predicted_next = model.transition(state_hat_t, model.target_node_stress(x_t), dt)
    return (
        nn.functional.mse_loss(state_hat_t, state_t)
        + nn.functional.mse_loss(predicted_next, state_next)
        + nn.functional.mse_loss(model.observe_target_nodes(predicted_next), channel_next)
    )


def validate_spatial_feature_schema(feature_path: Path) -> None:
    """在任何训练/评估前拒绝未重建的旧 target feature HDF5。"""
    required = {"scan_az_deg", "margin0_dB", "array_grid", "element_spacing_lambda", "subarray_block"}
    with h5py.File(feature_path, "r") as h5:
        if not h5.keys():
            raise ValueError(f"空间 target feature 为空；请执行：{_FEATURE_REBUILD_COMMAND}")
        for trajectory in h5.keys():
            group = h5[trajectory]
            if group.attrs.get("array_twin_metadata_schema", "") != "subarray_array_twin_v1":
                raise ValueError(f"空间 target {trajectory} 缺少 subarray_array_twin_v1 元数据；请执行：{_FEATURE_REBUILD_COMMAND}")
            if group.attrs.get("array_twin_metadata_access", "") != "twin_only_not_model_input":
                raise ValueError(f"空间 target {trajectory} 的阵列元数据访问标记非法；请执行：{_FEATURE_REBUILD_COMMAND}")
            missing = required - set(group.attrs)
            if missing:
                raise ValueError(f"空间 target {trajectory} 缺少阵列元数据: {', '.join(sorted(missing))}；请执行：{_FEATURE_REBUILD_COMMAND}")


def _load_array_twin_metadata(feature_path: Path, ids: np.ndarray) -> dict[str, dict[str, object]]:
    required = {"scan_az_deg", "margin0_dB", "array_grid", "element_spacing_lambda", "subarray_block"}
    validate_spatial_feature_schema(feature_path)
    metadata: dict[str, dict[str, object]] = {}
    with h5py.File(feature_path, "r") as h5:
        for trajectory in np.unique(ids):
            group = h5[str(trajectory)]
            missing = required - set(group.attrs)
            if missing:
                raise ValueError(f"空间 target {trajectory} 缺少阵列元数据: {', '.join(sorted(missing))}")
            metadata[str(trajectory)] = {
                "scan_az_deg": float(group.attrs["scan_az_deg"]), "margin0_dB": float(group.attrs["margin0_dB"]),
                "array_grid": tuple(int(value) for value in group.attrs["array_grid"]),
                "element_spacing_lambda": float(group.attrs["element_spacing_lambda"]),
                "subarray_block": int(group.attrs["subarray_block"]),
            }
    return metadata


def _validate_rollout_trajectory_lengths(rows: SpatialDomainRows) -> None:
    short = [str(trajectory) for trajectory in np.unique(rows.ids)
             if len(np.flatnonzero(rows.ids == trajectory)) < _MIN_ROLLOUT_OBSERVATIONS]
    if short:
        raise ValueError(f"阵列六步 rollout 的每条测试轨迹至少 {_MIN_ROLLOUT_OBSERVATIONS} 个观察点；不足轨迹: {', '.join(short)}")


def _rollout_channels(model: DamageStateModel, rows: SpatialDomainRows, index: np.ndarray, *, time_scale_s: float,
                      device: torch.device) -> np.ndarray:
    """只由首时刻节点遥测和后续遥测应力连续 rollout，不读通道/阵列标签。"""
    if len(index) < _MIN_ROLLOUT_OBSERVATIONS:
        raise ValueError(f"阵列六步 rollout 的每条测试轨迹至少 {_MIN_ROLLOUT_OBSERVATIONS} 个观察点（基线+{_ROLLOUT_STEPS} 步）")
    x = torch.as_tensor(rows.x_nodes[index], dtype=torch.float32, device=device)
    state = model.encode_target_nodes(x[:1])
    predicted = [model.observe_target_nodes(state).squeeze(0)]
    for position in range(_ROLLOUT_STEPS):
        dt_value = float((rows.time[index[position + 1]] - rows.time[index[position]]) / time_scale_s)
        dt = torch.full((1, x.shape[1], 1), dt_value, dtype=torch.float32, device=device)
        state = model.transition(state, model.target_node_stress(x[position:position + 1]), dt)
        predicted.append(model.observe_target_nodes(state).squeeze(0))
    return torch.stack(predicted).detach().cpu().numpy()


def array_rollout_errors(model: DamageStateModel, rows: SpatialDomainRows, metadata: dict[str, dict[str, object]], *,
                         time_scale_s: float, device: torch.device) -> dict[str, float]:
    """六步节点 rollout → 确定性阵列孪生；score-only 真值仅用于末端误差。"""
    squared: dict[str, list[np.ndarray]] = {name: [] for name in _ARRAY_METRICS}
    _validate_rollout_trajectory_lengths(rows)
    trajectory_indices = {str(trajectory): np.flatnonzero(rows.ids == trajectory) for trajectory in np.unique(rows.ids)}
    model.eval()
    with torch.no_grad():
        for trajectory, index in trajectory_indices.items():
            predictions = evaluate_subarray_array(_rollout_channels(model, rows, index, time_scale_s=time_scale_s, device=device),
                                                   metadata[trajectory])
            for name in _ARRAY_METRICS:
                residual = predictions[name][1:] - rows.array_score_truth[name][index[1:_MIN_ROLLOUT_OBSERVATIONS]]
                squared[name].append(np.square(residual))
    if not all(squared.values()):
        raise ValueError("测试轨迹不足以执行六步阵列 rollout")
    return {f"{name}_rmse": float(np.sqrt(np.concatenate(values).mean())) for name, values in squared.items()}


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _spatial_test_fingerprint(rows: SpatialDomainRows) -> str:
    """绑定同一测试轨迹、时间轴和 score-only 真值，防止组间偷换评分集。"""
    return _fingerprint({
        "ids": rows.ids.tolist(), "time": np.asarray(rows.time, float).round(6).tolist(),
        "event": rows.event.astype(int).tolist(), "rul": np.asarray(rows.rul, float).round(6).tolist(),
        "node_labels": np.asarray(rows.node_labels, float).round(6).tolist(),
        "array_score_truth": {key: np.asarray(value, float).round(6).tolist()
                              for key, value in sorted(rows.array_score_truth.items())},
    })


def spatial_trajectory_metrics(model: DamageStateModel, rows: SpatialDomainRows, metadata: dict[str, dict[str, object]], *,
                               channel_scale: np.ndarray, array_scale: dict[str, float], time_scale_s: float,
                               device: torch.device) -> list[dict]:
    """逐轨迹保存四通道、四阵列和服务/RUL评分，所有预测均为固定六步 rollout。"""
    _validate_rollout_trajectory_lengths(rows)
    channel_scale = np.asarray(channel_scale, dtype=float)
    if channel_scale.shape != (len(_CHANNEL_METRICS),) or np.any(channel_scale <= 0.0):
        raise ValueError("channel_scale 必须是四个正的训练集尺度")
    if set(array_scale) != set(_ARRAY_METRICS) or any(float(array_scale[name]) <= 0.0 for name in _ARRAY_METRICS):
        raise ValueError("array_scale 必须精确包含四个正的训练集阵列尺度")
    output: list[dict] = []
    model.eval()
    with torch.no_grad():
        for trajectory in sorted(np.unique(rows.ids).tolist()):
            index = np.flatnonzero(rows.ids == trajectory)
            channels = _rollout_channels(model, rows, index, time_scale_s=time_scale_s, device=device)
            twin = evaluate_subarray_array(channels, metadata[trajectory])
            channel_truth = rows.node_labels[index[1:_MIN_ROLLOUT_OBSERVATIONS]]
            channel_rmse = np.sqrt(np.square(channels[1:] - channel_truth).mean(axis=(0, 1)))
            array_rmse = {
                name: float(np.sqrt(np.square(twin[name][1:] - rows.array_score_truth[name][index[1:_MIN_ROLLOUT_OBSERVATIONS]]).mean()))
                for name in _ARRAY_METRICS
            }
            predicted_event = np.flatnonzero(twin["M_link_dB"][1:] <= 0.0)
            predicted_rul = (float(rows.time[index[int(predicted_event[0]) + 1]] - rows.time[index[0]])
                             if len(predicted_event) else None)
            observed = bool(rows.event[index[0]])
            truth_rul = float(rows.rul[index[0]])
            output.append({
                "trajectory_id": trajectory,
                "test_trajectory_fingerprint": _fingerprint({
                    "id": trajectory, "time": np.asarray(rows.time[index], float).round(6).tolist(),
                    "event": rows.event[index].astype(int).tolist(), "rul": np.asarray(rows.rul[index], float).round(6).tolist(),
                    "node_labels": np.asarray(rows.node_labels[index], float).round(6).tolist(),
                    "array_score_truth": {name: np.asarray(rows.array_score_truth[name][index], float).round(6).tolist()
                                          for name in _ARRAY_METRICS},
                }),
                "prediction_steps": _ROLLOUT_STEPS,
                "channel_6step_rmse_by_metric": {name: float(channel_rmse[position]) for position, name in enumerate(_CHANNEL_METRICS)},
                "channel_6step_nrmse_by_metric": {name: float(channel_rmse[position] / channel_scale[position])
                                                    for position, name in enumerate(_CHANNEL_METRICS)},
                "array_6step_rmse_by_metric": array_rmse,
                "array_6step_nrmse_by_metric": {name: float(array_rmse[name] / array_scale[name]) for name in _ARRAY_METRICS},
                "service_rul_s_truth": truth_rul, "service_rul_s_pred": predicted_rul,
                "service_rul_abs_error_s": (abs(predicted_rul - truth_rul) if observed and predicted_rul is not None else None),
                "service_event_observed": observed,
                "service_censor_lower_bound_violation": bool(not observed and predicted_rul is not None and predicted_rul < truth_rul),
            })
    return output


def _exact_signflip_pvalue(deltas: np.ndarray) -> float:
    if len(deltas) == 0:
        return float("nan")
    signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * len(deltas)))).T.reshape(-1, len(deltas))
    null_means = (signs * deltas.reshape(1, -1)).mean(axis=1)
    return float(np.mean(np.abs(null_means) >= abs(float(deltas.mean())) - 1e-12))


def signflip_pvalue(deltas: np.ndarray, *, exact_max_n: int = _EXACT_SIGNFLIP_MAX_N,
                    monte_carlo_samples: int = _MONTE_CARLO_SIGNFLIP_SAMPLES,
                    seed: int = 20260728) -> dict[str, object]:
    """双侧 sign-flip p 值：小样本精确枚举，大样本固定 seed Monte Carlo。"""
    values = np.asarray(deltas, dtype=float).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("sign-flip 差值必须为非空有限数组")
    threshold = min(int(exact_max_n), _EXACT_SIGNFLIP_MAX_N)
    if len(values) <= threshold:
        return {
            "p_value": _exact_signflip_pvalue(values), "p_value_method": "exact_enumeration",
            "p_value_samples": int(2 ** len(values)), "p_value_seed": None,
            "p_value_observed_sign_included": True,
        }
    samples = int(monte_carlo_samples)
    if samples < 2:
        raise ValueError("Monte Carlo sign-flip 至少需要 2 个随机符号样本")
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    # 只抽取 B 个随机符号组合；有限样本 p 值采用唯一的加一校正。
    extreme = 0
    for start in range(0, samples, 4096):
        count = min(4096, samples - start)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8).astype(np.float64) * 2.0 - 1.0
        extreme += int(np.sum(np.abs(signs @ values / len(values)) >= observed - 1e-12))
    return {
        "p_value": float((extreme + 1) / (samples + 1)), "p_value_method": "monte_carlo_signflip",
        "p_value_samples": samples, "p_value_seed": int(seed), "p_value_observed_sign_included": False,
    }


def _holm_adjust(items: list[dict]) -> None:
    ordered = sorted(items, key=lambda item: float(item["p_value_raw"]))
    for rank, item in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * float(item["p_value_raw"]))
        if rank:
            adjusted = max(adjusted, float(ordered[rank - 1]["holm_adjusted_p"]))
        item["holm_adjusted_p"] = adjusted


def summarize_spatial_paired_acceptance(entries: list[dict], *, expected_seeds=range(42, 52), n_bootstrap: int = 10_000,
                                        bootstrap_seed: int = 20260728, min_effect_nrmse: float = 0.02,
                                        non_degradation_min_delta: float = 0.0) -> dict[str, dict]:
    """以同一测试轨迹的 M_link 六步 nRMSE 为主指标，三项阵列指标不恶化为门控。"""
    expected = tuple(int(seed) for seed in expected_seeds)
    reasons: list[str] = []
    if len(expected) != _FORMAL_SEED_COUNT or len(set(expected)) != len(expected):
        raise ValueError("正式空间验收必须提供恰好 10 个不重复 seed")
    by_group: dict[str, dict[int, dict]] = {group: {} for group in SPATIAL_GROUPS}
    for entry in entries:
        group, seed = str(entry.get("group", "")), int(entry.get("seed", -1))
        if group not in by_group:
            continue
        if seed in by_group[group]:
            reasons.append("duplicate_seed")
        by_group[group].setdefault(seed, entry)
        if entry.get("smoke", False) or int(entry.get("evaluation_horizon_steps", -1)) != _ROLLOUT_STEPS:
            reasons.append("requires_non_smoke_fixed_6step_entries")
    for group in SPATIAL_GROUPS:
        if set(by_group[group]) != set(expected):
            reasons.append(f"requires_exact_10_seeds_{group}")
    trajectory_deltas: dict[str, list[float]] = {}
    seed_deltas: list[float] = []
    gate_deltas: dict[str, list[float]] = {name: [] for name in ("EIRP_norm", "SLL_dB", "theta_err_deg")}
    for seed in expected:
        if seed not in by_group["target_only"] or seed not in by_group["gan_transition_init_spatial"]:
            continue
        target, init = by_group["target_only"][seed], by_group["gan_transition_init_spatial"][seed]
        if target.get("test_fingerprint") != init.get("test_fingerprint"):
            reasons.append("test_fingerprint_mismatch")
            continue
        target_map = {str(item.get("trajectory_id")): item for item in target.get("trajectory_metrics", [])}
        init_map = {str(item.get("trajectory_id")): item for item in init.get("trajectory_metrics", [])}
        if not target_map or set(target_map) != set(init_map):
            reasons.append("test_trajectory_set_mismatch")
            continue
        per_seed: list[float] = []
        for trajectory in sorted(target_map):
            t_item, i_item = target_map[trajectory], init_map[trajectory]
            if (t_item.get("test_trajectory_fingerprint") != i_item.get("test_trajectory_fingerprint")
                    or int(t_item.get("prediction_steps", -1)) != _ROLLOUT_STEPS
                    or int(i_item.get("prediction_steps", -1)) != _ROLLOUT_STEPS):
                reasons.append("trajectory_fingerprint_or_horizon_mismatch")
                continue
            try:
                delta = float(t_item["array_6step_nrmse_by_metric"]["M_link_dB"] - i_item["array_6step_nrmse_by_metric"]["M_link_dB"])
            except (KeyError, TypeError):
                reasons.append("missing_mlink_primary_metric")
                continue
            if not np.isfinite(delta):
                reasons.append("nonfinite_mlink_delta")
                continue
            trajectory_deltas.setdefault(trajectory, []).append(delta); per_seed.append(delta)
            for name in gate_deltas:
                try:
                    gate_delta = float(t_item["array_6step_nrmse_by_metric"][name]
                                       - i_item["array_6step_nrmse_by_metric"][name])
                except (KeyError, TypeError):
                    reasons.append(f"missing_{name}_gate_metric")
                    continue
                if not np.isfinite(gate_delta):
                    reasons.append(f"{name}_nonfinite")
                    continue
                gate_deltas[name].append(gate_delta)
        if per_seed:
            seed_deltas.append(float(np.mean(per_seed)))
    deltas = np.asarray([np.mean(values) for _, values in sorted(trajectory_deltas.items())], dtype=float)
    valid = len(seed_deltas) == _FORMAL_SEED_COUNT and len(deltas) > 0 and not reasons
    result = {
        "formal_primary_metric": "M_link_dB_6step_nrmse", "bootstrap_unit": "test_trajectory",
        "expected_seeds": list(expected), "paired_seed_count": len(seed_deltas), "formal_valid": valid,
        "failure_reasons": reasons, "holm_adjusted_p": None,
        "primary_min_effect_nrmse": float(min_effect_nrmse),
        "non_degradation_min_delta": float(non_degradation_min_delta),
        "array_non_degradation_gate": {name: (float(np.mean(values)) if values else None) for name, values in gate_deltas.items()},
    }
    if len(deltas):
        rng = np.random.default_rng(bootstrap_seed)
        boot = rng.choice(deltas, size=(n_bootstrap, len(deltas)), replace=True).mean(axis=1)
        p_detail = signflip_pvalue(deltas, seed=bootstrap_seed)
        result.update(mean_delta=float(deltas.mean()), ci95_lo=float(np.percentile(boot, 2.5)),
                      ci95_hi=float(np.percentile(boot, 97.5)), p_value_raw=float(p_detail["p_value"]),
                      p_value_method=p_detail["p_value_method"], p_value_samples=p_detail["p_value_samples"],
                      p_value_seed=p_detail["p_value_seed"],
                      p_value_observed_sign_included=p_detail["p_value_observed_sign_included"],
                      positive_seed_count=int(np.sum(np.asarray(seed_deltas) > 0.0)),
                      n_bootstrap_trajectories=int(len(deltas)))
    else:
        result.update(mean_delta=None, ci95_lo=None, ci95_hi=None, p_value_raw=None, positive_seed_count=0,
                      n_bootstrap_trajectories=0, p_value_method=None, p_value_samples=None, p_value_seed=None,
                      p_value_observed_sign_included=None)
    if valid:
        if result["mean_delta"] < min_effect_nrmse:
            reasons.append("M_link_dB_primary_not_improved")
        if result["ci95_lo"] <= 0.0:
            reasons.append("M_link_dB_bootstrap_ci_not_above_zero")
        for name, value in result["array_non_degradation_gate"].items():
            if value is None or not np.isfinite(value):
                reasons.append(f"{name}_nonfinite")
            elif value < non_degradation_min_delta:
                reasons.append(f"{name}_degraded")
        result["acceptance_pass"] = not reasons
        result["acceptance_status"] = "accepted" if result["acceptance_pass"] else "rejected"
    else:
        result["acceptance_pass"] = False
        result["acceptance_status"] = "invalid"
    holm_items = [result] if result["formal_valid"] and result["p_value_raw"] is not None else []
    _holm_adjust(holm_items)
    result["holm_family_size"] = len(holm_items)
    result["holm_is_noop"] = len(holm_items) == 1
    return {"all": result}


def _per_seed_mlink_delta(entries: list[dict], expected_seeds, comparator_group: str) -> dict[str, float | None]:
    """结果透明性表：逐 seed 汇总同一轨迹的 target-only − 对照 M_link 差值，不参与正式验收统计。"""
    indexed: dict[tuple[str, int], dict] = {
        (str(entry.get("group", "")), int(entry.get("seed", -1))): entry for entry in entries
    }
    deltas: dict[str, float | None] = {}
    for seed in (int(value) for value in expected_seeds):
        target, comparator = indexed.get(("target_only", seed)), indexed.get((comparator_group, seed))
        if target is None or comparator is None or target.get("test_fingerprint") != comparator.get("test_fingerprint"):
            deltas[str(seed)] = None
            continue
        target_metrics = {str(item.get("trajectory_id")): item for item in target.get("trajectory_metrics", [])}
        comparator_metrics = {str(item.get("trajectory_id")): item for item in comparator.get("trajectory_metrics", [])}
        if not target_metrics or set(target_metrics) != set(comparator_metrics):
            deltas[str(seed)] = None
            continue
        values: list[float] = []
        for trajectory in sorted(target_metrics):
            try:
                value = float(target_metrics[trajectory]["array_6step_nrmse_by_metric"]["M_link_dB"]
                              - comparator_metrics[trajectory]["array_6step_nrmse_by_metric"]["M_link_dB"])
            except (KeyError, TypeError):
                values = []
                break
            if not np.isfinite(value):
                values = []
                break
            values.append(value)
        deltas[str(seed)] = float(np.mean(values)) if values else None
    return deltas


def _spatial_result_transparency(entries: list[dict], expected_seeds) -> dict:
    """记录逐 seed 对照与服务事件可评估性，避免把六步窗口的空事件误报成服务层结论。"""
    service_items = [item for entry in entries for item in entry.get("trajectory_metrics", [])]
    predicted_events = [item for item in service_items if item.get("service_rul_s_pred") is not None]
    errors = [float(item["service_rul_abs_error_s"]) for item in service_items
              if item.get("service_rul_abs_error_s") is not None and np.isfinite(float(item["service_rul_abs_error_s"]))]
    censor_flags = [bool(item.get("service_censor_lower_bound_violation", False)) for item in service_items
                    if not bool(item.get("service_event_observed", False))]
    if not predicted_events:
        status = "not_evaluable_no_predicted_event_in_six_step_window"
    elif not errors:
        status = "not_evaluable_no_observed_predicted_event_pair"
    else:
        status = "evaluable_observed_predicted_events"
    return {
        "seed_mlink_delta_target_minus_source_init": _per_seed_mlink_delta(
            entries, expected_seeds, "gan_transition_init_spatial"),
        "seed_mlink_delta_target_minus_random_scramble": _per_seed_mlink_delta(
            entries, expected_seeds, "random_source_control_spatial"),
        "service_layer": {
            "status": status, "trajectory_item_count": len(service_items),
            "predicted_service_event_count": len(predicted_events),
            "evaluable_service_event_count": len(errors),
            "service_event_mae_s": (float(np.mean(errors)) if errors else None),
            "censor_lower_bound_violation_rate": (float(np.mean(censor_flags)) if censor_flags else None),
        },
    }


def _ood_holdout_binding(ood_holdout: SpatialOODHoldout | None) -> dict | None:
    """抽取可写入 checkpoint/报告的最小 OOD 审计绑定，不重复保存轨迹条件原文。"""
    if ood_holdout is None:
        return None
    manifest = ood_holdout.manifest
    required = ("ood_protocol", "condition_fingerprint", "ood_test_fingerprint")
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(f"已核验 OOD holdout 缺少字段: {', '.join(missing)}")
    return {
        "manifest_fingerprint": str(ood_holdout.manifest_fingerprint),
        "ood_protocol": manifest["ood_protocol"],
        "condition_fingerprint": str(manifest["condition_fingerprint"]),
        "ood_test_fingerprint": str(manifest["ood_test_fingerprint"]),
        "counts": {
            "train": len(ood_holdout.split.train_ids), "val": len(ood_holdout.split.val_ids),
            "iid_test": len(ood_holdout.split.iid_test_ids), "ood_test": len(ood_holdout.split.ood_test_ids),
        },
    }


def _summarize_spatial_ood(entries: list[dict], ood_holdout: SpatialOODHoldout | None) -> dict:
    """独立汇总 OOD 六步误差；绝不产生 paired Δ、p 值或 IID 验收状态。"""
    binding = _ood_holdout_binding(ood_holdout)
    by_group: dict[str, dict[str, list[float]]] = {}
    for entry in entries:
        items = entry.get("ood_trajectory_metrics", [])
        if not items:
            continue
        metrics = by_group.setdefault(str(entry.get("group")), {})
        for item in items:
            for section, prefix in (("channel_6step_nrmse_by_metric", "channel"),
                                    ("array_6step_nrmse_by_metric", "array")):
                for name, value in item.get(section, {}).items():
                    numeric = float(value)
                    if np.isfinite(numeric):
                        metrics.setdefault(f"{prefix}:{name}", []).append(numeric)
    if binding is None:
        return {"status": "not_requested", "holdout": None, "metrics_by_group": {}}
    if not by_group:
        return {"status": "requested_but_not_scored", "holdout": binding, "metrics_by_group": {}}
    return {
        "status": "separate_not_in_iid_acceptance", "holdout": binding,
        "metrics_by_group": {
            group: {name: float(np.mean(values)) for name, values in sorted(metrics.items()) if values}
            for group, metrics in sorted(by_group.items())
        },
    }


def persist_spatial_formal_results(json_path: Path, markdown_path: Path, entries: list[dict], *, n_bootstrap: int = 10_000,
                                   expected_seeds=range(42, 52), min_effect_nrmse: float = 0.02,
                                   non_degradation_min_delta: float = 0.0,
                                   ood_holdout: SpatialOODHoldout | None = None) -> None:
    """持久化正式比较矩阵；此函数不生成数据或触发训练。"""
    expected = tuple(int(seed) for seed in expected_seeds)
    acceptance = summarize_spatial_paired_acceptance(
        entries, expected_seeds=expected, n_bootstrap=n_bootstrap,
        min_effect_nrmse=min_effect_nrmse, non_degradation_min_delta=non_degradation_min_delta,
    )
    transparency = _spatial_result_transparency(entries, expected)
    ood_evaluation = _summarize_spatial_ood(entries, ood_holdout)
    json_path.parent.mkdir(parents=True, exist_ok=True); markdown_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_scope": "formal_preregistered_spatial", "evaluation_horizon_steps": _ROLLOUT_STEPS,
               "metrics": entries, "acceptance": acceptance, "result_transparency": transparency,
               "ood_evaluation": ood_evaluation}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = acceptance["all"]
    rows = ["# GaN 空间子阵迁移正式比较矩阵", "", "主指标：同一测试轨迹 **M_link 六步 nRMSE** 的配对差值（target-only − source-init）。EIRP、SLL、指向误差均须不恶化。", "",
            "| 配对 seeds | 测试轨迹 | M_link 平均 Δ | 轨迹 bootstrap 95% CI | 原始 p | Holm p | 状态 | 失败原因 |",
            "| ---: | ---: | ---: | --- | ---: | ---: | --- | --- |"]
    ci = f"[{summary['ci95_lo']:.6f}, {summary['ci95_hi']:.6f}]" if summary["ci95_lo"] is not None else "—"
    raw = f"{summary['p_value_raw']:.6f}" if summary["p_value_raw"] is not None else "—"
    holm = f"{summary['holm_adjusted_p']:.6f}" if summary["holm_adjusted_p"] is not None else "—"
    reasons = ", ".join(summary["failure_reasons"]) or "—"
    rows.append(f"| {summary['paired_seed_count']} | {summary['n_bootstrap_trajectories']} | {summary['mean_delta'] if summary['mean_delta'] is not None else float('nan'):.6f} | {ci} | {raw} | {holm} | {summary['acceptance_status']} | {reasons} |")
    gate = summary["array_non_degradation_gate"]
    gate_values = " | ".join(
        f"{gate[name] if gate[name] is not None else float('nan'):.6f}" for name in ("EIRP_norm", "SLL_dB", "theta_err_deg")
    )
    random_entries = [entry for entry in entries if entry.get("group") == "random_source_control_spatial"]
    service = transparency["service_layer"]
    def _delta_text(value: float | None) -> str:
        return "—" if value is None else f"{value:.6f}"
    rows.extend([
        "", "## 三项阵列不恶化门控", "", "| EIRP_norm Δ | SLL_dB Δ | theta_err_deg Δ | 预注册方向门槛 |",
        "| ---: | ---: | ---: | ---: |", f"| {gate_values} | ≥{summary['non_degradation_min_delta']:.6f} |",
        "", "## Random 对照与服务指标", "",
        "`random_source_control_spatial` 为 **state-label-scramble** 对照：仅在每个源器件内打乱状态标签；观测与应力时间顺序仍真实，并与 source-init 使用相同预训练预算。",
        f"随机对照运行数：{len(random_entries)}；服务事件 MAE：{_delta_text(service['service_event_mae_s'])} s；删失下界违反率：{_delta_text(service['censor_lower_bound_violation_rate'])}。",
        f"最小效应量：M_link Δ≥{summary['primary_min_effect_nrmse']:.6f}；方向门控：三项 Δ≥{summary['non_degradation_min_delta']:.6f}。",
        (f"p 值方法：{summary['p_value_method']}（样本数 {summary['p_value_samples']}，seed {summary['p_value_seed']}）"
         + ("；Monte Carlo 仅采样随机符号组合，并采用唯一一次 (B+1) 分母/分子加一校正。"
            if summary['p_value_method'] == 'monte_carlo_signflip' else "；精确枚举包含观测符号组合。")),
        "Holm：本矩阵只有一个正式主检验，调整为 **no-op**（调整后 p 等于原始 p）。",
        "正式运行必须为非 smoke 且显式 `--n-seeds 10`；每条记录都保存四通道、EIRP/SLL/指向/M_link 六步真值误差及服务 RUL/删失违规。",
        "", "## 结果透明性：逐 seed M_link 差值（不参与正式验收统计）", "",
        "| seed | target-only − source-init（M_link nRMSE） | target-only − random scramble（M_link nRMSE） |",
        "| ---: | ---: | ---: |",
        *[f"| {seed} | {_delta_text(transparency['seed_mlink_delta_target_minus_source_init'][str(seed)])} | "
          f"{_delta_text(transparency['seed_mlink_delta_target_minus_random_scramble'][str(seed)])} |"
          for seed in expected],
    ])
    if service["status"] == "not_evaluable_no_predicted_event_in_six_step_window":
        rows.append(f"服务层当前不可评估：{service['trajectory_item_count']} 条六步窗口轨迹均未出现预测服务事件，"
                    "故服务事件 MAE 为 `null`；不得据此主张服务层验证。")
    elif service["status"] == "not_evaluable_no_observed_predicted_event_pair":
        rows.append("服务层当前不可评估：虽存在预测服务事件，但没有可配对的已观测服务事件；不得主张服务层验证。")
    else:
        rows.append(f"服务层可评估样本数：{service['evaluable_service_event_count']}（仅限预测与真实服务事件配对样本）。")
    rows.extend(["", "## OOD 保持集（独立、未与 IID 主验收合并）", "",
                 "OOD 六步指标只描述预注册扫描—占空比—结温组合角点的独立表现；**不参与 IID 主验收**的配对 Δ、bootstrap、p 值、Holm 或通过/拒绝状态。"])
    if ood_evaluation["status"] == "separate_not_in_iid_acceptance":
        rows.extend(["", "| 组别 | OOD 六步 nRMSE 均值 |", "| --- | --- |"])
        for group, metrics in ood_evaluation["metrics_by_group"].items():
            values = "; ".join(f"{name}={value:.6f}" for name, value in metrics.items()) or "—"
            rows.append(f"| {group} | {values} |")
    elif ood_evaluation["status"] == "not_requested":
        rows.append("本次正式矩阵未传入 OOD manifest；无 OOD 评分，不能主张 OOD 泛化。")
    else:
        rows.append("已传入 OOD manifest，但当前 entries 未含独立 OOD 轨迹评分；不能主张 OOD 泛化。")
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode("ascii")); digest.update(str(value.dtype).encode("ascii")); digest.update(value.tobytes())
    return digest.hexdigest()


def _source_rows_fingerprint(source: DomainRows) -> str:
    """绑定实际读取的源域训练行，防止仅替换源样本仍借用旧 checkpoint。"""
    return _fingerprint({
        "x": _array_fingerprint(source.x), "states": _array_fingerprint(source.states),
        "obs_labels": _array_fingerprint(source.obs_labels), "rul": _array_fingerprint(source.rul),
        "event": _array_fingerprint(source.event), "ids": [str(value) for value in source.ids.tolist()],
        "time": _array_fingerprint(source.time),
    })


def _array_twin_metadata_fingerprints(metadata: dict[str, dict[str, object]], trajectories: list[str]) -> dict[str, str]:
    """逐轨迹绑定阵列孪生的全部物理元数据；这些量虽非模型输入，却影响评分。"""
    fingerprints: dict[str, str] = {}
    for trajectory in trajectories:
        try:
            item = metadata[trajectory]
            canonical = {
                "scan_az_deg": float(item["scan_az_deg"]), "margin0_dB": float(item["margin0_dB"]),
                "array_grid": [int(value) for value in item["array_grid"]],
                "element_spacing_lambda": float(item["element_spacing_lambda"]),
                "subarray_block": int(item["subarray_block"]),
            }
        except KeyError as exc:
            raise ValueError(f"阵列孪生 metadata 缺少 {trajectory} 的 {exc.args[0]}") from exc
        fingerprints[trajectory] = _fingerprint(canonical)
    return fingerprints


def build_spatial_checkpoint_manifest(config: dict, source: DomainRows, rows: SpatialDomainRows,
                                      array_twin_metadata: dict[str, dict[str, object]], *, source_dynamics_id: str,
                                      target_dynamics_id: str, expected_seeds,
                                      ood_holdout: SpatialOODHoldout | None = None) -> dict:
    """绑定正式 checkpoint 与当前配置、目标特征和预注册条件。"""
    validate_formal_target_trajectory_count(rows)
    trajectories: dict[str, str] = {}
    for trajectory in sorted(np.unique(rows.ids).tolist()):
        index = np.flatnonzero(rows.ids == trajectory)
        trajectories[trajectory] = _array_fingerprint(
            rows.x_global[index], rows.x_nodes[index], rows.node_states[index], rows.node_labels[index],
            rows.rul[index], rows.event[index], rows.time[index],
            *(rows.array_score_truth[name][index] for name in sorted(rows.array_score_truth)),
        )
    twin_fingerprints = _array_twin_metadata_fingerprints(array_twin_metadata, sorted(trajectories))
    canonical_config = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected = [int(seed) for seed in expected_seeds]
    return {
        "run_scope": "formal_spatial_metrics_checkpoint", "schema_version": "spatial_checkpoint_manifest_v2",
        "config_canonical_hash": hashlib.sha256(canonical_config.encode("utf-8")).hexdigest(),
        "target_total_trajectories": len(trajectories),
        "target_trajectory_fingerprints": trajectories,
        "target_feature_fingerprint": _fingerprint(trajectories),
        "source_rows_fingerprint": _source_rows_fingerprint(source),
        "array_twin_metadata_fingerprints": twin_fingerprints,
        "array_twin_metadata_fingerprint": _fingerprint(twin_fingerprints),
        "source_dynamics_id": str(source_dynamics_id), "target_dynamics_id": str(target_dynamics_id),
        "expected_seeds": expected,
        "ood_holdout": _ood_holdout_binding(ood_holdout),
        "preregistered_thresholds": {
            "spatial_primary_min_delta_nrmse": float(config["transfer"]["spatial_primary_min_delta_nrmse"]),
            "spatial_non_degradation_min_delta_nrmse": float(config["transfer"]["spatial_non_degradation_min_delta_nrmse"]),
        },
    }


def validate_spatial_checkpoint_manifest(saved: dict, current: dict) -> None:
    """恢复汇总前逐项比较 manifest；拒绝旧 schema、伪造小数据或任意环境漂移。"""
    required = ("run_scope", "schema_version", "config_canonical_hash", "target_total_trajectories",
                "target_feature_fingerprint", "target_trajectory_fingerprints", "source_dynamics_id",
                "target_dynamics_id", "source_rows_fingerprint", "array_twin_metadata_fingerprints",
                "array_twin_metadata_fingerprint", "expected_seeds", "preregistered_thresholds", "ood_holdout")
    for key in required:
        if key not in saved:
            raise ValueError(f"checkpoint manifest 缺少 {key}")
        if saved[key] != current.get(key):
            raise ValueError(f"checkpoint manifest {key} 与当前运行不匹配")
    if int(saved["target_total_trajectories"]) < 300:
        raise ValueError("checkpoint manifest target_total_trajectories 必须至少 300")


def persist_spatial_metrics_checkpoint(path: Path, entries: list[dict], manifest: dict) -> None:
    """每完成一个训练组即原子持久化 entry 与严格 manifest，供统计失败后恢复汇总。"""
    validate_spatial_checkpoint_manifest(manifest, manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {"manifest": manifest, "metrics": entries}
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_spatial_metrics_checkpoint(path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("metrics") if isinstance(payload, dict) else None
    manifest = payload.get("manifest") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not isinstance(manifest, dict):
        raise ValueError("空间 metrics checkpoint 缺少 metrics 列表或 manifest")
    return entries, manifest


def _pretrained_spatial_model(source_rows: DomainRows, *, target_global_dim: int, node_dim: int, hidden_dim: int,
                              epochs: int, device: torch.device) -> tuple[DamageStateModel, int]:
    """源域仅预训练并迁移 Fθ；目标的节点 encoder/head 始终随机初始化。"""
    source_model = DamageStateModel(source_rows.x.shape[1], target_global_dim, hidden_dim=hidden_dim, target_node_input_dim=node_dim).to(device)
    steps = pretrain_source_transition(source_model, source_rows, epochs=epochs, device=device)
    target_model = DamageStateModel(source_rows.x.shape[1], target_global_dim, hidden_dim=hidden_dim, target_node_input_dim=node_dim).to(device)
    target_model.load_transition_state_dict(source_model.transition_state_dict())
    return target_model, steps


def train_spatial_model(model: DamageStateModel, train: SpatialDomainRows, val: SpatialDomainRows, *, time_scale_s: float,
                        epochs: int, device: torch.device, patience: int = 3) -> tuple[DamageStateModel, int]:
    """仅用验证节点损失 early-stop；测试轨迹不进入训练或选模。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    best, best_val, stale, steps = None, float("inf"), 0, 0
    for _ in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = spatial_node_one_step_loss(model, train, time_scale_s=time_scale_s, device=device)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); steps += 1
        model.eval()
        with torch.no_grad():
            val_loss = float(spatial_node_one_step_loss(model, val, time_scale_s=time_scale_s, device=device))
        if val_loss < best_val:
            best, best_val, stale = copy.deepcopy(model.state_dict()), val_loss, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best is not None:
        model.load_state_dict(best)
    return model, steps


def validate_spatial_run_args(*, smoke: bool, n_seeds: int | None) -> int:
    """smoke 保持非正式；正式矩阵只接受显式且固定的 10 seed。"""
    if smoke:
        if n_seeds == _FORMAL_SEED_COUNT:
            raise ValueError("固定 10 seed 正式矩阵必须为非 smoke 运行")
        return 1 if n_seeds is None else int(n_seeds)
    if n_seeds != _FORMAL_SEED_COUNT:
        raise ValueError("正式空间比较矩阵必须显式指定 --n-seeds 10")
    return _FORMAL_SEED_COUNT


def validate_formal_target_trajectory_count(rows: SpatialDomainRows, minimum: int = 300) -> None:
    count = len(np.unique(rows.ids))
    if count < minimum:
        raise ValueError(f"正式空间比较矩阵至少 {minimum} 条目标轨迹，当前仅 {count}；请增加目标仿真规模")


def _scrambled_source_rows(source: DomainRows, seed: int) -> DomainRows:
    states, time = scramble_source_state_pairs(source.states, source.ids, source.time, seed)
    return DomainRows(source.x, states, source.obs_labels, source.rul, source.event, source.ids, time,
                      source.transition_time_scale_s)


def _new_spatial_target_model(source: DomainRows, train: SpatialDomainRows, *, hidden_dim: int,
                              device: torch.device) -> DamageStateModel:
    return DamageStateModel(source.x.shape[1], train.x_global.shape[1], hidden_dim=hidden_dim,
                            target_node_input_dim=train.x_nodes.shape[-1]).to(device)


def make_spatial_target_initial_state(source: DomainRows, train: SpatialDomainRows, *, seed: int, hidden_dim: int,
                                      device: torch.device) -> dict[str, torch.Tensor]:
    """在源预训练前捕获目标模型完整随机初始状态，供同一 seed 的所有组恢复。"""
    set_seed(seed)
    model = _new_spatial_target_model(source, train, hidden_dim=hidden_dim, device=device)
    return copy.deepcopy(model.state_dict())


def initialize_spatial_group_model(group: str, source: DomainRows, train: SpatialDomainRows, *,
                                   target_initial_state: dict[str, torch.Tensor], source_epochs: int,
                                   hidden_dim: int, seed: int, device: torch.device) -> tuple[DamageStateModel, int, str]:
    """所有组先恢复同一目标初始 state；只有 init/random 可覆盖共享 Fθ。"""
    if group not in SPATIAL_GROUPS:
        raise ValueError(f"未知空间实验组: {group}")
    set_seed(seed)
    target_model = _new_spatial_target_model(source, train, hidden_dim=hidden_dim, device=device)
    target_model.load_state_dict(copy.deepcopy(target_initial_state), strict=True)
    if group == "target_only":
        return target_model, 0, "无源预训练；目标模型恢复同 seed 共享随机初始化"
    pretrain_rows = source if group == "gan_transition_init_spatial" else _scrambled_source_rows(source, seed)
    source_model = DamageStateModel(source.x.shape[1], train.x_global.shape[1], hidden_dim=hidden_dim,
                                    target_node_input_dim=train.x_nodes.shape[-1]).to(device)
    source_steps = pretrain_source_transition(source_model, pretrain_rows, epochs=source_epochs, device=device)
    target_model.load_transition_state_dict(source_model.transition_state_dict())
    note = "源预训练→仅覆盖Fθ（真实状态）→目标训练" if group == "gan_transition_init_spatial" else \
        "源预训练→仅覆盖Fθ（state-label-scramble，观测仍真实；与init等预算）→目标训练"
    return target_model, source_steps, note


def run_spatial_group(group: str, source: DomainRows, train: SpatialDomainRows, val: SpatialDomainRows,
                      test: SpatialDomainRows, metadata: dict[str, dict[str, object]], *, seed: int,
                      time_scale_s: float, channel_scale: np.ndarray, array_scale: dict[str, float], epochs: int,
                      source_epochs: int, device: torch.device,
                      target_initial_state: dict[str, torch.Tensor] | None = None,
                      ood_test: SpatialDomainRows | None = None,
                      ood_metadata: dict[str, dict[str, object]] | None = None) -> dict:
    """单个空间对照组：init/random 均等预算预训练，唯一差异是源状态监督是否打乱。"""
    if group not in SPATIAL_GROUPS:
        raise ValueError(f"未知空间实验组: {group}")
    hidden_dim = 16 if epochs <= 2 else 64
    if target_initial_state is None:
        target_initial_state = make_spatial_target_initial_state(source, train, seed=seed, hidden_dim=hidden_dim, device=device)
    model, source_steps, budget_note = initialize_spatial_group_model(
        group, source, train, target_initial_state=target_initial_state, source_epochs=source_epochs,
        hidden_dim=hidden_dim, seed=seed, device=device,
    )
    model, target_steps = train_spatial_model(model, train, val, time_scale_s=time_scale_s, epochs=epochs, device=device)
    trajectory_metrics = spatial_trajectory_metrics(model, test, metadata, channel_scale=channel_scale,
                                                     array_scale=array_scale, time_scale_s=time_scale_s, device=device)
    result = {
        "group": group, "seed": int(seed), "smoke": False, "evaluation_horizon_steps": _ROLLOUT_STEPS,
        "target_train_trajectory_count": int(len(np.unique(train.ids))), "source_pretrain_steps": int(source_steps),
        "target_steps": int(target_steps), "budget_note": budget_note, "test_fingerprint": _spatial_test_fingerprint(test),
        "target_initial_state_fingerprint": _fingerprint({key: value.detach().cpu().tolist()
                                                            for key, value in sorted(target_initial_state.items())}),
        "trajectory_metrics": trajectory_metrics,
    }
    if ood_test is not None:
        if ood_metadata is None:
            raise ValueError("OOD 空间评分必须提供独立 array twin metadata")
        result["ood_trajectory_metrics"] = spatial_trajectory_metrics(
            model, ood_test, ood_metadata, channel_scale=channel_scale, array_scale=array_scale,
            time_scale_s=time_scale_s, device=device,
        )
        result["ood_test_fingerprint"] = _spatial_test_fingerprint(ood_test)
    return result


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="GaN 空间子阵迁移：smoke 或正式比较矩阵")
    parser.add_argument("--config", default="configs/phased_array_gan.yaml")
    parser.add_argument("--smoke", action="store_true", help="小数据最小训练路径；结果不是正式验收")
    parser.add_argument("--n-seeds", type=int, default=None, help="正式矩阵必须显式指定 10；smoke 默认 1")
    parser.add_argument("--output", default=None, help="JSON 结果路径")
    parser.add_argument("--report", default=None, help="正式 Markdown 报告路径（正式模式默认 docs/）")
    parser.add_argument("--ood-holdout", default=None,
                        help="预注册 OOD split manifest 路径；训练、归一化和早停仅使用其中 IID 轨迹")
    parser.add_argument("--summarize-checkpoint", default=None, help="从已验证的正式 metrics checkpoint 仅汇总/报告，不续训")
    parser.add_argument("--resume-checkpoint", default=None, help="已弃用别名；等同 --summarize-checkpoint，仅汇总不续训")
    args = parser.parse_args()
    n_seeds = validate_spatial_run_args(smoke=args.smoke, n_seeds=args.n_seeds)
    if args.summarize_checkpoint and args.resume_checkpoint:
        raise ValueError("--summarize-checkpoint 与已弃用的 --resume-checkpoint 只能指定其一")
    checkpoint_arg = args.summarize_checkpoint or args.resume_checkpoint
    cfg = load_config(_resolve_path(args.config))
    output = _resolve_path(args.output) if args.output else ROOT / "docs" / "results_phased_array_gan_spatial.json"
    report = _resolve_path(args.report) if args.report else ROOT / "docs" / "results_phased_array_gan_spatial.md"
    validate_source_config(cfg["source"])
    source_path, target_path = _resolve_path(cfg["source"]["feature_path"]), _resolve_path(cfg["target"]["feature_path"])
    source, source_id = _source_rows(source_path, 12 if args.smoke else 160)
    validate_spatial_feature_schema(target_path)
    spatial, target_id = _spatial_target_rows(target_path, 12 if args.smoke else 256)
    validate_domain_separation(source_id, target_id)
    if not args.smoke:
        validate_formal_target_trajectory_count(spatial)
    array_twin_metadata = _load_array_twin_metadata(target_path, spatial.ids)
    ood_holdout = None
    if args.ood_holdout:
        if "ood_holdout" not in cfg.get("target", {}):
            raise ValueError("使用 --ood-holdout 时 target.ood_holdout 必须声明预注册协议")
        ood_holdout = load_spatial_ood_holdout(
            _resolve_path(args.ood_holdout), cfg["target"]["ood_holdout"], target_path, spatial,
        )
    expected_seeds = range(int(cfg["seed"]), int(cfg["seed"]) + n_seeds)
    manifest = None if args.smoke else build_spatial_checkpoint_manifest(
        cfg, source, spatial, array_twin_metadata, source_dynamics_id=source_id, target_dynamics_id=target_id,
        expected_seeds=expected_seeds, ood_holdout=ood_holdout,
    )
    if checkpoint_arg:
        if args.smoke:
            raise ValueError("smoke 不支持汇总正式 metrics checkpoint")
        entries, saved_manifest = load_spatial_metrics_checkpoint(_resolve_path(checkpoint_arg))
        validate_spatial_checkpoint_manifest(saved_manifest, manifest)
        persist_spatial_formal_results(
            output, report, entries, expected_seeds=expected_seeds,
            min_effect_nrmse=float(cfg["transfer"]["spatial_primary_min_delta_nrmse"]),
            non_degradation_min_delta=float(cfg["transfer"]["spatial_non_degradation_min_delta_nrmse"]),
            ood_holdout=ood_holdout,
        )
        print(f">> 空间 GaN 正式矩阵已由 checkpoint 恢复汇总 -> {output} / {report}")
        return
    time_scale_s = float(cfg["transfer"]["transition_time_scale_s"])
    source_scaler = NodeTrainOnlyStandardizer().fit(source.x[:, None, :])
    source = DomainRows(source_scaler.transform(source.x[:, None, :])[:, 0], source.states, source.obs_labels,
                        source.rul, source.event, source.ids, source.time, time_scale_s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        if ood_holdout is None:
            train, val, test = split_spatial_rows(spatial, int(cfg["seed"]))
            ood_test = None
        else:
            split = ood_holdout.split
            train = _select_spatial_rows(spatial, list(split.train_ids))
            val = _select_spatial_rows(spatial, list(split.val_ids))
            test = _select_spatial_rows(spatial, list(split.iid_test_ids))
            ood_test = _select_spatial_rows(spatial, list(split.ood_test_ids))
        _validate_rollout_trajectory_lengths(test)
        if ood_test is not None:
            _validate_rollout_trajectory_lengths(ood_test)
            train, val, test, ood_test, _node_scaler = _standardize_spatial(train, val, test, ood_test)
        else:
            train, val, test, _node_scaler = _standardize_spatial(train, val, test)
        metadata = {str(trajectory): array_twin_metadata[str(trajectory)] for trajectory in np.unique(test.ids)}
        set_seed(int(cfg["seed"]))
        model, source_steps = _pretrained_spatial_model(source, target_global_dim=train.x_global.shape[1], node_dim=train.x_nodes.shape[-1],
                                                         hidden_dim=16, epochs=1, device=device)
        model, target_steps = train_spatial_model(model, train, val, time_scale_s=time_scale_s, epochs=2, device=device)
        metrics = array_rollout_errors(model, test, metadata, time_scale_s=time_scale_s, device=device)
        output = _resolve_path(args.output) if args.output else ROOT / "runs" / "gan_spatial_transfer_smoke.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_scope": "smoke_only_non_formal", "smoke": True, "group": "gan_transition_init_spatial",
            "source_pretrain_steps": source_steps, "target_steps": target_steps,
            "evaluation_horizon_steps": _ROLLOUT_STEPS, "array_rollout_rmse": metrics,
            "trajectory_split": {"train": sorted(np.unique(train.ids).tolist()), "val": sorted(np.unique(val.ids).tolist()),
                                 "test": sorted(np.unique(test.ids).tolist())},
            "note": "仅验证空间训练与确定性阵列孪生链路；非10-seed、非300轨迹、非正式正迁移验收。",
        }
        if ood_holdout is not None and ood_test is not None:
            ood_metadata = {str(trajectory): array_twin_metadata[str(trajectory)] for trajectory in np.unique(ood_test.ids)}
            payload.update({
                "evaluation_scope": "iid_with_pre_registered_ood_holdout",
                "ood_array_rollout_rmse": array_rollout_errors(
                    model, ood_test, ood_metadata, time_scale_s=time_scale_s, device=device),
                "ood_holdout": {
                    "manifest_fingerprint": ood_holdout.manifest_fingerprint,
                    "condition_fingerprint": ood_holdout.manifest["condition_fingerprint"],
                    "ood_test_fingerprint": ood_holdout.manifest["ood_test_fingerprint"],
                },
                "trajectory_split": {
                    "train": sorted(np.unique(train.ids).tolist()), "val": sorted(np.unique(val.ids).tolist()),
                    "iid_test": sorted(np.unique(test.ids).tolist()), "ood_test": sorted(np.unique(ood_test.ids).tolist()),
                },
                "note": "IID 轨迹单独用于训练、归一化、早停与主验收；OOD 保持集仅作独立六步评分，绝不与 IID 主验收混合。"
                        "该 smoke 非10-seed、非300轨迹，不能用于正迁移结论。",
            })
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f">> 空间 GaN 迁移 smoke 完成 -> {output}（非正式验收）")
        return

    entries: list[dict] = []
    checkpoint = output.with_name(f"{output.stem}_checkpoint.json")
    for seed in range(int(cfg["seed"]), int(cfg["seed"]) + n_seeds):
        if ood_holdout is None:
            train, val, test = split_spatial_rows(spatial, seed)
            ood_test = None
        else:
            split = ood_holdout.split
            train = _select_spatial_rows(spatial, list(split.train_ids))
            val = _select_spatial_rows(spatial, list(split.val_ids))
            test = _select_spatial_rows(spatial, list(split.iid_test_ids))
            ood_test = _select_spatial_rows(spatial, list(split.ood_test_ids))
        _validate_rollout_trajectory_lengths(test)
        if ood_test is not None:
            _validate_rollout_trajectory_lengths(ood_test)
            train, val, test, ood_test, _node_scaler = _standardize_spatial(train, val, test, ood_test)
        else:
            train, val, test, _node_scaler = _standardize_spatial(train, val, test)
        metadata = {str(trajectory): array_twin_metadata[str(trajectory)] for trajectory in np.unique(test.ids)}
        ood_metadata = ({str(trajectory): array_twin_metadata[str(trajectory)] for trajectory in np.unique(ood_test.ids)}
                        if ood_test is not None else None)
        channel_scale = train.node_labels.reshape(-1, len(_CHANNEL_METRICS)).std(axis=0).astype(np.float32) + 1e-6
        array_scale = {name: float(np.std(train.array_score_truth[name]) + 1e-6) for name in _ARRAY_METRICS}
        target_initial_state = make_spatial_target_initial_state(source, train, seed=seed, hidden_dim=64, device=device)
        for group in SPATIAL_GROUPS:
            entries.append(run_spatial_group(group, source, train, val, test, metadata, seed=seed,
                                             time_scale_s=time_scale_s, channel_scale=channel_scale, array_scale=array_scale,
                                             epochs=30, source_epochs=20, device=device,
                                             target_initial_state=target_initial_state,
                                             ood_test=ood_test, ood_metadata=ood_metadata))
            persist_spatial_metrics_checkpoint(checkpoint, entries, manifest)
    persist_spatial_formal_results(
        output, report, entries, expected_seeds=expected_seeds,
        min_effect_nrmse=float(cfg["transfer"]["spatial_primary_min_delta_nrmse"]),
        non_degradation_min_delta=float(cfg["transfer"]["spatial_non_degradation_min_delta_nrmse"]),
        ood_holdout=ood_holdout,
    )
    print(f">> 空间 GaN 正式比较矩阵完成 -> {output} / {report}")


if __name__ == "__main__":
    main()
