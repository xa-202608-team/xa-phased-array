"""诊断 1 脚本的逻辑测试（合成数据，不依赖真实仿真产物）。"""
from __future__ import annotations

import numpy as np
import torch

from src.experiments.run_gan_diagnostics import (
    _nrmse_per_dim, _persistence_pred, _rollout6, _rollout_indices,
    diagnostic1, split_devices,
)
from src.experiments.run_gan_transfer import DomainRows
from src.transfer.damage_state import DamageStateModel


def _make_source_rows(n_devices=5, n_points=20, seed=0):
    rng = np.random.default_rng(seed)
    xs, states, labels, ruls, events, ids, times = [], [], [], [], [], [], []
    for d in range(n_devices):
        t = np.arange(n_points, dtype=np.float32) * 600.0
        x = rng.normal(0, 1, (n_points, 6)).astype(np.float32)
        s = np.stack([
            np.linspace(0, 0.5 + 0.1 * d, n_points),
            0.5 + 0.1 * np.sin(np.arange(n_points) + d),
            np.linspace(0, 0.2, n_points),
        ], axis=1).astype(np.float32)
        xs.append(x); states.append(s)
        labels.append(rng.normal(0, 1, (n_points, 4)).astype(np.float32))
        ruls.append((n_points - np.arange(n_points)).astype(np.float32))
        events.append(np.zeros(n_points, dtype=bool))
        ids.append(np.array([f"dev-{d:03d}"] * n_points))
        times.append(t)
    return DomainRows(
        np.concatenate(xs), np.concatenate(states), np.concatenate(labels),
        np.concatenate(ruls), np.concatenate(events), np.concatenate(ids),
        np.concatenate(times), 21600.0, None)


def test_split_devices_disjoint_and_complete():
    ids = np.array([f"dev-{i:03d}" for i in range(12)])
    train, val, holdout = split_devices(ids, n_holdout=3, n_val=2, seed=1)
    assert len(train) + len(val) + len(holdout) == 12
    assert set(train).isdisjoint(val) and set(train).isdisjoint(holdout)
    assert set(val).isdisjoint(holdout)
    assert len(holdout) == 3 and len(val) == 2 and len(train) == 7


def test_rollout_indices_respect_device_boundary():
    ids = np.array(["a"] * 8 + ["b"] * 8)
    time = np.concatenate([np.arange(8.0), np.arange(8.0)])
    starts = _rollout_indices(ids)
    # 每器件 8 点 -> 起点为 index 0,1（t..t+6 同器件），共 2 器件 = 4 起点
    assert starts.tolist() == [0, 1, 8, 9]


def test_rollout6_shape_and_persistence_constant():
    rows = _make_source_rows(n_devices=3, n_points=12)
    device = torch.device("cpu")
    model = DamageStateModel(rows.x.shape[1], rows.x.shape[1]).to(device)
    pred, target = _rollout6(model, rows, device=device, transition_time_scale_s=21600.0)
    assert pred.shape == target.shape
    assert pred.shape[1] == 6 and pred.shape[2] == 3
    starts = _rollout_indices(rows.ids)
    pp, tp = _persistence_pred(rows, starts)
    # persistence 第 k 步等于初态
    assert np.allclose(pp[:, 0], pp[:, 5])
    assert pp.shape == tp.shape


def test_nrmse_per_dim_zero_for_exact_match():
    target = np.ones((4, 6, 3))
    nrmse = _nrmse_per_dim(target, target, scale=np.ones(3))
    assert np.allclose(nrmse, 0.0)


def test_diagnostic1_runs_on_synthetic_smoke():
    rows = _make_source_rows(n_devices=8, n_points=24, seed=3)
    config = {
        "transfer": {"transition_time_scale_s": 21600.0},
        "damage_state": {"names": ["d_perm", "q_trap", "r_th"]},
    }
    # 直接用已加载的 rows 构造一个最小入口，避开 h5 依赖
    import src.experiments.run_gan_diagnostics as diag
    orig_source_rows = diag._source_rows

    class _Stub:
        def __call__(self, path, max_points):
            return rows, "rfalt_lumped_v1_test"

    diag._source_rows = _Stub()
    try:
        result = diag.diagnostic1(
            config, source_path=None, n_holdout=2, n_val=1, epochs=3, seed=7,
            device=torch.device("cpu"), max_points=999)
    finally:
        diag._source_rows = orig_source_rows

    assert result["diagnostic"] == 1
    assert result["device_split"]["holdout"] == 2
    for key in ("trained_Ftheta", "random_transition", "persistence"):
        per = result["nrmse_6step"][key]["per_dim"]
        assert len(per) == 3 and all(np.isfinite(v) for v in per)
    # persistence 在合成的单调 d_perm 上应有正误差
    assert result["nrmse_6step"]["persistence"]["mean"] > 0


def test_u_support_audit_quantiles_and_coverage():
    """u 支持域审计：目标在源范围内时 coverage=1.0，分位数单调，recovery 占比正确。"""
    from src.experiments.run_gan_diagnostics import u_support_audit
    rng = np.random.default_rng(0)
    source_u = np.stack([
        rng.uniform(0.7, 3.4, 1000), rng.uniform(0.02, 1.2, 1000),
        rng.integers(0, 2, 1000).astype(float)], axis=1)
    target_u = np.stack([
        rng.uniform(1.0, 2.0, 500), rng.uniform(0.3, 0.7, 500), np.zeros(500)], axis=1)
    result = u_support_audit(source_u, target_u)
    assert result["coverage_in_source_range"]["a_T"]["target_in_range_fraction"] == 1.0
    assert result["coverage_in_source_range"]["s"]["target_in_range_fraction"] == 1.0
    assert result["recovery_active_fraction"]["source"] > 0.3
    assert result["recovery_active_fraction"]["target"] == 0.0
    for name in ("a_T", "s", "recovery"):
        s = result["source_u_stats"][name]
        assert s[0.0] <= s[0.5] <= s[1.0]
    assert result["target_nn_distance_normalized"]["median"] >= 0.0
    assert result["physical_stress_schema"] == "gan_physical_stress_v1"


def test_u_support_audit_detects_ood_target():
    """目标 a_T 超出源范围时 coverage=0.0（OOD 外推检测）。"""
    from src.experiments.run_gan_diagnostics import u_support_audit
    source_u = np.array([[1.0, 0.5, 0.0], [2.0, 0.8, 1.0]])
    target_u = np.array([[10.0, 0.5, 0.0]])
    result = u_support_audit(source_u, target_u)
    assert result["coverage_in_source_range"]["a_T"]["target_in_range_fraction"] == 0.0
    assert result["coverage_in_source_range"]["s"]["target_in_range_fraction"] == 1.0


def test_u_support_audit_rejects_wrong_dim():
    import pytest
    from src.experiments.run_gan_diagnostics import u_support_audit
    with pytest.raises(ValueError):
        u_support_audit(np.zeros((3, 2)), np.zeros((3, 3)))
