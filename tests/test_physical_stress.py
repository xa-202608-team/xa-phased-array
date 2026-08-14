"""共享 build_physical_stress 契约（源/目标同坐标系，Gate 1.1/2 跨域基础）。"""
from __future__ import annotations

import numpy as np

from src.sim.physical_stress import (
    PHYSICAL_STRESS_SCHEMA,
    TARGET_PIN_DBM_REFERENCE,
    TARGET_VSWR_REFERENCE,
    build_physical_stress,
)


def test_build_physical_stress_matches_sim_temp_and_stress_formula():
    """公式与 gan_rfalt_sim.py:115/117 逐字一致（temp_factor / stress_factor）。"""
    tj = np.array([95.0, 130.0, 60.0, 150.0])
    duty = np.array([0.5, 0.8, 0.2, 0.05])
    pin = np.array([35.0, 40.0, 38.0, 30.0])
    vswr = np.array([1.0, 2.0, 1.5, 3.0])
    rec = np.array([0, 0, 1, 1])
    u = build_physical_stress(tj, duty, pin_dbm=pin, vswr=vswr, recovery=rec)
    expected_a = np.clip(np.exp((tj - 95.0) / 42.0), 0.20, 8.0)
    expected_s = duty * (1.0 + 0.18 * np.maximum(pin - 35.0, 0.0) + 0.25 * np.maximum(vswr - 1.0, 0.0))
    np.testing.assert_allclose(u[:, 0], expected_a, rtol=1e-6)
    np.testing.assert_allclose(u[:, 1], expected_s, rtol=1e-6)
    np.testing.assert_allclose(u[:, 2], rec, rtol=1e-6)
    assert u.dtype == np.float32
    assert u.shape == (4, 3)


def test_target_baseline_pin_vswr_collapses_s_to_duty():
    """目标域基准 pin=35/vswr=1 → 压缩/驻波项归零，s=duty（GPT §5 预注册基准）。"""
    tj = np.array([113.0, 120.0])
    duty = np.array([0.55, 0.6])
    u = build_physical_stress(tj, duty, pin_dbm=TARGET_PIN_DBM_REFERENCE,
                              vswr=TARGET_VSWR_REFERENCE, recovery=np.array([0, 0]))
    np.testing.assert_allclose(u[:, 1], duty, rtol=1e-6)


def test_a_t_clipped_to_physical_bounds():
    """a_T 截断到 [0.2, 8.0]：极端 Tj 既不爆炸也不归零。"""
    tj = np.array([-1000.0, 1000.0])
    u = build_physical_stress(tj, np.array([0.5, 0.5]), pin_dbm=35.0, vswr=1.0,
                              recovery=np.array([0, 0]))
    assert np.isclose(u[0, 0], 0.20, atol=1e-6)
    assert np.isclose(u[1, 0], 8.0, atol=1e-5)


def test_diagnostic_physical_stress_delegates_to_shared_function():
    """run_gan_diagnostics._physical_stress 委托后输出与 build_physical_stress 完全一致。"""
    from src.experiments.run_gan_diagnostics import (
        _PHYS_DUTY_IDX, _PHYS_PIN_IDX, _PHYS_TJ_IDX, _PHYS_VSWR_IDX, _physical_stress,
    )
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 20)).astype(np.float32)
    x[:, _PHYS_TJ_IDX] = rng.uniform(80, 130, 50)
    x[:, _PHYS_DUTY_IDX] = rng.uniform(0.02, 0.8, 50)   # 含 <0.1 的 recovery 段
    x[:, _PHYS_VSWR_IDX] = rng.uniform(1.0, 3.0, 50)
    x[:, _PHYS_PIN_IDX] = rng.uniform(30, 42, 50)

    out = _physical_stress(x)
    expected = build_physical_stress(
        tj_c=x[:, _PHYS_TJ_IDX], duty=x[:, _PHYS_DUTY_IDX],
        pin_dbm=x[:, _PHYS_PIN_IDX], vswr=x[:, _PHYS_VSWR_IDX],
        recovery=(x[:, _PHYS_DUTY_IDX] < 0.10))
    np.testing.assert_allclose(out, expected, rtol=1e-6)
    # recovery 列由源域 (duty<0.10) 判定
    np.testing.assert_allclose(out[:, 2], (x[:, _PHYS_DUTY_IDX] < 0.10).astype(np.float32))


def test_schema_and_reference_constants_exported():
    assert PHYSICAL_STRESS_SCHEMA == "gan_physical_stress_v1"
    assert TARGET_PIN_DBM_REFERENCE == 35.0
    assert TARGET_VSWR_REFERENCE == 1.0
