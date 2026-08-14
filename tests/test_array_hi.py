"""PA4 阵列 HI 测试 (二轮: latent 隔离 + 子阵节点 + 固定 HI + 右删失)。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sim.build_array_hi import build_features, X_GLOBAL_COLS, X_NODE_COLS      # noqa: E402


def _fake(n=200, seed=0, fail=True):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    damage = (t / n) ** 0.5
    EIRP = np.clip(1.0 - 0.7 * damage + 0.02 * rng.standard_normal(n), 0, 1)
    G = -2.0 * (1.0 - EIRP)
    Mobs = 2.0 - 3.0 * (1.0 - EIRP)
    Mtrue = 2.0 - 3.0 * (1.0 - EIRP)
    SLL_true = -13.0 + 8.0 * damage
    SLL_obs = SLL_true + 0.1 * rng.standard_normal(n)
    th_true = 0.1 * damage
    th_obs = th_true + 0.05 * rng.standard_normal(n)
    lf = np.zeros(n, dtype=np.int8)
    if fail:
        lf[160:] = 1
    cols = {"damage": damage, "EIRP_norm": EIRP, "G_array_dB": G, "M_link_dB": Mobs,
            "M_link_dB_true": Mtrue, "SLL_dB": SLL_obs, "SLL_dB_true": SLL_true,
            "theta_err_deg": th_obs, "theta_err_deg_true": th_true,
            "IDSS_ratio": 1.0 - 0.1 * damage, "Tj": 400.0 + 5.0 * rng.standard_normal(n),
            "label_fail": lf}
    sa = rng.random((n, 16, 8)).astype(np.float32)
    params = {"margin0_dB": 2.0}
    lim = {"SLL_max_dB": -8.0, "theta_err_max_deg": 0.5}
    return cols, sa, params, lim


def test_x_shapes_and_obs_only():
    """失效轨迹在 EOL 截断，且特征仅含可观测量。"""
    cols, sa, params, lim = _fake(seed=0)
    rng = np.random.default_rng(0)
    xg, xn, HI, dmg, rul, eol, ev = build_features(cols, sa, params, lim, 0.35, rng)
    n = len(cols["EIRP_norm"])
    expected_t = eol + 1 if ev else n
    assert xg.shape == (expected_t, 6) and xn.shape == (expected_t, 16, 6)
    assert len(X_GLOBAL_COLS) == 6 and len(X_NODE_COLS) == 6
    # x 第一维 G_obs 不应等于无噪 latent EIRP_norm (防答案泄漏, GPT 八)
    assert not np.allclose(xg[:, 0], cols["EIRP_norm"][:expected_t]), "x 不应含无噪 EIRP_norm 答案变量"
    assert not np.any(np.isnan(xg)) and not np.any(np.isnan(xn))


def test_hi_fixed_scale_reaches_one_at_failure():
    cols, sa, params, lim = _fake(seed=1, fail=True)
    rng = np.random.default_rng(1)
    xg, xn, HI, dmg, rul, eol, ev = build_features(cols, sa, params, lim, 0.35, rng)
    assert HI.min() >= -1e-6 and HI.max() <= 1 + 1e-6
    assert HI.max() >= 0.9, "失效轨迹 HI 末态应接近 1 (服务越限)"


def test_event_observed_and_censor():
    cols, sa, params, lim = _fake(seed=2, fail=True)
    rng = np.random.default_rng(2)
    xg, xn, HI, dmg, rul, eol, ev = build_features(cols, sa, params, lim, 0.35, rng)
    assert ev is True and rul[eol] == 0
    cols2, sa2, p2, lim2 = _fake(seed=3, fail=False)
    xg2, xn2, HI2, dmg2, rul2, eol2, ev2 = build_features(cols2, sa2, p2, lim2, 0.35, np.random.default_rng(3))
    assert ev2 is False, "未越限应 event_observed=0 (右删失)"
    assert rul2[-1] == 0, "右删失 rul 下界末端=0"


def test_damage_norm_in_unit():
    cols, sa, params, lim = _fake(seed=4)
    xg, xn, HI, dmg, rul, eol, ev = build_features(cols, sa, params, lim, 0.35, np.random.default_rng(4))
    assert dmg.min() >= -1e-6 and dmg.max() <= 1 + 1e-6


def test_hi_tracks_damage():
    cols, sa, params, lim = _fake(seed=5, fail=True)
    xg, xn, HI, dmg, rul, eol, ev = build_features(cols, sa, params, lim, 0.35, np.random.default_rng(5))
    corr = np.corrcoef(HI, dmg)[0, 1]
    assert corr > 0.7, f"HI-damage 相关 {corr:.3f} 应 > 0.7"
