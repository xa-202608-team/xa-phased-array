"""§4h 误差方向/风险偏置诊断契约测试.

_bias_diag 的分箱语义与指标口径:
  ① HI 真值三分箱 early(<1/3)/middle(1/3-2/3)/late(>=2/3), 仅失效样本进箱;
  ② e = RUL_hat - RUL: mean/median bias、over_rate = P(e>0);
  ③ 删失样本不进箱, 单独报 lb_violation_rate / mean_bias_vs_lb (相对下界)。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.run_groups import _bias_diag                       # noqa: E402


def test_bins_and_metrics():
    # 失效 6 样本: early(hi=0.1,0.2) middle(0.4,0.5) late(0.7,0.8); 删失 2 样本
    p = np.array([0.6, 0.4, 0.3, 0.5, 0.2, 0.4, 0.9, 0.1])
    t = np.array([0.5, 0.5, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5])
    ev = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    hi = np.array([0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.5, 0.5])
    d = _bias_diag(p, t, ev, hi)

    assert d["early"]["n"] == 2
    np.testing.assert_allclose(d["early"]["mean_bias"], np.mean([0.6 - 0.5, 0.4 - 0.5]))
    assert d["early"]["over_rate"] == 0.5
    assert d["middle"]["n"] == 2
    assert d["late"]["n"] == 2
    np.testing.assert_allclose(d["late"]["mean_bias"], np.mean([0.2 - 0.3, 0.4 - 0.3]))
    assert d["failed_all"]["n"] == 6
    np.testing.assert_allclose(d["failed_all"]["mean_bias"], np.mean(p[:6] - t[:6]))

    # 删失: pred vs lower_bound
    np.testing.assert_allclose(d["censored"]["lb_violation_rate"], 0.5)   # 0.9>0.5 ok, 0.1<0.5 违反
    np.testing.assert_allclose(d["censored"]["mean_bias_vs_lb"], np.mean([0.4, -0.4]))
    assert d["censored"]["n"] == 2


def test_empty_bins_omitted_and_censored_boundary():
    # 无 late 样本 -> late 键缺失 (不产出 n=0 箱); 删失边界 pred==lb 不算违反
    p = np.array([0.3, 0.4])
    t = np.array([0.3, 0.2])
    ev = np.array([1, 0])
    hi = np.array([0.1, 0.1])
    d = _bias_diag(p, t, ev, hi)
    assert "late" not in d and "middle" not in d
    assert d["early"]["n"] == 1
    assert d["censored"]["lb_violation_rate"] == 0.0     # pred(0.4) >= lb(0.2) -> 不违反


def test_all_censored():
    p = np.array([0.2, 0.8])
    t = np.array([0.5, 0.5])
    ev = np.array([0, 0])
    hi = np.array([0.2, 0.8])
    d = _bias_diag(p, t, ev, hi)
    assert set(d.keys()) == {"censored"}
    np.testing.assert_allclose(d["censored"]["lb_violation_rate"], 0.5)
