"""提前性指标测试 (T5/M5)。

解析轨迹验证边界行为 (文档 T5 单测要求):
  - 完美预测 (pred=true) → PH=eol, α-λ=True, CM=0, RA=1
  - 恒定偏置 25% (pred=1.25·true, α=0.2) → PH=0, α-λ=False
  - PH 随偏置递减单调递增 (偏置 ≤α → PH=eol; >α → PH=0)
  - RA ≤ 1, CM ≥ 0 边界
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics.prognostic import (                              # noqa: E402
    alpha_lambda_accuracy, prognostic_horizon, convergence_metric, relative_accuracy)


def _linear_rul(T=100):
    """线性 RUL 真值: T-1, T-2, ..., 1, 0 (EOL 在末尾索引 T-1)。"""
    return np.arange(T - 1, -1, -1, dtype=np.float64)


def test_perfect_prediction():
    """完美预测 → PH=eol(99), α-λ=True, CM=0, RA=1。"""
    rt = _linear_rul(100)
    rp = rt.copy()
    assert prognostic_horizon(rp, rt, alpha=0.2) == 99
    assert alpha_lambda_accuracy(rp, rt, alpha=0.2, lam=0.5) is True
    assert convergence_metric(rp, rt, alpha=0.2) == 0.0
    assert abs(relative_accuracy(rp, rt, lam=0.5) - 1.0) < 1e-9


def test_constant_bias_25pct():
    """恒定偏置 +25% (pred=1.25·true, α=0.2) → PH=0, α-λ=False, RA=0.75。"""
    rt = _linear_rul(100)
    rp = 1.25 * rt
    assert prognostic_horizon(rp, rt, alpha=0.2) == 0
    assert alpha_lambda_accuracy(rp, rt, alpha=0.2, lam=0.5) is False
    # t_lam = int(0.5*99) = 49, rt[49]=50, rp[49]=62.5 → RA = 1-12.5/50 = 0.75
    assert abs(relative_accuracy(rp, rt, lam=0.5) - 0.75) < 1e-9


def test_ph_monotone_with_bias():
    """PH 随偏置递减单调递增: 偏置>α→PH=0; ≤α→PH=eol。"""
    rt = _linear_rul(100)
    phs = [prognostic_horizon((1 + b) * rt, rt, alpha=0.2) for b in (0.30, 0.20, 0.10)]
    assert phs == [0, 99, 99], f"PH 阶跃失败: {phs} (期望 [0,99,99])"
    # 严格: 偏置递减 PH 非减
    assert phs[2] >= phs[1] >= phs[0]


def test_partial_horizon():
    """前半 outside、后半 inside → PH = 后半长度 (首次持续 inside 至 EOL)。"""
    rt = _linear_rul(100)
    rp = rt.copy()
    rp[:50] = 1.5 * rt[:50]    # 前 50 窗偏置 50% (outside α=0.2 带)
    ph = prognostic_horizon(rp, rt, alpha=0.2)
    assert ph == 49, f"部分 PH 应=49 (后 49 窗持续 inside), 实={ph}"


def test_ra_cm_bounds():
    """RA ≤ 1, CM ≥ 0 边界 (小偏置不越界)。"""
    rt = _linear_rul(50)
    rp = rt + 3.0
    assert relative_accuracy(rp, rt, lam=0.5) <= 1.0
    assert convergence_metric(rp, rt, alpha=0.2) >= 0.0
