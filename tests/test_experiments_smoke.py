"""实验冒烟测试 (plan P6): 指标函数 + 物理外推逻辑。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.physical_extrap import physical_extrap_rul, phm_score, rmse, mae   # noqa: E402


def test_rmse_mae_basic():
    assert abs(rmse([1, 2, 3], [1, 2, 3])) < 1e-9
    assert mae([1, 2], [3, 4]) == 2.0


def test_phm_score_penalizes_late_prediction():
    """晚预测 (pred>true) 的 PHM 分数应高于早预测 (重罚晚预测)。"""
    s_late = phm_score([10.0], [5.0])     # 晚预测 5
    s_early = phm_score([5.0], [10.0])    # 早预测 5
    assert s_late > s_early


def test_physical_extrap_recovers_decreasing_rul():
    """线性退化 b̂ 外推: RUL 应为正且随时间递减。"""
    t = np.arange(100, dtype=float)
    b_hat = 0.01 * t + 1.0                # 线性增, 到 t=100 达 2.0
    rul = physical_extrap_rul(b_hat, t, b_fail=2.0, window=20)
    assert not np.isnan(rul[50]) and rul[50] > 0
    assert rul[80] < rul[20], "RUL 应随时间递减"


def test_physical_extrap_handles_flat():
    """平坦 b̂ (slope≈0) 应返回 NaN (无法外推)。"""
    t = np.arange(50, dtype=float)
    b_hat = np.ones(50)
    rul = physical_extrap_rul(b_hat, t, b_fail=2.0, window=20)
    assert np.all(np.isnan(rul))
