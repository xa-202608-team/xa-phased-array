"""源/目标共享的物理应力构造（Gate 1.1 / Gate 2 跨域同坐标系）。

公式与 ``gan_rfalt_sim.integrate_damage_states`` 的 ``temp_factor`` / ``stress_factor``
逐字一致（``gan_rfalt_sim.py:115/117``，已核实），确保源域 Fθ 学到的 (u→Δz) 映射
在目标域可物理解释。目标域缺逐时刻 Pin/VSWR 时传**预注册物理基准**
（``pin=35`` → 压缩项归零、``vswr=1`` → 驻波项归零，故 ``s=duty``），基准值须在跑
Gate 2 前冻结，不得在看到结果后回改（GPT §5）。结果 JSON 须记录
``physical_stress_schema`` 与目标域 Pin/VSWR 假设。
"""
from __future__ import annotations

import numpy as np

PHYSICAL_STRESS_SCHEMA = "gan_physical_stress_v1"
# 目标域 Pin/VSWR 缺失时的预注册基准（s 退化为 duty；不得在看到 Gate 2 结果后修改）。
TARGET_PIN_DBM_REFERENCE = 35.0
TARGET_VSWR_REFERENCE = 1.0


def build_physical_stress(
    tj_c: np.ndarray,
    duty: np.ndarray,
    *,
    pin_dbm: np.ndarray | float,
    vswr: np.ndarray | float,
    recovery: np.ndarray,
) -> np.ndarray:
    """构造 ``[a_T, s, recovery]`` 物理应力，与 RFALT 仿真器同坐标系。

    参数:
        tj_c: 结温 °C。源域用 sim 自洽 ``t_j``（含 RF 自热·(1+r_th) 反馈），
            即 RFALT 特征的 ``T_j_C`` 列；目标域用 ``x_global`` 的 ``Tj`` 列。
        duty: 占空比 [0,1]。源域用 ``duty_cycle`` 列；目标域用 ``physical_duty`` 辅助量。
        pin_dbm: 输入功率 dBm。源域用 ``Pin_dBm`` 列；目标域缺则传
            ``TARGET_PIN_DBM_REFERENCE``（35.0）。
        vswr: 驻波比。源域用 ``VSWR`` 列；目标域缺则传 ``TARGET_VSWR_REFERENCE``（1.0）。
        recovery: 释放工况标志（布尔或 0/1）。源域 ``(duty<0.10)``；目标域恒 0
            （LEO 无释放段）。

    返回:
        (..., 3) float32 数组，列序 ``[a_T, s, recovery]``。
    """
    tj_c = np.asarray(tj_c, dtype=np.float64)
    duty = np.asarray(duty, dtype=np.float64)
    pin_dbm = np.asarray(pin_dbm, dtype=np.float64)
    vswr = np.asarray(vswr, dtype=np.float64)
    # temp_factor: gan_rfalt_sim.py:115
    a_t = np.clip(np.exp((tj_c - 95.0) / 42.0), 0.20, 8.0)
    # stress_factor: gan_rfalt_sim.py:117
    compression = np.maximum(pin_dbm - 35.0, 0.0)
    mismatch = np.maximum(vswr - 1.0, 0.0)
    s = duty * (1.0 + 0.18 * compression + 0.25 * mismatch)
    recovery = np.asarray(recovery, dtype=np.float64)
    return np.stack([a_t, s, recovery], axis=-1).astype(np.float32)
