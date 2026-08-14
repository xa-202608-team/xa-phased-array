"""工况分段工具（Phase 1 修复核心模块）。

用于检测 MOSFET/IGBT 数据中的工况切换点，并按工况分段处理。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class OperatingRegime:
    """单个工况段的元数据。"""
    regime_id: str
    start_idx: int
    end_idx: int
    supply_V: float
    temp_low: float
    temp_high: float
    duration_s: float
    n_points: int


def detect_operating_regimes(
    df: pd.DataFrame,
    supply_col: str = "supply_V",
    temp_low_col: str = "temp_low",
    temp_high_col: str = "temp_high",
    time_col: str = "elapsed_time_s",
    supply_threshold: float = 0.5,
    temp_threshold: float = 10.0,
    min_regime_points: int = 10,
) -> list[OperatingRegime]:
    """检测工况切换点并分段。

    Args:
        df: 包含供电、温控设定、时间的 DataFrame
        supply_col: 供电电压列名
        temp_low_col: 温控下限列名
        temp_high_col: 温控上限列名
        time_col: 时间列名
        supply_threshold: 供电电压变化阈值（V）
        temp_threshold: 温控变化阈值（°C）
        min_regime_points: 最小工况段点数

    Returns:
        工况段列表（按时间排序）
    """
    # 检查必需列
    required_cols = [supply_col, temp_low_col, temp_high_col, time_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 检测供电电压变化点
    supply_changes = []
    if supply_col in df.columns:
        supply_diff = df[supply_col].diff().abs()
        supply_changes = supply_diff[supply_diff > supply_threshold].index.tolist()

    # 检测温控变化点
    temp_changes = []
    if temp_low_col in df.columns and temp_high_col in df.columns:
        temp_diff_low = df[temp_low_col].diff().abs()
        temp_diff_high = df[temp_high_col].diff().abs()
        temp_changes = (temp_diff_low[temp_diff_low > temp_threshold].index |
                        temp_diff_high[temp_diff_high > temp_threshold].index).tolist()

    # 合并所有切分点
    split_points = sorted(set([0] + supply_changes + temp_changes + [len(df)]))

    # 生成分段
    regimes = []
    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i + 1]
        segment = df.iloc[start_idx:end_idx]

        if len(segment) < min_regime_points:
            continue

        # 提取该段的工况参数
        supply_V = segment[supply_col].median()
        temp_low = segment[temp_low_col].median()
        temp_high = segment[temp_high_col].median()
        duration_s = segment[time_col].max() - segment[time_col].min()

        regime_id = f"V{int(supply_V)}_T{int(temp_low)}_{int(temp_high)}"

        regimes.append(OperatingRegime(
            regime_id=regime_id,
            start_idx=start_idx,
            end_idx=end_idx,
            supply_V=supply_V,
            temp_low=temp_low,
            temp_high=temp_high,
            duration_s=duration_s,
            n_points=len(segment),
        ))

    return regimes


def match_regime_metadata(
    df: pd.DataFrame,
    pwm_data: list[dict],
    time_col: str = "elapsed_time_s",
) -> pd.DataFrame:
    """从 pwmTempControllerState 匹配工况元数据到主数据框。

    Args:
        df: 主数据框（包含 transient RDS 数据）
        pwm_data: pwmTempControllerState 数据列表（每个元素是一个 dict）
        time_col: 时间列名

    Returns:
        添加了 supply_V / temp_low / temp_high 列的 DataFrame
    """
    if not pwm_data:
        # 如果没有 pwm 数据，返回默认值
        df["supply_V"] = np.nan
        df["temp_low"] = np.nan
        df["temp_high"] = np.nan
        return df

    # 构建 pwm 时间序列
    pwm_df = pd.DataFrame(pwm_data)

    # 最近邻匹配
    supply_matched = []
    temp_low_matched = []
    temp_high_matched = []

    for t in df[time_col]:
        # 找到最近的 pwm 状态
        idx = (pwm_df["time_s"] - t).abs().idxmin()
        supply_matched.append(pwm_df.loc[idx, "supply_V"])
        temp_low_matched.append(pwm_df.loc[idx, "lowTemp"])
        temp_high_matched.append(pwm_df.loc[idx, "highTemp"])

    df["supply_V"] = supply_matched
    df["temp_low"] = temp_low_matched
    df["temp_high"] = temp_high_matched

    return df


def regime_conditioned_temperature_correction(
    df: pd.DataFrame,
    regimes: list[OperatingRegime],
    rds_col: str = "RDS_raw_ohm",
    temp_col: str = "T_case_C",
) -> pd.DataFrame:
    """工况条件化温度校正（核心修复）。

    在每个工况段内独立拟合温度系数，避免跨工况污染。

    Args:
        df: 包含 RDS 和温度的数据框
        regimes: 工况段列表
        rds_col: RDS 列名
        temp_col: 温度列名

    Returns:
        添加了 RDS_corrected / RDS_baseline / RDS_drift 列的 DataFrame
    """
    df["RDS_corrected"] = np.nan
    df["RDS_baseline"] = np.nan

    for regime in regimes:
        # 提取该工况段的数据
        segment = df.iloc[regime.start_idx:regime.end_idx]

        if len(segment) < 10:
            continue

        rds_raw = segment[rds_col].to_numpy()
        temp_raw = segment[temp_col].to_numpy()

        # 过滤无效值
        valid = np.isfinite(rds_raw) & np.isfinite(temp_raw) & (rds_raw > 0)
        if valid.sum() < 10:
            continue

        rds_valid = rds_raw[valid]
        temp_valid = temp_raw[valid]

        # 在该工况内拟合温度系数
        # 简单线性回归：RDS = a + b * T
        try:
            # 使用稳健拟合（中位数 + MAD）
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                temp_valid, rds_valid
            )

            # 温度校正到该工况的平均温度
            T_ref = np.median(temp_valid)
            corrected = rds_raw - slope * (temp_raw - T_ref)

            # 基线：该工况前 10% 截尾中位数（非全局前 20%）
            n_base = max(5, int(0.1 * len(corrected)))
            early = corrected[:n_base]
            early_valid = early[np.isfinite(early) & (early > 0)]

            if len(early_valid) >= 5:
                # 10% 截尾中位数
                trim = max(1, len(early_valid) // 10)
                r0 = float(np.median(np.sort(early_valid)[trim:-trim]))
            else:
                r0 = float(np.nanmedian(early_valid))

            if not np.isfinite(r0) or r0 <= 0:
                # 退化到该工况的中位数
                r0 = float(np.nanmedian(corrected[np.isfinite(corrected) & (corrected > 0)]))

            # 写入结果
            df.loc[regime.start_idx:regime.end_idx - 1, "RDS_corrected"] = corrected
            df.loc[regime.start_idx:regime.end_idx - 1, "RDS_baseline"] = r0

        except Exception as exc:
            # 拟合失败，跳过该段
            print(f"Warning: Temperature fit failed for regime {regime.regime_id}: {exc}")
            continue

    # P1 修复: 跳过的段 (valid<10 或 linregress 失败) corrected/baseline 留 NaN,
    # 用 rds_raw + 全局健康基线兜底, 防 z-score mean=NaN 致 pretrain 全 NaN
    nan_mask = df["RDS_corrected"].isna()
    if nan_mask.any():
        good = df.loc[~nan_mask, "RDS_corrected"].dropna()
        global_r0 = float(good.iloc[:max(5, int(0.1 * len(good)))].median()) if len(good) else 1.0
        df.loc[nan_mask, "RDS_corrected"] = df.loc[nan_mask, rds_col]
        df.loc[nan_mask, "RDS_baseline"] = global_r0
    # 计算相对漂移（跨工况可比）
    df["RDS_drift"] = (df["RDS_corrected"] - df["RDS_baseline"]) / df["RDS_baseline"]

    return df


def visualize_regimes(
    df: pd.DataFrame,
    regimes: list[OperatingRegime],
    rds_col: str = "RDS_raw_ohm",
    time_col: str = "elapsed_time_s",
    out_path: str | None = None,
):
    """可视化工况分段结果（调试用）。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # 1. RDS 原始值（标记工况切换点）
    ax = axes[0]
    ax.plot(df[time_col], df[rds_col], "o-", markersize=2, label="RDS_raw")

    # 标记工况切换
    for regime in regimes:
        ax.axvline(df[time_col].iloc[regime.start_idx], color="red", linestyle="--", alpha=0.5)

    ax.set_ylabel("RDS (Ω)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"工况分段：{len(regimes)} 段")

    # 2. 温度
    ax = axes[1]
    if "T_case_C" in df.columns:
        ax.plot(df[time_col], df["T_case_C"], "o-", markersize=2, color="red", label="T_case")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 供电电压
    ax = axes[2]
    if "supply_V" in df.columns:
        ax.plot(df[time_col], df["supply_V"], "o-", markersize=2, color="green", label="Supply_V")
    ax.set_ylabel("Supply Voltage (V)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {out_path}")
    else:
        plt.show()

    plt.close()