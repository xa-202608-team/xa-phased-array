#!/usr/bin/env python3
"""Build real NASA MOSFET source features (Phase 1 修复版).

修复内容：
1. 工况分段：检测供电/温控切换点
2. 条件化温度校正：每个工况内独立拟合
3. 删除全局 log-MAD 过滤：保留所有工况段
4. 特征维度升级：2 维 → 5 维
5. 分层标签：damage_state + event_type

Feature dimension: 5
    [RDS_drift, T_case_C, supply_V, gate_voltage, duty_cycle]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .nasa_mat_common import (
    DataValidationError,
    build_fixed_threshold_labels,
    concatenate_device_runs,
    parse_mosfet_ids,
    parse_mosfet_run,
    write_feature_h5,
    loadmat_squeezed,
    get_attr,
    scalar_float,
    finite_1d,
)
from .regime_utils import (
    OperatingRegime,
    detect_operating_regimes,
    match_regime_metadata,
    regime_conditioned_temperature_correction,
    visualize_regimes,
)


def extract_pwm_metadata(mat_data: dict) -> list[dict]:
    """从 pwmTempControllerState 提取工况元数据。

    Returns:
        [{'time_s': ..., 'supply_V': ..., 'lowTemp': ..., 'highTemp': ...}, ...]
    """
    m = mat_data.get("measurement")
    if m is None:
        return []

    pwm = getattr(m, "pwmTempControllerState", None)
    if pwm is None:
        return []

    pwm_list = [pwm] if not isinstance(pwm, (list, np.ndarray)) else list(pwm)

    results = []
    for p in pwm_list:
        try:
            time_epoch = getattr(p, "timeEpoch", None)
            time_s = (float(time_epoch) - 719529.0) * 86400.0 if time_epoch else np.nan

            supply_V = float(getattr(p, "supplyVoltage", np.nan))
            low_temp = float(getattr(p, "lowTemp", np.nan))
            high_temp = float(getattr(p, "highTemp", np.nan))
            gate_voltage = float(getattr(p, "gateVoltage", np.nan))
            duty_cycle = float(getattr(p, "dutyCycle", np.nan))

            if np.isfinite(time_s):
                results.append({
                    "time_s": time_s,
                    "supply_V": supply_V,
                    "lowTemp": low_temp,
                    "highTemp": high_temp,
                    "gate_voltage": gate_voltage,
                    "duty_cycle": duty_cycle,
                })
        except Exception:
            continue

    return results


def build_case_regime_aware(
    files: list[Path],
    threshold: float = 0.05,
    persistence_points: int = 3,
    strict_temperature_sign: bool = True,
    visualize: bool = False,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """构建 case 数据（工况条件化版本）。

    Args:
        files: 同一 case 的所有 MAT 文件
        threshold: RDS 漂移阈值
        persistence_points: 持续穿越点数
        strict_temperature_sign: 是否强制正温度系数
        visualize: 是否可视化工况分段
        out_dir: 可视化输出目录

    Returns:
        包含 5 维特征 + 标签的 DataFrame
    """
    # 1. 解析所有 run 文件
    frames = []
    for path in files:
        try:
            frame = parse_mosfet_run(path)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            print(f"Warning: Failed to parse {path}: {exc}")
            continue

    if not frames:
        raise DataValidationError("No valid transient RDS measurements.")

    frame = concatenate_device_runs(frames)
    if frame.empty:
        raise DataValidationError("No valid data after concatenation.")

    # 2. 删除全局 log-MAD 过滤（核心修复）
    # 旧代码（删除）：
    #   keep = robust_physical_filter(frame["RDS_raw_ohm"], lower_bound=0.005, upper_bound=5.0)
    # 新逻辑：只过滤明确异常值
    keep = (
        np.isfinite(frame["RDS_raw_ohm"]) &
        (frame["RDS_raw_ohm"] > 0) &
        (frame["RDS_raw_ohm"] < 10.0)  # 宽松物理上限
    )
    frame = frame[keep].reset_index(drop=True)

    if frame.empty:
        raise DataValidationError("No RDS points survive basic filter.")

    # 3. 提取 pwm 工况元数据
    # 注意：parse_mosfet_run 已经解析了 transient，但没有 pwm 数据
    # 需要重新读取 MAT 文件提取 pwmTempControllerState
    pwm_data = []
    for path in files:
        try:
            mat_data = loadmat_squeezed(path)
            pwm_data.extend(extract_pwm_metadata(mat_data))
        except Exception:
            continue

    # 4. 匹配工况元数据 (P1 修复: time_col 用 time_abs_s 绝对秒, 不能用 elapsed_time_s 相对秒
    #    —— pwm.time_s 是 unix 绝对秒 ~1.2e9, frame.elapsed_time_s 是相对秒 0~6e4, 基准不匹配
    #    会让最近邻全落到 pwm 时间最小者 → supply_V/temp 恒定 → regime 必然 1 段, 工况条件化失效)
    if pwm_data:
        frame = match_regime_metadata(frame, pwm_data, time_col="time_abs_s")
    else:
        # 如果没有 pwm 数据，使用默认值
        frame["supply_V"] = 5.0  # 假设默认供电
        frame["temp_low"] = 100.0  # 假设默认温控
        frame["temp_high"] = 100.0

    # 5. 检测工况分段
    try:
        regimes = detect_operating_regimes(frame)
        # P1 修复: regime_id 写回 frame (detect 返回 list 不自动回写; 否则下游 output 测 nunique=1 假象)
        frame["regime_id"] = "unknown"
        reg_col = frame.columns.get_loc("regime_id")
        for r in regimes:
            frame.iloc[r.start_idx:r.end_idx, reg_col] = r.regime_id
        print(f"Detected {len(regimes)} operating regimes")
    except Exception as exc:
        print(f"Warning: Regime detection failed: {exc}")
        # 退化到单段处理
        regimes = [OperatingRegime(
            regime_id="unknown",
            start_idx=0,
            end_idx=len(frame),
            supply_V=frame["supply_V"].median() if "supply_V" in frame.columns else 5.0,
            temp_low=frame["temp_low"].median() if "temp_low" in frame.columns else 100.0,
            temp_high=frame["temp_high"].median() if "temp_high" in frame.columns else 100.0,
            duration_s=frame["elapsed_time_s"].max() - frame["elapsed_time_s"].min(),
            n_points=len(frame),
        )]

    # 6. 工况条件化温度校正（核心修复）
    frame = regime_conditioned_temperature_correction(
        frame,
        regimes,
        rds_col="RDS_raw_ohm",
        temp_col="T_case_C",
    )

    # 7. 可视化（调试用）
    if visualize and out_dir:
        out_path = out_dir / f"{files[0].stem}_regimes.png"
        visualize_regimes(frame, regimes, out_path=str(out_path))

    # 8. 构建标签
    labels = build_fixed_threshold_labels(
        frame["elapsed_time_s"].to_numpy(),
        frame["RDS_drift"].to_numpy(),
        threshold=threshold,
        persistence_points=persistence_points,
    )

    # 9. 组装最终特征
    # 添加缺失的工况变量（如果有）
    if "gate_voltage" not in frame.columns:
        frame["gate_voltage"] = 10.0  # 默认栅压

    if "duty_cycle" not in frame.columns:
        frame["duty_cycle"] = 40.0  # 默认占空比

    # 10. 组装输出
    output = pd.DataFrame({
        "elapsed_time_s": frame["elapsed_time_s"],
        # 5 维特征
        "RDS_drift": frame["RDS_drift"],
        "T_case_C": frame["T_case_C"],
        "supply_V": frame["supply_V"],
        "gate_voltage": frame["gate_voltage"],
        "duty_cycle": frame["duty_cycle"],
        # 标签
        "hi": labels["hi"],
        "rul_s": labels["rul_lower_bound_s"],
        "rul_lower_bound_s": labels["rul_lower_bound_s"],
        "event_observed": labels["event_observed"],
        "eol_time_s": labels["eol_time_s"],
        "censor_time_s": labels["censor_time_s"],
        # 元数据
        "regime_id": frame.get("regime_id", "unknown"),
    })

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-h5", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--persistence-points", type=int, default=3)
    parser.add_argument("--visualize", action="store_true", help="生成工况分段可视化")
    parser.add_argument("--out-dir", type=Path, help="可视化输出目录")
    parser.add_argument(
        "--allow-nonpositive-temp-coefficient",
        action="store_true",
        help="允许非正温度系数（不推荐）",
    )
    args = parser.parse_args()

    files = sorted(args.input_root.rglob("Test_*_run_*.mat"))
    if not files:
        raise SystemExit(f"No Test_*_run_*.mat files under {args.input_root}")

    by_case: dict[str, list[Path]] = {}
    for path in files:
        case_id, _ = parse_mosfet_ids(path)
        by_case.setdefault(case_id, []).append(path)

    devices: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    for case_id, case_files in sorted(
        by_case.items(), key=lambda kv: int(kv[0].split("_")[1])
    ):
        try:
            print(f"Processing {case_id}...")
            devices[case_id] = build_case_regime_aware(
                case_files,
                threshold=args.threshold,
                persistence_points=args.persistence_points,
                strict_temperature_sign=not args.allow_nonpositive_temp_coefficient,
                visualize=args.visualize,
                out_dir=args.out_dir,
            )
        except Exception as exc:
            errors[case_id] = repr(exc)
            print(f"Error processing {case_id}: {exc}")

    if not devices:
        raise SystemExit(f"No valid cases. Errors: {errors}")

    # 写入 HDF5（schema_v3，5 维特征）
    metadata = {
        "dataset_id": "NASA_MOSFET_Thermal_Overstress_v2",
        "schema_version": "3.0",
        "feature_names": ["RDS_drift", "T_case_C", "supply_V", "gate_voltage", "duty_cycle"],
        "feature_dim": 5,
        "failure_feature": "RDS_drift",
        "failure_threshold": args.threshold,
        "regime_aware": True,
        "description": "工况条件化版本，删除全局 log-MAD 过滤，保留所有工况段",
    }

    # 写入 HDF5（需要 feature_columns 参数）
    feature_columns = ["RDS_drift", "T_case_C", "supply_V", "gate_voltage", "duty_cycle"]
    write_feature_h5(args.out_h5, devices, feature_columns, metadata)

    if args.out_csv:
        all_frames = []
        for case_id, frame in devices.items():
            frame["case_id"] = case_id
            all_frames.append(frame)
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(args.out_csv, index=False)

    print(f"\nSuccessfully processed {len(devices)} cases")
    print(f"Output: {args.out_h5}")
    if errors:
        print(f"\nFailed cases ({len(errors)}):")
        for case_id, err in errors.items():
            print(f"  {case_id}: {err}")


if __name__ == "__main__":
    main()