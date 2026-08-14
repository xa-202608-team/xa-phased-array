#!/usr/bin/env python3
"""Build real NASA MOSFET source features.

Feature dimension is exactly two:
    [RDS_drift, T_case_C]

Vth, gm and leakage current are intentionally absent because the fixed-gate
waveforms do not contain a VGS sweep from which they could be identified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .nasa_mat_common import (
    DataValidationError,
    apply_temperature_correction,
    build_fixed_threshold_labels,
    concatenate_device_runs,
    parse_mosfet_ids,
    parse_mosfet_run,
    robust_linear_temperature_fit,
    robust_physical_filter,
    write_feature_h5,
)


def build_case(
    files: list[Path],
    threshold: float,
    persistence_points: int,
    strict_temperature_sign: bool,
) -> pd.DataFrame:
    frames = [parse_mosfet_run(path) for path in files]
    frame = concatenate_device_runs(frames)
    if frame.empty:
        raise DataValidationError("No valid transient RDS measurements.")

    # Filter physically implausible RDS: failed ON-state extraction on
    # end-of-life waveforms can yield ~20 Ω spikes (IRF520N healthy ~0.18 Ω,
    # fully failed ~1-2 Ω). Bounds + log-MAD remove run-tail garbage before
    # the temperature fit, otherwise outliers corrupt slope and baseline.
    keep = robust_physical_filter(
        frame["RDS_raw_ohm"].to_numpy(),
        lower_bound=0.005,
        upper_bound=5.0,
    )
    frame = frame[keep].reset_index(drop=True)
    if frame.empty:
        raise DataValidationError("No RDS points survive physical outlier filter.")

    fit = robust_linear_temperature_fit(
        frame["RDS_raw_ohm"].to_numpy(),
        frame["T_case_C"].to_numpy(),
    )
    if strict_temperature_sign and fit.slope_per_c <= 0:
        raise DataValidationError(
            f"dRDS/dT={fit.slope_per_c:.6g} Ω/°C is not positive."
        )

    corrected = apply_temperature_correction(
        frame["RDS_raw_ohm"].to_numpy(),
        frame["T_case_C"].to_numpy(),
        fit,
    )
    n_base = max(5, int(np.ceil(0.05 * len(frame))))
    early = corrected[:n_base]
    early_valid = np.sort(early[np.isfinite(early) & (early > 0)])
    if early_valid.size >= 10:
        trim = max(1, early_valid.size // 10)
        r0 = float(np.median(early_valid[trim:-trim]))
    else:
        r0 = float(np.nanmedian(early))
    if not np.isfinite(r0) or r0 <= 0:
        raise DataValidationError(f"Invalid corrected healthy RDS baseline: {r0}")
    drift = (corrected - r0) / r0

    labels = build_fixed_threshold_labels(
        frame["elapsed_time_s"].to_numpy(),
        drift,
        threshold=threshold,
        persistence_points=persistence_points,
    )
    frame["RDS_corrected_ohm"] = corrected
    frame["RDS_drift"] = drift
    frame["hi"] = labels["hi"]
    # rul_s 统一用 lower_bound: 失效器件=精确 rul (lower_bound==exact_rul), 删失器件=下界
    # (build_fixed_threshold_labels 的 exact_rul 对删失器件是 NaN, 直接写会让下游训练出 NaN)
    frame["rul_s"] = labels["rul_lower_bound_s"]
    frame["rul_lower_bound_s"] = labels["rul_lower_bound_s"]
    frame["event_observed"] = labels["event_observed"]
    frame["eol_time_s"] = labels["eol_time_s"]
    frame["censor_time_s"] = labels["censor_time_s"]
    frame["temperature_slope_per_c"] = fit.slope_per_c
    frame["reference_temperature_c"] = fit.reference_temperature_c
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-h5", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--persistence-points", type=int, default=3)
    parser.add_argument(
        "--allow-nonpositive-temp-coefficient",
        action="store_true",
        help="Not recommended; keeps a case even if early dRDS/dT is non-positive.",
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
            devices[case_id] = build_case(
                case_files,
                threshold=args.threshold,
                persistence_points=args.persistence_points,
                strict_temperature_sign=not args.allow_nonpositive_temp_coefficient,
            )
        except Exception as exc:
            errors[case_id] = repr(exc)

    if not devices:
        raise SystemExit(f"No valid cases. Errors: {errors}")

    metadata = {
        "dataset_id": "NASA_MOSFET_Thermal_Overstress",
        "schema_version": "2.0",
        "feature_names": ["RDS_drift", "T_case_C"],
        "feature_dim": 2,
        "failure_feature": "RDS_drift",
        "failure_threshold": args.threshold,
        "persistence_points": args.persistence_points,
        "group_split_key": "case_id",
        "right_censoring": True,
        "temperature_semantics": "package/case temperature, not junction temperature",
        "excluded_features": ["Vth", "gm", "I_leak"],
        "n_valid_cases": len(devices),
        "n_failed_cases": len(errors),
        "errors": errors,
    }
    write_feature_h5(
        args.out_h5,
        devices,
        feature_columns=["RDS_drift", "T_case_C"],
        metadata=metadata,
    )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(
            [frame.assign(device_id=device_id) for device_id, frame in devices.items()],
            ignore_index=True,
        )
        combined.to_csv(args.out_csv, index=False)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
