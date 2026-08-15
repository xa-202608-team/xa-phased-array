#!/usr/bin/env python3
"""Build real NASA IGBT auxiliary source features.

Default source is transient ON-state VCE(sat), because node2Voltage is not
trusted until validate_nasa_electronics.py confirms it against transient VCE.

Feature dimension is two:
    [VCE_drift, T_case_C]

The single DC-gate MAT file is exported only as OOD metadata and is excluded
from RUL training.
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
    concatenate_device_runs,
    nearest_values,
    parse_igbt_device_id,
    parse_igbt_steady,
    parse_igbt_transient,
    read_node2_validation_status,
    robust_linear_temperature_fit,
    robust_physical_filter,
    write_feature_h5,
)


def build_device_transient(files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in files:
        transient = parse_igbt_transient(path)
        steady = parse_igbt_steady(path)
        if transient.empty:
            continue
        if not steady.empty:
            tcase, gaps = nearest_values(
                steady["time_abs_s"].to_numpy(),
                steady["T_case_C"].to_numpy(),
                transient["time_abs_s"].to_numpy(),
            )
            transient["T_case_C"] = tcase
            transient["temperature_match_gap_s"] = gaps
        else:
            transient["T_case_C"] = np.nan
            transient["temperature_match_gap_s"] = np.nan
        transient = transient.rename(
            columns={"VCE_sat_transient_V": "VCE_raw_V"}
        )
        frames.append(transient)
    return concatenate_device_runs(frames)


def build_device_node2(files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in files:
        steady = parse_igbt_steady(path)
        if steady.empty:
            continue
        steady = steady.rename(columns={"node2Voltage_V": "VCE_raw_V"})
        frames.append(steady)
    return concatenate_device_runs(frames)


def add_precursor_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(subset=["VCE_raw_V", "T_case_C"]).copy()
    if len(frame) < 10:
        raise DataValidationError("Too few finite IGBT VCE/Tcase points.")

    # Filter physically implausible VCE (failed ON-state extraction on
    # degraded waveforms). IGBT VCE(sat) typically 1-3 V; bounds + log-MAD
    # remove tail garbage before the temperature fit.
    keep = robust_physical_filter(
        frame["VCE_raw_V"].to_numpy(),
        lower_bound=0.05,
        upper_bound=10.0,
    )
    frame = frame[keep].reset_index(drop=True)
    if len(frame) < 10:
        raise DataValidationError("Too few IGBT VCE points after outlier filter.")

    fit = robust_linear_temperature_fit(
        frame["VCE_raw_V"].to_numpy(),
        frame["T_case_C"].to_numpy(),
        min_points=min(20, max(8, len(frame) // 3)),
        min_temperature_span_c=1.0,
    )
    corrected = apply_temperature_correction(
        frame["VCE_raw_V"].to_numpy(),
        frame["T_case_C"].to_numpy(),
        fit,
    )
    n_base = max(5, int(np.ceil(0.05 * len(frame))))
    v0 = float(np.nanmedian(corrected[:n_base]))
    if not np.isfinite(v0) or abs(v0) < 1e-9:
        raise DataValidationError(f"Invalid IGBT VCE baseline: {v0}")

    frame["VCE_corrected_V"] = corrected
    frame["VCE_drift"] = (corrected - v0) / abs(v0)

    # No universal IGBT EOL threshold is invented here. IGBT is auxiliary:
    # use it for representation learning / trend regularization unless a
    # dataset-specific terminal label is later validated.
    frame["hi"] = np.nan
    frame["rul_s"] = np.nan
    frame["rul_lower_bound_s"] = np.maximum(
        frame["elapsed_time_s"].max() - frame["elapsed_time_s"], 0.0
    )
    frame["event_observed"] = False
    frame["eol_time_s"] = np.nan
    frame["censor_time_s"] = float(frame["elapsed_time_s"].max())
    frame["temperature_slope_per_c"] = fit.slope_per_c
    frame["reference_temperature_c"] = fit.reference_temperature_c
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-h5", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument(
        "--vce-source",
        choices=["transient", "node2"],
        default="transient",
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        help="Required when --vce-source node2; must report overall_status=confirmed.",
    )
    args = parser.parse_args()

    if args.vce_source == "node2":
        if not args.validation_json:
            raise SystemExit("--validation-json is required for node2.")
        status = read_node2_validation_status(args.validation_json)
        if status != "confirmed":
            raise SystemExit(
                f"node2Voltage is not enabled: validation status is {status!r}."
            )

    all_files = sorted(args.input_root.rglob("*.mat"))
    square_files = [
        p for p in all_files if parse_igbt_device_id(p) not in {"Device_DC", "Device_unknown"}
    ]
    if not square_files:
        raise SystemExit(f"No IGBT square-gate MAT files under {args.input_root}")

    by_device: dict[str, list[Path]] = {}
    for path in square_files:
        by_device.setdefault(parse_igbt_device_id(path), []).append(path)

    devices: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for device_id, files in sorted(by_device.items()):
        try:
            frame = (
                build_device_transient(files)
                if args.vce_source == "transient"
                else build_device_node2(files)
            )
            devices[device_id] = add_precursor_features(frame)
        except Exception as exc:
            errors[device_id] = repr(exc)

    if not devices:
        raise SystemExit(f"No valid IGBT devices. Errors: {errors}")

    metadata = {
        "dataset_id": "NASA_IGBT_Accelerated_Aging",
        "schema_version": "2.0",
        "feature_names": ["VCE_drift", "T_case_C"],
        "feature_dim": 2,
        "vce_source": args.vce_source,
        "group_split_key": "device_id",
        "supervision_role": "auxiliary_unlabeled_or_trend_regularization",
        "dc_device_policy": "ood_only",
        "right_censoring": True,
        "warning": (
            "No universal VCE-drift failure threshold is hard-coded. "
            "Do not train an exact IGBT RUL head until a dataset-specific "
            "terminal criterion is validated."
        ),
        "n_valid_devices": len(devices),
        "n_failed_devices": len(errors),
        "errors": errors,
    }
    write_feature_h5(
        args.out_h5,
        devices,
        feature_columns=["VCE_drift", "T_case_C"],
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
