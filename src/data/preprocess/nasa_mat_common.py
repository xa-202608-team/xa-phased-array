"""Shared utilities for NASA MOSFET / IGBT accelerated-aging MAT files.

The implementation is intentionally conservative:
- MOSFET RDS(ON) is extracted only from transient ON-state waveforms.
- IGBT VCE(sat) is extracted from transient ON-state waveforms by default.
- steadyState.node2Voltage is gated behind an explicit validation report.
- MATLAB datenum is used through elapsed-time differences, not row indices.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence
import json
import math
import re
import warnings

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio


MATLAB_UNIX_EPOCH_DAYS = 719529.0


class DataValidationError(RuntimeError):
    """Raised when a required physical/data validation fails."""


def loadmat_squeezed(path: str | Path) -> dict[str, Any]:
    """Load a pre-v7.3 MAT file and remove MATLAB metadata variables."""
    data = sio.loadmat(
        str(path),
        struct_as_record=False,
        squeeze_me=True,
    )
    return {k: v for k, v in data.items() if not k.startswith("__")}


def as_sequence(value: Any) -> list[Any]:
    """Convert a scalar or MATLAB struct array to a flat Python list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()]
        return list(value.reshape(-1))
    return [value]


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def first_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        value = get_attr(obj, name, None)
        if value is not None:
            return value
    return default


def get_nested(obj: Any, path: str, default: Any = None) -> Any:
    current = obj
    for part in path.split("."):
        current = get_attr(current, part, None)
        if current is None:
            return default
    return current


def finite_1d(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=float)
    return arr[np.isfinite(arr)]


def scalar_float(value: Any, default: float = np.nan) -> float:
    arr = finite_1d(value)
    return float(arr[0]) if arr.size else float(default)


def matlab_datenum_to_unix_seconds(value: Any) -> float:
    """Convert MATLAB datenum to Unix seconds.

    Absolute Unix time is useful for cross-run ordering. Elapsed time is always
    computed by subtracting the first valid timestamp, so the absolute epoch
    offset does not affect RUL.
    """
    day = scalar_float(value)
    if not np.isfinite(day):
        return np.nan
    return (day - MATLAB_UNIX_EPOCH_DAYS) * 86400.0


def largest_contiguous_block(indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return indices
    split_at = np.where(np.diff(indices) > 1)[0] + 1
    blocks = np.split(indices, split_at)
    return max(blocks, key=len)


def on_state_mask(
    gate_signal: Any,
    main_current: Any,
    current_fraction: float = 0.10,
    edge_trim_fraction: float = 0.05,
    min_points: int = 8,
) -> np.ndarray:
    gate = np.asarray(gate_signal, dtype=float).reshape(-1)
    current = np.asarray(main_current, dtype=float).reshape(-1)
    n = min(gate.size, current.size)
    gate, current = gate[:n], current[:n]
    finite = np.isfinite(gate) & np.isfinite(current)
    if finite.sum() < min_points:
        raise DataValidationError("Too few finite gate/current samples.")

    gmin = float(np.nanmin(gate[finite]))
    gmax = float(np.nanmax(gate[finite]))
    imax = float(np.nanmax(current[finite]))
    if not np.isfinite(gmin + gmax + imax) or gmax <= gmin or imax <= 0:
        raise DataValidationError("Invalid gate/current waveform range.")

    midpoint = 0.5 * (gmin + gmax)
    raw = finite & (gate >= midpoint) & (current >= current_fraction * imax)
    idx = largest_contiguous_block(np.flatnonzero(raw))
    if idx.size < min_points:
        raise DataValidationError(
            f"ON-state block too short: {idx.size} < {min_points}."
        )

    trim = int(math.floor(edge_trim_fraction * idx.size))
    if trim > 0 and idx.size - 2 * trim >= min_points:
        idx = idx[trim:-trim]

    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def extract_on_state_ratio(
    gate_signal: Any,
    main_voltage: Any,
    main_current: Any,
    current_fraction: float = 0.10,
) -> float:
    """Extract MOSFET RDS(ON) as median(VDS / ID) in the stable ON block."""
    gate = np.asarray(gate_signal, dtype=float).reshape(-1)
    voltage = np.asarray(main_voltage, dtype=float).reshape(-1)
    current = np.asarray(main_current, dtype=float).reshape(-1)
    n = min(gate.size, voltage.size, current.size)
    gate, voltage, current = gate[:n], voltage[:n], current[:n]
    mask = on_state_mask(gate, current, current_fraction=current_fraction)
    mask &= np.isfinite(voltage) & np.isfinite(current) & (current > 1e-9)
    values = voltage[mask] / current[mask]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 5:
        raise DataValidationError("Too few valid ON-state V/I samples.")
    return float(np.median(values))


def extract_on_state_voltage(
    gate_signal: Any,
    main_voltage: Any,
    main_current: Any,
    current_fraction: float = 0.10,
) -> float:
    """Extract IGBT VCE(sat) as median VCE in the stable ON block."""
    gate = np.asarray(gate_signal, dtype=float).reshape(-1)
    voltage = np.asarray(main_voltage, dtype=float).reshape(-1)
    current = np.asarray(main_current, dtype=float).reshape(-1)
    n = min(gate.size, voltage.size, current.size)
    gate, voltage, current = gate[:n], voltage[:n], current[:n]
    mask = on_state_mask(gate, current, current_fraction=current_fraction)
    mask &= np.isfinite(voltage) & np.isfinite(current)
    values = voltage[mask]
    values = values[np.isfinite(values)]
    if values.size < 5:
        raise DataValidationError("Too few valid ON-state VCE samples.")
    return float(np.median(values))


def nearest_values(
    source_time_s: np.ndarray,
    source_values: np.ndarray,
    target_time_s: np.ndarray,
    max_gap_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour match source values onto target timestamps.

    Returns (matched_values, absolute_gap_seconds).
    """
    st = np.asarray(source_time_s, dtype=float)
    sv = np.asarray(source_values, dtype=float)
    tt = np.asarray(target_time_s, dtype=float)
    valid = np.isfinite(st) & np.isfinite(sv)
    st, sv = st[valid], sv[valid]
    order = np.argsort(st)
    st, sv = st[order], sv[order]

    matched = np.full(tt.shape, np.nan, dtype=float)
    gaps = np.full(tt.shape, np.nan, dtype=float)
    if st.size == 0:
        return matched, gaps

    pos = np.searchsorted(st, tt)
    left = np.clip(pos - 1, 0, st.size - 1)
    right = np.clip(pos, 0, st.size - 1)
    choose_right = np.abs(st[right] - tt) < np.abs(st[left] - tt)
    idx = np.where(choose_right, right, left)
    matched[:] = sv[idx]
    gaps[:] = np.abs(st[idx] - tt)
    if max_gap_s is not None:
        matched[gaps > max_gap_s] = np.nan
    return matched, gaps


@dataclass
class LinearCorrection:
    slope_per_c: float
    intercept: float
    reference_temperature_c: float
    n_fit: int
    temperature_span_c: float
    residual_mad: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def robust_linear_temperature_fit(
    precursor: np.ndarray,
    temperature_c: np.ndarray,
    early_fraction: float = 0.20,
    min_points: int = 20,
    min_temperature_span_c: float = 3.0,
) -> LinearCorrection:
    """Robust two-pass early-life fit: precursor = a + b*T_case."""
    y = np.asarray(precursor, dtype=float)
    x = np.asarray(temperature_c, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    indices = np.flatnonzero(valid)
    if indices.size < min_points:
        raise DataValidationError(
            f"Only {indices.size} finite precursor/temperature pairs."
        )

    n_early = max(min_points, int(math.ceil(early_fraction * indices.size)))
    indices = indices[: min(n_early, indices.size)]
    xe, ye = x[indices], y[indices]
    span = float(np.nanmax(xe) - np.nanmin(xe))
    if span < min_temperature_span_c:
        raise DataValidationError(
            f"Early temperature span {span:.3f} °C is too small for correction."
        )

    slope, intercept = np.polyfit(xe, ye, deg=1)
    residual = ye - (intercept + slope * xe)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    if mad > 0:
        inlier = np.abs(residual - median) <= 4.0 * 1.4826 * mad
        if inlier.sum() >= min_points:
            slope, intercept = np.polyfit(xe[inlier], ye[inlier], deg=1)
            residual = ye[inlier] - (intercept + slope * xe[inlier])
            mad = float(np.median(np.abs(residual - np.median(residual))))
            n_fit = int(inlier.sum())
        else:
            n_fit = int(xe.size)
    else:
        n_fit = int(xe.size)

    return LinearCorrection(
        slope_per_c=float(slope),
        intercept=float(intercept),
        reference_temperature_c=float(np.median(xe)),
        n_fit=n_fit,
        temperature_span_c=span,
        residual_mad=mad,
    )


def apply_temperature_correction(
    precursor: np.ndarray,
    temperature_c: np.ndarray,
    fit: LinearCorrection,
) -> np.ndarray:
    """Correct precursor to fit.reference_temperature_c."""
    y = np.asarray(precursor, dtype=float)
    temp = np.asarray(temperature_c, dtype=float)
    return y - fit.slope_per_c * (temp - fit.reference_temperature_c)


def persistent_first_crossing(
    values: np.ndarray,
    threshold: float,
    persistence_points: int = 3,
) -> int | None:
    v = np.asarray(values, dtype=float)
    above = np.isfinite(v) & (v >= threshold)
    if persistence_points <= 1:
        idx = np.flatnonzero(above)
        return int(idx[0]) if idx.size else None
    kernel = np.ones(persistence_points, dtype=int)
    runs = np.convolve(above.astype(int), kernel, mode="valid")
    idx = np.flatnonzero(runs >= persistence_points)
    return int(idx[0]) if idx.size else None


def robust_physical_filter(
    values: np.ndarray,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    mad_sigma: float = 4.0,
    min_keep: int = 20,
) -> np.ndarray:
    """Boolean mask (True = keep) for positive-skewed precursor series.

    Removes non-finite values, physical bound violations, and log-domain
    statistical outliers. RDS_ON / VCE(sat) are positive and right-skewed, so
    outlier rejection runs in log space. A fallback keeps the filter from
    deleting too much when the series is noisy (end-of-life waveforms can
    otherwise dominate the tail).
    """
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    if lower_bound is not None:
        keep &= v >= lower_bound
    if upper_bound is not None:
        keep &= v <= upper_bound
    positive = keep & (v > 0)
    if int(positive.sum()) >= min_keep:
        lv = np.log(v[positive])
        med = float(np.median(lv))
        mad = float(np.median(np.abs(lv - med)))
        if mad > 0:
            z = np.abs(lv - med) / (1.4826 * mad)
            keep_log = z <= mad_sigma
            if int(keep_log.sum()) >= min_keep:
                log_mask = np.zeros_like(keep, dtype=bool)
                log_mask[positive] = keep_log
                keep &= log_mask
    return keep


def parse_mosfet_ids(path: str | Path) -> tuple[str, str]:
    name = Path(path).name
    match = re.search(r"Test_(\d+)_run_(\d+)", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse MOSFET case/run from {name}")
    return f"Test_{int(match.group(1))}", f"run_{int(match.group(2))}"


def parse_igbt_device_id(path: str | Path) -> str:
    p = Path(path)
    joined = "/".join(p.parts)
    match = re.search(r"Device\s*([2-5])", joined, flags=re.IGNORECASE)
    if match:
        return f"Device_{match.group(1)}"
    if "Square Signal at gate and SMU" not in joined and "Square Signal at gate" in joined:
        return "Device_1"
    if "DC at gate" in joined:
        return "Device_DC"
    return "Device_unknown"


def get_measurement(path: str | Path) -> Any:
    data = loadmat_squeezed(path)
    if "measurement" not in data:
        raise DataValidationError(f"{path} has no 'measurement' struct.")
    return data["measurement"]


def parse_mosfet_run(path: str | Path) -> pd.DataFrame:
    """Parse one MOSFET MAT run into transient-level RDS and matched Tcase."""
    measurement = get_measurement(path)
    case_id, run_id = parse_mosfet_ids(path)

    steady_rows: list[dict[str, float]] = []
    for item in as_sequence(get_attr(measurement, "steadyState")):
        td = get_attr(item, "timeDomain")
        steady_rows.append(
            {
                "time_abs_s": matlab_datenum_to_unix_seconds(
                    first_attr(item, ["timeEpoch"])
                ),
                "T_case_C": scalar_float(
                    first_attr(td, ["packageTemperature", "packageTempurature"])
                ),
            }
        )
    steady = pd.DataFrame(steady_rows).dropna(subset=["time_abs_s"])

    transient_rows: list[dict[str, float | str]] = []
    for item in as_sequence(get_attr(measurement, "transient")):
        td = get_attr(item, "timeDomain")
        try:
            rds = extract_on_state_ratio(
                get_attr(td, "gateSignalVoltage"),
                get_attr(td, "drainSourceVoltage"),
                get_attr(td, "drainCurrent"),
            )
        except (DataValidationError, TypeError, ValueError):
            continue
        transient_rows.append(
            {
                "case_id": case_id,
                "run_id": run_id,
                "source_file": str(path),
                "time_abs_s": matlab_datenum_to_unix_seconds(
                    first_attr(item, ["timeEpoch"])
                ),
                "RDS_raw_ohm": rds,
            }
        )
    if not transient_rows:
        return pd.DataFrame(
            columns=["case_id", "run_id", "source_file", "time_abs_s", "RDS_raw_ohm"]
        )
    transient = pd.DataFrame(transient_rows).dropna(
        subset=["time_abs_s", "RDS_raw_ohm"]
    )
    if transient.empty:
        return transient

    if not steady.empty:
        matched, gaps = nearest_values(
            steady["time_abs_s"].to_numpy(),
            steady["T_case_C"].to_numpy(),
            transient["time_abs_s"].to_numpy(),
        )
        transient["T_case_C"] = matched
        transient["temperature_match_gap_s"] = gaps
    else:
        transient["T_case_C"] = np.nan
        transient["temperature_match_gap_s"] = np.nan
    return transient.sort_values("time_abs_s").reset_index(drop=True)


def igbt_temperature_from_td(td: Any) -> float:
    return scalar_float(
        first_attr(
            td,
            [
                "packageTemperature",
                "packageTempurature",
                "package_temperture",
                "internalTemperature",
                "heatSinkTemperature",
                "heatSinkTempurature",
            ],
        )
    )


def parse_igbt_steady(path: str | Path) -> pd.DataFrame:
    measurement = get_measurement(path)
    device_id = parse_igbt_device_id(path)
    rows: list[dict[str, float | str]] = []
    for item in as_sequence(get_attr(measurement, "steadyState")):
        td = get_attr(item, "timeDomain")
        rows.append(
            {
                "device_id": device_id,
                "run_id": Path(path).stem,
                "source_file": str(path),
                "time_abs_s": matlab_datenum_to_unix_seconds(
                    first_attr(item, ["timeEpoch", "timeSinceEpoch"])
                ),
                "node2Voltage_V": scalar_float(get_attr(td, "node2Voltage")),
                "collector_current_A": scalar_float(
                    first_attr(
                        td,
                        ["collectorEmitterCurrent", "collectorEmitterCurrentSignal"],
                    )
                ),
                "T_case_C": igbt_temperature_from_td(td),
            }
        )
    return (
        pd.DataFrame(rows)
        .dropna(subset=["time_abs_s"])
        .sort_values("time_abs_s")
        .reset_index(drop=True)
    )


def parse_igbt_transient(path: str | Path) -> pd.DataFrame:
    measurement = get_measurement(path)
    device_id = parse_igbt_device_id(path)
    rows: list[dict[str, float | str]] = []
    for item in as_sequence(get_attr(measurement, "transient")):
        td = get_attr(item, "timeDomain")
        try:
            vce_sat = extract_on_state_voltage(
                get_attr(td, "gateSignalVoltage"),
                get_attr(td, "collectorEmitterVoltage"),
                get_attr(td, "collectorEmitterCurrentSignal"),
            )
            current = np.asarray(
                get_attr(td, "collectorEmitterCurrentSignal"), dtype=float
            ).reshape(-1)
            gate = np.asarray(get_attr(td, "gateSignalVoltage"), dtype=float).reshape(-1)
            mask = on_state_mask(gate, current)
            ic_on = float(np.median(current[: mask.size][mask]))
        except (DataValidationError, TypeError, ValueError):
            continue
        rows.append(
            {
                "device_id": device_id,
                "run_id": Path(path).stem,
                "source_file": str(path),
                "time_abs_s": matlab_datenum_to_unix_seconds(
                    first_attr(item, ["timeEpoch", "timeSinceEpoch"])
                ),
                "VCE_sat_transient_V": vce_sat,
                "collector_current_A": ic_on,
            }
        )
    return (
        pd.DataFrame(rows)
        .dropna(subset=["time_abs_s", "VCE_sat_transient_V"])
        .sort_values("time_abs_s")
        .reset_index(drop=True)
    )


def concatenate_device_runs(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["time_abs_s", "run_id"]).drop_duplicates(
        subset=["time_abs_s"], keep="last"
    )
    out["elapsed_time_s"] = out["time_abs_s"] - out["time_abs_s"].min()
    return out.reset_index(drop=True)


def build_fixed_threshold_labels(
    elapsed_time_s: np.ndarray,
    drift: np.ndarray,
    threshold: float,
    persistence_points: int = 3,
    denoise_for_eol: bool = True,
) -> dict[str, Any]:
    time = np.asarray(elapsed_time_s, dtype=float)
    d = np.asarray(drift, dtype=float)
    d_for_eol = d
    if denoise_for_eol:
        # Isotonic regression on the time-ordered drift suppresses early
        # fluctuations that would otherwise trip persistent_first_crossing
        # before real degradation sets in. Only the EOL crossing uses the
        # denoised curve; the returned hi/rul keep the raw drift.
        finite = np.isfinite(time) & np.isfinite(d)
        if int(finite.sum()) >= 10:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            d_for_eol = np.array(d, dtype=float)
            d_for_eol[finite] = iso.fit_transform(time[finite], d[finite])
    eol_idx = persistent_first_crossing(d_for_eol, threshold, persistence_points)
    event_observed = eol_idx is not None
    censor_time = float(np.nanmax(time)) if time.size else np.nan
    exact_rul = np.full(time.shape, np.nan, dtype=float)
    lower_bound = np.maximum(censor_time - time, 0.0)
    if event_observed:
        eol_time = float(time[eol_idx])
        exact_rul = np.maximum(eol_time - time, 0.0)
        lower_bound = exact_rul.copy()
    else:
        eol_time = np.nan

    hi = np.clip(d / threshold, 0.0, 1.0)
    return {
        "hi": hi,
        "rul_s": exact_rul,
        "rul_lower_bound_s": lower_bound,
        "event_observed": bool(event_observed),
        "eol_index": int(eol_idx) if eol_idx is not None else -1,
        "eol_time_s": eol_time,
        "censor_time_s": censor_time,
    }


def write_feature_h5(
    output_path: str | Path,
    devices: dict[str, pd.DataFrame],
    feature_columns: Sequence[str],
    metadata: dict[str, Any],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as h5:
        h5.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        h5.attrs["feature_names"] = np.asarray(feature_columns, dtype=string_dtype)
        root = h5.create_group("devices")
        for device_id, frame in devices.items():
            grp = root.create_group(device_id)
            grp.create_dataset(
                "x",
                data=frame[list(feature_columns)].to_numpy(dtype=np.float32),
                compression="gzip",
                chunks=True,
            )
            for col in [
                "elapsed_time_s",
                "time_abs_s",
                "hi",
                "rul_s",
                "rul_lower_bound_s",
            ]:
                if col in frame:
                    grp.create_dataset(
                        col,
                        data=frame[col].to_numpy(dtype=np.float64),
                        compression="gzip",
                        chunks=True,
                    )
            grp.attrs["event_observed"] = bool(
                frame["event_observed"].iloc[0]
                if "event_observed" in frame and len(frame)
                else False
            )
            for attr_name in [
                "eol_time_s",
                "censor_time_s",
                "temperature_slope_per_c",
                "reference_temperature_c",
            ]:
                if attr_name in frame and len(frame):
                    value = frame[attr_name].iloc[0]
                    grp.attrs[attr_name] = float(value) if np.isfinite(value) else np.nan


def read_node2_validation_status(path: str | Path) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    node = data.get("igbt_node2_validation", {})
    return str(node.get("overall_status", "unconfirmed"))
