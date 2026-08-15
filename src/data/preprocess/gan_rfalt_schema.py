"""GaN RFALT 源域轨迹的数据字段与输入校验。"""
from __future__ import annotations

import pandas as pd


SCHEMA_VERSION = "gan_rfalt_v1"

REQUIRED_COLUMNS = (
    "time_s", "device_id", "lot_id", "T_base_C", "T_j_C",
    "VDS", "VGS", "ID", "IG", "duty_cycle", "waveform", "PAPR_dB", "VSWR",
    "Pin_dBm", "Pout_dBm", "gain_dB", "PAE", "AM_AM_dB", "AM_PM_deg",
    "EVM_pct", "ACPR_dBc", "RDS_dynamic_ohm", "gm_S", "Vth_V", "stress_mode",
    "event_observed", "rul_lower_bound_s",
)

LATENT_COLUMNS = ("d_perm", "q_trap", "r_th")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"缺少 GaN RFALT 必需字段: {', '.join(missing)}")


def validate_rfalt_frame(frame: pd.DataFrame, require_latent: bool = False) -> pd.DataFrame:
    """返回通过契约校验的副本，不改变任一观测量的数值趋势。"""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("GaN RFALT 输入必须是 pandas.DataFrame")

    _require_columns(frame, REQUIRED_COLUMNS)
    validated = frame.copy(deep=True)
    if validated.empty:
        raise ValueError("GaN RFALT 轨迹不能为空")

    for device_id, trajectory in validated.groupby("device_id", sort=False):
        time_s = trajectory["time_s"].to_numpy()
        if len(time_s) > 1 and (time_s[1:] <= time_s[:-1]).any():
            raise ValueError(f"device_id={device_id} 的 time_s 必须严格递增")

    if require_latent:
        _require_columns(validated, LATENT_COLUMNS)
        if (validated["d_perm"] < 0).any():
            raise ValueError("d_perm 必须非负")
        if ((validated["q_trap"] < 0) | (validated["q_trap"] > 1)).any():
            raise ValueError("q_trap 必须位于 [0, 1]")
        if (validated["r_th"] < 0).any():
            raise ValueError("r_th 必须非负")

    validated.attrs["schema_version"] = SCHEMA_VERSION
    return validated
