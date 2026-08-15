"""GaN RFALT 源域数据契约测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocess.gan_rfalt_schema import validate_rfalt_frame


def _valid_frame() -> pd.DataFrame:
    base = {
        "device_id": "dev-01",
        "lot_id": "lot-a",
        "T_base_C": 40.0,
        "T_j_C": 115.0,
        "VDS": 28.0,
        "VGS": -2.7,
        "ID": 0.5,
        "IG": 1e-6,
        "duty_cycle": 0.6,
        "waveform": "cw",
        "PAPR_dB": 0.0,
        "VSWR": 1.2,
        "Pin_dBm": 30.0,
        "Pout_dBm": 39.0,
        "gain_dB": 9.0,
        "PAE": 0.55,
        "AM_AM_dB": -0.1,
        "AM_PM_deg": 0.2,
        "EVM_pct": 1.0,
        "ACPR_dBc": -45.0,
        "RDS_dynamic_ohm": 0.12,
        "gm_S": 0.35,
        "Vth_V": -2.0,
        "stress_mode": "rfalt_cw",
        "event_observed": True,
        "rul_lower_bound_s": 3600.0,
    }
    rows = []
    for time_s in (0.0, 600.0):
        rows.append({"time_s": time_s, **base})
    return pd.DataFrame(rows)


def test_valid_rfalt_frame_returns_copy_with_schema_metadata():
    frame = _valid_frame()

    validated = validate_rfalt_frame(frame)

    assert validated is not frame
    assert validated.attrs["schema_version"] == "gan_rfalt_v1"
    assert list(validated["time_s"]) == [0.0, 600.0]


def test_missing_required_field_is_rejected():
    frame = _valid_frame().drop(columns="device_id")

    with pytest.raises(ValueError, match="device_id"):
        validate_rfalt_frame(frame)


def test_time_must_increase_within_device():
    frame = _valid_frame()
    frame.loc[1, "time_s"] = 0.0

    with pytest.raises(ValueError, match="time_s"):
        validate_rfalt_frame(frame)


def test_latent_trap_state_must_be_bounded_when_requested():
    frame = _valid_frame().assign(d_perm=[0.1, 0.2], q_trap=[0.2, 1.1], r_th=[0.0, 0.01])

    with pytest.raises(ValueError, match="q_trap"):
        validate_rfalt_frame(frame, require_latent=True)
