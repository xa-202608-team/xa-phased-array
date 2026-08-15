"""确定性子阵阵列孪生的方向性与输入隔离契约。"""
from __future__ import annotations

import numpy as np
import pytest


def _metadata():
    return {"scan_az_deg": 0.0, "margin0_dB": 2.0, "array_grid": (16, 16),
            "element_spacing_lambda": 0.5, "subarray_block": 4}


def test_deterministic_twin_has_reasonable_power_and_phase_directions():
    from src.sim.subarray_array_twin import evaluate_subarray_array

    healthy = np.zeros((2, 16, 4), dtype=np.float32)
    power_loss = healthy.copy()
    power_loss[1, :, 2] = -3.0
    phase_gradient = healthy.copy()
    phase_gradient[1, :, 1] = np.tile(np.arange(4, dtype=float), 4) * 20.0

    base = evaluate_subarray_array(healthy, _metadata())
    powered = evaluate_subarray_array(power_loss, _metadata())
    phased = evaluate_subarray_array(phase_gradient, _metadata())

    assert base["G_array_dB"][0] == pytest.approx(0.0, abs=1e-5)
    assert base["EIRP_norm"][0] == pytest.approx(1.0, abs=1e-6)
    assert powered["G_array_dB"][1] < base["G_array_dB"][1]
    assert powered["EIRP_norm"][1] < base["EIRP_norm"][1]
    assert powered["M_link_dB"][1] < base["M_link_dB"][1]
    assert abs(phased["theta_err_deg"][1]) > abs(base["theta_err_deg"][1]) + 0.1
    assert phased["EIRP_norm"][1] < base["EIRP_norm"][1]


def test_twin_accepts_only_predicted_channels_and_explicit_metadata():
    from src.sim.subarray_array_twin import evaluate_subarray_array

    channels = np.zeros((1, 16, 4), dtype=np.float32)
    with pytest.raises(TypeError):
        evaluate_subarray_array(channels, _metadata(), score_labels=np.ones((1, 5)))
