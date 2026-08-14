"""GaN RFALT 源域仿真的物理不变量测试。"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from src.data.preprocess.gan_rfalt_schema import validate_rfalt_frame
from src.sim.gan_rfalt_sim import (
    integrate_damage_states,
    sample_rfalt_profile,
    simulate_rfalt_trajectory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phased_array_gan.yaml"


def test_same_seed_produces_identical_trajectory():
    first = simulate_rfalt_trajectory(
        device_id="rfalt-01", duration_s=7200, sample_period_s=600, seed=17,
    )
    second = simulate_rfalt_trajectory(
        device_id="rfalt-01", duration_s=7200, sample_period_s=600, seed=17,
    )

    np.testing.assert_allclose(
        first.select_dtypes(include=["number"]).to_numpy(),
        second.select_dtypes(include=["number"]).to_numpy(),
    )
    assert first["stress_mode"].tolist() == second["stress_mode"].tolist()


def test_permanent_damage_and_thermal_resistance_never_decrease():
    frame = simulate_rfalt_trajectory(
        device_id="rfalt-02", duration_s=14400, sample_period_s=600, seed=3,
    )

    assert (np.diff(frame["d_perm"]) >= -1e-12).all()
    assert (np.diff(frame["r_th"]) >= -1e-12).all()


def test_trap_charge_recovers_during_recovery_segment():
    profile = sample_rfalt_profile(
        duration_s=7200,
        sample_period_s=600,
        stress_modes=["rfalt_cw", "recovery"],
        seed=5,
    )
    states = integrate_damage_states(profile)
    recovery = np.flatnonzero(profile["stress_mode"].to_numpy() == "recovery")

    assert len(recovery) >= 2
    assert states.loc[recovery[-1], "q_trap"] < states.loc[recovery[0], "q_trap"]


def test_profile_time_grid_never_exceeds_duration_and_appends_exact_endpoint():
    profile = sample_rfalt_profile(
        duration_s=1000.0, sample_period_s=600.0, stress_modes=["rfalt_cw"], seed=4,
    )

    np.testing.assert_allclose(profile["time_s"].to_numpy(), [0.0, 600.0, 1000.0])
    assert profile["time_s"].max() <= 1000.0


@pytest.mark.parametrize("sample_period_h", [12.0, 24.0])
def test_active_trap_dynamics_remains_physical_at_large_time_steps(sample_period_h):
    profile = sample_rfalt_profile(
        duration_s=72.0 * 3600.0,
        sample_period_s=sample_period_h * 3600.0,
        stress_modes=["rfalt_cw"],
        base_temperature_c=80.0,
        seed=6,
    )
    states = integrate_damage_states(profile)

    assert ((states["q_trap"] >= 0.0) & (states["q_trap"] <= 1.0)).all()
    assert (states["q_trap"].iloc[1:] > 0.0).all()
    assert states["q_trap"].iloc[-1] > 0.40


def test_higher_temperature_or_compression_accelerates_permanent_damage():
    cool = sample_rfalt_profile(
        duration_s=7200, sample_period_s=600, stress_modes=["rfalt_cw"],
        base_temperature_c=55.0, compression_dB=0.0, seed=8,
    )
    hot = sample_rfalt_profile(
        duration_s=7200, sample_period_s=600, stress_modes=["rfalt_cw"],
        base_temperature_c=115.0, compression_dB=0.0, seed=8,
    )
    compressed = sample_rfalt_profile(
        duration_s=7200, sample_period_s=600, stress_modes=["rfalt_cw"],
        base_temperature_c=55.0, compression_dB=3.0, seed=8,
    )

    assert integrate_damage_states(hot)["d_perm"].iloc[-1] > integrate_damage_states(cool)["d_perm"].iloc[-1]
    assert integrate_damage_states(compressed)["d_perm"].iloc[-1] > integrate_damage_states(cool)["d_perm"].iloc[-1]


def test_simulated_observations_satisfy_rfalt_schema():
    frame = simulate_rfalt_trajectory(
        device_id="rfalt-03", duration_s=7200, sample_period_s=600, seed=9,
    )

    validated = validate_rfalt_frame(frame, require_latent=True)
    assert validated.attrs["schema_version"] == "gan_rfalt_v1"


def test_rfalt_sim_cli_smoke_writes_contract_raw_h5(tmp_path):
    raw_path = tmp_path / "rfalt_raw.h5"

    subprocess.run(
        [
            sys.executable, "-m", "src.sim.gan_rfalt_sim", "--config", str(CONFIG_PATH),
            "--smoke", "--output", str(raw_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    with h5py.File(raw_path, "r") as h5:
        assert h5.attrs["schema_version"] == "gan_rfalt_v1"
        assert "trajectories" in h5
        assert h5["trajectories"].keys()
        trajectory = h5["trajectories"][next(iter(h5["trajectories"].keys()))]
        assert "event_observed" in trajectory
        assert "rul_lower_bound_s" in trajectory
