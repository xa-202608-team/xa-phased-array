"""GaN RFALT 特征 HDF5 写入的契约测试。"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

from src.data.preprocess.gan_rfalt_features import write_rfalt_h5
from src.sim.gan_rfalt_sim import simulate_rfalt_trajectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phased_array_gan.yaml"


def test_feature_h5_has_schema_devices_labels_and_no_latent_inputs(tmp_path):
    trajectories = [
        simulate_rfalt_trajectory("rfalt-01", duration_s=7200, sample_period_s=600, seed=1),
        simulate_rfalt_trajectory("rfalt-02", duration_s=7200, sample_period_s=600, seed=2),
    ]
    output_path = tmp_path / "rfalt_source.h5"

    write_rfalt_h5(trajectories, output_path)

    with h5py.File(output_path, "r") as h5:
        assert h5.attrs["schema_version"] == "gan_rfalt_v1"
        names = [name.decode() if isinstance(name, bytes) else str(name)
                 for name in h5.attrs["feature_names"]]
        assert "d_perm" not in names
        assert "q_trap" not in names
        assert "r_th" not in names
        assert set(h5["devices"].keys()) == {"rfalt-01", "rfalt-02"}
        for device_id in h5["devices"]:
            group = h5["devices"][device_id]
            assert group["x"].shape[1] == len(names)
            assert "event_observed" in group.attrs
            assert "rul_lower_bound_s" in group
            assert "time_s" in group
            assert np.array_equal(group["time_s"][:], group["elapsed_time_s"][:])
            assert "latent_d_perm" in group
            assert "latent_q_trap" in group
            assert "latent_r_th" in group


def test_rfalt_features_cli_reads_raw_h5_without_latent_training_inputs(tmp_path):
    raw_path = tmp_path / "rfalt_raw.h5"
    feature_path = tmp_path / "rfalt_features.h5"
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

    subprocess.run(
        [
            sys.executable, "-m", "src.data.preprocess.gan_rfalt_features",
            "--config", str(CONFIG_PATH), "--input", str(raw_path), "--output", str(feature_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    with h5py.File(feature_path, "r") as h5:
        names = [name.decode() if isinstance(name, bytes) else str(name)
                 for name in h5.attrs["feature_names"]]
        assert h5["devices"].keys()
        assert not {"d_perm", "q_trap", "r_th"}.intersection(names)
        for group in h5["devices"].values():
            assert group["x"].shape[1] == len(names)
