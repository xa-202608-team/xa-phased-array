"""GaN 损伤状态驱动的 LEO 目标 T/R 通道回归测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sim import phased_array_sim as target_sim  # noqa: E402
from src.sim.build_array_hi import X_GLOBAL_COLS, main as build_hi_main  # noqa: E402
from src.utils import load_config  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "phased_array_gan.yaml"


def _short_target_sim(hotspots: bool = True) -> dict:
    cfg = load_config(CONFIG_PATH)
    sim_cfg = dict(cfg["sim"])
    sim_cfg.update(duration_years=0.08, sample_period_s=43200.0, physics_dt_s=900.0)
    sim_cfg["physics"] = dict(sim_cfg["physics"])
    sim_cfg["physics"]["hotspot_field"] = dict(sim_cfg["physics"]["hotspot_field"], enabled=hotspots)
    sim_cfg["service_limits"] = dict(sim_cfg["service_limits"], theta_err_max_deg=1.0e-4)
    return sim_cfg


def _trajectory(seed: int = 19, hotspots: bool = True):
    sim_cfg = _short_target_sim(hotspots)
    rng = np.random.default_rng(seed)
    params = target_sim.sample_gan_target_params(rng, sim_cfg)
    return target_sim.simulate_gan_state(params, sim_cfg, np.random.default_rng(params["seed_traj"]))


def test_gan_state_emits_finite_subarray_rf_labels():
    """目标路径输出无噪 T/R 子阵 RF 标签，状态与观测标签相互隔离。"""
    df, sa, _, _, labels, states = _trajectory()
    assert target_sim.DYNAMICS_ID == "leo_coupled_v1"
    assert set(labels) == {"gain_dB", "phase_deg", "Pout_dBm", "PAE"}
    for values in labels.values():
        assert values.shape == sa.shape[:2]
        assert np.isfinite(values).all()
    assert set(states) == {"d_perm", "q_trap", "r_th"}
    assert all(values.shape[0] == len(df) for values in states.values())
    assert not {"d_perm", "q_trap", "r_th", "gain_dB", "phase_deg", "Pout_dBm", "PAE"}.intersection(df.columns)


def test_hotspot_changes_target_state_spatial_distribution():
    """轨道热循环以外，热点耦合还必须改变 T/R 空间退化图。"""
    _, _, _, _, _, hotspot_states = _trajectory(seed=23, hotspots=True)
    _, _, _, _, _, flat_states = _trajectory(seed=23, hotspots=False)
    hot_map = hotspot_states["d_perm"][-1]
    flat_map = flat_states["d_perm"][-1]
    assert np.std(hot_map) > np.std(flat_map)
    assert not np.allclose(hot_map, flat_map)


def test_feature_writer_truncates_all_gan_arrays_and_keeps_labels_out_of_x(tmp_path, monkeypatch):
    """EOL 截断后每个特征、状态和标签长度相同，x 仍只含遥测。"""
    df, sa, eol, failed, labels, states = _trajectory(seed=31)
    assert failed and eol < len(df) - 1, "测试需要含 EOL 后尾段的目标轨迹"
    raw = tmp_path / "target_raw.h5"
    features = tmp_path / "target_features.h5"
    target_sim.write_target_raw_h5([(df, sa, eol, failed, labels, states, {
        "margin0_dB": 1.8, "scan_az_deg": 12.0, "duty": 0.55, "Tj_base_C": 115.0,
    })], raw)

    cfg = load_config(CONFIG_PATH)
    cfg["target"]["raw_path"] = str(raw)
    cfg["target"]["feature_path"] = str(features)
    cfg_path = tmp_path / "gan.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["build_array_hi", "--config", str(cfg_path)])
    build_hi_main()

    with h5py.File(features, "r") as h5:
        group = h5["traj_000"]
        assert h5.attrs["target_condition_schema"] == "target_ood_conditions_v1"
        lengths = {name: len(group[name]) for name in group if group[name].ndim >= 1}
        assert set(lengths.values()) == {eol + 1}
        assert "time_s" in group
        np.testing.assert_allclose(group["time_s"][:], df["t"].to_numpy()[:eol + 1])
        # Gate 2 物理 u_phys 辅助量：逐步 duty_out（随扫描角变化），与 time_s 同 EOL 截断，不进 x。
        assert "physical_duty" in group
        np.testing.assert_allclose(group["physical_duty"][:], df["duty"].to_numpy()[:eol + 1])
        assert len(group["physical_duty"]) == eol + 1
        assert group["physical_duty"].attrs["label_level"] == "physical_stress_aux"
        assert group["physical_duty"].attrs["access"] == "stress_aux_not_model_input"
        assert "physical_duty" not in group["x_global"].attrs["feature_names"]
        assert group["x_global"].shape[1] == 6 and group["x_nodes"].shape[2] == 6
        assert {"label_channel_gain_dB", "label_channel_phase_deg", "label_channel_Pout_dBm", "label_channel_PAE"}.issubset(group)
        array_labels = {
            "label_array_G_array_dB": ("G_array_dB_true", "dB"),
            "label_array_EIRP_norm": ("EIRP_norm", "ratio"),
            "label_array_SLL_dB": ("SLL_dB_true", "dB"),
            "label_array_theta_err_deg": ("theta_err_deg_true", "deg"),
            "label_array_M_link_dB": ("M_link_dB_true", "dB"),
        }
        assert set(array_labels).issubset(group)
        for label_name, (raw_name, unit) in array_labels.items():
            np.testing.assert_allclose(group[label_name][:], df[raw_name].to_numpy()[:eol + 1])
            assert group[label_name].attrs["units"] == unit
            assert group[label_name].attrs["label_level"] == "array_scoring_truth"
            assert group[label_name].attrs["access"] == "score_only_not_model_input"
        assert group.attrs["array_scoring_label_schema"] == "array_scoring_truth_v1"
        assert group.attrs["array_twin_metadata_schema"] == "subarray_array_twin_v1"
        assert group.attrs["scan_az_deg"] == 12.0 and group.attrs["margin0_dB"] == 1.8
        assert group.attrs["duty_cycle"] == 0.55 and group.attrs["Tj_base_C"] == 115.0
        assert tuple(group.attrs["array_grid"]) == (16, 16)
        assert group.attrs["element_spacing_lambda"] == 0.5 and group.attrs["subarray_block"] == 4
        assert group.attrs["array_twin_metadata_access"] == "twin_only_not_model_input"
        node_states = {"d_perm": "latent_d_perm", "q_trap": "latent_q_trap", "r_th": "latent_r_th"}
        subarray_ids = target_sim._subarray_ids(cfg["sim"]["array"]["grid"], cfg["sim"]["array"]["subarray_block"])
        for state_name, raw_name in node_states.items():
            expected = target_sim._subarray_reduce(group[raw_name][:], subarray_ids, 16)
            saved = group[f"label_node_{state_name}"][:]
            np.testing.assert_allclose(saved, expected)
            assert saved.shape == group["label_channel_gain_dB"].shape
            assert group[f"label_node_{state_name}"].attrs["label_level"] == "subarray_train_label"
            assert group[f"label_node_{state_name}"].attrs["access"] == "train_label_not_model_input"
        assert {"latent_d_perm", "latent_q_trap", "latent_r_th"}.issubset(group)
        assert group["x_global"].attrs["feature_names"].split(",") == X_GLOBAL_COLS
        assert "G_array_dB_true" not in group["x_global"].attrs["feature_names"]
        assert "M_link_dB_true" not in group["x_global"].attrs["feature_names"]


def test_spatial_reader_loads_only_observations_node_labels_and_score_truth(tmp_path, monkeypatch):
    """空间 reader 保持节点结构，并拒绝长度/EOL 不一致的特征契约。"""
    from src.experiments.run_gan_transfer import _spatial_target_rows, make_target_observations

    df, sa, eol, _, labels, states = _trajectory(seed=34)
    raw, features = tmp_path / "spatial_raw.h5", tmp_path / "spatial_features.h5"
    target_sim.write_target_raw_h5([(df, sa, eol, True, labels, states, {
        "margin0_dB": 1.8, "scan_az_deg": 0.0, "duty": 0.55, "Tj_base_C": 115.0,
    })], raw)
    cfg = load_config(CONFIG_PATH)
    cfg["target"]["raw_path"], cfg["target"]["feature_path"] = str(raw), str(features)
    cfg_path = tmp_path / "spatial.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["build_array_hi", "--config", str(cfg_path)])
    build_hi_main()

    rows, _ = _spatial_target_rows(features, max_points_per_traj=32)
    assert rows.x_global.shape[1] == 6 and rows.x_nodes.shape[1:] == (16, 6)
    assert rows.node_states.shape[1:] == (16, 3) and rows.node_labels.shape[1:] == (16, 4)
    assert set(rows.array_score_truth) == {"G_array_dB", "EIRP_norm", "SLL_dB", "theta_err_deg", "M_link_dB"}
    global_obs, node_obs = make_target_observations(rows.x_global, rows.x_nodes, preserve_node_axis=True)
    assert np.array_equal(rows.x_global, global_obs) and np.array_equal(rows.x_nodes, node_obs)
    with h5py.File(features, "r+") as h5:
        h5["traj_000"]["label_array_M_link_dB"][0] += 99.0
    changed, _ = _spatial_target_rows(features, max_points_per_traj=32)
    assert np.array_equal(rows.x_global, changed.x_global) and np.array_equal(rows.x_nodes, changed.x_nodes)
    assert not np.array_equal(rows.array_score_truth["M_link_dB"], changed.array_score_truth["M_link_dB"])


def test_target_dynamics_is_independent_of_rfalt_integrator():
    """目标积分器不能导入或复用台架 RFALT 集总状态积分器。"""
    source = Path(target_sim.__file__).read_text(encoding="utf-8")
    assert "gan_rfalt_sim" not in source
    assert target_sim.DYNAMICS_ID != "rfalt_lumped_v1"


def test_physics_substeps_set_time_axis_and_change_converged_states():
    """physics_dt_s 决定窗口内积分分辨率，默认 8 年时间轴不应漂移。"""
    full_cfg = load_config(CONFIG_PATH)["sim"]
    full_params = target_sim.sample_gan_target_params(np.random.default_rng(101), full_cfg)
    full_df, _, _, _, _, full_states = target_sim.simulate_gan_state(
        full_params, full_cfg, np.random.default_rng(full_params["seed_traj"]))
    duration_s = full_cfg["duration_years"] * target_sim.SEC_PER_YEAR
    assert full_df["t"].iloc[-1] == duration_s - full_cfg["sample_period_s"]
    assert all(np.isfinite(values).all() and values.min() >= 0.0 and values.max() < 1.0
               for values in full_states.values())

    coarse = _short_target_sim()
    fine = _short_target_sim()
    coarse["physics_dt_s"] = 1800.0
    fine["physics_dt_s"] = 300.0
    params = target_sim.sample_gan_target_params(np.random.default_rng(103), fine)
    _, _, _, _, _, coarse_states = target_sim.simulate_gan_state(
        params, coarse, np.random.default_rng(params["seed_traj"]))
    _, _, _, _, _, fine_states = target_sim.simulate_gan_state(
        params, fine, np.random.default_rng(params["seed_traj"]))
    delta = np.abs(fine_states["d_perm"][-1] - coarse_states["d_perm"][-1])
    assert delta.max() > 0.0 and delta.max() < 2.0e-3


def test_target_raw_arrays_are_chunked_and_compressed(tmp_path):
    """状态、标签和子阵原始大数组必须使用可读的块压缩存储。"""
    df, sa, eol, failed, labels, states = _trajectory(seed=41)
    raw = tmp_path / "compressed_target_raw.h5"
    target_sim.write_target_raw_h5([(df, sa, eol, failed, labels, states, {
        "margin0_dB": 1.8, "scan_az_deg": 0.0, "duty": 0.55, "Tj_base_C": 115.0,
    })], raw)
    with h5py.File(raw, "r") as h5:
        group = h5["traj_000"]
        for name in ["subarray_features", "latent_d_perm", "latent_q_trap", "latent_r_th",
                     "label_channel_gain_dB", "label_channel_phase_deg", "label_channel_Pout_dBm", "label_channel_PAE"]:
            assert group[name].chunks is not None and group[name].compression is not None
            assert group[name][:].shape[0] == len(df)
    assert raw.stat().st_size < 1_000_000


def test_cli_small_smoke_writes_then_builds_target_features(tmp_path, monkeypatch):
    """端到端 CLI：目标仿真原始文件可被 build_array_hi 直接消费。"""
    cfg = load_config(CONFIG_PATH)
    cfg["sim"].update(n_traj=2, duration_years=0.02, sample_period_s=43200.0, physics_dt_s=900.0)
    raw, features = tmp_path / "cli_raw.h5", tmp_path / "cli_features.h5"
    cfg["target"]["raw_path"], cfg["target"]["feature_path"] = str(raw), str(features)
    cfg_path = tmp_path / "cli_gan.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["phased_array_sim", "--config", str(cfg_path), "--n_traj", "2"])
    target_sim.main()
    monkeypatch.setattr(sys, "argv", ["build_array_hi", "--config", str(cfg_path)])
    build_hi_main()
    with h5py.File(features, "r") as h5:
        assert len(h5) == 2
        assert all("x_global" in h5[name] and "label_channel_gain_dB" in h5[name] for name in h5)
