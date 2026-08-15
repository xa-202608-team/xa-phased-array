"""空间阶段 smoke 训练入口的轨迹隔离与阵列评分契约。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


def _spatial_rows(*, n_trajectories: int = 4, n_time: int = 9):
    from src.experiments.run_gan_transfer import SpatialDomainRows

    ids = np.repeat([f"traj_{index:03d}" for index in range(n_trajectories)], n_time)
    time = np.tile(np.arange(n_time, dtype=np.float32) * 21600.0, n_trajectories)
    x_nodes = np.zeros((len(ids), 16, 6), dtype=np.float32)
    x_nodes[..., 0] = np.repeat(np.arange(n_trajectories, dtype=np.float32), n_time)[:, None]
    states = np.zeros((len(ids), 16, 3), dtype=np.float32)
    labels = np.zeros((len(ids), 16, 4), dtype=np.float32)
    score = {
        "G_array_dB": np.zeros(len(ids), dtype=np.float32),
        "EIRP_norm": np.ones(len(ids), dtype=np.float32),
        "SLL_dB": np.full(len(ids), -12.0, dtype=np.float32),
        "theta_err_deg": np.zeros(len(ids), dtype=np.float32),
        "M_link_dB": np.full(len(ids), 2.0, dtype=np.float32),
    }
    return SpatialDomainRows(
        x_global=np.zeros((len(ids), 6), dtype=np.float32), x_nodes=x_nodes,
        node_states=states, node_labels=labels, array_score_truth=score,
        rul=np.arange(len(ids), 0, -1, dtype=np.float32), event=np.zeros(len(ids), dtype=bool),
        ids=ids, time=time,
    )


def _write_source_h5(path: Path):
    names = (
        "T_base_C", "T_j_C", "VDS", "VGS", "ID", "IG", "duty_cycle", "PAPR_dB", "VSWR", "Pin_dBm",
        "Pout_dBm", "gain_dB", "PAE", "AM_AM_dB", "AM_PM_deg", "EVM_pct", "ACPR_dBc", "RDS_dynamic_ohm", "gm_S", "Vth_V",
    )
    with h5py.File(path, "w") as h5:
        h5.attrs.update(schema_version="gan_rfalt_v1", dynamics_id="rfalt_lumped_v1", feature_names=names, feature_dim=20)
        group = h5.create_group("devices").create_group("dev_00")
        x = np.zeros((9, 20), dtype=np.float32)
        group.create_dataset("x", data=x); group.create_dataset("time_s", data=np.arange(9) * 600.0)
        group.create_dataset("rul_lower_bound_s", data=np.arange(9, 0, -1, dtype=np.float32))
        for name in ("latent_d_perm", "latent_q_trap", "latent_r_th"):
            group.create_dataset(name, data=np.zeros(9, dtype=np.float32))
        group.attrs["event_observed"] = 0


def _write_spatial_h5(path: Path, rows):
    with h5py.File(path, "w") as h5:
        h5.attrs.update(dynamics_id="leo_coupled_v1", target_condition_schema="target_ood_conditions_v1")
        for trajectory in np.unique(rows.ids):
            index = np.flatnonzero(rows.ids == trajectory)
            group = h5.create_group(str(trajectory)); n = len(index)
            group.create_dataset("x_global", data=rows.x_global[index])
            group.create_dataset("x_nodes", data=rows.x_nodes[index])
            group.create_dataset("rul", data=rows.rul[index])
            group.create_dataset("label_fail", data=rows.event[index])
            group.create_dataset("time_s", data=rows.time[index])
            for position, name in enumerate(("d_perm", "q_trap", "r_th")):
                group.create_dataset(f"label_node_{name}", data=rows.node_states[index, :, position])
            for position, name in enumerate(("gain_dB", "phase_deg", "Pout_dBm", "PAE")):
                group.create_dataset(f"label_channel_{name}", data=rows.node_labels[index, :, position])
            for name, values in rows.array_score_truth.items():
                dataset = group.create_dataset(f"label_array_{name}", data=values[index])
                dataset.attrs["access"] = "score_only_not_model_input"
            group.attrs.update(
                eol_idx=n - 1, event_observed=0, array_twin_metadata_schema="subarray_array_twin_v1",
                array_twin_metadata_access="twin_only_not_model_input", scan_az_deg=0.0, margin0_dB=2.0,
                duty_cycle=(0.8 if str(trajectory) == "traj_005" else 0.4),
                Tj_base_C=(130.0 if str(trajectory) == "traj_005" else 100.0),
                array_grid=np.asarray([16, 16], dtype=np.int32), element_spacing_lambda=0.5, subarray_block=4,
            )


def test_spatial_ood_holdout_rejects_config_data_or_trajectory_drift(tmp_path):
    from src.experiments.run_gan_spatial_transfer import load_spatial_ood_holdout
    from src.experiments.run_gan_transfer import (
        build_ood_split_manifest, read_target_ood_conditions, split_target_iid_ood,
    )

    rows = _spatial_rows(n_trajectories=6)
    target = tmp_path / "target.h5"; _write_spatial_h5(target, rows)
    protocol = {"schema_version": "target_ood_conditions_v1", "abs_scan_az_deg_min": 30.0,
                "duty_cycle_min": 0.7, "Tj_base_C_min": 125.0}
    conditions = read_target_ood_conditions(target, protocol)
    conditions["traj_005"]["scan_az_deg"] = 35.0
    with h5py.File(target, "a") as h5:
        h5["traj_005"].attrs["scan_az_deg"] = 35.0
    split = split_target_iid_ood(sorted(conditions), conditions, protocol, seed=3)
    manifest_path = tmp_path / "ood_manifest.json"
    manifest_path.write_text(json.dumps(build_ood_split_manifest(protocol, conditions, split)), encoding="utf-8")

    holdout = load_spatial_ood_holdout(manifest_path, protocol, target, rows)
    assert set(holdout.split.ood_test_ids) == {"traj_005"}
    assert set(holdout.split.ood_test_ids).isdisjoint(holdout.iid_ids)
    with pytest.raises(ValueError, match="OOD split manifest ood_protocol"):
        load_spatial_ood_holdout(manifest_path, {**protocol, "duty_cycle_min": 0.75}, target, rows)
    with h5py.File(target, "a") as h5:
        h5["traj_000"].attrs["Tj_base_C"] = 101.0
    with pytest.raises(ValueError, match="condition_fingerprint"):
        load_spatial_ood_holdout(manifest_path, protocol, target, rows)
    with h5py.File(target, "a") as h5:
        h5["traj_000"].attrs["Tj_base_C"] = 100.0
    with pytest.raises(ValueError, match="空间轨迹集合"):
        load_spatial_ood_holdout(manifest_path, protocol, target, _spatial_rows(n_trajectories=5))


def test_spatial_split_and_node_scaler_are_train_trajectory_only():
    from src.experiments.run_gan_spatial_transfer import NodeTrainOnlyStandardizer, split_spatial_rows

    rows = _spatial_rows()
    train, val, test = split_spatial_rows(rows, seed=7)
    assert set(np.unique(train.ids)).isdisjoint(np.unique(val.ids))
    assert set(np.unique(train.ids)).isdisjoint(np.unique(test.ids))
    scaler = NodeTrainOnlyStandardizer().fit(train.x_nodes)
    assert np.allclose(scaler.mean_, train.x_nodes.mean(axis=(0, 1)))
    assert not np.allclose(scaler.transform(test.x_nodes).mean(axis=(0, 1)), 0.0)


def test_node_loss_excludes_score_only_truth_and_rollout_scores_four_array_metrics():
    from src.experiments.run_gan_spatial_transfer import (
        array_rollout_errors, spatial_node_one_step_loss,
    )
    from src.transfer.damage_state import DamageStateModel

    rows = _spatial_rows(n_trajectories=1)
    model = DamageStateModel(20, 6, hidden_dim=8, target_node_input_dim=6)
    loss = spatial_node_one_step_loss(model, rows, time_scale_s=21600.0, device=torch.device("cpu"))
    modified = _spatial_rows(n_trajectories=1)
    modified.array_score_truth["M_link_dB"] += 100.0
    changed_loss = spatial_node_one_step_loss(model, modified, time_scale_s=21600.0, device=torch.device("cpu"))
    assert torch.equal(loss, changed_loss), "score-only 阵列真值不得参与节点训练损失"
    errors = array_rollout_errors(model, modified, {"traj_000": {
        "scan_az_deg": 0.0, "margin0_dB": 2.0, "array_grid": (16, 16),
        "element_spacing_lambda": 0.5, "subarray_block": 4,
    }}, time_scale_s=21600.0, device=torch.device("cpu"))
    assert set(errors) == {"EIRP_norm_rmse", "SLL_dB_rmse", "theta_err_deg_rmse", "M_link_dB_rmse"}
    assert all(np.isfinite(value) for value in errors.values())


def test_array_rollout_requires_baseline_plus_exactly_six_steps():
    from src.experiments.run_gan_spatial_transfer import array_rollout_errors
    from src.transfer.damage_state import DamageStateModel

    rows = _spatial_rows(n_trajectories=1, n_time=6)
    model = DamageStateModel(20, 6, hidden_dim=8, target_node_input_dim=6)
    metadata = {"traj_000": {"scan_az_deg": 0.0, "margin0_dB": 2.0, "array_grid": (16, 16),
                              "element_spacing_lambda": 0.5, "subarray_block": 4}}
    with pytest.raises(ValueError, match="至少 7"):
        array_rollout_errors(model, rows, metadata, time_scale_s=21600.0, device=torch.device("cpu"))


def test_spatial_feature_schema_rejects_legacy_h5_with_rebuild_instruction(tmp_path):
    from src.experiments.run_gan_spatial_transfer import validate_spatial_feature_schema

    legacy = tmp_path / "legacy_target.h5"
    with h5py.File(legacy, "w") as h5:
        h5.create_group("traj_000")
    with pytest.raises(ValueError, match=r"python -m src\.sim\.build_array_hi --config"):
        validate_spatial_feature_schema(legacy)


def test_spatial_cli_smoke_writes_nonformal_independent_result(tmp_path, monkeypatch):
    from src.experiments import run_gan_spatial_transfer as spatial

    rows = _spatial_rows(n_trajectories=4)
    source, target, result = tmp_path / "source.h5", tmp_path / "target.h5", tmp_path / "spatial_smoke.json"
    _write_source_h5(source); _write_spatial_h5(target, rows)
    monkeypatch.setattr(spatial, "ROOT", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "source:\n  name: synthetic-GaN-RFALT\n  schema_version: gan_rfalt_v1\n  dynamics_id: rfalt_lumped_v1\n"
        f"  feature_path: {source.as_posix()}\ntarget:\n  dynamics_id: leo_coupled_v1\n  feature_path: {target.as_posix()}\n"
        "seed: 3\ntransfer:\n  transition_time_scale_s: 21600.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["run_gan_spatial_transfer", "--config", str(cfg), "--smoke", "--output", str(result)])
    spatial.main()
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["run_scope"] == "smoke_only_non_formal"
    assert payload["evaluation_horizon_steps"] == 6
    assert payload["smoke"] is True and set(payload["array_rollout_rmse"]) == {
        "EIRP_norm_rmse", "SLL_dB_rmse", "theta_err_deg_rmse", "M_link_dB_rmse",
    }


def test_spatial_cli_smoke_ood_holdout_trains_iid_and_reports_ood_separately(tmp_path, monkeypatch):
    from src.experiments import run_gan_spatial_transfer as spatial
    from src.experiments.run_gan_transfer import (
        build_ood_split_manifest, read_target_ood_conditions, split_target_iid_ood,
    )

    rows = _spatial_rows(n_trajectories=6)
    source, target = tmp_path / "source.h5", tmp_path / "target.h5"
    result, ood_manifest = tmp_path / "spatial_ood_smoke.json", tmp_path / "ood_manifest.json"
    _write_source_h5(source); _write_spatial_h5(target, rows)
    protocol = {"schema_version": "target_ood_conditions_v1", "abs_scan_az_deg_min": 30.0,
                "duty_cycle_min": 0.7, "Tj_base_C_min": 125.0}
    with h5py.File(target, "a") as h5:
        h5["traj_005"].attrs["scan_az_deg"] = 35.0
    conditions = read_target_ood_conditions(target, protocol)
    split = split_target_iid_ood(sorted(conditions), conditions, protocol, seed=3)
    ood_manifest.write_text(json.dumps(build_ood_split_manifest(protocol, conditions, split)), encoding="utf-8")
    monkeypatch.setattr(spatial, "ROOT", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "source:\n  name: synthetic-GaN-RFALT\n  schema_version: gan_rfalt_v1\n  dynamics_id: rfalt_lumped_v1\n"
        f"  feature_path: {source.as_posix()}\ntarget:\n  dynamics_id: leo_coupled_v1\n  feature_path: {target.as_posix()}\n"
        "  ood_holdout:\n    schema_version: target_ood_conditions_v1\n    abs_scan_az_deg_min: 30.0\n"
        "    duty_cycle_min: 0.7\n    Tj_base_C_min: 125.0\nseed: 3\n"
        "transfer:\n  transition_time_scale_s: 21600.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["run_gan_spatial_transfer", "--config", str(cfg), "--smoke",
                                        "--ood-holdout", str(ood_manifest), "--output", str(result)])
    spatial.main()
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["evaluation_scope"] == "iid_with_pre_registered_ood_holdout"
    assert set(payload["array_rollout_rmse"]) == set(payload["ood_array_rollout_rmse"])
    split_payload = payload["trajectory_split"]
    assert split_payload["ood_test"] == ["traj_005"]
    assert not ({*split_payload["train"], *split_payload["val"], *split_payload["iid_test"]} & set(split_payload["ood_test"]))
    assert "主验收" in payload["note"] and "OOD" in payload["note"]


def test_spatial_cli_rejects_short_test_trajectory_without_writing_result(tmp_path, monkeypatch):
    from src.experiments import run_gan_spatial_transfer as spatial

    rows = _spatial_rows(n_trajectories=4, n_time=6)
    source, target, result = tmp_path / "source.h5", tmp_path / "target.h5", tmp_path / "must_not_exist.json"
    _write_source_h5(source); _write_spatial_h5(target, rows)
    monkeypatch.setattr(spatial, "ROOT", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "source:\n  name: synthetic-GaN-RFALT\n  schema_version: gan_rfalt_v1\n  dynamics_id: rfalt_lumped_v1\n"
        f"  feature_path: {source.as_posix()}\ntarget:\n  dynamics_id: leo_coupled_v1\n  feature_path: {target.as_posix()}\n"
        "seed: 3\ntransfer:\n  transition_time_scale_s: 21600.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["run_gan_spatial_transfer", "--config", str(cfg), "--smoke", "--output", str(result)])
    with pytest.raises(ValueError, match="至少 7"):
        spatial.main()
    assert not result.exists()


def _formal_spatial_entry(group: str, seed: int, *, mlink: float, eirp: float = 0.2,
                          sll: float = 0.2, theta: float = 0.2):
    return {
        "group": group, "seed": seed, "smoke": False, "evaluation_horizon_steps": 6,
        "test_fingerprint": "same-test", "target_train_trajectory_count": 3,
        "trajectory_metrics": [{
            "trajectory_id": "traj_000", "test_trajectory_fingerprint": "same-traj", "prediction_steps": 6,
            "channel_6step_nrmse_by_metric": {"gain_dB": 0.2, "phase_deg": 0.2, "Pout_dBm": 0.2, "PAE": 0.2},
            "array_6step_nrmse_by_metric": {
                "EIRP_norm": eirp, "SLL_dB": sll, "theta_err_deg": theta, "M_link_dB": mlink,
            },
            "service_rul_s_truth": 100.0, "service_rul_s_pred": 100.0,
            "service_rul_abs_error_s": 0.0, "service_event_observed": True,
            "service_censor_lower_bound_violation": False,
        }],
    }


def test_spatial_formal_groups_require_ten_seeds_and_matched_source_budget():
    from src.experiments.run_gan_spatial_transfer import SPATIAL_GROUPS, validate_spatial_run_args

    assert SPATIAL_GROUPS == ("target_only", "gan_transition_init_spatial", "random_source_control_spatial")
    with pytest.raises(ValueError, match="--n-seeds 10"):
        validate_spatial_run_args(smoke=False, n_seeds=None)
    with pytest.raises(ValueError, match="非 smoke"):
        validate_spatial_run_args(smoke=True, n_seeds=10)
    assert validate_spatial_run_args(smoke=False, n_seeds=10) == 10


def test_init_and_random_source_controls_use_matched_pretraining_budget():
    from src.experiments.run_gan_spatial_transfer import _standardize_spatial, run_spatial_group, split_spatial_rows
    from src.experiments.run_gan_transfer import DomainRows

    rows = _spatial_rows(n_trajectories=4)
    train, val, test = split_spatial_rows(rows, seed=3)
    train, val, test, _ = _standardize_spatial(train, val, test)
    source = DomainRows(
        x=np.zeros((9, 20), dtype=np.float32), states=np.zeros((9, 3), dtype=np.float32),
        obs_labels=np.zeros((9, 4), dtype=np.float32), rul=np.arange(9, 0, -1, dtype=np.float32),
        event=np.zeros(9, dtype=bool), ids=np.array(["dev_00"] * 9), time=np.arange(9, dtype=np.float32) * 600.0,
        transition_time_scale_s=21600.0,
    )
    metadata = {trajectory: {"scan_az_deg": 0.0, "margin0_dB": 2.0, "array_grid": (16, 16),
                              "element_spacing_lambda": 0.5, "subarray_block": 4}
                for trajectory in np.unique(test.ids)}
    array_scale = {name: 1.0 for name in ("EIRP_norm", "SLL_dB", "theta_err_deg", "M_link_dB")}
    init = run_spatial_group("gan_transition_init_spatial", source, train, val, test, metadata, seed=9,
                             time_scale_s=21600.0, channel_scale=np.ones(4), array_scale=array_scale,
                             epochs=1, source_epochs=1, device=torch.device("cpu"))
    random = run_spatial_group("random_source_control_spatial", source, train, val, test, metadata, seed=9,
                               time_scale_s=21600.0, channel_scale=np.ones(4), array_scale=array_scale,
                               epochs=1, source_epochs=1, device=torch.device("cpu"))
    assert init["source_pretrain_steps"] == random["source_pretrain_steps"] == 1
    assert "等预算" in random["budget_note"]


def test_all_spatial_groups_share_identical_target_initial_state_before_ftheta_load():
    from src.experiments.run_gan_spatial_transfer import (
        _standardize_spatial, initialize_spatial_group_model, make_spatial_target_initial_state, split_spatial_rows,
    )
    from src.experiments.run_gan_transfer import DomainRows

    rows = _spatial_rows(n_trajectories=4)
    train, _, _ = split_spatial_rows(rows, seed=3)
    train, _ = _standardize_spatial(train)
    source = DomainRows(
        x=np.zeros((9, 20), dtype=np.float32), states=np.zeros((9, 3), dtype=np.float32),
        obs_labels=np.zeros((9, 4), dtype=np.float32), rul=np.arange(9, 0, -1, dtype=np.float32),
        event=np.zeros(9, dtype=bool), ids=np.array(["dev_00"] * 9), time=np.arange(9, dtype=np.float32) * 600.0,
        transition_time_scale_s=21600.0,
    )
    initial = make_spatial_target_initial_state(source, train, seed=11, hidden_dim=8, device=torch.device("cpu"))
    models = {
        group: initialize_spatial_group_model(group, source, train, target_initial_state=initial,
                                              source_epochs=1, hidden_dim=8, seed=11, device=torch.device("cpu"))[0]
        for group in ("target_only", "gan_transition_init_spatial", "random_source_control_spatial")
    }
    for name, value in initial.items():
        assert torch.equal(models["target_only"].state_dict()[name], value)
        if not name.startswith("transition."):
            assert torch.equal(models["gan_transition_init_spatial"].state_dict()[name], value)
            assert torch.equal(models["random_source_control_spatial"].state_dict()[name], value)


def test_spatial_formal_acceptance_uses_mlink_primary_and_gates_other_array_metrics():
    from src.experiments.run_gan_spatial_transfer import summarize_spatial_paired_acceptance

    entries = []
    for seed in range(10):
        entries.extend([
            _formal_spatial_entry("target_only", seed, mlink=0.4),
            _formal_spatial_entry("gan_transition_init_spatial", seed, mlink=0.2),
            _formal_spatial_entry("random_source_control_spatial", seed, mlink=0.3),
        ])
    accepted = summarize_spatial_paired_acceptance(entries, expected_seeds=range(10), n_bootstrap=100)["all"]
    assert accepted["formal_primary_metric"] == "M_link_dB_6step_nrmse"
    assert accepted["acceptance_pass"] is True and accepted["holm_adjusted_p"] is not None

    entries[1] = _formal_spatial_entry("gan_transition_init_spatial", 0, mlink=0.2, eirp=0.5)
    rejected = summarize_spatial_paired_acceptance(entries, expected_seeds=range(10), n_bootstrap=100)["all"]
    assert rejected["acceptance_pass"] is False
    assert "EIRP_norm_degraded" in rejected["failure_reasons"]

    entries[1] = _formal_spatial_entry("gan_transition_init_spatial", 0, mlink=0.2, eirp=float("nan"))
    nonfinite = summarize_spatial_paired_acceptance(entries, expected_seeds=range(10), n_bootstrap=100)["all"]
    assert nonfinite["acceptance_pass"] is False
    assert "EIRP_norm_nonfinite" in nonfinite["failure_reasons"]


def test_formal_spatial_mode_requires_at_least_300_target_trajectories():
    from src.experiments.run_gan_spatial_transfer import validate_formal_target_trajectory_count

    with pytest.raises(ValueError, match="至少 300"):
        validate_formal_target_trajectory_count(_spatial_rows(n_trajectories=4))


def test_signflip_uses_exact_small_n_and_deterministic_monte_carlo_for_270_trajectories():
    from src.experiments.run_gan_spatial_transfer import _exact_signflip_pvalue, signflip_pvalue

    small = np.array([-0.3, 0.1, 0.4, 0.2], dtype=float)
    exact = signflip_pvalue(small, exact_max_n=20, monte_carlo_samples=1_000, seed=17)
    assert exact["p_value_method"] == "exact_enumeration"
    assert exact["p_value"] == pytest.approx(_exact_signflip_pvalue(small))
    approximate_small = signflip_pvalue(small, exact_max_n=0, monte_carlo_samples=100_000, seed=17)
    assert approximate_small["p_value_method"] == "monte_carlo_signflip"
    assert approximate_small["p_value"] == pytest.approx(exact["p_value"], abs=0.01)
    large = np.linspace(-0.3, 0.4, 270)
    first = signflip_pvalue(large, exact_max_n=20, monte_carlo_samples=10_000, seed=20260728)
    second = signflip_pvalue(large, exact_max_n=20, monte_carlo_samples=10_000, seed=20260728)
    assert first == second
    assert first["p_value_method"] == "monte_carlo_signflip" and first["p_value_samples"] == 10_000
    assert first["p_value_seed"] == 20260728 and 0.0 < first["p_value"] <= 1.0
    values, samples, seed = np.array([1.0, 2.0]), 2, 9
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(samples, len(values)), dtype=np.int8).astype(float) * 2.0 - 1.0
    expected_extreme = int(np.sum(np.abs(signs @ values / len(values)) >= abs(values.mean()) - 1e-12))
    corrected = signflip_pvalue(values, exact_max_n=0, monte_carlo_samples=samples, seed=seed)
    assert corrected["p_value"] == pytest.approx((expected_extreme + 1) / (samples + 1))
    assert corrected["p_value_observed_sign_included"] is False


def test_spatial_checkpoint_preserves_entries_for_later_summary(tmp_path):
    from src.experiments.run_gan_spatial_transfer import (
        build_spatial_checkpoint_manifest, load_spatial_metrics_checkpoint, persist_spatial_metrics_checkpoint,
        validate_spatial_checkpoint_manifest,
    )
    from src.experiments.run_gan_transfer import DomainRows

    entries = [_formal_spatial_entry("target_only", 0, mlink=0.4)]
    rows = _spatial_rows(n_trajectories=300)
    config = {"seed": 42, "transfer": {"spatial_primary_min_delta_nrmse": 0.02,
                                           "spatial_non_degradation_min_delta_nrmse": 0.0}}
    source = DomainRows(
        x=np.zeros((9, 20), dtype=np.float32), states=np.zeros((9, 3), dtype=np.float32),
        obs_labels=np.zeros((9, 4), dtype=np.float32), rul=np.arange(9, 0, -1, dtype=np.float32),
        event=np.zeros(9, dtype=bool), ids=np.array(["dev_00"] * 9), time=np.arange(9, dtype=np.float32) * 600.0,
    )
    metadata = {trajectory: {"scan_az_deg": 0.0, "margin0_dB": 2.0, "array_grid": (16, 16),
                             "element_spacing_lambda": 0.5, "subarray_block": 4}
                for trajectory in np.unique(rows.ids)}
    manifest = build_spatial_checkpoint_manifest(config, source, rows, metadata, source_dynamics_id="rfalt_lumped_v1",
                                                 target_dynamics_id="leo_coupled_v1", expected_seeds=range(42, 52))
    checkpoint = tmp_path / "spatial_checkpoint.json"
    persist_spatial_metrics_checkpoint(checkpoint, entries, manifest)
    loaded_entries, loaded_manifest = load_spatial_metrics_checkpoint(checkpoint)
    assert loaded_entries == entries
    validate_spatial_checkpoint_manifest(loaded_manifest, manifest)
    changed = {**manifest, "config_canonical_hash": "different"}
    with pytest.raises(ValueError, match="config_canonical_hash"):
        validate_spatial_checkpoint_manifest(loaded_manifest, changed)
    changed_rows = _spatial_rows(n_trajectories=300)
    changed_rows.node_states[0, 0, 0] += 1.0
    changed_feature = build_spatial_checkpoint_manifest(config, source, changed_rows, metadata,
                                                        source_dynamics_id="rfalt_lumped_v1", target_dynamics_id="leo_coupled_v1",
                                                        expected_seeds=range(42, 52))
    with pytest.raises(ValueError, match="target_feature_fingerprint"):
        validate_spatial_checkpoint_manifest(loaded_manifest, changed_feature)
    changed_metadata = {key: dict(value) for key, value in metadata.items()}
    changed_metadata["traj_000"]["scan_az_deg"] = 1.0
    changed_twin = build_spatial_checkpoint_manifest(config, source, rows, changed_metadata,
                                                     source_dynamics_id="rfalt_lumped_v1", target_dynamics_id="leo_coupled_v1",
                                                     expected_seeds=range(42, 52))
    with pytest.raises(ValueError, match="array_twin_metadata_fingerprint"):
        validate_spatial_checkpoint_manifest(loaded_manifest, changed_twin)
    source_fields = ("x", "states", "obs_labels", "rul", "event", "ids", "time")
    for field in source_fields:
        values = {name: np.array(getattr(source, name), copy=True) for name in source_fields}
        if field == "event":
            values[field][0] = True
        elif field == "ids":
            values[field][0] = "dev_changed"
        else:
            values[field].flat[0] += 1.0
        changed_source = DomainRows(**values)
        changed_manifest = build_spatial_checkpoint_manifest(config, changed_source, rows, metadata,
                                                              source_dynamics_id="rfalt_lumped_v1", target_dynamics_id="leo_coupled_v1",
                                                              expected_seeds=range(42, 52))
        with pytest.raises(ValueError, match="source_rows_fingerprint"):
            validate_spatial_checkpoint_manifest(loaded_manifest, changed_manifest)
    with pytest.raises(ValueError, match="至少 300"):
        build_spatial_checkpoint_manifest(config, source, _spatial_rows(n_trajectories=4), metadata,
                                           source_dynamics_id="rfalt_lumped_v1", target_dynamics_id="leo_coupled_v1",
                                           expected_seeds=range(42, 52))


def test_spatial_trajectory_metrics_save_channel_array_and_service_fields():
    from src.experiments.run_gan_spatial_transfer import spatial_trajectory_metrics
    from src.transfer.damage_state import DamageStateModel

    rows = _spatial_rows(n_trajectories=1, n_time=9)
    rows.event[:] = True
    model = DamageStateModel(20, 6, hidden_dim=8, target_node_input_dim=6)
    metadata = {"traj_000": {"scan_az_deg": 0.0, "margin0_dB": 2.0, "array_grid": (16, 16),
                              "element_spacing_lambda": 0.5, "subarray_block": 4}}
    metrics = spatial_trajectory_metrics(model, rows, metadata, channel_scale=np.ones(4),
                                         array_scale={name: 1.0 for name in ("EIRP_norm", "SLL_dB", "theta_err_deg", "M_link_dB")},
                                         time_scale_s=21600.0, device=torch.device("cpu"))
    item = metrics[0]
    assert item["prediction_steps"] == 6
    assert set(item["channel_6step_nrmse_by_metric"]) == {"gain_dB", "phase_deg", "Pout_dBm", "PAE"}
    assert set(item["array_6step_nrmse_by_metric"]) == {"EIRP_norm", "SLL_dB", "theta_err_deg", "M_link_dB"}
    assert {"service_rul_s_truth", "service_rul_s_pred", "service_censor_lower_bound_violation"}.issubset(item)


def test_formal_spatial_result_writes_json_and_markdown(tmp_path):
    from src.experiments.run_gan_spatial_transfer import persist_spatial_formal_results

    entries = [_formal_spatial_entry("target_only", seed, mlink=0.4) for seed in range(10)]
    entries += [_formal_spatial_entry("gan_transition_init_spatial", seed, mlink=(0.5 if seed == 4 else 0.4))
                for seed in range(10)]
    entries += [_formal_spatial_entry("random_source_control_spatial", seed, mlink=(0.5 if seed == 4 else 0.4))
                for seed in range(10)]
    for entry in entries:
        for item in entry["trajectory_metrics"]:
            item["service_rul_s_pred"] = None
            item["service_rul_abs_error_s"] = None
    json_path, md_path = tmp_path / "formal.json", tmp_path / "formal.md"
    persist_spatial_formal_results(json_path, md_path, entries, n_bootstrap=100, expected_seeds=range(10))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run_scope"] == "formal_preregistered_spatial" and payload["acceptance"]["all"]["formal_primary_metric"] == "M_link_dB_6step_nrmse"
    assert payload["acceptance"]["all"]["p_value_method"] == "exact_enumeration"
    transparency = payload["result_transparency"]
    assert transparency["seed_mlink_delta_target_minus_source_init"]["4"] == pytest.approx(-0.1)
    assert transparency["seed_mlink_delta_target_minus_source_init"]["0"] == pytest.approx(0.0)
    assert transparency["seed_mlink_delta_target_minus_random_scramble"]["4"] == pytest.approx(-0.1)
    assert transparency["service_layer"]["status"] == "not_evaluable_no_predicted_event_in_six_step_window"
    assert transparency["service_layer"]["service_event_mae_s"] is None
    report = md_path.read_text(encoding="utf-8")
    assert "M_link" in report and "Holm" in report and "state-label-scramble" in report
    assert "服务事件 MAE" in report and "EIRP_norm" in report and "no-op" in report
    assert "逐 seed M_link 差值" in report and "服务层当前不可评估" in report


def test_spatial_checkpoint_binds_ood_holdout_and_report_keeps_it_out_of_iid_acceptance(tmp_path):
    from src.experiments.run_gan_spatial_transfer import (
        SpatialOODHoldout, build_spatial_checkpoint_manifest, persist_spatial_formal_results,
        validate_spatial_checkpoint_manifest,
    )
    from src.experiments.run_gan_transfer import DomainRows, TargetOODSplit

    protocol = {"schema_version": "target_ood_conditions_v1", "abs_scan_az_deg_min": 30.0,
                "duty_cycle_min": 0.7, "Tj_base_C_min": 125.0}
    holdout = SpatialOODHoldout(
        split=TargetOODSplit(("traj_000",), ("traj_001",), ("traj_002",), ("traj_299",)),
        manifest={"ood_protocol": protocol, "condition_fingerprint": "conditions", "ood_test_fingerprint": "ood"},
        manifest_fingerprint="manifest",
    )
    rows = _spatial_rows(n_trajectories=300)
    source = DomainRows(
        x=np.zeros((9, 20), dtype=np.float32), states=np.zeros((9, 3), dtype=np.float32),
        obs_labels=np.zeros((9, 4), dtype=np.float32), rul=np.arange(9, 0, -1, dtype=np.float32),
        event=np.zeros(9, dtype=bool), ids=np.array(["dev_00"] * 9), time=np.arange(9, dtype=np.float32) * 600.0,
    )
    config = {"seed": 42, "target": {"ood_holdout": protocol}, "transfer": {
        "spatial_primary_min_delta_nrmse": 0.02, "spatial_non_degradation_min_delta_nrmse": 0.0,
    }}
    metadata = {trajectory: {"scan_az_deg": 0.0, "margin0_dB": 2.0, "array_grid": (16, 16),
                             "element_spacing_lambda": 0.5, "subarray_block": 4}
                for trajectory in np.unique(rows.ids)}
    manifest = build_spatial_checkpoint_manifest(
        config, source, rows, metadata, source_dynamics_id="rfalt_lumped_v1", target_dynamics_id="leo_coupled_v1",
        expected_seeds=range(10), ood_holdout=holdout,
    )
    changed = dict(manifest); changed["ood_holdout"] = {**manifest["ood_holdout"], "manifest_fingerprint": "changed"}
    with pytest.raises(ValueError, match="ood_holdout"):
        validate_spatial_checkpoint_manifest(manifest, changed)

    entries = []
    for seed in range(10):
        for group, mlink in (("target_only", 0.4), ("gan_transition_init_spatial", 0.2),
                             ("random_source_control_spatial", 0.3)):
            entry = _formal_spatial_entry(group, seed, mlink=mlink)
            entry["ood_trajectory_metrics"] = [{
                **entry["trajectory_metrics"][0], "trajectory_id": "traj_299",
                "array_6step_nrmse_by_metric": {"EIRP_norm": 0.3, "SLL_dB": 0.3,
                                                   "theta_err_deg": 0.3, "M_link_dB": 0.3},
            }]
            entries.append(entry)
    json_path, markdown_path = tmp_path / "formal.json", tmp_path / "formal.md"
    persist_spatial_formal_results(json_path, markdown_path, entries, n_bootstrap=100, expected_seeds=range(10),
                                   ood_holdout=holdout)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["acceptance"]["all"]["formal_primary_metric"] == "M_link_dB_6step_nrmse"
    assert payload["ood_evaluation"]["status"] == "separate_not_in_iid_acceptance"
    assert payload["ood_evaluation"]["holdout"]["manifest_fingerprint"] == "manifest"
    report = markdown_path.read_text(encoding="utf-8")
    assert "OOD 保持集" in report and "不参与 IID 主验收" in report
