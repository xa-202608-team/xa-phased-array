"""GaN 实验的数据隔离和对照组卫生契约。"""
from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_split_is_trajectory_level_and_disjoint():
    from src.experiments.run_gan_transfer import split_target_trajectories

    train, val, test = split_target_trajectories([f"traj_{i:03d}" for i in range(20)], seed=42)

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert len(train) + len(val) + len(test) == 20


def test_stratified_target_split_keeps_failed_and_censored_trajectories_in_test():
    from src.experiments.run_gan_transfer import split_target_trajectories

    ids = [f"traj_{i:03d}" for i in range(20)]
    failed = {"traj_002", "traj_005", "traj_006", "traj_007", "traj_009", "traj_012", "traj_015"}
    events = {trajectory: trajectory in failed for trajectory in ids}
    train, val, test = split_target_trajectories(ids, seed=42, event_by_id=events)

    assert any(events[trajectory] for trajectory in test)
    assert any(not events[trajectory] for trajectory in test)
    assert set(train).isdisjoint(test) and set(val).isdisjoint(test)


def test_scaler_fit_uses_train_rows_only():
    from src.experiments.run_gan_transfer import TrainOnlyStandardizer

    train = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    test = np.array([[1000.0, 1000.0]], dtype=np.float32)
    scaler = TrainOnlyStandardizer().fit(train)

    assert np.allclose(scaler.mean_, [1.0, 2.0])
    assert np.allclose(scaler.transform(test)[0], [999.0, 998.0])


def test_random_source_control_scrambles_pairs_but_preserves_time_order():
    from src.experiments.run_gan_transfer import scramble_source_state_pairs

    states = np.arange(18, dtype=np.float32).reshape(6, 3)
    device_ids = np.array(["a", "a", "a", "b", "b", "b"])
    time = np.array([0, 1, 2, 0, 1, 2])
    scrambled, returned_time = scramble_source_state_pairs(states, device_ids, time, seed=9)

    assert np.array_equal(returned_time, time)
    for device in ("a", "b"):
        mask = device_ids == device
        assert {tuple(row) for row in scrambled[mask]} == {tuple(row) for row in states[mask]}
    assert not np.array_equal(scrambled, states)


def test_training_inputs_exclude_latent_labels_and_future_values():
    from src.experiments.run_gan_transfer import make_target_observations

    x_global = np.arange(24, dtype=np.float32).reshape(4, 6)
    x_nodes = np.arange(96, dtype=np.float32).reshape(4, 4, 6)
    latent = np.random.default_rng(0).normal(size=(4, 256)).astype(np.float32)
    labels = np.random.default_rng(1).normal(size=(4, 4)).astype(np.float32)
    x_before = make_target_observations(x_global, x_nodes, latent, labels)
    x_after = make_target_observations(x_global, x_nodes, latent[::-1], labels[::-1])

    assert np.array_equal(x_before, x_after)
    assert x_before.shape == (4, 12)


def test_spatial_observation_interface_preserves_nodes_without_labels():
    from src.experiments.run_gan_transfer import make_target_observations

    x_global = np.arange(24, dtype=np.float32).reshape(4, 6)
    x_nodes = np.arange(384, dtype=np.float32).reshape(4, 16, 6)
    latent = np.random.default_rng(2).normal(size=(4, 256)).astype(np.float32)
    score_labels = np.random.default_rng(3).normal(size=(4, 5)).astype(np.float32)
    global_obs, node_obs = make_target_observations(
        x_global, x_nodes, latent, score_labels, preserve_node_axis=True)
    changed_global, changed_nodes = make_target_observations(
        x_global, x_nodes, latent[::-1], score_labels[::-1], preserve_node_axis=True)

    assert global_obs.shape == (4, 6) and node_obs.shape == (4, 16, 6)
    assert np.array_equal(global_obs, changed_global)
    assert np.array_equal(node_obs, changed_nodes)
    assert np.array_equal(node_obs, x_nodes)


def test_main_experiment_uses_only_declared_damage_state_groups():
    source = (ROOT / "src" / "experiments" / "run_gan_transfer.py").read_text(encoding="utf-8")

    assert "target_only" in source
    assert "gan_transition_init" in source
    assert "gan_joint_no_align" in source
    assert "random_source_control" in source
    assert "latent_legacy" not in source


def test_dynamics_ids_are_checked_as_distinct_before_training():
    from src.experiments.run_gan_transfer import validate_domain_separation

    validate_domain_separation("rfalt_lumped_v1", "leo_coupled_v1")
    try:
        validate_domain_separation("same", "same")
    except ValueError as exc:
        assert "dynamics_id" in str(exc)
    else:
        raise AssertionError("相同 dynamics_id 必须拒绝")


def test_early_stop_uses_validation_not_test_static():
    source = (ROOT / "src" / "experiments" / "run_gan_transfer.py").read_text(encoding="utf-8")
    assert "val_loss" in source
    assert "test_loader" not in source.split("def train_with_early_stop", 1)[1].split("def ", 1)[0]


def test_transition_pretraining_accepts_source_only_rows():
    """source-init 的 Fθ 预训练不能借用目标 train/val 轨迹。"""
    import torch
    from src.experiments.run_gan_transfer import DomainRows, pretrain_source_transition
    from src.transfer.damage_state import DamageStateModel

    rows = DomainRows(
        x=np.random.default_rng(3).normal(size=(5, 20)).astype(np.float32),
        states=np.random.default_rng(4).random(size=(5, 3)).astype(np.float32),
        obs_labels=np.random.default_rng(5).normal(size=(5, 4)).astype(np.float32),
        rul=np.ones(5, dtype=np.float32), event=np.zeros(5, dtype=bool),
        ids=np.array(["source"] * 5), time=np.arange(5),
    )
    model = DamageStateModel(20, 12)
    pretrain_source_transition(model, rows, epochs=1, device=torch.device("cpu"))


def test_target_rul_normalization_fits_train_rows_only():
    from src.experiments.run_gan_transfer import DomainRows, normalize_target_rul

    train = DomainRows(np.zeros((2, 1), np.float32), np.zeros((2, 3), np.float32),
                       np.zeros((2, 4), np.float32), np.array([10.0, 5.0], np.float32),
                       np.array([True, True]), np.array(["tr", "tr"]), np.arange(2))
    test = DomainRows(np.zeros((1, 1), np.float32), np.zeros((1, 3), np.float32),
                      np.zeros((1, 4), np.float32), np.array([1000.0], np.float32),
                      np.array([True]), np.array(["te"]), np.array([0]))
    normalized_train, normalized_test, scale = normalize_target_rul(train, test)

    assert scale == 10.0
    assert np.allclose(normalized_train.rul, [1.0, 0.5])
    assert np.allclose(normalized_test.rul, [100.0])


def test_target_loss_uses_transition_static():
    source = (ROOT / "src" / "experiments" / "run_gan_transfer.py").read_text(encoding="utf-8")
    body = source.split("def _target_loss", 1)[1].split("def ", 1)[0]
    assert "model.transition" in body and "state_anchor_loss" in body


def _mini_rows(input_dim: int, prefix: str):
    rng = np.random.default_rng(input_dim)
    return __import__("src.experiments.run_gan_transfer", fromlist=["DomainRows"]).DomainRows(
        x=rng.normal(size=(7, input_dim)).astype(np.float32),
        states=rng.random(size=(7, 3)).astype(np.float32),
        obs_labels=rng.normal(size=(7, 4)).astype(np.float32),
        rul=rng.random(7).astype(np.float32), event=np.array([True] * 7),
        ids=np.array([prefix] * 7), time=np.arange(7),
    )


def test_init_and_random_control_have_identical_two_stage_step_budgets():
    import torch
    from src.experiments.run_gan_transfer import run_group

    source = _mini_rows(20, "source")
    train = _mini_rows(12, "train")
    val = _mini_rows(12, "val")
    test = _mini_rows(12, "test")
    init = run_group("gan_transition_init", source, train, val, test, seed=7, epochs=2, device=torch.device("cpu"))
    random = run_group("random_source_control", source, train, val, test, seed=7, epochs=2, device=torch.device("cpu"))

    assert init["source_pretrain_steps"] == random["source_pretrain_steps"] == 2
    assert init["target_steps"] == random["target_steps"] == 2
    assert init["source_joint_steps"] == random["source_joint_steps"] == 0
    assert init["source_observation_supervision"] is True
    assert random["source_observation_supervision"] is False


def test_budget_records_explain_target_only_and_joint_designs():
    import torch
    from src.experiments.run_gan_transfer import run_group

    source = _mini_rows(20, "source")
    train = _mini_rows(12, "train")
    val = _mini_rows(12, "val")
    test = _mini_rows(12, "test")
    target = run_group("target_only", source, train, val, test, seed=8, epochs=1, device=torch.device("cpu"))
    joint = run_group("gan_joint_no_align", source, train, val, test, seed=8, epochs=1, device=torch.device("cpu"))

    assert target["source_pretrain_steps"] == 0 and target["target_steps"] == 1
    assert joint["source_pretrain_steps"] == 0 and joint["source_joint_steps"] == 1
    assert "design" in target["budget_note"] and "design" in joint["budget_note"]
    assert target["source_observation_supervision"] is False
    assert joint["source_observation_supervision"] is True


def test_oracle_and_nasa_groups_are_rejected_at_runtime():
    from src.experiments.run_gan_transfer import validate_experiment_groups

    for forbidden in ("shared_simulator_oracle", "nasa_negative_control"):
        try:
            validate_experiment_groups(["target_only", forbidden])
        except ValueError as exc:
            assert "禁止" in str(exc)
        else:
            raise AssertionError(f"{forbidden} 必须拒绝")


def test_source_config_whitelist_rejects_nasa_replacement():
    from src.experiments.run_gan_transfer import validate_source_config

    valid = {"name": "synthetic-GaN-RFALT", "schema_version": "gan_rfalt_v1", "dynamics_id": "rfalt_lumped_v1"}
    validate_source_config(valid)
    invalid = {**valid, "name": "NASA-MOSFET"}
    try:
        validate_source_config(invalid)
    except ValueError as exc:
        assert "GaN RFALT" in str(exc)
    else:
        raise AssertionError("NASA 配置替换必须拒绝")


def test_source_h5_whitelist_rejects_non_rfalt_metadata(tmp_path):
    import h5py
    from src.experiments.run_gan_transfer import validate_source_h5_contract

    path = tmp_path / "nasa_like.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "nasa_mosfet_v2"
        h5.attrs["dynamics_id"] = "mosfet_switching_v1"
    try:
        validate_source_h5_contract(path)
    except ValueError as exc:
        message = str(exc)
        assert "schema_version" in message or "dynamics_id" in message
    else:
        raise AssertionError("伪 NASA HDF5 必须在字段读取前拒绝")


def _write_minimal_rfalt_h5(path):
    import h5py
    names = [
        "T_base_C", "T_j_C", "VDS", "VGS", "ID", "IG", "duty_cycle", "PAPR_dB", "VSWR", "Pin_dBm",
        "Pout_dBm", "gain_dB", "PAE", "AM_AM_dB", "AM_PM_deg", "EVM_pct", "ACPR_dBc", "RDS_dynamic_ohm", "gm_S", "Vth_V",
    ]
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "gan_rfalt_v1"
        h5.attrs["dynamics_id"] = "rfalt_lumped_v1"
        h5.attrs["feature_names"] = names
        h5.attrs["feature_dim"] = len(names)
        group = h5.create_group("devices").create_group("rfalt-0")
        group.create_dataset("x", data=np.zeros((3, len(names)), np.float32))
        for name in ("time_s", "rul_lower_bound_s", "latent_d_perm", "latent_q_trap", "latent_r_th"):
            group.create_dataset(name, data=np.arange(3, dtype=np.float32))
        group.attrs["event_observed"] = False


def test_source_h5_whitelist_accepts_minimal_rfalt(tmp_path):
    from src.experiments.run_gan_transfer import validate_source_h5_contract

    path = tmp_path / "rfalt.h5"
    _write_minimal_rfalt_h5(path)
    validate_source_h5_contract(path)


def test_source_h5_contract_rejects_missing_required_dataset(tmp_path):
    from src.experiments.run_gan_transfer import validate_source_h5_contract
    import h5py

    path = tmp_path / "incomplete.h5"
    _write_minimal_rfalt_h5(path)
    with h5py.File(path, "a") as h5:
        del h5["devices/rfalt-0/latent_q_trap"]
    try:
        validate_source_h5_contract(path)
    except ValueError as exc:
        assert "latent_q_trap" in str(exc)
    else:
        raise AssertionError("缺少必需 latent 数据集必须拒绝")


def test_source_h5_contract_decodes_bytes_names_and_rejects_wrong_columns_or_nonfinite(tmp_path):
    import h5py
    from src.experiments.run_gan_transfer import validate_source_h5_contract

    path = tmp_path / "rfalt.h5"
    _write_minimal_rfalt_h5(path)
    with h5py.File(path, "a") as h5:
        names = [name.encode("utf-8") for name in h5.attrs["feature_names"]]
        del h5.attrs["feature_names"]
        h5.attrs["feature_names"] = names
    validate_source_h5_contract(path)
    with h5py.File(path, "a") as h5:
        names = [name if isinstance(name, bytes) else str(name).encode("utf-8") for name in h5.attrs["feature_names"]]
        names[0] = b"not_T_base_C"
        del h5.attrs["feature_names"]; h5.attrs["feature_names"] = np.asarray(names, dtype="S32")
    try:
        validate_source_h5_contract(path)
    except ValueError as exc:
        assert "feature_names" in str(exc)
    else:
        raise AssertionError("错误特征列必须拒绝")
    finite_path = tmp_path / "nonfinite.h5"
    _write_minimal_rfalt_h5(finite_path)
    with h5py.File(finite_path, "a") as h5:
        h5["devices/rfalt-0/x"][0, 0] = np.nan
    try:
        validate_source_h5_contract(finite_path)
    except ValueError as exc:
        assert "非有限值" in str(exc)
    else:
        raise AssertionError("非有限源特征必须拒绝")


def test_transition_pairs_align_t_plus_one_and_never_cross_trajectory():
    from src.experiments.run_gan_transfer import DomainRows, make_transition_pairs

    rows = DomainRows(
        x=np.arange(20, dtype=np.float32).reshape(5, 4),
        states=np.arange(15, dtype=np.float32).reshape(5, 3),
        obs_labels=np.arange(20, dtype=np.float32).reshape(5, 4),
        rul=np.arange(5, dtype=np.float32), event=np.ones(5, dtype=bool),
        ids=np.array(["a", "a", "b", "b", "b"]), time=np.array([0., 4., 0., 2., 6.]),
    )
    pairs = make_transition_pairs(rows)

    assert pairs.x_t.shape[0] == 3
    assert np.array_equal(pairs.state_t, rows.states[[0, 2, 3]])
    assert np.array_equal(pairs.state_t_plus_1, rows.states[[1, 3, 4]])
    assert np.array_equal(pairs.stress_t, rows.x[[0, 2, 3]])
    assert np.array_equal(pairs.time_t, np.array([0., 0., 2.]))
    assert np.array_equal(pairs.time_t_plus_1, np.array([4., 2., 6.]))
    assert np.array_equal(pairs.obs_t_plus_1, rows.obs_labels[[1, 3, 4]])
    assert np.all(pairs.normalized_dt > 0)
    assert np.array_equal(pairs.ids, np.array(["a", "b", "b"]))


def test_state_t_anchor_labels_change_loss_but_never_model_inputs():
    import torch
    from src.experiments.run_gan_transfer import DomainRows, make_transition_pairs, state_anchor_loss

    rows = DomainRows(
        x=np.arange(16, dtype=np.float32).reshape(4, 4), states=np.zeros((4, 3), np.float32),
        obs_labels=np.zeros((4, 4), np.float32), rul=np.ones(4, np.float32), event=np.ones(4, bool),
        ids=np.array(["a"] * 4), time=np.arange(4, dtype=np.float32),
    )
    changed = DomainRows(rows.x, rows.states.copy(), rows.obs_labels, rows.rul, rows.event, rows.ids, rows.time)
    changed.states[0] = 3.0
    original_pairs, changed_pairs = make_transition_pairs(rows), make_transition_pairs(changed)
    pred = torch.zeros((3, 3))

    assert np.array_equal(original_pairs.x_t, changed_pairs.x_t)
    assert np.array_equal(original_pairs.stress_t, changed_pairs.stress_t)
    assert state_anchor_loss(pred, original_pairs).item() != state_anchor_loss(pred, changed_pairs).item()


def test_fixed_transition_time_scale_is_not_per_domain_median():
    from src.experiments.run_gan_transfer import DomainRows, make_transition_pairs

    common = dict(x=np.zeros((3, 2), np.float32), states=np.zeros((3, 3), np.float32),
                  obs_labels=np.zeros((3, 4), np.float32), rul=np.ones(3, np.float32), event=np.ones(3, bool),
                  ids=np.array(["a"] * 3), transition_time_scale_s=21600.0)
    source = DomainRows(**common, time=np.array([0.0, 600.0, 1200.0]))
    target = DomainRows(**common, time=np.array([0.0, 21600.0, 43200.0]))

    assert np.allclose(make_transition_pairs(source).normalized_dt, 600.0 / 21600.0)
    assert np.allclose(make_transition_pairs(target).normalized_dt, 1.0)


def test_formal_run_uses_all_requested_counts_and_persists_count_seed_metrics(tmp_path):
    from src.experiments.run_gan_transfer import build_experiment_schedule, persist_metrics
    import json

    assert build_experiment_schedule([1, 3, 5, 10], smoke=False) == [1, 3, 5, 10]
    assert build_experiment_schedule([1, 3, 5, 10], smoke=True) == [1, 3, 5, 10]
    output = tmp_path / "metrics.json"
    persist_metrics(output, [{"target_train_count": 3, "seed": 42, "rmse": 0.1}])
    assert json.loads(output.read_text(encoding="utf-8"))[0]["target_train_count"] == 3


def _paired_metrics(count: int, *, delta: float = 0.03, scope: str = "failed_rows", failed: int = 12):
    metrics = []
    for seed in range(42, 52):
        target_rmse = 0.50 + seed * 0.001
        trajectories = []
        for index, trajectory_delta in enumerate((delta * 0.5, delta * 1.5)):
            base = 0.20 + index * 0.05 + seed * 1e-5
            trajectories.append((f"traj_{index:03d}", base, trajectory_delta))
        target_trajectory_metrics = [
            {"trajectory_id": trajectory, "test_trajectory_fingerprint": f"fp-{trajectory}",
             "n_test_rows": 7, "n_failed_rows": 7 if failed else 0,
             "channel_multistep_nrmse": base, "array_link_margin_mae_dB": 0.1,
             "channel_6step_nrmse_by_metric": {"gain_dB": base, "phase_deg": base,
                                                "Pout_dBm": base, "PAE": base},
             "channel_6step_nrmse_mean": base,
             "prediction_steps": 6, "service_event_mae_s": 3.0 if failed else None,
             "service_censor_lower_bound_violation_rate": None}
            for trajectory, base, _ in trajectories
        ]
        init_trajectory_metrics = [
            {**item, "channel_multistep_nrmse": item["channel_multistep_nrmse"] - trajectory_delta,
             "channel_6step_nrmse_by_metric": {name: value - trajectory_delta
                                                for name, value in item["channel_6step_nrmse_by_metric"].items()},
             "channel_6step_nrmse_mean": item["channel_6step_nrmse_mean"] - trajectory_delta}
            for item, (_, _, trajectory_delta) in zip(target_trajectory_metrics, trajectories)
        ]
        metrics.extend([
            {"group": "target_only", "target_train_count": count, "seed": seed, "rmse": target_rmse,
             "metric_scope": scope, "n_failed_rows": failed, "n_test_rows": 14,
             "test_fingerprint": "same-test", "trajectory_metrics": target_trajectory_metrics,
             "evaluation_horizon_steps": 6},
            {"group": "gan_transition_init", "target_train_count": count, "seed": seed,
             "rmse": target_rmse - delta, "metric_scope": scope, "n_failed_rows": failed},
        ])
        metrics[-1].update(n_test_rows=14, test_fingerprint="same-test", trajectory_metrics=init_trajectory_metrics,
                           evaluation_horizon_steps=6)
    return metrics


def test_primary_paired_acceptance_uses_fixed_bootstrap_and_preregistered_gates():
    from src.experiments.run_gan_transfer import summarize_paired_acceptance

    options = dict(n_bootstrap=10_000, bootstrap_seed=20260728, primary_counts=[3, 5], min_delta_nrmse=0.02)
    first = summarize_paired_acceptance(_paired_metrics(3), **options)
    second = summarize_paired_acceptance(_paired_metrics(3), **options)
    stats = first["3"]

    assert stats == second["3"]
    assert stats["acceptance_tier"] == "primary"
    assert stats["mean_delta"] >= 0.02
    assert stats["ci95_lo"] > 0
    assert stats["positive_seed_count"] == 10
    assert stats["acceptance_pass"] is True
    assert stats["bootstrap_unit"] == "test_trajectory"
    assert stats["n_bootstrap_trajectories"] == 2


def test_formal_channel_primary_is_arithmetic_mean_not_legacy_joint_rms():
    from src.experiments.run_gan_transfer import summarize_paired_acceptance

    entries = _paired_metrics(3)
    for entry in entries:
        for item in entry["trajectory_metrics"]:
            if entry["group"] == "target_only":
                item["channel_6step_nrmse_by_metric"] = {"gain_dB": 1.0, "phase_deg": 3.0, "Pout_dBm": 5.0, "PAE": 7.0}
                item["channel_6step_nrmse_mean"] = 4.0
            else:
                item["channel_6step_nrmse_by_metric"] = {"gain_dB": 1.0, "phase_deg": 1.0, "Pout_dBm": 1.0, "PAE": 1.0}
                item["channel_6step_nrmse_mean"] = 1.0
            item["channel_multistep_nrmse"] = -99.0  # 旧整体 RMS 不得进入正式验收。
    stats = summarize_paired_acceptance(entries, primary_counts=[3, 5], min_delta_nrmse=0.02)["3"]

    assert stats["formal_primary_metric"] == "channel_6step_nrmse_arithmetic_mean"
    assert stats["mean_delta"] == 3.0


def test_acceptance_is_invalid_without_failed_rows_and_supportive_for_k1_k10():
    from src.experiments.run_gan_transfer import summarize_paired_acceptance

    options = dict(primary_counts=[3, 5], min_delta_nrmse=0.02)
    invalid = summarize_paired_acceptance(_paired_metrics(3, scope="all_rows_pipeline_connectivity_only", failed=0), **options)["3"]
    supportive = summarize_paired_acceptance(_paired_metrics(1), **options)["1"]

    assert invalid["formal_valid"] is False
    assert invalid["acceptance_status"] == "invalid"
    assert "failed_rows" in invalid["failure_reasons"]
    assert supportive["acceptance_tier"] == "supportive_holm_exploratory"
    assert supportive["acceptance_pass"] is None


def test_holm_adjusts_all_four_k_when_all_results_are_formally_valid():
    from src.experiments.run_gan_transfer import summarize_paired_acceptance

    entries = sum((_paired_metrics(count) for count in (1, 3, 5, 10)), [])
    summary = summarize_paired_acceptance(entries, primary_counts=[3, 5], min_delta_nrmse=0.02)

    assert all(summary[str(count)]["holm_adjusted_p"] is not None for count in (1, 3, 5, 10))
    assert summary["1"]["acceptance_tier"] == "supportive_holm_exploratory"
    assert summary["3"]["acceptance_tier"] == "primary"


def test_fixed_six_step_metrics_include_four_channels_and_censor_violation():
    import torch
    from src.experiments.run_gan_transfer import DomainRows, _evaluate
    from src.transfer.damage_state import DamageStateModel

    rng = np.random.default_rng(61)
    rows = DomainRows(
        x=rng.normal(size=(14, 12)).astype(np.float32), states=rng.random((14, 3), dtype=np.float32),
        obs_labels=rng.normal(size=(14, 4)).astype(np.float32), rul=np.ones(14, dtype=np.float32),
        event=np.array([True] * 7 + [False] * 7), ids=np.array(["failed"] * 7 + ["censored"] * 7),
        time=np.tile(np.arange(7, dtype=float), 2),
    )
    torch.manual_seed(61)
    metrics = _evaluate(DamageStateModel(20, 12).eval(), rows, torch.device("cpu"),
                        channel_scale=np.ones(4, dtype=np.float32), rul_scale_s=10.0, link_margin_bridge=None)

    assert metrics["evaluation_horizon_steps"] == 6
    assert set(metrics["channel_6step_nrmse_by_metric"]) == {"gain_dB", "phase_deg", "Pout_dBm", "PAE"}
    assert metrics["channel_6step_nrmse_mean"] == np.mean(list(metrics["channel_6step_nrmse_by_metric"].values()))
    assert all(item["prediction_steps"] == 6 for item in metrics["trajectory_metrics"])
    assert all(item["channel_6step_nrmse_mean"] == np.mean(list(item["channel_6step_nrmse_by_metric"].values()))
               for item in metrics["trajectory_metrics"])
    assert 0.0 <= metrics["service_censor_lower_bound_violation_rate"] <= 1.0


def test_scoring_truth_changes_fingerprint_and_json_replaces_nonfinite_with_null(tmp_path):
    import json
    from src.experiments.run_gan_transfer import DomainRows, persist_results_with_acceptance, test_trajectory_fingerprint

    rows = _mini_rows(12, "fingerprint")
    changed_label = DomainRows(rows.x, rows.states, rows.obs_labels.copy(), rows.rul, rows.event, rows.ids, rows.time)
    changed_label.obs_labels[0, 0] += 1.0
    changed_rul = DomainRows(rows.x, rows.states, rows.obs_labels, rows.rul.copy(), rows.event, rows.ids, rows.time)
    changed_rul.rul[0] += 1.0
    assert test_trajectory_fingerprint(rows, "fingerprint") != test_trajectory_fingerprint(changed_label, "fingerprint")
    assert test_trajectory_fingerprint(rows, "fingerprint") != test_trajectory_fingerprint(changed_rul, "fingerprint")

    output = tmp_path / "strict.json"
    persist_results_with_acceptance(output, [{"nan": float("nan"), "inf": float("inf")}], {"1": {"value": float("-inf")}})
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics"][0] == {"nan": None, "inf": None}
    assert payload["acceptance"]["1"]["value"] is None


def test_acceptance_requires_exactly_ten_paired_seeds_for_formal_claim():
    from src.experiments.run_gan_transfer import summarize_paired_acceptance

    nine_pairs = _paired_metrics(3)[:-2]
    stats = summarize_paired_acceptance(nine_pairs, primary_counts=[3, 5], min_delta_nrmse=0.02)["3"]

    assert stats["paired_seed_count"] == 9
    assert stats["formal_valid"] is False
    assert stats["acceptance_status"] == "invalid"
    assert "requires_10_paired_seeds" in stats["failure_reasons"]


def test_acceptance_rejects_duplicate_seed_nonfinite_rmse_and_test_fingerprint_mismatch():
    from src.experiments.run_gan_transfer import summarize_paired_acceptance

    entries = _paired_metrics(3)
    entries.append(entries[0].copy())
    entries[1]["rmse"] = float("nan")
    entries[3]["test_fingerprint"] = "different-test"
    stats = summarize_paired_acceptance(entries, primary_counts=[3, 5], min_delta_nrmse=0.02)["3"]

    assert stats["formal_valid"] is False
    assert stats["acceptance_status"] == "invalid"
    assert "duplicate_seed" in stats["failure_reasons"]
    assert "nonfinite_rmse" in stats["failure_reasons"]
    assert "test_fingerprint_mismatch" in stats["failure_reasons"]


def test_batched_rollout_matches_reference_per_trajectory_for_unequal_lengths():
    import torch
    from src.experiments.run_gan_transfer import (
        DomainRows, _rollout_target_trajectories_batched, _rollout_target_trajectories_reference,
    )
    from src.transfer.damage_state import DamageStateModel

    rng = np.random.default_rng(20260728)
    ids = np.array(["traj_a"] * 4 + ["traj_b"] * 6)
    rows = DomainRows(
        x=rng.normal(size=(10, 12)).astype(np.float32), states=rng.random((10, 3), dtype=np.float32),
        obs_labels=rng.normal(size=(10, 4)).astype(np.float32), rul=rng.random(10, dtype=np.float32),
        event=np.ones(10, dtype=bool), ids=ids, time=np.array([0, 2, 4, 6, 0, 1, 2, 3, 4, 5], dtype=float),
    )
    torch.manual_seed(17)
    model = DamageStateModel(20, 12).eval()
    reference = _rollout_target_trajectories_reference(model, rows, torch.device("cpu"))
    batched = _rollout_target_trajectories_batched(model, rows, torch.device("cpu"))

    assert reference.keys() == batched.keys()
    for trajectory in reference:
        for expected, actual in zip(reference[trajectory], batched[trajectory]):
            assert np.allclose(expected, actual, rtol=1e-6, atol=1e-7)


def test_early_stop_honors_patience_behavior(monkeypatch):
    import torch
    import src.experiments.run_gan_transfer as gan
    from src.transfer.damage_state import DamageStateModel

    rows = _mini_rows(12, "x")
    model = DamageStateModel(20, 12)
    val_losses = iter([1.0, 2.0, 3.0])
    def controlled_loss(model, rows, device):
        if model.training:
            return sum((parameter * 0).sum() for parameter in model.parameters())
        return torch.tensor(next(val_losses))
    monkeypatch.setattr(gan, "_target_loss", controlled_loss)
    _, steps = gan.train_with_early_stop(model, rows, rows, source_rows=None, epochs=5, patience=1, device=torch.device("cpu"))
    assert steps == 2


def test_subset_groups_write_report_without_key_error(tmp_path):
    from src.experiments.run_gan_transfer import _write_results

    results = {"target_only": [{"rmse": 0.2, "n_failed_rows": 0, "source_pretrain_steps": 0, "target_steps": 1, "budget_note": "design"}]}
    _write_results(tmp_path / "report.md", results, smoke=True)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "target_only" in report
    assert "0" in report and "不可解释" in report
