"""GaN 目标域 OOD 组合保持集与完整数据非劣汇总契约。"""
from __future__ import annotations

import h5py
import numpy as np
import pytest


OOD_PROTOCOL = {
    "schema_version": "target_ood_conditions_v1",
    "abs_scan_az_deg_min": 30.0,
    "duty_cycle_min": 0.70,
    "Tj_base_C_min": 125.0,
}


def _write_condition_h5(path, conditions: dict[str, tuple[float, float, float]]) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["dynamics_id"] = "leo_coupled_v1"
        h5.attrs["target_condition_schema"] = "target_ood_conditions_v1"
        for trajectory, (scan, duty, tj_base) in conditions.items():
            group = h5.create_group(trajectory)
            group.attrs["scan_az_deg"] = scan
            group.attrs["duty_cycle"] = duty
            group.attrs["Tj_base_C"] = tj_base


def _full_data_entries(candidate_value: float) -> list[dict]:
    entries: list[dict] = []
    for seed in range(42, 52):
        for group, value in (("target_only", 0.400), ("gan_transition_init", candidate_value)):
            entries.append({
                "group": group, "seed": seed, "evaluation_scope": "iid_full_data",
                "test_fingerprint": "same-iid-full-data",
                "trajectory_metrics": [
                    {"trajectory_id": "traj_a", "test_trajectory_fingerprint": "a",
                     "channel_6step_nrmse_mean": value},
                    {"trajectory_id": "traj_b", "test_trajectory_fingerprint": "b",
                     "channel_6step_nrmse_mean": value},
                ],
            })
    return entries


def test_ood_condition_metadata_and_split_keep_corner_trajectories_out_of_iid_pools(tmp_path):
    from src.experiments.run_gan_transfer import (
        TargetOODSplit,
        build_ood_split_manifest,
        read_target_ood_conditions,
        split_target_iid_ood,
        validate_ood_split_manifest,
    )

    conditions = {
        "iid_0": (0.0, 0.40, 100.0), "iid_1": (35.0, 0.40, 100.0),
        "iid_2": (0.0, 0.75, 100.0), "iid_3": (0.0, 0.40, 130.0),
        "iid_4": (20.0, 0.60, 120.0), "ood_0": (30.0, 0.70, 125.0),
        "ood_1": (-35.0, 0.75, 130.0),
    }
    path = tmp_path / "target_features.h5"
    _write_condition_h5(path, conditions)
    loaded = read_target_ood_conditions(path, OOD_PROTOCOL)
    split = split_target_iid_ood(sorted(conditions), loaded, OOD_PROTOCOL, seed=9)

    assert set(split.ood_test_ids) == {"ood_0", "ood_1"}
    assert not (set(split.ood_test_ids) & (set(split.train_ids) | set(split.val_ids) | set(split.iid_test_ids)))
    assert set(split.train_ids) | set(split.val_ids) | set(split.iid_test_ids) == set(conditions) - set(split.ood_test_ids)
    manifest = build_ood_split_manifest(OOD_PROTOCOL, loaded, split)
    validate_ood_split_manifest(manifest, OOD_PROTOCOL, loaded, split)
    changed = dict(loaded)
    changed["ood_0"] = {**changed["ood_0"], "duty_cycle": 0.69}
    with pytest.raises(ValueError, match="OOD split manifest"):
        validate_ood_split_manifest(manifest, OOD_PROTOCOL, changed, split)
    forged = TargetOODSplit(split.train_ids, split.val_ids, split.iid_test_ids, ("iid_0",))
    with pytest.raises(ValueError, match="ood_test_ids"):
        build_ood_split_manifest(OOD_PROTOCOL, loaded, forged)


def test_full_data_noninferiority_summary_is_separate_from_iid_primary_acceptance():
    from src.experiments.run_gan_transfer import summarize_full_data_noninferiority

    accepted = summarize_full_data_noninferiority(
        _full_data_entries(0.405), max_allowed_increase_nrmse=0.01, expected_seeds=range(42, 52), n_bootstrap=100,
    )
    rejected = summarize_full_data_noninferiority(
        _full_data_entries(0.420), max_allowed_increase_nrmse=0.01, expected_seeds=range(42, 52), n_bootstrap=100,
    )
    assert accepted["scope"] == "iid_full_data_noninferiority"
    assert accepted["status"] == "accepted" and accepted["ci95_hi"] <= 0.01
    assert rejected["status"] == "rejected" and rejected["ci95_hi"] > 0.01


def test_iid_result_report_explicitly_states_that_ood_performance_is_not_evaluated(tmp_path):
    from src.experiments.run_gan_transfer import _write_results

    entry = {
        "target_train_count": 3, "channel_6step_nrmse_mean": 0.4,
        "channel_6step_nrmse_by_metric": {"gain_dB": 0.4, "phase_deg": 0.4, "Pout_dBm": 0.4, "PAE": 0.4},
        "array_link_margin_mae_dB": None, "service_event_mae_s": None,
        "service_censor_lower_bound_violation_rate": None, "n_failed_rows": 1,
        "evaluation_scope": "iid_with_pre_registered_ood_holdout", "ood_test_trajectory_count": 2,
    }
    report = tmp_path / "iid.md"
    _write_results(report, {"target_only": [entry]}, smoke=False)
    text = report.read_text(encoding="utf-8")
    assert "OOD 保持集" in text and "未计算 OOD 性能" in text
