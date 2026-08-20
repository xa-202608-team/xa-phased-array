"""A3 validation-only 分析器的行为契约测试。"""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_layerwise_a3 import (  # noqa: E402
    GROUPS,
    analyze,
    load_records,
    paired_stats,
    select_depth,
)


def _record(group, seed, val_rmse, rmse):
    return {
        "group": GROUPS[group],
        "seed": seed,
        "val_rmse": val_rmse,
        "rmse": rmse,
    }


def _base_data():
    p1_test = [0.24, 0.25, 0.23, 0.24, 0.25]
    data = {group_name: {} for group_name in GROUPS.values()}
    for offset, seed in enumerate(range(42, 47)):
        for arm, val_rmse, rmse in (
            ("R", 0.30, 0.30),
            ("P1", 0.20, p1_test[offset]),
            ("P2", 0.25, 0.10),
            ("Full", 0.35, 0.35),
        ):
            record = _record(arm, seed, val_rmse, rmse)
            data[record["group"]][seed] = record
    return data


def test_selection_uses_only_validation_seeds_and_never_test_rmse():
    data = _base_data()

    assert select_depth(data) == "P1"
    report = analyze(data)
    assert report["selected_depth"] == "P1"
    assert report["primary"]["left"] == "P1"
    assert report["primary"]["right"] == "R"
    assert report["primary"]["n"] == 5
    assert report["extension_required"] is True
    assert report["verdict"] == "shallow_candidate"

    # 扩展 seed 的 validation 即使极差，也不得反向改变 42--46 的 depth*。
    for seed in range(47, 52):
        record = _record("P1", seed, 9.0, 0.01)
        data[record["group"]][seed] = record
    assert select_depth(data) == "P1"


def test_load_records_deduplicates_identical_rows_and_rejects_conflicts(tmp_path):
    original = _record("R", 42, 0.30, 0.31)
    same = dict(reversed(list(original.items())))
    input_path = tmp_path / "same.jsonl"
    input_path.write_text(
        "\n"
        + json.dumps(original)
        + "\n"
        + json.dumps(same, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )

    loaded = load_records([input_path])
    assert loaded[GROUPS["R"]][42] == original
    assert len(loaded[GROUPS["R"]]) == 1

    conflict_path = tmp_path / "conflict.jsonl"
    changed = dict(original, rmse=0.99)
    conflict_path.write_text(
        json.dumps(original) + "\n" + json.dumps(changed) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="冲突记录"):
        load_records([conflict_path])


@pytest.mark.parametrize(
    ("left_value", "right_value"),
    [(1, 1.0), (0.0, -0.0)],
)
def test_load_records_rejects_numeric_type_and_signed_zero_conflicts(
    tmp_path, left_value, right_value
):
    left = _record("R", 42, 0.30, left_value)
    right = _record("R", 42, 0.30, right_value)
    input_path = tmp_path / "numeric-conflict.jsonl"
    input_path.write_text(
        json.dumps(left) + "\n" + json.dumps(right) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="冲突记录"):
        load_records([input_path])


def test_random_validation_win_has_no_primary_or_extension_and_directions_are_fixed():
    data = _base_data()
    for seed in range(42, 47):
        data[GROUPS["R"]][seed]["val_rmse"] = 0.10

    report = analyze(data)
    assert report["primary"] is None
    assert report["extension_required"] is False
    assert report["verdict"] == "no_source_candidate"
    assert set(report["incremental"]) == {"P2-P1", "Full-P2"}
    assert report["incremental"]["P2-P1"]["left"] == "P2"
    assert report["incremental"]["P2-P1"]["right"] == "P1"
    assert report["incremental"]["P2-P1"]["mean"] == pytest.approx(-0.142)
    assert report["incremental"]["Full-P2"]["left"] == "Full"
    assert report["incremental"]["Full-P2"]["right"] == "P2"
    assert report["incremental"]["Full-P2"]["mean"] == pytest.approx(0.25)


def test_paired_stats_reports_sample_statistics_and_rejects_incomplete_pairs():
    data = _base_data()
    data[GROUPS["P1"]][42]["rmse"] = 0.20
    data[GROUPS["P1"]][43]["rmse"] = 0.40
    data[GROUPS["R"]][42]["rmse"] = 0.30
    data[GROUPS["R"]][43]["rmse"] = 0.30

    stats = paired_stats(data, "P1", "R", [42, 43])
    assert stats["left"] == "P1"
    assert stats["right"] == "R"
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(0.0, abs=1e-12)
    assert stats["std"] == pytest.approx(math.sqrt(0.02))
    assert stats["ci95_lo"] == pytest.approx(-1.2706204736)
    assert stats["ci95_hi"] == pytest.approx(1.2706204736)
    assert stats["negative_count"] == 1

    with pytest.raises(ValueError):
        paired_stats(data, "P1", "R", [42])
    with pytest.raises(ValueError):
        paired_stats(data, "P1", "R", [42, 99])


def test_full_validation_win_never_reopens_full_endpoint():
    data = _base_data()
    for seed in range(42, 47):
        data[GROUPS["Full"]][seed]["val_rmse"] = 0.05

    report = analyze(data)
    assert report["selected_depth"] == "Full"
    assert report["primary"]["left"] == "Full"
    assert report["primary"]["right"] == "R"
    assert report["extension_required"] is False
    assert report["verdict"] == "full_endpoint_not_reopened"


def test_non_finite_initial_ci_cannot_trigger_extension():
    data = _base_data()
    data[GROUPS["P1"]][42]["rmse"] = math.nan

    report = analyze(data)
    assert report["selected_depth"] == "P1"
    assert report["primary"]["n"] == 5
    assert report["extension_required"] is False
    assert report["verdict"] == "no_confirmed_gain"


def test_complete_extension_cannot_promote_primary_before_initial_gate():
    data = _base_data()
    data[GROUPS["P1"]][42]["rmse"] = 0.50
    for seed in range(47, 52):
        for arm, val_rmse, rmse in (
            ("R", 0.30, 0.30),
            ("P1", 0.20, 0.10),
        ):
            record = _record(arm, seed, val_rmse, rmse)
            data[record["group"]][seed] = record

    report = analyze(data)
    assert report["selected_depth"] == "P1"
    assert report["primary"]["n"] == 5
    assert report["primary"]["ci95_hi"] >= 0.0
    assert report["extension_required"] is False
    assert report["verdict"] == "no_confirmed_gain"


def test_ten_seed_confirmation_keeps_preselected_depth_and_uses_only_primary_extension():
    data = _base_data()
    for seed in range(47, 52):
        for arm, val_rmse, rmse in (
            ("R", 0.30, 0.30),
            ("P1", 0.20, 0.20),
        ):
            record = _record(arm, seed, val_rmse, rmse)
            data[record["group"]][seed] = record

    report = analyze(data)
    assert report["selected_depth"] == "P1"
    assert report["primary"]["n"] == 10
    assert report["primary"]["negative_count"] == 10
    assert report["primary"]["ci95_hi"] < 0.0
    assert report["direct"]["P1-R"]["n"] == 5
    assert report["incremental"]["P2-P1"]["n"] == 5
    assert report["extension_required"] is False
    assert report["verdict"] == "confirmed_positive_transfer"

    # depth* 仍为 P1，但扩展 validation 均值不再胜 R 时，不得确认翻正。
    for seed in range(47, 52):
        data[GROUPS["P1"]][seed]["val_rmse"] = 9.0
    report = analyze(data)
    assert report["selected_depth"] == "P1"
    assert report["primary"]["n"] == 10
    assert report["verdict"] == "no_confirmed_gain"


def test_cli_writes_recomputable_json_and_markdown_from_records(tmp_path):
    data = _base_data()
    a1_path = tmp_path / "a1.jsonl"
    a3_path = tmp_path / "a3.jsonl"
    a1_rows = []
    a3_rows = []
    for arm in ("R", "P1", "P2", "Full"):
        rows = [data[GROUPS[arm]][seed] for seed in range(42, 47)]
        (a1_rows if arm in ("R", "Full") else a3_rows).extend(rows)
    a1_path.write_text("\n".join(json.dumps(row) for row in a1_rows) + "\n", encoding="utf-8")
    a3_path.write_text("\n".join(json.dumps(row) for row in a3_rows) + "\n", encoding="utf-8")
    json_out = tmp_path / "nested" / "analysis.json"
    markdown_out = tmp_path / "nested" / "analysis.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_layerwise_a3.py"),
            "--a1-jsonl",
            str(a1_path),
            "--a3-jsonl",
            str(a3_path),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    persisted = json.loads(json_out.read_text(encoding="utf-8"))
    assert persisted["selected_depth"] == "P1"
    markdown = markdown_out.read_text(encoding="utf-8")
    for label in ("R", "P1", "P2", "Full"):
        assert f"| {label} |" in markdown
    for label in ("P1-R", "P2-R", "Full-R", "P2-P1", "Full-P2"):
        assert label in markdown
    assert "depth*：P1" in markdown
    assert "是否扩测：是" in markdown
    assert "n<10 探索性" in markdown
