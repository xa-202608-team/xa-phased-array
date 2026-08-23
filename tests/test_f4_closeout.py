# -*- coding: utf-8 -*-
"""F4 收口行为契约: B2/B3 必须用 matched seeds 42–44 配对口径, 不得混用 F2 五种子聚合均值。

配对 CI 口径: 差值样本标准差 ddof=1, t(2) 0.975 分位 4.3026527299;
n=3 仅描述性统计, 不升级为确认性假设检验。
"""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GROUP = "ch_target_only_gru"
F2_JSONL = ROOT / "outputs" / "f2_formal_5seed" / "results_partial.jsonl"
ABL_DIR = ROOT / "outputs" / "f4_ablation"
ARTIFACTS = {
    "full": F2_JSONL,
    "b2_count": ABL_DIR / "b2_count" / "results_partial.jsonl",
    "b3_subagg": ABL_DIR / "b3_subagg" / "results_partial.jsonl",
    "b3_sparse": ABL_DIR / "b3_sparse" / "results_partial.jsonl",
}


def test_read_group_rmse_by_seed_filters_group_and_casts(tmp_path):
    from scripts.f4_analyze import read_group_rmse_by_seed

    rows = [
        {"group": GROUP, "seed": 42, "rmse": 0.10},
        {"group": "other_group", "seed": 42, "rmse": 9.9},
        {"group": GROUP, "seed": 43, "rmse": 0.12},
    ]
    p = tmp_path / "results_partial.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert read_group_rmse_by_seed(p, GROUP) == {42: 0.10, 43: 0.12}
    assert read_group_rmse_by_seed(p, "other_group") == {42: 9.9}


def test_paired_rmse_summary_matches_hand_computed_stats():
    from scripts.f4_analyze import paired_rmse_summary

    baseline = {42: 0.10, 43: 0.12, 44: 0.14}
    variant = {42: 0.13, 43: 0.11, 44: 0.19}          # delta = +0.03, -0.01, +0.05
    s = paired_rmse_summary(baseline, variant)

    assert s["seeds"] == [42, 43, 44]
    assert s["n"] == 3
    assert s["baseline_mean"] == pytest.approx(0.12)
    assert s["baseline_std"] == pytest.approx(0.02)
    assert s["variant_mean"] == pytest.approx(0.43 / 3)
    m = 0.07 / 3
    delta_std = math.sqrt(((0.03 - m) ** 2 + (-0.01 - m) ** 2 + (0.05 - m) ** 2) / 2)
    half = 4.3026527299 * delta_std / math.sqrt(3)
    assert s["delta_mean"] == pytest.approx(m)
    assert s["delta_std"] == pytest.approx(delta_std)
    assert s["delta_ci"] == pytest.approx((m - half, m + half))


def test_paired_rmse_summary_requires_two_common_seeds():
    from scripts.f4_analyze import paired_rmse_summary

    with pytest.raises(ValueError, match="至少需要 2 个共同 seed"):
        paired_rmse_summary({42: 0.1}, {42: 0.2})
    with pytest.raises(ValueError, match="至少需要 2 个共同 seed"):
        paired_rmse_summary({42: 0.1, 43: 0.2}, {44: 0.3, 45: 0.4})


def test_f4_matched_seed_paired_numbers_locked():
    """锁定 F4 收口的 matched-seed (42–44) 配对数字; 与 F2 五种子聚合口径不可混用。"""
    missing = [str(p) for p in ARTIFACTS.values() if not p.exists()]
    if missing:
        pytest.skip(f"F4/F2 本地产物缺失 (outputs/ 不入库): {', '.join(missing)}")

    from scripts.f4_analyze import paired_rmse_summary, read_group_rmse_by_seed

    data = {k: read_group_rmse_by_seed(p, GROUP) for k, p in ARTIFACTS.items()}
    assert set(data["full"]) >= {42, 43, 44}          # F2 jsonl 含五种子, 配对只取交集

    b2 = paired_rmse_summary(data["full"], data["b2_count"])
    assert b2["seeds"] == [42, 43, 44]
    assert b2["baseline_mean"] == pytest.approx(0.1471296946)
    assert b2["baseline_std"] == pytest.approx(0.0096240430)
    assert b2["variant_mean"] == pytest.approx(0.3152185380)
    assert b2["delta_mean"] == pytest.approx(0.1680888434)
    assert b2["delta_ci"] == pytest.approx((0.1537326399, 0.1824450469))

    subagg = paired_rmse_summary(data["full"], data["b3_subagg"])
    assert subagg["delta_mean"] == pytest.approx(0.0185102870)
    assert subagg["delta_ci"] == pytest.approx((0.0043184166, 0.0327021574))

    sparse = paired_rmse_summary(data["full"], data["b3_sparse"])
    assert sparse["delta_mean"] == pytest.approx(-0.0226510863)
    assert sparse["delta_ci"] == pytest.approx((-0.0648503437, 0.0195481712))
