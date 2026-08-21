"""固化 A3 validation-only 选层、配对统计与扩测判读。"""
import argparse
import json
import math
import statistics
from pathlib import Path

from scipy import stats as scipy_stats


GROUPS = {
    "R": "ch_random_full_finetune_k3",
    "P1": "ch_layerwise_gru_p1_k3",
    "P2": "ch_layerwise_gru_p2_k3",
    "Full": "ch_source_mmd_physics_k3",
}
SELECTION_SEEDS = frozenset(range(42, 47))
EXTENSION_SEEDS = frozenset(range(42, 52))


def _exactly_equal(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exactly_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exactly_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, float) and left == 0.0 and right == 0.0:
        return math.copysign(1.0, left) == math.copysign(1.0, right)
    if isinstance(left, float) and math.isnan(left) and math.isnan(right):
        return True
    return left == right


def load_records(paths: list[Path]) -> dict[str, dict[int, dict]]:
    """读取 JSONL，并拒绝同一组与 seed 的非完全相同记录。"""
    data: dict[str, dict[int, dict]] = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            group = record["group"]
            seed = int(record["seed"])
            group_records = data.setdefault(group, {})
            previous = group_records.get(seed)
            if previous is not None:
                if not _exactly_equal(previous, record):
                    raise ValueError(
                        f"冲突记录：group={group}, seed={seed}，位置={path}:{line_number}"
                    )
                continue
            group_records[seed] = record
    return data


def select_depth(data: dict) -> str:
    """仅以固定 42--46 validation 均值选择 depth*。"""
    return min(
        GROUPS,
        key=lambda arm: statistics.mean(
            data[GROUPS[arm]][seed]["val_rmse"] for seed in SELECTION_SEEDS
        ),
    )


def paired_stats(data, left, right, seeds) -> dict:
    """计算 left - right 的逐 seed test RMSE 配对统计。"""
    selected_seeds = tuple(seeds)
    if len(selected_seeds) < 2 or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("配对统计要求至少两个互异 seed")
    if left not in GROUPS or right not in GROUPS:
        raise ValueError("配对统计使用了未知臂")
    left_records = data.get(GROUPS[left], {})
    right_records = data.get(GROUPS[right], {})
    missing = [
        seed for seed in selected_seeds
        if seed not in left_records or seed not in right_records
    ]
    if missing:
        raise ValueError(f"配对统计缺少共同 seed：{missing}")

    differences = [
        float(left_records[seed]["rmse"]) - float(right_records[seed]["rmse"])
        for seed in selected_seeds
    ]
    n = len(differences)
    if all(math.isfinite(difference) for difference in differences):
        mean = statistics.mean(differences)
        std = statistics.stdev(differences)
        critical = float(scipy_stats.t.ppf(0.975, n - 1))
        half_width = critical * std / math.sqrt(n)
        ci95_lo = mean - half_width
        ci95_hi = mean + half_width
    else:
        mean = std = ci95_lo = ci95_hi = math.nan
    return {
        "left": left,
        "right": right,
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_lo": ci95_lo,
        "ci95_hi": ci95_hi,
        "negative_count": sum(difference < 0.0 for difference in differences),
    }


def _arm_summaries(data: dict) -> dict:
    summaries = {}
    for arm, group in GROUPS.items():
        records = data[group]
        summaries[arm] = {
            "n": len(SELECTION_SEEDS),
            "val_mean": statistics.mean(
                float(records[seed]["val_rmse"]) for seed in SELECTION_SEEDS
            ),
            "test_mean": statistics.mean(
                float(records[seed]["rmse"]) for seed in SELECTION_SEEDS
            ),
        }
    return summaries


def _has_complete_extension(data: dict, selected_depth: str) -> bool:
    selected_records = data.get(GROUPS[selected_depth], {})
    random_records = data.get(GROUPS["R"], {})
    return all(
        seed in selected_records and seed in random_records
        for seed in EXTENSION_SEEDS
    )


def analyze(data) -> dict:
    """生成只由输入记录复算得到的 A3 分析报告。"""
    selected_depth = select_depth(data)
    selection_seeds = sorted(SELECTION_SEEDS)
    direct = {
        "P1-R": paired_stats(data, "P1", "R", selection_seeds),
        "P2-R": paired_stats(data, "P2", "R", selection_seeds),
        "Full-R": paired_stats(data, "Full", "R", selection_seeds),
    }
    incremental = {
        "P2-P1": paired_stats(data, "P2", "P1", selection_seeds),
        "Full-P2": paired_stats(data, "Full", "P2", selection_seeds),
    }
    report = {
        "selected_depth": selected_depth,
        "selection_seeds": selection_seeds,
        "arms": _arm_summaries(data),
        "primary": None,
        "direct": direct,
        "incremental": incremental,
        "extension_required": False,
        "verdict": "no_confirmed_gain",
    }

    if selected_depth == "R":
        report["verdict"] = "no_source_candidate"
        return report
    if selected_depth == "Full":
        report["primary"] = direct["Full-R"]
        report["verdict"] = "full_endpoint_not_reopened"
        return report

    initial_primary = direct[f"{selected_depth}-R"]
    report["primary"] = initial_primary

    # 只有初始 n=5 CI 全负才允许进入浅层候选与其后扩测判读。
    initial_candidate = (
        math.isfinite(initial_primary["ci95_hi"])
        and initial_primary["ci95_hi"] < 0.0
    )
    if not initial_candidate:
        return report
    has_extension = _has_complete_extension(data, selected_depth)
    if not has_extension:
        report["extension_required"] = True
        report["verdict"] = "shallow_candidate"
        return report

    report["primary"] = paired_stats(
        data, selected_depth, "R", sorted(EXTENSION_SEEDS)
    )
    primary = report["primary"]
    selected_records = data[GROUPS[selected_depth]]
    random_records = data[GROUPS["R"]]
    extension_val_better = statistics.mean(
        float(selected_records[seed]["val_rmse"]) for seed in EXTENSION_SEEDS
    ) < statistics.mean(
        float(random_records[seed]["val_rmse"]) for seed in EXTENSION_SEEDS
    )
    if (
        primary["n"] == 10
        and primary["ci95_hi"] < 0.0
        and primary["negative_count"] >= 7
        and extension_val_better
    ):
        report["verdict"] = "confirmed_positive_transfer"
    return report


def render_markdown(report: dict) -> str:
    """仅从结构化 report 渲染 Markdown，不另行计算统计量。"""
    lines = [
        "# A3 分层迁移 validation-only 分析",
        "",
        "## 四臂汇总（固定 seeds 42--46）",
        "",
        "| 臂 | n | validation RMSE 均值 | test RMSE 均值 |",
        "|---|---:|---:|---:|",
    ]
    for arm in GROUPS:
        summary = report["arms"][arm]
        lines.append(
            f"| {arm} | {summary['n']} | {summary['val_mean']:.8f} | "
            f"{summary['test_mean']:.8f} |"
        )

    lines.extend(["", "## Direct 配对 CI（固定 seeds 42--46）", ""])
    lines.extend(_format_ci(name, stats) for name, stats in report["direct"].items())
    lines.extend(["", "## Incremental 配对 CI（固定 seeds 42--46）", ""])
    lines.extend(_format_ci(name, stats) for name, stats in report["incremental"].items())

    primary = report["primary"]
    extension = (
        "是/已完成" if primary is not None and primary["n"] == 10
        else "是" if report["extension_required"] else "否"
    )
    if primary is None:
        sample_label = "无主比较（未进入扩测门）"
    elif primary["n"] == 5:
        sample_label = "n<10 探索性"
    elif primary["n"] == 10:
        sample_label = "n=10 确认门"
    else:
        sample_label = f"n={primary['n']} 非预注册样本规模"
    lines.extend(
        [
            "",
            "## 冻结判读",
            "",
            f"- depth*：{report['selected_depth']}",
            f"- 是否扩测：{extension}",
            f"- 样本标签：{sample_label}",
            f"- 判读：{report['verdict']}",
            "",
        ]
    )
    return "\n".join(lines)


def _format_ci(name: str, stats: dict) -> str:
    label = "；n<10 探索性" if stats["n"] < 10 else ""
    return (
        f"- {name}：mean={stats['mean']:+.8f}，std={stats['std']:.8f}，"
        f"95% CI=[{stats['ci95_lo']:+.8f}, {stats['ci95_hi']:+.8f}]，"
        f"负差值={stats['negative_count']}/{stats['n']}{label}"
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="复算 A3 validation-only 分层迁移分析")
    parser.add_argument(
        "--a1-jsonl",
        type=Path,
        default=Path("outputs/alpha_soft_a1/results_partial.jsonl"),
    )
    parser.add_argument(
        "--a3-jsonl",
        type=Path,
        default=Path("outputs/layerwise_a3/results_partial.jsonl"),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("outputs/layerwise_a3/analysis.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("outputs/layerwise_a3/analysis.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data = load_records([args.a1_jsonl, args.a3_jsonl])
    report = analyze(data)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(f"已写入 JSON：{args.json_out}")
    print(f"已写入 Markdown：{args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
