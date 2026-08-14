"""合并 channel + service level 矩阵结果 → 完整报告 (T6.4/M7)。

channel level (k=all, 3 seed) 和 service level (cross_level_transfer, 3 seed) 分两次跑,
本脚本合并两者的 by 字典 → 调 aggregate (算 level_control + paired) → write_results。

用法:
  python scripts/merge_matrix_results.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                                  # noqa: E402
from src.experiments.run_groups import aggregate, write_results    # noqa: E402

CKPT = ROOT / "checkpoints"
ch_json = CKPT / "all_metrics_phased_array_channel.json"
svc_json = CKPT / "all_metrics_phased_array_service.json"
assert ch_json.exists() and svc_json.exists(), "先跑 channel + service level"

# 读两个 JSON (结构: by_dict[group] = [metrics, ...]; 我们之前存的是 agg, 需要 by)
# 实际上 all_metrics 存的是 agg (聚合后), 不是 by (原始 per-seed)。
# 重构 by: 从 agg["_raw_by"] 反推 per-seed RMSE, 但只有 rmse, 缺 phm/mae 等。
# 更好: 重跑时把 by 也存。这里用 agg 直接合并 + 手动算 level_control。
ch_agg = json.loads(ch_json.read_text(encoding="utf-8"))["agg"]
svc_agg = json.loads(svc_json.read_text(encoding="utf-8"))["agg"]

# 合并 agg (channel + service)
# channel JSON 只取 ch_* 组 (cross_level_transfer_kall 是 --level channel bug 残留,
# 正确的 cross_level_transfer 在 service JSON)
merged_agg = {}
for g in [k for k in ch_agg if k.startswith("ch_")]:
    merged_agg[g] = ch_agg[g]
for g in [k for k in svc_agg if not k.startswith("_")]:
    merged_agg[g] = svc_agg[g]

# 合并 _raw_by (per-seed rmse, level_control 配对用)
# channel 只取 ch_* (同上, 排除 bug 版 cross_level_transfer_kall)
merged_raw = {}
for g, seed_rmse in ch_agg.get("_raw_by", {}).items():
    if g.startswith("ch_"):
        merged_raw[g] = seed_rmse
for g, seed_rmse in svc_agg.get("_raw_by", {}).items():
    if g not in merged_raw:
        merged_raw[g] = seed_rmse
merged_agg["_raw_by"] = merged_raw

# 手动算 level_control (channel ch_source_mmd_physics_kall vs service cross_level_transfer)
from src.experiments.run_groups import _paired_delta_ci
ch_main = "ch_source_mmd_physics_kall"
svc_main = "cross_level_transfer"
level_ctrl = None
if ch_main in merged_raw and svc_main in merged_raw:
    level_ctrl = _paired_delta_ci(merged_raw[svc_main], merged_raw[ch_main])
merged_agg["_level_control"] = {"channel_group": ch_main, "service_group": svc_main,
                                "paired": level_ctrl}

# 手动算 init/full/mmd control (channel level)
def _safe_pair(a, b):
    if a in merged_raw and b in merged_raw:
        return _paired_delta_ci(merged_raw[a], merged_raw[b])
    return None

merged_agg["_paired"] = {
    "primary_target": "ch_target_only_gru_kall",
    "target_groups": [g for g in merged_agg if g.startswith("ch_target_only")],
    "primary_source_group": "ch_source_mmd_physics_kall",
    "all_pairs": {},
    "init_control": _safe_pair("ch_source_pretrain_frozen_kall", "ch_random_frozen_kall"),
    "full_control": _safe_pair("ch_source_mmd_physics_kall", "ch_random_full_finetune_kall"),
    "mmd_control":  _safe_pair("ch_random_full_finetune_kall", "ch_random_nommd_kall"),
}

# 各 target 配对各 source
for tg in merged_agg["_paired"]["target_groups"]:
    merged_agg["_paired"]["all_pairs"][tg] = {}
    for sg in [g for g in merged_agg if g.startswith("ch_source_")]:
        ps = _safe_pair(tg, sg)
        if ps:
            merged_agg["_paired"]["all_pairs"][tg][sg] = ps

# k-shot rmse 表
merged_agg["_k_shot_rmse"] = {"kall": {g: merged_agg[g]["rmse_mean"]
                                        for g in merged_agg
                                        if isinstance(merged_agg[g], dict) and "rmse_mean" in merged_agg[g]}}

# 写报告
cfg = load_config(ROOT / "configs" / "phased_array.yaml")
report_groups = list(merged_agg["_paired"]["target_groups"]) + \
                [g for g in merged_agg if g.startswith("ch_source_")] + \
                [g for g in merged_agg if g.startswith("ch_random_")] + \
                ["cross_level_transfer"]
write_results(merged_agg, None, ROOT / "docs" / "results_phased_array_channel.md",
              smoke=False, n_seeds=3, group_names=report_groups, component="phased_array",
              primary_model_group="ch_target_only_gru_kall")

# 存合并 JSON
(CKPT / "all_metrics_phased_array_merged.json").write_text(
    json.dumps({"agg": merged_agg}, indent=2, ensure_ascii=False), encoding="utf-8")
print(">> 合并完成: docs/results_phased_array_channel.md")
print(f">> level_control delta (service - channel): "
      f"{level_ctrl['delta_mean']:+.4f} +/- {level_ctrl['delta_std']:.4f}, "
      f"CI [{level_ctrl['ci95_lo']:+.4f}, {level_ctrl['ci95_hi']:+.4f}]")
print(f">> init_control (source_pretrain - random_frozen): "
      f"{merged_agg['_paired']['init_control']['delta_mean']:+.4f}")
print(f">> full_control (source_mmd - random_full): "
      f"{merged_agg['_paired']['full_control']['delta_mean']:+.4f}")
print(f">> mmd_control (random_full - random_nommd): "
      f"{merged_agg['_paired']['mmd_control']['delta_mean']:+.4f}")
