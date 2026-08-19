"""B 路线 (§4e): sim_v1 通道级特征 → device_canonical_v1 源域文件.

把 --subdose off 仿真 + build_channel_hi 产出的 v1 channel_features.h5
(legacy 标量动力学, 16 子阵共享损伤) 展开为器件级 canonical 源域:
每 traj×sub = 1 个 device (200 traj × 16 sub = 3200 序列), 与
mosfet_canonical.h5 完全同构 (devices/Test_N: x/hi/rul_s/rul_lower_bound_s/
elapsed_time_s + attrs event_observed/original_T), pretrain/load_source 零改动消费。

语义要点:
- x 直接透传 x_ch (build_canonical_x 同函数产出) — 源/目标输入语义严格统一;
- rul_s = rul_ch(步) × sample_period_s; rul_lower_bound_s ≡ rul_s (照抄 mosfet
  canonical 口径: 记录值即删失下界, pretrain 删失 hinge 消费);
- val 划分按 traj 整组 (最后 val_n traj, 默认 20): 同 traj 的 16 sub 必同 split,
  杜绝源域 val 泄漏 (任务 3 纪律); Test 编号 = traj*16+sub+1, val 集中于
  Test_{(n_traj-val_n)*16+1 .. n_traj*16}, 名单写 val_device_ids.txt 供
  pretrain --val-device-ids 使用。
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import load_config                                     # noqa: E402

FEATURE_NAMES = ["p_drift_norm", "T_dev_C", "duty", "drive_norm"]
SCHEMA = "device_canonical_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--in", dest="in_h5", required=True,
                    help="v1 channel_features.h5 (build_channel_hi 产出)")
    ap.add_argument("--out", required=True, help="输出 canonical h5")
    ap.add_argument("--val-n-traj", type=int, default=20,
                    help="val 划分用的 traj 数 (取最后 N 条, 整 traj 进 val)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dt = float(cfg["sim"]["sample_period_s"])          # 21600 s/步 (6h 遥测窗)

    src = h5py.File(args.in_h5, "r")
    traj_keys = sorted(k for k in src.keys() if k.startswith("traj_"))
    n_traj = len(traj_keys)
    val_trajs = set(range(n_traj - args.val_n_traj, n_traj))
    val_ids = []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_dev = n_fail = 0
    with h5py.File(out_path, "w") as out:
        out.attrs["schema"] = SCHEMA
        out.attrs["feature_names"] = np.asarray(FEATURE_NAMES, dtype=object)
        out.attrs["source_dynamics_id"] = str(src.attrs.get("dynamics_id", ""))
        out.attrs["source_note"] = ("B route (§4e): sim_v1 legacy scalar dynamics -> "
                                    "per-subarray canonical devices")
        devs = out.create_group("devices")
        for key in traj_keys:
            traj_id = int(key.split("_")[1])
            g = src[key]
            sub_keys = sorted(k for k in g.keys() if k.startswith("sub_"))
            for s_i, sk in enumerate(sub_keys):
                s = g[sk]
                x = s["x_ch"][:]
                hi = s["hi_ch"][:]
                rul_s = s["rul_ch"][:].astype(np.float64) * dt
                ev = int(s.attrs["event_observed"])
                test_id = traj_id * len(sub_keys) + s_i + 1
                name = f"Test_{test_id}"
                dg = devs.create_group(name)
                dg.create_dataset("x", data=x)
                dg.create_dataset("hi", data=hi)
                dg.create_dataset("rul_s", data=rul_s)
                dg.create_dataset("rul_lower_bound_s", data=rul_s)   # ≡ rul_s (mosfet 同口径)
                dg.create_dataset("elapsed_time_s", data=np.arange(len(hi), dtype=np.float64) * dt)
                dg.attrs["event_observed"] = ev
                dg.attrs["original_T"] = len(hi)
                dg.attrs["schema"] = SCHEMA
                n_dev += 1
                n_fail += ev
                if traj_id in val_trajs:
                    val_ids.append(name)
    src.close()

    val_file = out_path.parent / (out_path.stem + "_val_ids.txt")
    val_file.write_text(",".join(val_ids), encoding="utf-8")
    print(f">> {n_dev} devices ({n_fail} 失效, {100*n_fail/n_dev:.0f}%) -> {out_path}")
    print(f">> val = 最后 {args.val_n_traj} traj 整组 ({len(val_ids)} devices) -> {val_file}")
    print(f">> pretrain 用: --canonical-source {out_path} --val-device-ids $(cat {val_file})")


if __name__ == "__main__":
    main()
