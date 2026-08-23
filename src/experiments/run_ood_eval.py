"""experiments/run_ood_eval.py

T9 OOD 保持集评估 (M8 加分项): 预注册组合角点 OOD, 验证模型非"死记仿真公式"。

OOD 角点 (plan §T9 预注册, 与 GaN 路线一致):
  |scan_az_deg| ≥ 30°  ∧  duty ≥ 0.70  ∧  Tj_base_C ≥ 125°C

协议 (plan §T9):
  - OOD 轨迹不进 IID train/val/test split, 不进 scaler/early stopping
  - 不进 IID 配对 Δ/bootstrap/CI
  - 独立报告段, 标注"未进入 IID 主验收"
  - 作用: 回应"模型只是在学你自己写的仿真公式"质疑

本脚本:
  1. 从 channel_features.h5 traj attrs 算 OOD flag + 写 manifest (预注册)
  2. 评估模型 (pretrain ckpt + IID train 微调) 在 OOD 轨迹上的 RMSE
  3. 报告 OOD RMSE vs IID test RMSE + 覆盖度

诚实性: OOD 样本数 0 时写 null + "OOD 层未验证"。

用法:
  python -m src.experiments.run_ood_eval --config configs/phased_array.yaml \
      --ckpt checkpoints/source_phased_array_tcn_pretrain.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed                                  # noqa: E402
from src.transfer.channel_dataset import (  # noqa: E402
    load_target_channel, ChannelSeqDataset,
    read_channel_label_meta, CHANNEL_LABEL_SCHEMA_V2)
from src.transfer.train_transfer import split_trajectories, SourceWindowDataset  # noqa: E402
from src.experiments.run_groups import _build_model, _train_with_early_stop, eval_test  # noqa: E402
from src.baselines.physical_extrap import phm_score, mae as mae_fn           # noqa: E402

# OOD 角点 (plan §T9 预注册, 不得改)
OOD_SCAN_AZ_MIN = 30.0
OOD_DUTY_MIN = 0.70
OOD_TJ_BASE_MIN = 125.0


def identify_ood_trajectories(h5_path: Path) -> dict:
    """从 channel_features.h5 traj attrs 算 OOD flag。

    返回 {traj_id: bool is_ood} + 统计。
    """
    ood_flags = {}
    with h5py.File(h5_path, "r") as f:
        for key in sorted(f.keys()):
            g = f[key]
            traj_id = int(key.split("_")[1])
            scan_az = abs(float(g.attrs.get("scan_az_deg", 0)))
            duty = float(g.attrs.get("duty", 0))
            Tj_base = float(g.attrs.get("Tj_base_C", 0))
            is_ood = (scan_az >= OOD_SCAN_AZ_MIN and
                      duty >= OOD_DUTY_MIN and
                      Tj_base >= OOD_TJ_BASE_MIN)
            ood_flags[traj_id] = is_ood
    n_ood = sum(ood_flags.values())
    n_iid = len(ood_flags) - n_ood
    return {"ood_flags": ood_flags, "n_ood": n_ood, "n_iid": n_iid,
            "n_total": len(ood_flags)}


def write_ood_manifest(ood_info: dict, out_path: Path):
    """写 OOD 切分 manifest (预注册, 供 run_groups --ood-holdout 用)。"""
    ood_ids = sorted([tid for tid, is_ood in ood_info["ood_flags"].items() if is_ood])
    iid_ids = sorted([tid for tid, is_ood in ood_info["ood_flags"].items() if not is_ood])
    manifest = {
        "schema_version": "channel_ood_split_manifest_v1",
        "ood_protocol": {
            "scan_az_deg_abs_min": OOD_SCAN_AZ_MIN,
            "duty_min": OOD_DUTY_MIN,
            "Tj_base_C_min": OOD_TJ_BASE_MIN,
        },
        "ood_test_ids": ood_ids,
        "iid_ids": iid_ids,
        "n_ood": len(ood_ids),
        "n_iid": len(iid_ids),
    }
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def evaluate_ood(cfg: dict, ckpt_path: Path, seed: int, verbose: bool = True) -> dict:
    """评估模型在 OOD 轨迹上的 RMSE (不进 IID 训练)。

    协议: OOD 轨迹从 IID split 排除 (train/val/test 只用 IID);
    模型在 IID train 上训练 (target_only + 微调); 在 OOD 上单独评估。
    """
    set_seed(seed, cfg["reproducibility"]["deterministic"])
    ch_cfg = cfg["channel_level"]
    tc = cfg["transfer"]
    mc = cfg["model"]
    L = int(mc["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    target_h5 = ROOT / ch_cfg["feature_path"]
    if not target_h5.exists():
        raise FileNotFoundError(f"缺 {target_h5}; 先 build_channel_hi")

    # 识别 OOD
    ood_info = identify_ood_trajectories(target_h5)
    ood_ids_set = {tid for tid, is_ood in ood_info["ood_flags"].items() if is_ood}
    iid_ids = [tid for tid, is_ood in ood_info["ood_flags"].items() if not is_ood]
    if verbose:
        print(f">> OOD 识别: {ood_info['n_ood']}/{ood_info['n_total']} 轨迹 "
              f"(角点: |scan_az|≥{OOD_SCAN_AZ_MIN}° ∧ duty≥{OOD_DUTY_MIN} ∧ Tj_base≥{OOD_TJ_BASE_MIN}°C)")

    if ood_info["n_ood"] == 0:
        print("!! 无 OOD 轨迹 (当前仿真参数域未覆盖角点); OOD 层未验证")
        return {"ood_info": ood_info, "ood_metrics": None,
                "note": "无 OOD 轨迹, 当前仿真参数域未覆盖预注册角点"}

    # 加载数据
    # F1-A: v2 loader 已返回 rul_ch_norm (窗口数/H); v1 返回窗口数需再除 rul_max_norm
    with h5py.File(target_h5, "r") as _f:
        ch_meta = read_channel_label_meta(_f)
    xT, hiT, rulT, ckT, tidT, evT, lbT, n_traj, sidT = load_target_channel(target_h5)
    # 划分只在 IID 内做 (OOD 排除)
    iid_mask = np.isin(tidT, list(iid_ids))
    # IID 内 train/val/test split (复用 split_trajectories 但只对 IID traj)
    iid_traj_arr = np.array(sorted(iid_ids))
    tr_iid, va_iid, te_iid = split_trajectories(
        len(iid_traj_arr),
        [tc["split"]["train"], tc["split"]["val"], tc["split"]["test"]], seed)
    tr_ids = iid_traj_arr[tr_iid]
    va_ids = iid_traj_arr[va_iid]
    te_ids = iid_traj_arr[te_iid]

    # 归一 (IID train)
    tr_mask = np.isin(tidT, tr_ids)
    fm = xT[tr_mask].mean(axis=0)
    fs = xT[tr_mask].std(axis=0) + 1e-6
    xT_norm = (xT - fm) / fs
    # F1-A: v2 已由 loader 归一到 H, factor=1.0; v1 用 transfer.rul_max_norm
    if ch_meta["channel_label_schema"] == CHANNEL_LABEL_SCHEMA_V2:
        rul_factor = 1.0
    else:
        rul_factor = float(tc["rul_max_norm"])
    rulT_norm = rulT / rul_factor
    lbT_norm = lbT / rul_factor

    # 构造数据集
    tstride = int(tc.get("target_stride", 50))
    def mkDS(ids):
        m = np.isin(tidT, ids)
        return ChannelSeqDataset(xT_norm[m], hiT[m], rulT_norm[m], ckT[m], L, K,
                                 stride=tstride, event_observed=evT[m],
                                 rul_lower_bound=lbT_norm[m])
    bs = int(cfg["pretrain"]["batch_size"])
    from torch.utils.data import DataLoader
    ltr = DataLoader(mkDS(tr_ids), batch_size=bs, shuffle=True)
    lva = DataLoader(mkDS(va_ids), batch_size=bs, shuffle=False)
    lte_iid = DataLoader(mkDS(te_ids), batch_size=bs, shuffle=False)
    lte_ood = DataLoader(mkDS(sorted(ood_ids_set)), batch_size=bs, shuffle=False)

    # 加载模型 + ckpt
    device = "cuda" if (cfg["pretrain"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    model = _build_model(cfg, int(ch_cfg["n_features"]), int(ch_cfg["n_features"]), device)
    if ckpt_path.exists():
        model.load_pretrained(str(ckpt_path), device)
    # target_only 微调 (在 IID train 上)
    import torch.nn as nn
    model.freeze_encoder(False)
    opt = torch.optim.Adam(model.parameters(), lr=float(tc["finetune_lr"]))
    lam = (float(cfg["loss"].get("beta_hi", 1.0)),
           float(cfg["loss"].get("mu_mono", 0.1)),
           float(cfg["loss"].get("nu_smooth", 0.1)))
    e = int(tc.get("epochs_s2", 20))
    _train_with_early_stop(model, ltr, lva, opt, device,
                           nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"])),
                           nn.MSELoss(), lam, e, f"OOD-target_only seed{seed}")

    # 评估
    m_iid = eval_test(model, lte_iid, device)
    m_ood = eval_test(model, lte_ood, device)
    if verbose:
        print(f"\n===== seed {seed} OOD 评估 (未进入 IID 主验收) =====")
        print(f"  IID test: RMSE={m_iid['rmse']:.4f} (n_failed={m_iid['n_failed']})")
        print(f"  OOD:      RMSE={m_ood['rmse']:.4f} (n_failed={m_ood['n_failed']})")
        if m_iid['n_failed'] > 0 and m_ood['n_failed'] > 0:
            ratio = m_ood['rmse'] / max(m_iid['rmse'], 1e-9)
            print(f"  OOD/IID 比值: {ratio:.2f}x ({'OOD 泛化良好' if ratio < 1.3 else '⚠ OOD 显著差, 模型可能过拟合 IID'})")
    return {"ood_info": ood_info,
            "iid_metrics": m_iid, "ood_metrics": m_ood,
            "note": "OOD 轨迹未进入 IID 训练/early stopping/配对 CI"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--ckpt", default="checkpoints/source_phased_array_tcn_pretrain.pt")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--manifest-out", default="docs/channel_ood_split_manifest.json")
    ap.add_argument("--out", default="docs/results_ood_eval.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ckpt = ROOT / args.ckpt

    # 先写 manifest (预注册)
    ch_cfg = cfg["channel_level"]
    target_h5 = ROOT / ch_cfg["feature_path"]
    if not target_h5.exists():
        print(f"!! 缺 {target_h5}; 先 build_channel_hi")
        sys.exit(1)
    ood_info = identify_ood_trajectories(target_h5)
    manifest = write_ood_manifest(ood_info, ROOT / args.manifest_out)
    print(f">> OOD manifest: {ROOT / args.manifest_out} "
          f"({manifest['n_ood']} OOD / {manifest['n_iid']} IID)")

    all_results = []
    for s in range(args.seeds):
        seed = cfg["seed"] + s
        res = evaluate_ood(cfg, ckpt, seed, verbose=True)
        res["seed"] = seed
        all_results.append(res)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"manifest": manifest, "results": all_results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n>> {out}")


if __name__ == "__main__":
    main()
