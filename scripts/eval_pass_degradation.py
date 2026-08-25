# -*- coding: utf-8 -*-
"""Pass 观测协议性能衰减评估 (预注册: docs/pass_observation_degradation_prereg.md)。

对照臂: C0 原始复现校验 (≈bundle test_rmse) / C1 pass-ffill 主口径 / C1b per-status 细分。
纪律: normalizer 用 bundle 训练统计量 (禁重拟合); 原始 h5 只读; F2 既有指标不动。

用法:
  /f/anaconda3/envs/pytorch_gpu/python.exe scripts/eval_pass_degradation.py \
      --pass-h5 tmp/channel_features_pass.h5 \
      --out outputs/pass_degradation/pass_degradation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from component.predict_gru import load_bundle, rebuild_model  # noqa: E402
from src.experiments.run_groups import eval_test  # noqa: E402
from src.transfer.channel_dataset import load_target_channel  # noqa: E402
from src.transfer.train_transfer import TargetSeqDataset, split_trajectories  # noqa: E402
from src.utils import load_config  # noqa: E402

BUNDLE_TEST_RMSE = 0.14521145820617676   # outputs/f2_formal_5seed/inference_bundle (校验门)
CALIB_TOL = 1e-3
BOOTSTRAP_N = 1000


def load_status_flat(h5_path: Path) -> np.ndarray:
    """按 load_target_channel 同序 (sorted traj → sorted sub → 拼接) 读 obs_status (N,)。"""
    out = []
    with h5py.File(h5_path, "r") as f:
        for tk in sorted(k for k in f.keys() if k.startswith("traj_")):
            for sk in sorted(k for k in f[tk].keys() if k.startswith("sub_")):
                out.append(f[f"{tk}/{sk}/obs_status"][:].astype(np.uint8))
    return np.concatenate(out)


def ffill_by_channel(x: np.ndarray, channel_keys: np.ndarray) -> np.ndarray:
    """每通道序列内 NaN 前向填充 (首段后向); 返回副本。"""
    x = x.copy()
    df_keys = pd.DataFrame({"ck": channel_keys})
    for _, g in df_keys.groupby("ck"):
        idxs = g.index.to_numpy()
        seq = pd.DataFrame(x[idxs]).ffill().bfill()
        if seq.isna().any().any():
            raise ValueError("整条通道无有效观测 (命中率参数异常)")
        x[idxs] = seq.values.astype(np.float32)
    return x


def window_end_rows(x, hi, rul, keys, L, K, stride):
    """复刻 TargetSeqDataset 滑窗循环, 返回每样本 (K,) 窗末全局行号列表 (与 ds.samples 同序)。"""
    rows_per_sample = []
    df = pd.DataFrame({"tid": keys})
    for _, g in df.groupby("tid"):
        idxs = g.index.to_numpy()
        T = len(idxs)
        if T < L:
            continue
        starts = list(range(0, T - L + 1, stride))
        ends = [idxs[s + L - 1] for s in starts]
        for s2 in range(0, len(starts) - K + 1):
            rows_per_sample.append(np.asarray(ends[s2:s2 + K]))
    return rows_per_sample


def collect_preds(model, ds, device="cpu"):
    """与 eval_test 相同的前向, 但额外返回窗末 (pred,label,event) 供 per-status 细分。"""
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    preds, labels, evs = [], [], []
    with torch.no_grad():
        for x, h, r, ev, lb, dmg in loader:
            x = x.to(device)
            B, Kk = x.size(0), x.size(1)
            _, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
            preds.append(rul_p.cpu().numpy())
            labels.append(r.reshape(-1).numpy())
            evs.append(ev.reshape(-1).numpy())
    return (np.concatenate(preds), np.concatenate(labels),
            np.concatenate(evs).astype(bool))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--bundle", default="outputs/f2_formal_5seed/inference_bundle")
    ap.add_argument("--pass-h5", default="tmp/channel_features_pass.h5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/pass_degradation/pass_degradation.json")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / args.config))
    tc = cfg["transfer"]
    L = int(cfg["model"]["input_len_L"])
    K = int(cfg["pretrain"].get("seq_block_K", 8))
    stride = int(tc.get("target_stride", 50))

    bundle = load_bundle(ROOT / args.bundle)
    model = rebuild_model(bundle, ROOT / args.bundle)
    mean = np.asarray(bundle["normalizer"]["mean"], dtype=np.float32)
    std = np.asarray(bundle["normalizer"]["std"], dtype=np.float32)

    src_h5 = ROOT / cfg["channel_level"]["feature_path"]
    pass_h5 = ROOT / args.pass_h5

    # ---- 数据 (原始 + pass), 同 split, 同归一 ----
    xO, hiO, rulO, ckO, tidO, evO, lbO, n_traj, _ = load_target_channel(src_h5)
    xP, hiP, rulP, ckP, tidP, evP, lbP, _, _ = load_target_channel(pass_h5)
    assert np.array_equal(ckO, ckP) and np.array_equal(tidO, tidP)
    assert np.array_equal(rulO, rulP) and np.array_equal(evO, evP)   # 标签不变 (协议只改可观测性)
    status = load_status_flat(pass_h5)
    assert status.shape == xP.shape[:1]

    xP = ffill_by_channel(xP, ckP)

    tr, va, te = split_trajectories(
        n_traj, [tc["split"]["train"], tc["split"]["val"], tc["split"]["test"]], args.seed)
    te_mask = np.isin(tidO, te)

    def mk(x):
        xn = (x[te_mask] - mean) / std
        return TargetSeqDataset(xn, hiO[te_mask], rulO[te_mask], ckO[te_mask], L, K,
                                stride=stride, event_observed=evO[te_mask],
                                rul_lower_bound=lbO[te_mask])

    def run_arm(tag, x, keep_preds):
        """构 Dataset → eval_test → (可选) collect → 立即释放 Dataset (峰值内存 ~1 dataset)。

        两次 run_arm 之间不并存 Dataset: samples python 列表 ~3GB/臂, 并存会顶爆内存
        进交换区 (2026-08-25 首跑教训)。
        """
        ds = mk(x)
        n_samples = len(ds)
        m = eval_test(model, DataLoader(ds, batch_size=256, shuffle=False), "cpu")
        pc = collect_preds(model, ds) if keep_preds else None
        del ds
        return m, pc, n_samples

    mO, (pO, tO, eO), nO = run_arm("C0", xO, keep_preds=True)
    print(f"[C0] samples={nO} rmse={mO['rmse']:.5f} (bundle 基线 {BUNDLE_TEST_RMSE:.5f}) "
          f"phm={mO['phm']:.2f} n_fail={mO['n_failed']}")
    if abs(mO["rmse"] - BUNDLE_TEST_RMSE) >= CALIB_TOL:
        raise SystemExit(f"C0 校验门未过: |{mO['rmse']:.5f}-{BUNDLE_TEST_RMSE:.5f}|>= {CALIB_TOL}")

    mP, (pP, tP, eP), nP = run_arm("C1", xP, keep_preds=True)
    print(f"[C1] samples={nP} rmse={mP['rmse']:.5f} phm={mP['phm']:.2f} n_fail={mP['n_failed']} "
          f"censor_viol={mP['censor_violation_rate']:.4f}")
    assert nO == nP, f"C0/C1 样本数不等 {nO} != {nP}"
    assert np.array_equal(tO, tP) and np.array_equal(eO, eP)

    # ---- paired bootstrap Δrmse (失效样本, 逐窗末点) ----
    fo = eO
    sqO, sqP = (pO - tO) ** 2, (pP - tP) ** 2
    rng = np.random.default_rng(20260825)
    n = int(fo.sum())
    sqO_f, sqP_f = sqO[fo], sqP[fo]
    boots = np.empty(BOOTSTRAP_N)
    for b in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, n)
        boots[b] = np.sqrt(np.mean(sqP_f[idx])) - np.sqrt(np.mean(sqO_f[idx]))
    delta = mP["rmse"] - mO["rmse"]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"[Δ] rmse差={delta:+.5f}  bootstrap95CI=[{ci[0]:+.5f},{ci[1]:+.5f}]")

    # ---- C1b per-status 细分 (窗末点级, 行号复刻自检: 与 collect 的标签逐点对齐) ----
    rows = window_end_rows((xO[te_mask] - mean) / std, hiO[te_mask], rulO[te_mask],
                           ckO[te_mask], L, K, stride)
    assert len(rows) * K == len(tP), f"行号复刻 {len(rows)}x{K} != 样本 {len(tP)}"
    rulO_te = rulO[te_mask]            # 预计算: 23.9M fancy index 严禁进 29 万次循环
    status_te = status[te_mask]
    rul_ends = np.concatenate([rulO_te[r_] for r_ in rows])
    assert np.array_equal(rul_ends, tP), "行号复刻与样本标签未对齐"
    st_flat = np.concatenate([status_te[r_] for r_ in rows])
    assert st_flat.shape == pP.shape
    fail_mask = eP
    rmse_by = {}
    for tag, m in [("normal", st_flat == 1), ("filled", st_flat != 1)]:
        mm = m & fail_mask
        rmse_by[tag] = (float(np.sqrt(np.mean((pP[mm] - tP[mm]) ** 2))) if mm.any() else None,
                        int(mm.sum()))
    print(f"[C1b] normal rmse={rmse_by['normal'][0]:.5f} (n={rmse_by['normal'][1]}) "
          f"filled rmse={rmse_by['filled'][0]:.5f} (n={rmse_by['filled'][1]})")

    result = {
        "prereg": "docs/pass_observation_degradation_prereg.md",
        "bundle": str(args.bundle), "seed_split": args.seed,
        "L_K_stride": [L, K, stride],
        "protocol_hit_rate_expected": 0.735,
        "C0_reproduction": mO,
        "C1_pass_ffill": mP,
        "delta_rmse": delta, "delta_rmse_ci95": ci,
        "C1b_per_status_rmse": {k: {"rmse": v[0], "n": v[1]} for k, v in rmse_by.items()},
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
