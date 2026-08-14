#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/run_feature_space_transfer.py — 特征空间迁移受控对照 (§6.6)。

唯一变量: scaler (z-score mean/std) 来源。其余全固定 (同模型/同 split/同训练超参)。
  A: 源域迁移 scaler — mosfet_canonical.h5 全部 42 器件拟合
  B: 目标域原生 scaler — channel_features.h5 train split (traj 0-29) 拟合
3 seed 配对比较 RMSE。

与电池分支的"冻结源域坐标系"对照 (源域迁移 0.003328 vs 目标域原生 0.004271, 6/6 族优于)
形成跨组件统一结论: 坐标系迁移是否在相控阵域对上同样有效。

用法: python scripts/run_feature_space_transfer.py
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC_H5 = ROOT / "data/features/phased_array/schema_v4/source/mosfet_canonical.h5"
NOMINAL_H5 = ROOT / "data/features/phased_array/schema_ch_v1/target/channel_features.h5"
TRAIN_IDS = list(range(0, 30))
EVAL_IDS = list(range(70, 85))
L_WIN, STRIDE = 64, 50
HIDDEN, N_EPOCHS, LR, BATCH = 64, 20, 1e-3, 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUL_NORM = 4088.0
SEEDS = [42, 43, 44]


class GRUModel(nn.Module):
    def __init__(self, n_feat=4, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def fit_source_scaler():
    """从源域 mosfet_canonical.h5 拟合 mean/std (4 维)。"""
    all_x = []
    with h5py.File(SRC_H5, "r") as f:
        for dev_name in f["devices"].keys():
            g = f["devices"][dev_name]
            if "x" in g:
                all_x.append(g["x"][:])
    all_x = np.concatenate(all_x, axis=0)
    mean = all_x.mean(axis=0)
    std = np.maximum(all_x.std(axis=0), 1e-8)  # 防 duty std=0 除零
    return mean.astype(np.float32), std.astype(np.float32)


def fit_target_scaler():
    """从目标域 channel_features.h5 train split 拟合 mean/std。"""
    all_x = []
    with h5py.File(NOMINAL_H5, "r") as f:
        for ti in TRAIN_IDS:
            for si in range(16):
                key = f"traj_{ti:03d}/sub_{si:02d}"
                if key in f:
                    all_x.append(f[key]["x_ch"][:])
    all_x = np.concatenate(all_x, axis=0)
    mean = all_x.mean(axis=0)
    std = np.maximum(all_x.std(axis=0), 1e-8)
    return mean.astype(np.float32), std.astype(np.float32)


def build_windows(h5_path, traj_ids, scaler_mean, scaler_std, n_sub=16):
    """构建 z-score 归一化的滑窗。"""
    xs, rs, evs = [], [], []
    with h5py.File(h5_path, "r") as f:
        for ti in traj_ids:
            for si in range(n_sub):
                key = f"traj_{ti:03d}/sub_{si:02d}"
                if not (f"traj_{ti:03d}" in f and key in f):
                    continue
                g = f[key]
                x_ch = (g["x_ch"][:] - scaler_mean) / scaler_std  # z-score
                rul = g["rul_ch"][:] / RUL_NORM
                event = bool(g.attrs.get("event_observed", 0))
                T = len(x_ch)
                for s in range(0, max(1, T - L_WIN + 1), STRIDE):
                    end = min(s + L_WIN, T)
                    xs.append(x_ch[s:end])
                    rs.append(rul[min(end - 1, T - 1)])
                    evs.append(event)
    x_arr = np.zeros((len(xs), L_WIN, 4), dtype=np.float32)
    for i, w in enumerate(xs):
        x_arr[i, :len(w)] = w
    x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=3.0, neginf=-3.0)  # 防极端值
    return x_arr, np.array(rs, dtype=np.float32), np.array(evs, dtype=bool)


def train_one(x_tr, r_tr, ev_tr, x_va, r_va, ev_va, seed):
    from src.utils.seed import set_seed
    set_seed(seed, True, False)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = GRUModel().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    huber = nn.HuberLoss(delta=1.0)
    xt = torch.tensor(x_tr[ev_tr], device=DEVICE)
    rt = torch.tensor(r_tr[ev_tr], device=DEVICE)
    xv = torch.tensor(x_va[ev_va], device=DEVICE) if ev_va.any() else None
    rv = torch.tensor(r_va[ev_va], device=DEVICE) if ev_va.any() else None
    best_val, best_state = 1e9, None
    n = len(xt)
    for ep in range(N_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            pred = model(xt[idx]); loss = huber(pred, rt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if xv is not None:
            model.eval()
            with torch.no_grad():
                vp = model(xv)
                val_rmse = torch.sqrt(((vp - rv) ** 2).mean()).item()
            if val_rmse < best_val:
                best_val = val_rmse; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_model(model, x, rul, event):
    if not event.any():
        return float("nan")
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x[event], device=DEVICE)
        pred = model(xt).cpu().numpy() * RUL_NORM
        true = rul[event] * RUL_NORM
    return float(np.sqrt(np.mean((pred - true) ** 2))) / RUL_NORM  # 归一化 RMSE


def main():
    print("=== 拟合 scaler ===")
    src_mean, src_std = fit_source_scaler()
    tgt_mean, tgt_std = fit_target_scaler()
    print(f"源域: mean={src_mean}, std={src_std}")
    print(f"目标: mean={tgt_mean}, std={tgt_std}")
    print(f"std 比 (src/tgt): {src_std / tgt_std}")

    results = {}
    for scaler_name, (mean, std) in [("source", (src_mean, src_std)),
                                      ("target", (tgt_mean, tgt_std))]:
        print(f"\n=== Scaler: {scaler_name} ===")
        seed_rmses = []
        for seed in SEEDS:
            x_tr, r_tr, ev_tr = build_windows(NOMINAL_H5, TRAIN_IDS, mean, std)
            x_va, r_va, ev_va = build_windows(NOMINAL_H5, list(range(30, 50)), mean, std)
            x_te, r_te, ev_te = build_windows(NOMINAL_H5, EVAL_IDS, mean, std)
            model = train_one(x_tr, r_tr, ev_tr, x_va, r_va, ev_va, seed)
            rmse = eval_model(model, x_te, r_te, ev_te)
            seed_rmses.append(rmse)
            print(f"  seed {seed}: test RMSE = {rmse:.4f}")
        results[scaler_name] = seed_rmses

    # 配对统计
    src_rmses = results["source"]
    tgt_rmses = results["target"]
    deltas = [s - t for s, t in zip(src_rmses, tgt_rmses)]
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas, ddof=1) if len(deltas) > 1 else 0
    # t(2) CI
    from scipy import stats as sp_stats
    t_crit = sp_stats.t.ppf(0.975, len(deltas) - 1) if len(deltas) > 1 else 0
    ci_lo = mean_delta - t_crit * std_delta / math.sqrt(len(deltas))
    ci_hi = mean_delta + t_crit * std_delta / math.sqrt(len(deltas))

    print(f"\n=== 配对统计 (source − target, 正=源域scaler更差) ===")
    print(f"  source RMSE: {np.mean(src_rmses):.4f} ± {np.std(src_rmses, ddof=1):.4f}")
    print(f"  target RMSE: {np.mean(tgt_rmses):.4f} ± {np.std(tgt_rmses, ddof=1):.4f}")
    print(f"  Δ (source−target): {mean_delta:+.4f} ± {std_delta:.4f}")
    print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    cross0 = ci_lo <= 0 <= ci_hi
    print(f"  CI {'跨' if cross0 else '不跨'}0 → {'无显著差异' if cross0 else '有显著差异'}")

    out_data = dict(
        source_rmse_per_seed=src_rmses, target_rmse_per_seed=tgt_rmses,
        source_mean=float(np.mean(src_rmses)), target_mean=float(np.mean(tgt_rmses)),
        delta_mean=float(mean_delta), delta_std=float(std_delta),
        ci=[float(ci_lo), float(ci_hi)], ci_cross0=bool(cross0),
        source_scaler=dict(mean=src_mean.tolist(), std=src_std.tolist()),
        target_scaler=dict(mean=tgt_mean.tolist(), std=tgt_std.tolist()),
    )
    out = ROOT / "docs/实验结果汇总/12_特征空间迁移/results_feature_space_transfer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
