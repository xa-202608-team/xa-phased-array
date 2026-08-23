#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/run_fault_model_eval.py — 故障注入 §5.7 预测器降级评估。

加载/训练冻结 target_only GRU, 在标称 test (traj 70-84) + 故障 test 上推理,
配对统计 RMSE/PHM/MAE。不重训任何迁移组。

用法: python scripts/run_fault_model_eval.py
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

NOMINAL_H5 = ROOT / "data/features/phased_array/schema_ch_v1/target/channel_features.h5"
FAULT_H5S = {
    "rth_step":     ROOT / "data/simulated/phased_array/fault_rth_step_test/channel_features.h5",
    "thermal_bias": ROOT / "data/simulated/phased_array/fault_thermal_bias_test/channel_features.h5",
    "channel_open": ROOT / "data/simulated/phased_array/fault_channel_open_test/channel_features.h5",
    "cal_freeze":   ROOT / "data/simulated/phased_array/fault_cal_freeze_test/channel_features.h5",
}
TRAIN_IDS = list(range(0, 30))    # train split (15%)
EVAL_IDS = list(range(70, 85))    # test split 故障注入轨迹
L_WIN, STRIDE, K_BLK = 64, 50, 8
HIDDEN, N_EPOCHS, LR, BATCH = 64, 20, 1e-3, 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GRUModel(nn.Module):
    def __init__(self, n_feat=4, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def build_windows(h5_path, traj_ids, n_sub=16, L=L_WIN, stride=STRIDE, rul_norm=None):
    """从 h5 选择性构建滑窗。返回 (x, rul, event, rul_norm) numpy 数组 / float。

    F1-A: rul_norm 默认从 h5 schema 读 — v2 用 H (rul_scale_windows) 除 rul_ch_windows,
    v1 用 transfer.rul_max_norm(4088) 除 rul_ch。显式传 rul_norm 可覆盖。
    第 4 个返回值 rul_norm (float) 供 eval_rmse_phm 反归一用, 保证评估侧与
    build_windows 用同一个 H, 禁止评估侧默认值漂移。
    """
    from src.transfer.channel_dataset import (
        read_channel_label_meta, CHANNEL_LABEL_SCHEMA_V2)
    xs, rs, evs = [], [], []
    with h5py.File(h5_path, "r") as f:
        if rul_norm is None:
            meta = read_channel_label_meta(f)
            if meta["channel_label_schema"] == CHANNEL_LABEL_SCHEMA_V2:
                rul_norm = float(meta["rul_scale_windows"])
                rul_field = "rul_ch_windows"
            else:
                rul_norm = 4088.0
                rul_field = "rul_ch"
        else:
            meta = read_channel_label_meta(f)
            rul_field = ("rul_ch_windows"
                         if meta["channel_label_schema"] == CHANNEL_LABEL_SCHEMA_V2
                         else "rul_ch")
        for ti in traj_ids:
            for si in range(n_sub):
                key = f"traj_{ti:03d}/sub_{si:02d}" if f"traj_{ti:03d}" in f else None
                if not key or key not in f:
                    continue
                g = f[key]
                x_ch = g["x_ch"][:]
                rul = g[rul_field][:] / rul_norm
                event = bool(g.attrs.get("event_observed", 0))
                T = len(x_ch)
                for s in range(0, max(1, T - L + 1), stride):
                    end = min(s + L, T)
                    xs.append(x_ch[s:end])
                    idx = min(end - 1, T - 1)
                    rs.append(rul[idx])
                    evs.append(event)
    # pad sequences to L
    x_arr = np.zeros((len(xs), L, 4), dtype=np.float32)
    for i, w in enumerate(xs):
        x_arr[i, :len(w)] = w
    return x_arr, np.array(rs, dtype=np.float32), np.array(evs, dtype=bool), float(rul_norm)


def train_model(x_tr, r_tr, ev_tr, x_va, r_va, ev_va):
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
        tr_loss = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            pred = model(xt[idx])
            loss = huber(pred, rt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(idx)
        # val
        val_rmse = 0
        if xv is not None:
            model.eval()
            with torch.no_grad():
                vp = model(xv)
                val_rmse = torch.sqrt(((vp - rv) ** 2).mean()).item()
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == N_EPOCHS - 1:
            print(f"  ep{ep:02d} train_loss={tr_loss/n:.6f} val_rmse={val_rmse:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_rmse_phm(model, x, rul, event, rul_norm):
    """计算 RMSE (仅失效) + PHM Score + MAE + 删失违反率。

    rul_norm 为必填: 与 build_windows 返回的归一因子一致 (v2=H, v1=4088),
    用于把归一化预测/标签反归一回绝对窗口数后算 RMSE/PHM。
    """
    if not event.any():
        return dict(rmse=float("nan"), mae=float("nan"), phm=float("nan"), n_fail=0, n_cens=0)
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x[event], device=DEVICE)
        pred = model(xt).cpu().numpy() * rul_norm
        true = rul[event] * rul_norm
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    # PHM Score (Saxena 2008): exp(-d/13-1) for late, exp(d/10-1) for early
    d = np.clip(pred - true, -500, 500)
    s = np.where(d < 0, np.exp(-d / 13.0 - 1.0) - 1, np.exp(d / 10.0 - 1) - 1)
    phm = float(np.mean(s))
    # 删失违反率
    n_cens = (~event).sum()
    cens_viol = 0
    if n_cens > 0:
        xc = torch.tensor(x[~event], device=DEVICE)
        with torch.no_grad():
            pc = model(xc).cpu().numpy() * rul_norm
            tc = rul[~event] * rul_norm
            cens_viol = float(np.mean(pc < tc))  # 预测 < 下界 = 违反
    return dict(rmse=rmse, mae=mae, phm=phm, n_fail=int(event.sum()),
                n_cens=int(n_cens), censor_violation=cens_viol)


def main():
    set_seed_fn = __import__("src.utils.seed", fromlist=["set_seed"]).set_seed
    set_seed_fn(42, True, False)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=== 加载标称数据 ===")
    x_tr, r_tr, ev_tr, _rn_tr = build_windows(NOMINAL_H5, TRAIN_IDS)
    x_va, r_va, ev_va, _rn_va = build_windows(NOMINAL_H5, list(range(30, 50)))  # val
    print(f"  train: {len(x_tr)} 窗 (失效 {ev_tr.sum()})")
    print(f"  val:   {len(x_va)} 窗 (失效 {ev_va.sum()})")

    print("\n=== 训练 target_only GRU ===")
    model = train_model(x_tr, r_tr, ev_tr, x_va, r_va, ev_va)

    print("\n=== 评估标称 test (traj 70-84) ===")
    x_nom, r_nom, ev_nom, rn_nom = build_windows(NOMINAL_H5, EVAL_IDS)
    m_nom = eval_rmse_phm(model, x_nom, r_nom, ev_nom, rn_nom)
    print(f"  RMSE={m_nom['rmse']:.4f} MAE={m_nom['mae']:.4f} PHM={m_nom['phm']:.2f} "
          f"失效={m_nom['n_fail']} 删失={m_nom['n_cens']} 违反率={m_nom['censor_violation']:.3f}")

    results = dict(nominal=m_nom, faults={})
    print("\n=== 评估故障 test ===")
    for ft, h5_path in FAULT_H5S.items():
        if not h5_path.exists():
            print(f"  [跳过] {ft}")
            continue
        x_f, r_f, ev_f, rn_f = build_windows(h5_path, list(range(15)))  # 故障 h5 中是 traj_000~014
        m_f = eval_rmse_phm(model, x_f, r_f, ev_f, rn_f)
        ratio = m_f["rmse"] / m_nom["rmse"] if m_nom["rmse"] > 0 else float("nan")
        print(f"  {ft:14s}: RMSE={m_f['rmse']:.4f} (×{ratio:.2f}) MAE={m_f['mae']:.4f} "
              f"PHM={m_f['phm']:.2f} 违反率={m_f['censor_violation']:.3f}")
        m_f["rmse_ratio"] = ratio
        results["faults"][ft] = m_f

    # 保存
    out = ROOT / "docs/实验结果汇总/11_故障注入/results_fault_model_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")

    # 汇总表
    print("\n=== 预测器降级汇总 ===")
    print(f"{'故障类型':<16} {'标称RMSE':<10} {'故障RMSE':<10} {'误差比':<8} {'PHM变化':<10}")
    for ft, m in results["faults"].items():
        phm_delta = m["phm"] - m_nom["phm"]
        print(f"{ft:<16} {m_nom['rmse']:<10.4f} {m['rmse']:<10.4f} {m['rmse_ratio']:<8.2f} {phm_delta:+.2f}")


if __name__ == "__main__":
    main()
