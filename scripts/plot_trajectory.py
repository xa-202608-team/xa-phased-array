#!/usr/bin/env python
"""scripts/plot_trajectory.py — 相控阵 RUL/HI 预测轨迹图

产出三张图 (PNG), 填补相控阵交付包缺少预测轨迹可视化的空白 (对标电池分支 v1_02/v1_03):

  1. rul_trajectory_panel.png  (2×3 面板)
     每面板一条测试轨迹: 横轴=任务时间(年), 纵轴=归一化RUL
     曲线: 真值 / target_only_gru / source_pretrain_finetune / hi_extrap基线 + EOL竖线

  2. hi_tracking_panel.png     (1×2 面板)
     横轴=时间, 纵轴=HI; 真实 HI vs 模型预测 HI

  3. rul_scatter.png           (单图)
     横轴=true RUL, 纵轴=predicted RUL, 全 test 集散点 + y=x 对角线 + RMSE 标注

数据加载 / 划分 / 归一化与 run_groups.py 完全一致 (同 seed, 同 split 比例),
仅追加逐轨迹推理 + 可视化, 不修改实验代码。

用法:
  python scripts/plot_trajectory.py [--config configs/phased_array.yaml]
                                     [--output-dir docs/figures]
                                     [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed                               # noqa: E402
from src.transfer.train_transfer import TargetSeqDataset, split_trajectories, load_target  # noqa: E402
from src.models.factory import build_transfer_model                        # noqa: E402
from src.train.pretrain import _rul_loss                                   # noqa: E402
from src.baselines.phased_array_baselines import _hi_extrap_predict        # noqa: E402

CKPT_DIR = ROOT / "checkpoints"

# ---------- 中文字体 ----------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 颜色 ----------
C_TRUE = "#1a1a1a"
C_TARGET = "#2563eb"    # 蓝
C_SOURCE = "#dc2626"    # 红
C_EXTRAP = "#16a34a"    # 绿
C_EOL = "#f59e0b"       # 橙


# ===================================================================== 数据
def load_and_prepare(cfg, seed):
    """加载数据 + 划分 + 归一化 (与 run_groups.py 完全一致)。"""
    tc = cfg["transfer"]
    mc = cfg["model"]
    L = int(mc["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    rul_max = float(tc.get("rul_max_norm", 1.0))

    xT, hiT, rulT, tidT, n_traj, evT, damageT = load_target(
        ROOT / tc.get("target_feature_path",
                       "data/features/phased_array/schema_v1/target/target_features.h5"),
        has_nodes=True, return_damage=True)

    tr, va, te = split_trajectories(n_traj, [tc["split"]["train"],
                                              tc["split"]["val"],
                                              tc["split"]["test"]], seed)

    # z-score 归一 (train 集)
    _tr_mask = np.isin(tidT, tr)
    _fm = xT[_tr_mask].mean(axis=0)
    _fs = xT[_tr_mask].std(axis=0) + 1e-6
    xT = (xT - _fm) / _fs

    # RUL 归一
    rulT = rulT / max(rul_max, 1.0)
    lbT = rulT.copy()

    return dict(xT=xT, hiT=hiT, rulT=rulT, tidT=tidT, evT=evT, damageT=damageT,
                tr=tr, va=va, te=te, n_traj=n_traj, L=L, K=K,
                rul_max=rul_max, n_features=xT.shape[1])


# ===================================================================== 模型训练
def build_model(cfg, n_features, n_target, device, encoder="tcn"):
    # F1-A: 复用唯一构造函数 (与训练/评估/导出/推理同架构, 防漂移)
    return build_transfer_model(cfg, n_features=n_features, n_target=n_target,
                                encoder_type=encoder, device=device)


def train_epoch(model, loader, opt, device, huber, mse, lam):
    model.train()
    for batch in loader:
        x, h, r, ev, lb, dmg = batch
        x, h, r = x.to(device), h.to(device), r.to(device)
        ev, lb = ev.to(device), lb.to(device)
        B, Kk = x.size(0), x.size(1)
        hi_p, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
        hi_p = hi_p.view(B, Kk)
        rul_p = rul_p.view(B, Kk)
        Lr, _, _ = _rul_loss(rul_p, r, ev, lb, huber, 1.0)
        Lh = mse(hi_p, h)
        d = hi_p[:, 1:] - hi_p[:, :-1]
        loss = Lr + lam[0] * Lh + lam[1] * torch.relu(-d).mean() + lam[2] * (d * d).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def eval_rmse(model, loader, device):
    model.eval()
    preds, labels, evs = [], [], []
    for x, h, r, ev, lb, dmg in loader:
        x = x.to(device)
        B, Kk = x.size(0), x.size(1)
        _, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
        preds.append(rul_p.cpu().numpy())
        labels.append(r.reshape(-1).numpy())
        evs.append(ev.reshape(-1).numpy())
    p = np.concatenate(preds) if preds else np.array([0.0])
    t = np.concatenate(labels) if labels else np.array([0.0])
    e = np.concatenate(evs) if evs else np.array([True])
    m = e.astype(bool)
    return float(np.sqrt(np.mean((p[m] - t[m]) ** 2))) if m.any() else 0.0


def train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, epochs, tag):
    best = (float("inf"), None)
    for ep in range(epochs):
        train_epoch(model, ltr, opt, device, huber, mse, lam)
        vm = eval_rmse(model, lva, device)
        if vm < best[0]:
            best = (vm, {k: v.detach().clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
    print(f"  [{tag}] best val_rmse={best[0]:.4f}")
    return best[0]


def train_model(model_name, data, cfg, device):
    """训练一个模型并返回。model_name: 'target_only_gru' | 'source_pretrain_finetune'。"""
    tc = cfg["transfer"]
    _lc = cfg["loss"]
    lam = (float(_lc.get("beta_hi", 1.0)),
           float(_lc.get("mu_mono", 1.0)),
           float(_lc.get("nu_smooth", 0.1)))
    huber = torch.nn.HuberLoss(delta=float(_lc["huber_delta"]))
    mse = torch.nn.MSELoss()
    L, K = data["L"], data["K"]
    tstride = int(tc.get("target_stride", 50))
    bs = int(cfg["pretrain"]["batch_size"])
    epochs = int(tc.get("epochs_s2", 20))

    def mkDS(ids):
        m = np.isin(data["tidT"], ids)
        return TargetSeqDataset(data["xT"][m], data["hiT"][m], data["rulT"][m],
                                data["tidT"][m], L, K, stride=tstride,
                                event_observed=data["evT"][m],
                                rul_lower_bound=data["rulT"][m],
                                damage_b=data["damageT"][m])

    ltr = DataLoader(mkDS(data["tr"]), batch_size=bs, shuffle=True)
    lva = DataLoader(mkDS(data["va"]), batch_size=bs, shuffle=False)

    if model_name == "target_only_gru":
        model = build_model(cfg, 4, data["n_features"], device, encoder="gru")
        model.freeze_encoder(False)
        opt = torch.optim.Adam(model.parameters(), lr=float(tc["finetune_lr"]))
        train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, epochs, "target_only_gru")
    elif model_name == "source_pretrain_finetune":
        model = build_model(cfg, 4, data["n_features"], device, encoder="tcn")
        ckpt = str(CKPT_DIR / "source_phased_array_tcn_pretrain.pt")
        if Path(ckpt).exists():
            model.load_pretrained(ckpt, device)
            print(f"  加载源域 checkpoint: {ckpt}")
        else:
            print(f"  [warning] checkpoint 不存在: {ckpt}, 使用随机权重")
        model.freeze_encoder(True)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=float(tc["finetune_lr"]))
        train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, epochs, "source_pretrain_finetune")
    return model


# ===================================================================== 逐轨迹推理
@torch.no_grad()
def predict_trajectory(model, traj_x, traj_hi, traj_rul, traj_ev, traj_lb, traj_dmg,
                       L, K, device):
    """对单条轨迹逐时间步推理。stride=1, 取每个 K-block 最后一窗的预测。

    返回: time_idx (M,), pred_rul (M,), pred_hi (M,), true_rul (M,), true_hi (M,)
    """
    model.eval()
    n = len(traj_x)
    if n < L * K:
        return (np.array([]),) * 5
    tid = np.zeros(n, dtype=int)
    ds = TargetSeqDataset(traj_x, traj_hi, traj_rul, tid, L, K, stride=1,
                          event_observed=traj_ev, rul_lower_bound=traj_lb,
                          damage_b=traj_dmg)
    if len(ds) == 0:
        return (np.array([]),) * 5
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    t_indices, p_ruls, p_his = [], [], []
    offset = 0   # 每个 sample 的最后一窗对应 time_idx = s2 + K + L - 2
    for x, h, r, ev, lb, dmg in loader:
        x = x.to(device)
        B, Kk = x.size(0), x.size(1)
        hi_p, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
        hi_p = hi_p.view(B, Kk)
        rul_p = rul_p.view(B, Kk)
        last_rul = rul_p[:, -1].cpu().numpy()   # 取最后一窗
        last_hi = hi_p[:, -1].cpu().numpy()
        for i in range(B):
            s2 = offset + i
            t_idx = s2 + K + L - 2
            if t_idx < n:
                t_indices.append(t_idx)
                p_ruls.append(float(last_rul[i]))
                p_his.append(float(last_hi[i]))
        offset += B

    t_arr = np.array(t_indices)
    # 可视化裁剪: 归一化 RUL 物理范围为 [0,1], 冻结 encoder 基线未学到退化时
    # 预测值溢出到 >1 (甚至 ~27), 裁剪到 [−0.02, 1.05] 保证面板可读。
    # 图注中注明裁剪 + 解释越界基线的含义。
    p_ruls_arr = np.clip(np.array(p_ruls), -0.02, 1.05)
    return (t_arr, p_ruls_arr, np.array(p_his),
            traj_rul[t_arr], traj_hi[t_arr])


# ===================================================================== 选轨迹
def select_trajectories(te_ids, h5_path, n=6):
    """从 test 集选代表性轨迹: 按寿命排序取早/中/晚/删失。"""
    infos = []
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys())
        for tid in te_ids:
            g = f[keys[tid]]
            eol = int(g.attrs.get("eol_idx", len(g["rul"])))
            event = int(g.attrs.get("event_observed", 1))
            infos.append(dict(tid=int(tid), eol=eol, event=event,
                              T=len(g["rul"])))
    min_T = 600   # L*K=512 + 余量, 太短无法开窗
    failed = sorted([i for i in infos if i["event"] == 1 and i["T"] >= min_T],
                    key=lambda x: x["eol"])
    censored = sorted([i for i in infos if i["event"] == 0 and i["T"] >= min_T],
                      key=lambda x: x["T"])

    picks = []
    if len(failed) >= 3:
        picks.append(failed[0])                  # 最早失效
        picks.append(failed[len(failed) // 2])   # 中位失效
        picks.append(failed[-1])                 # 最晚失效
    elif failed:
        picks.extend(failed)
    if censored:
        picks.append(censored[len(censored) // 2])  # 中位删失
    # 补齐到 n
    picked_ids = {p["tid"] for p in picks}
    all_remaining = [i for i in failed + censored if i["tid"] not in picked_ids]
    picks.extend(all_remaining[:n - len(picks)])
    return picks[:n]


# ===================================================================== 画图
def plot_rul_trajectory(models_pred, traj_infos, h5_path, rul_max, out_path):
    """图1: RUL 预测轨迹面板。"""
    n = len(traj_infos)
    nrows = (n + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 4.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    keys = sorted(h5py.File(h5_path, "r").keys())
    f = h5py.File(h5_path, "r")

    for idx, info in enumerate(traj_infos):
        ax = axes[idx // 3][idx % 3]
        tid = info["tid"]
        g = f[keys[tid]]
        time_s = g["time_s"][:]
        time_years = time_s / (365.25 * 24 * 3600)
        true_rul_norm = g["rul"][:] / rul_max
        eol = info["eol"]
        event = info["event"]

        # 真值
        ax.plot(time_years, true_rul_norm, color=C_TRUE, linewidth=1.5, label="真值 RUL", zorder=5)

        # 模型预测
        for label, preds in models_pred.items():
            if tid in preds:
                t_idx, p_rul, _ = preds[tid]
                if len(t_idx) > 0:
                    ax.plot(time_years[t_idx], p_rul, color=preds["_color"],
                            linewidth=1.2, alpha=0.85, label=label, zorder=4)

        # hi_extrap 基线
        hi_raw = g["hi_array"][:]
        hi_p = _hi_extrap_predict(hi_raw, rul_max, window=50)
        valid = np.isfinite(hi_p)
        if valid.any():
            hi_filled = np.where(valid, hi_p, np.nan)
            ax.plot(time_years, hi_filled, color=C_EXTRAP, linewidth=1.0,
                    alpha=0.7, linestyle="--", label="HI 外推基线", zorder=3)

        # EOL 竖线
        if event == 1 and eol < len(time_years):
            ax.axvline(time_years[eol], color=C_EOL, linestyle=":", linewidth=1.2,
                       label=f"EOL ({time_years[eol]:.2f} 年)")

        tag = "失效" if event == 1 else "删失"
        ax.set_title(f"轨迹 #{tid} ({tag}, T={time_years[-1]:.2f}年)", fontsize=10)
        ax.set_xlabel("任务时间 (年)")
        ax.set_ylabel("归一化 RUL")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(-0.02, 1.08)

    # 隐藏多余子图
    for idx in range(len(traj_infos), nrows * 3):
        axes[idx // 3][idx % 3].set_visible(False)

    f.close()
    fig.suptitle("相控阵天线 RUL 预测轨迹 — 模型 vs 真值 vs 基线", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_hi_tracking(models_pred, traj_infos, h5_path, out_path):
    """图2: HI 退化追踪面板。"""
    n = min(2, len(traj_infos))
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 4.5))
    if n == 1:
        axes = [axes]
    keys = sorted(h5py.File(h5_path, "r").keys())
    f = h5py.File(h5_path, "r")

    for idx in range(n):
        ax = axes[idx]
        info = traj_infos[idx]
        tid = info["tid"]
        g = f[keys[tid]]
        time_s = g["time_s"][:]
        time_years = time_s / (365.25 * 24 * 3600)
        true_hi = g["hi_array"][:]

        ax.plot(time_years, true_hi, color=C_TRUE, linewidth=1.5, label="真实 HI", zorder=5)

        for label, preds in models_pred.items():
            if tid in preds:
                t_idx, _, p_hi = preds[tid]
                if len(t_idx) > 0:
                    ax.plot(time_years[t_idx], p_hi, color=preds["_color"],
                            linewidth=1.2, alpha=0.85, label=f"{label} 预测 HI", zorder=4)

        ax.axhline(1.0, color=C_EOL, linestyle=":", linewidth=1, label="HI=1.0 越限")
        tag = "失效" if info["event"] == 1 else "删失"
        ax.set_title(f"轨迹 #{tid} ({tag})", fontsize=10)
        ax.set_xlabel("任务时间 (年)")
        ax.set_ylabel("HI (健康指标)")
        ax.legend(fontsize=8)

    f.close()
    fig.suptitle("相控阵天线 HI 退化追踪 — 模型预测 vs 真值", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_rul_scatter(model, data, device, out_path, model_label="target_only_gru"):
    """图3: 全测试集 RUL 散点图 (hexbin 密度 + 分位数校准曲线)。

    254k+ 失效点密集 scatter → hexbin 密度 + P10/P50/P90 校准带, 解决:
    (1) 预测值越界 (归一化 RUL >1) 使 lim 飙到 ~27;
    (2) 密集蓝块看不出分布; (3) 离散横条伪影。
    """
    L, K = data["L"], data["K"]
    tc_ratio = data["rul_max"]
    tstride = 1   # 全覆盖

    all_true, all_pred, all_ev = [], [], []
    te = data["te"]

    for tid in te:
        m = data["tidT"] == tid
        if m.sum() < L * K:
            continue
        t_idx, p_rul, _, t_rul, _ = predict_trajectory(
            model, data["xT"][m], data["hiT"][m], data["rulT"][m],
            data["evT"][m], data["rulT"][m], data["damageT"][m],
            L, K, device)
        if len(t_idx) == 0:
            continue
        all_true.append(t_rul)
        all_pred.append(p_rul)
        all_ev.append(data["evT"][m][t_idx])

    true = np.concatenate(all_true)
    pred = np.concatenate(all_pred)
    ev = np.concatenate(all_ev)

    mf = ev.astype(bool)
    mc = ~mf
    # 裁剪到 [0, 1.05] (归一化 RUL 物理范围; 越界预测不影响密度图)
    true_f = np.clip(true[mf], 0, 1.05)
    pred_f = np.clip(pred[mf], 0, 1.05)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    # hexbin 密度图
    hb = ax.hexbin(true_f, pred_f, gridsize=45, cmap="Blues", mincnt=1,
                   extent=[0, 1.05, 0, 1.05])
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("样本数 (log)", fontsize=10)

    # y=x 完美预测线
    ax.plot([0, 1.05], [0, 1.05], "k--", linewidth=1.2, alpha=0.6, label="y=x (完美预测)")

    # 分位数校准曲线 (每 0.1 bin 算 P10/P50/P90)
    bins = np.linspace(0, 1.0, 11)
    centers, p10, p50, p90 = [], [], [], []
    for i in range(len(bins) - 1):
        bmask = (true_f >= bins[i]) & (true_f < bins[i + 1])
        if bmask.sum() >= 5:
            centers.append((bins[i] + bins[i + 1]) / 2)
            p10.append(np.percentile(pred_f[bmask], 10))
            p50.append(np.percentile(pred_f[bmask], 50))
            p90.append(np.percentile(pred_f[bmask], 90))
    if centers:
        ax.plot(centers, p50, "r-", linewidth=2, label="P50 (中位)", zorder=5)
        ax.fill_between(centers, p10, p90, alpha=0.12, color="red",
                        label="P10–P90 带", zorder=4)

    if mf.any():
        rmse = float(np.sqrt(np.mean((pred[mf] - true[mf]) ** 2)))
        ax.text(0.03, 0.97, f"失效 RMSE = {rmse:.4f}\nN = {mf.sum()} 点\n(裁剪至 [0,1.05] 可视化)",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("真实 RUL (归一化)", fontsize=11)
    ax.set_ylabel("预测 RUL (归一化)", fontsize=11)
    ax.set_title(f"全测试集 RUL 散点 — {model_label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser(description="相控阵 RUL/HI 预测轨迹图")
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--output-dir", default="docs/figures")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(args.seed, cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])
    device = "cuda" if (cfg["pretrain"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("1. 加载数据 + 划分")
    data = load_and_prepare(cfg, args.seed)
    h5_path = ROOT / cfg["transfer"].get(
        "target_feature_path",
        "data/features/phased_array/schema_v1/target/target_features.h5")
    print(f"  n_traj={data['n_traj']}, train={len(data['tr'])}, "
          f"val={len(data['va'])}, test={len(data['te'])}")
    print(f"  n_features={data['n_features']}, L={data['L']}, K={data['K']}")
    print(f"  device={device}")

    print("=" * 60)
    print("2. 训练 target_only_gru")
    model_tgt = train_model("target_only_gru", data, cfg, device)

    print("=" * 60)
    print("3. 训练 source_pretrain_finetune")
    model_src = train_model("source_pretrain_finetune", data, cfg, device)

    print("=" * 60)
    print("4. 选代表性测试轨迹")
    traj_infos = select_trajectories(data["te"], h5_path, n=6)
    for info in traj_infos:
        tag = "失效" if info["event"] == 1 else "删失"
        print(f"  轨迹 #{info['tid']}: {tag}, EOL={info['eol']}, T={info['T']}")

    print("=" * 60)
    print("5. 逐轨迹推理")
    L, K = data["L"], data["K"]

    models_pred = {
        "target_only_gru": {"_color": C_TARGET},
        "source_pretrain_finetune": {"_color": C_SOURCE},
    }

    for info in traj_infos:
        tid = info["tid"]
        m = data["tidT"] == tid
        for name, model in [("target_only_gru", model_tgt),
                            ("source_pretrain_finetune", model_src)]:
            t_idx, p_rul, p_hi, t_rul, t_hi = predict_trajectory(
                model, data["xT"][m], data["hiT"][m], data["rulT"][m],
                data["evT"][m], data["rulT"][m], data["damageT"][m],
                L, K, device)
            models_pred[name][tid] = (t_idx, p_rul, p_hi)
            print(f"  #{tid} {name}: {len(t_idx)} 个预测点")

    print("=" * 60)
    print("6. 画图")
    plot_rul_trajectory(models_pred, traj_infos, h5_path, data["rul_max"],
                        out_dir / "rul_trajectory_panel.png")
    plot_hi_tracking(models_pred, traj_infos[:2], h5_path,
                     out_dir / "hi_tracking_panel.png")
    plot_rul_scatter(model_tgt, data, device,
                     out_dir / "rul_scatter.png", "target_only_gru")

    print("=" * 60)
    print("完成! 图表保存到:", out_dir)


if __name__ == "__main__":
    main()
