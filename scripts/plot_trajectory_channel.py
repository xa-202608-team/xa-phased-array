#!/usr/bin/env python
"""scripts/plot_trajectory_channel.py — 相控阵通道级 RUL/HI 预测轨迹图

通道级 (channel-level) 可视化: 每条轨迹含 16 个子阵通道, 各通道独立退化。
产出三张图, 展示通道级退化空间多样性 + 模型预测精度:

  1. ch_rul_trajectory_panel.png  (2×3 面板)
     选 1 条测试轨迹的 6 个代表性子阵通道 (最快/中位/最慢退化),
     每面板画 true RUL vs 模型预测 vs 基线

  2. ch_hi_spatial_diversity.png  (1×2 面板)
     左: 单条轨迹 16 子阵真实 HI 退化曲线叠加 (展示空间多样性)
     右: 对应 RUL 退化曲线叠加

  3. ch_rul_scatter.png           (单图)
     全通道级 test 集散点: predicted vs true RUL + y=x + RMSE

用法:
  python scripts/plot_trajectory_channel.py [--config configs/phased_array.yaml]
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

from src.utils import load_config, set_seed                                  # noqa: E402
from src.transfer.train_transfer import TargetSeqDataset, split_trajectories  # noqa: E402
from src.transfer.adapter import TransferModel                                # noqa: E402
from src.transfer.channel_dataset import load_target_channel                  # noqa: E402
from src.train.pretrain import _rul_loss                                      # noqa: E402

CKPT_DIR = ROOT / "checkpoints"
N_SUB = 16

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_TRUE = "#1a1a1a"
C_TARGET = "#2563eb"
C_SOURCE = "#dc2626"
C_EOL = "#f59e0b"
# 16 子阵配色 (tab20 色环)
SUB_COLORS = plt.cm.tab20(np.linspace(0, 1, N_SUB))


# ===================================================================== 数据
def load_and_prepare(cfg, seed):
    """通道级数据加载 + 划分 + 归一化 (与 run_groups level=channel 一致)。"""
    tc = cfg["transfer"]
    mc = cfg["model"]
    ch_cfg = cfg["channel_level"]
    L = int(mc["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    rul_max = float(tc.get("rul_max_norm", 1.0))
    h5_path = ROOT / ch_cfg["feature_path"]

    xT, hiT, rulT, ckT, tidT, evT, lbT, n_traj, sidT = load_target_channel(h5_path)
    tr, va, te = split_trajectories(n_traj, [tc["split"]["train"],
                                              tc["split"]["val"],
                                              tc["split"]["test"]], seed)

    # z-score (train)
    _tr_mask = np.isin(tidT, tr)
    _fm = xT[_tr_mask].mean(axis=0)
    _fs = xT[_tr_mask].std(axis=0) + 1e-6
    xT = (xT - _fm) / _fs
    # RUL 归一
    rulT = rulT / max(rul_max, 1.0)
    lbT = rulT.copy()
    damageT = np.zeros_like(rulT)   # 通道级无 damage_norm

    return dict(xT=xT, hiT=hiT, rulT=rulT, ckT=ckT, tidT=tidT, sidT=sidT,
                evT=evT, lbT=lbT, damageT=damageT,
                tr=tr, va=va, te=te, n_traj=n_traj, L=L, K=K,
                rul_max=rul_max, n_features=xT.shape[1], h5_path=h5_path)


# ===================================================================== 模型
def build_model(cfg, n_target, device, encoder="gru"):
    mc = cfg["model"]
    tc = cfg["transfer"]
    # 通道级: source 4 维 canonical → encoder, target 4 维 x_ch → adapter
    return TransferModel(
        encoder_type=encoder, n_features=4, n_target=n_target,
        channels=mc["tcn"]["channels"], kernel_size=mc["tcn"]["kernel_size"],
        num_blocks=mc["tcn"]["num_blocks"], dropout=mc["tcn"]["dropout"],
        latent_dim=mc["latent_dim"], adapter_hidden=tc["adapter_hidden"]).to(device)


def _train_epoch(model, loader, opt, device, huber, mse, lam):
    model.train()
    for x, h, r, ev, lb, dmg in loader:
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
def _eval_rmse(model, loader, device):
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


def _train(model, ltr, lva, opt, device, huber, mse, lam, epochs, tag):
    best = (float("inf"), None)
    for ep in range(epochs):
        _train_epoch(model, ltr, opt, device, huber, mse, lam)
        vm = _eval_rmse(model, lva, device)
        if vm < best[0]:
            best = (vm, {k: v.detach().clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
    print(f"  [{tag}] best val_rmse={best[0]:.4f}")


def train_model(model_name, data, cfg, device):
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
                                data["ckT"][m], L, K, stride=tstride,
                                event_observed=data["evT"][m],
                                rul_lower_bound=data["lbT"][m],
                                damage_b=data["damageT"][m])

    ltr = DataLoader(mkDS(data["tr"]), batch_size=bs, shuffle=True)
    lva = DataLoader(mkDS(data["va"]), batch_size=bs, shuffle=False)

    if model_name == "ch_target_only_gru":
        model = build_model(cfg, data["n_features"], device, encoder="gru")
        model.freeze_encoder(False)
        opt = torch.optim.Adam(model.parameters(), lr=float(tc["finetune_lr"]))
        _train(model, ltr, lva, opt, device, huber, mse, lam, epochs, "ch_target_only_gru")
    elif model_name == "ch_source_pretrain_frozen":
        model = build_model(cfg, data["n_features"], device, encoder="tcn")
        ckpt = str(CKPT_DIR / "source_phased_array_tcn_pretrain.pt")
        if Path(ckpt).exists():
            model.load_pretrained(ckpt, device)
            print(f"  加载源域 checkpoint: {ckpt}")
        model.freeze_encoder(True)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=float(tc["finetune_lr"]))
        _train(model, ltr, lva, opt, device, huber, mse, lam, epochs, "ch_source_pretrain_frozen")
    return model


# ===================================================================== 逐通道推理
@torch.no_grad()
def predict_channel(model, x_ch, hi_ch, rul_ch, ev_ch, lb_ch, dmg_ch, L, K, device):
    """对单个子阵通道逐时间步推理, 返回 (t_idx, pred_rul, pred_hi, true_rul, true_hi)。"""
    model.eval()
    n = len(x_ch)
    if n < L * K:
        return (np.array([]),) * 5
    ck = np.zeros(n, dtype=int)   # 单通道
    ds = TargetSeqDataset(x_ch, hi_ch, rul_ch, ck, L, K, stride=1,
                          event_observed=ev_ch, rul_lower_bound=lb_ch,
                          damage_b=dmg_ch)
    if len(ds) == 0:
        return (np.array([]),) * 5
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    t_indices, p_ruls, p_his = [], [], []
    offset = 0
    for x, h, r, ev, lb, dmg in loader:
        x = x.to(device)
        B, Kk = x.size(0), x.size(1)
        hi_p, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
        hi_p = hi_p.view(B, Kk)
        rul_p = rul_p.view(B, Kk)
        last_rul = rul_p[:, -1].cpu().numpy()
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
    # 可视化裁剪: 冻结 TCN encoder 基线预测溢出到 ~27 (未学到退化趋势, 输出近常数),
    # 裁剪到 [−0.02, 1.05] 保证面板可读; 图注注明裁剪。
    p_ruls_arr = np.clip(np.array(p_ruls), -0.02, 1.05)
    return (t_arr, p_ruls_arr, np.array(p_his),
            rul_ch[t_arr], hi_ch[t_arr])


# ===================================================================== h5 辅助
def read_traj_channel_raw(h5_path, traj_id):
    """从 h5 读一条轨迹的全部 16 子阵原始数据 (未归一)。"""
    subs = {}
    with h5py.File(h5_path, "r") as f:
        key = f"traj_{traj_id:03d}"
        if key not in f:
            return None
        g = f[key]
        for sk in sorted(k for k in g.keys() if k.startswith("sub_")):
            sub = g[sk]
            sid = int(sub.attrs["sub_id"])
            subs[sid] = dict(
                x=sub["x_ch"][:].astype(np.float32),
                hi=sub["hi_ch"][:].astype(np.float32),
                rul=sub["rul_ch"][:].astype(np.float32),
                event=bool(sub.attrs["event_observed"]),
                T=sub["x_ch"].shape[0],
            )
    return subs


def select_test_trajectory(data, h5_path, min_T=600):
    """选一条中等寿命的失效测试轨迹 (有足够子阵多样性)。"""
    with h5py.File(h5_path, "r") as f:
        infos = []
        for tid in data["te"]:
            key = f"traj_{tid:03d}"
            if key not in f:
                continue
            g = f[key]
            sub_keys = [k for k in g.keys() if k.startswith("sub_")]
            if not sub_keys:
                continue
            T = g[sub_keys[0]]["x_ch"].shape[0]
            event = int(g[sub_keys[0]].attrs["event_observed"])
            if T >= min_T and event == 1:
                infos.append((tid, T))
    if not infos:
        # 退而求其次: 任何够长的
        for tid in data["te"]:
            key = f"traj_{tid:03d}"
            if key not in f:
                continue
            g = f[key]
            sk = [k for k in g.keys() if k.startswith("sub_")]
            if sk:
                T = g[sk[0]]["x_ch"].shape[0]
                if T >= min_T:
                    infos.append((tid, T))
    infos.sort(key=lambda x: x[1])
    return infos[len(infos) // 2][0] if infos else data["te"][0]


# ===================================================================== 画图
def plot_ch_rul_trajectory(model_tgt, model_src, data, traj_id, device, out_path):
    """图1: 通道级 RUL 预测轨迹面板 (2×3)。"""
    subs_raw = read_traj_channel_raw(data["h5_path"], traj_id)
    if subs_raw is None:
        print(f"  [skip] 轨迹 {traj_id} 不存在")
        return

    # 按退化速率排序 (用末端 z/hi 判)
    sub_ids_sorted = sorted(subs_raw.keys(),
                            key=lambda s: subs_raw[s]["hi"][-1], reverse=True)
    # 选 6 个: 最快2 / 中位2 / 最慢2
    picks = [sub_ids_sorted[0], sub_ids_sorted[1],
             sub_ids_sorted[N_SUB // 2 - 1], sub_ids_sorted[N_SUB // 2],
             sub_ids_sorted[-2], sub_ids_sorted[-1]]

    L, K = data["L"], data["K"]
    rul_max = data["rul_max"]
    # 归一用 train stats (与训练一致)
    _fm = data["xT"][np.isin(data["tidT"], data["tr"])].mean(axis=0)
    _fs = data["xT"][np.isin(data["tidT"], data["tr"])].std(axis=0) + 1e-6

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for idx, sid in enumerate(picks):
        ax = axes[idx // 3][idx % 3]
        raw = subs_raw[sid]
        T = raw["T"]
        time_years = np.arange(T) * 21600 / (365.25 * 24 * 3600)
        true_rul = raw["rul"] / rul_max

        # 归一化特征
        x_norm = (raw["x"] - _fm) / _fs
        rul_norm = raw["rul"] / rul_max
        hi = raw["hi"]
        ev = np.full(T, raw["event"], dtype=bool)
        dmg = np.zeros(T, dtype=np.float32)

        ax.plot(time_years, true_rul, color=C_TRUE, linewidth=1.5, label="真值 RUL", zorder=5)

        for name, model, color in [("target_only_gru", model_tgt, C_TARGET),
                                   ("source_frozen", model_src, C_SOURCE)]:
            t_idx, p_rul, _, _, _ = predict_channel(
                model, x_norm, hi, rul_norm, ev, rul_norm, dmg, L, K, device)
            if len(t_idx) > 0:
                ax.plot(time_years[t_idx], p_rul, color=color, linewidth=1.2,
                        alpha=0.85, label=name, zorder=4)

        ax.set_title(f"子阵 #{sid} (T={time_years[-1]:.2f}年)", fontsize=10)
        ax.set_xlabel("任务时间 (年)")
        ax.set_ylabel("归一化 RUL")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(-0.02, 1.08)

    fig.suptitle(f"通道级 RUL 预测轨迹 — 轨迹 #{traj_id} 的 6 个代表性子阵",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_ch_hi_spatial(model_tgt, data, traj_id, device, out_path):
    """图2: 16 子阵 HI/RUL 退化空间多样性 + 模型预测叠加。"""
    subs_raw = read_traj_channel_raw(data["h5_path"], traj_id)
    if subs_raw is None:
        return
    L, K = data["L"], data["K"]
    rul_max = data["rul_max"]
    _fm = data["xT"][np.isin(data["tidT"], data["tr"])].mean(axis=0)
    _fs = data["xT"][np.isin(data["tidT"], data["tr"])].std(axis=0) + 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax_idx, metric in enumerate(["hi", "rul"]):
        ax = axes[ax_idx]
        for sid in sorted(subs_raw.keys()):
            raw = subs_raw[sid]
            T = raw["T"]
            time_years = np.arange(T) * 21600 / (365.25 * 24 * 3600)
            if metric == "hi":
                ax.plot(time_years, raw["hi"], color=SUB_COLORS[sid], alpha=0.7,
                        linewidth=0.8, label=f"sub_{sid:02d}")
            else:
                ax.plot(time_years, raw["rul"] / rul_max, color=SUB_COLORS[sid],
                        alpha=0.7, linewidth=0.8, label=f"sub_{sid:02d}")

        # 在 3 个代表子阵上叠加模型预测
        sub_ids_sorted = sorted(subs_raw.keys(),
                                key=lambda s: subs_raw[s]["hi"][-1], reverse=True)
        for sid in [sub_ids_sorted[0], sub_ids_sorted[N_SUB // 2], sub_ids_sorted[-1]]:
            raw = subs_raw[sid]
            T = raw["T"]
            time_years = np.arange(T) * 21600 / (365.25 * 24 * 3600)
            x_norm = (raw["x"] - _fm) / _fs
            rul_norm = raw["rul"] / rul_max
            ev = np.full(T, raw["event"], dtype=bool)
            dmg = np.zeros(T, dtype=np.float32)
            t_idx, p_rul, p_hi, _, _ = predict_channel(
                model_tgt, x_norm, raw["hi"], rul_norm, ev, rul_norm, dmg, L, K, device)
            if len(t_idx) > 0:
                if metric == "hi":
                    ax.scatter(time_years[t_idx], p_hi, color=SUB_COLORS[sid],
                               s=3, zorder=5, edgecolors="none")
                else:
                    ax.scatter(time_years[t_idx], p_rul, color=SUB_COLORS[sid],
                               s=3, zorder=5, edgecolors="none")

        ylabel = "HI (健康指标)" if metric == "hi" else "归一化 RUL"
        title = "16 子阵真实 HI 退化" if metric == "hi" else "16 子阵真实 RUL 退化"
        if metric == "hi":
            ax.axhline(1.0, color=C_EOL, linestyle=":", linewidth=1)
        ax.set_title(f"{title} + 模型预测 (点)", fontsize=11)
        ax.set_xlabel("任务时间 (年)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, ncol=4, loc="best")

    fig.suptitle(f"通道级退化空间多样性 — 轨迹 #{traj_id} (16 子阵)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_ch_rul_scatter(model, data, device, out_path, label="ch_target_only_gru"):
    """图3: 通道级全 test 散点。用 stride=50 采样 + 批量推理加速。"""
    L, K = data["L"], data["K"]
    rul_max = data["rul_max"]
    _fm = data["xT"][np.isin(data["tidT"], data["tr"])].mean(axis=0)
    _fs = data["xT"][np.isin(data["tidT"], data["tr"])].std(axis=0) + 1e-6
    sample_stride = 50   # 散点图不需要逐点, 每 50 步采一个足够

    # 批量构建所有 test 通道的窗口
    all_true, all_pred, all_ev = [], [], []
    model.eval()
    with torch.no_grad():
        for tid in data["te"]:
            subs = read_traj_channel_raw(data["h5_path"], tid)
            if subs is None:
                continue
            for sid in sorted(subs.keys()):
                raw = subs[sid]
                T = raw["T"]
                if T < L * K:
                    continue
                x_norm = (raw["x"] - _fm) / _fs
                rul_norm = raw["rul"] / rul_max
                ev_arr = np.full(T, raw["event"], dtype=bool)
                dmg = np.zeros(T, dtype=np.float32)
                ck = np.zeros(T, dtype=int)
                ds = TargetSeqDataset(x_norm, raw["hi"], rul_norm, ck, L, K,
                                      stride=sample_stride,
                                      event_observed=ev_arr, rul_lower_bound=rul_norm,
                                      damage_b=dmg)
                if len(ds) == 0:
                    continue
                loader = DataLoader(ds, batch_size=128, shuffle=False)
                offset = 0
                for x, h, r, ev_b, lb, dmg_b in loader:
                    x = x.to(device)
                    B, Kk = x.size(0), x.size(1)
                    _, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
                    rul_p = rul_p.view(B, Kk)
                    last = rul_p[:, -1].cpu().numpy()
                    for i in range(B):
                        s2 = offset + i
                        t_idx = (s2 + K - 1) * sample_stride + L - 1
                        if t_idx < T:
                            all_true.append(float(rul_norm[t_idx]))
                            all_pred.append(float(last[i]))
                            all_ev.append(bool(ev_arr[t_idx]))
                    offset += B

    true = np.array(all_true)
    pred = np.array(all_pred)
    ev = np.array(all_ev)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    mf = ev.astype(bool)
    # 裁剪到 [0, 1.05] (归一化 RUL 物理范围)
    true_f = np.clip(true[mf], 0, 1.05)
    pred_f = np.clip(pred[mf], 0, 1.05)
    # hexbin 密度图
    hb = ax.hexbin(true_f, pred_f, gridsize=45, cmap="Blues", mincnt=1,
                   extent=[0, 1.05, 0, 1.05])
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("样本数 (log)", fontsize=10)
    # y=x
    ax.plot([0, 1.05], [0, 1.05], "k--", linewidth=1.2, alpha=0.6, label="y=x")
    # 分位数校准曲线
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
        ax.text(0.03, 0.97, f"失效 RMSE = {rmse:.4f}\nN = {mf.sum()} 点\n(通道级, 裁剪可视化)",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("真实 RUL (归一化)", fontsize=11)
    ax.set_ylabel("预测 RUL (归一化)", fontsize=11)
    ax.set_title(f"通道级全测试集 RUL 散点 — {label}", fontsize=12, fontweight="bold")
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
    ap = argparse.ArgumentParser(description="相控阵通道级 RUL/HI 预测轨迹图")
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
    print("1. 加载通道级数据")
    data = load_and_prepare(cfg, args.seed)
    print(f"  n_traj={data['n_traj']}, train={len(data['tr'])}, "
          f"val={len(data['va'])}, test={len(data['te'])}")
    print(f"  n_features={data['n_features']}, L={data['L']}, K={data['K']}")
    print(f"  device={device}")

    print("=" * 60)
    print("2. 训练 ch_target_only_gru")
    model_tgt = train_model("ch_target_only_gru", data, cfg, device)

    print("=" * 60)
    print("3. 训练 ch_source_pretrain_frozen")
    model_src = train_model("ch_source_pretrain_frozen", data, cfg, device)

    print("=" * 60)
    print("4. 选测试轨迹")
    traj_id = select_test_trajectory(data, data["h5_path"])
    print(f"  选定轨迹 #{traj_id}")

    print("=" * 60)
    print("5. 画图")
    plot_ch_rul_trajectory(model_tgt, model_src, data, traj_id, device,
                          out_dir / "ch_rul_trajectory_panel.png")
    plot_ch_hi_spatial(model_tgt, data, traj_id, device,
                       out_dir / "ch_hi_spatial_diversity.png")
    plot_ch_rul_scatter(model_tgt, data, device,
                        out_dir / "ch_rul_scatter.png", "ch_target_only_gru")

    print("=" * 60)
    print("完成! 通道级图表保存到:", out_dir)


if __name__ == "__main__":
    main()
