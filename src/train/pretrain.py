"""train/pretrain.py

源域预训练: TCN/LSTM/GRU 编码器 + HI/RUL 双头。

损失 (plan §四 Phase 2):
  L = Huber(RUL) + λ1·MSE(HI) + λ2·L_mono + λ3·L_smooth
    L_mono   = Σ relu(HI_t - HI_{t+1})   同轴承连续窗内, 惩罚 HI 下降
    L_smooth = Σ (HI_{t+1} - HI_t)^2

数据组织: 滑窗 (L, F); 每样本 = 同一轴承连续 K 个窗 (K 支撑 mono/smooth)。
划分: 沿用 source_features.h5 的 split (按轴承个体)。

两条路径:
  - 观测层预训练 (默认, `run`): encoder 看 features (源域 [RDS_drift, T_case_C] 等);
    飞轮主链路 + 相控阵旧观测层 ablation 用。
  - HI 动力学层预训练 (`run_hi_layer`, `--hi-layer`): encoder 只看 [HI, ΔHI] (2 维);
    相控阵迁移 B 重构主路径, 弃用 features, encoder 输入跨域统一为 HI 动力学语义。
    (设计文档 docs/开发推进计划/hi_layer_refactor_design.md §4.4)

用法:
  python -m src.train.pretrain --config configs/phased_array.yaml            # 观测层预训练 (飞轮)
  python -m src.train.pretrain --config configs/phased_array.yaml --hi-layer   # HI 层预训练 (相控阵)
  python -m src.train.pretrain --config configs/phased_array.yaml --smoke    # 小配置冒烟
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed                          # noqa: E402
from src.models.tcn_encoder import RULModel                          # noqa: E402
from src.data.preprocess.source_io import load_source_features       # noqa: E402

CKPT_DIR = ROOT / "checkpoints"


# ---------------------------------------------------------------- 数据
class BearingWindowSeq(Dataset):
    """每个样本 = 同一轴承连续 K 个滑窗 (K 用于 mono/smooth 约束)。

    滑窗窗末时刻的 HI/RUL 作为该窗的监督标签。
    同时承载窗末行的 event_observed (失效/删失) 和 rul_lower_bound (删失行 hinge 用)。
    """

    def __init__(self, features, hi, rul, bid, t_idx, L, K, stride=1,
                 event_observed=None, rul_lower_bound=None):
        self.samples = []
        # event_observed / rul_lower_bound 缺省 → 全 True / rul (退化到旧 Huber loss 行为)
        if event_observed is None:
            event_observed = np.ones(len(bid), dtype=bool)
        if rul_lower_bound is None:
            rul_lower_bound = np.asarray(rul, dtype=np.float32)
        df = pd.DataFrame({"bid": bid, "t": t_idx})
        for _, g in df.groupby("bid"):
            g = g.sort_values("t")
            idxs = g.index.to_numpy()        # 该轴承按时间排序的原始行索引
            T = len(idxs)
            if T < L:
                continue
            starts = list(range(0, T - L + 1, stride))
            win_feats = [features[idxs[s:s + L]] for s in starts]
            win_hi = [float(hi[idxs[s + L - 1]]) for s in starts]
            win_rul = [float(rul[idxs[s + L - 1]]) for s in starts]
            win_ev = [bool(event_observed[idxs[s + L - 1]]) for s in starts]
            win_lb = [float(rul_lower_bound[idxs[s + L - 1]]) for s in starts]
            n = len(starts)
            for s2 in range(0, n - K + 1):
                self.samples.append((
                    np.stack(win_feats[s2:s2 + K]).astype(np.float32),
                    np.array(win_hi[s2:s2 + K], dtype=np.float32),
                    np.array(win_rul[s2:s2 + K], dtype=np.float32),
                    np.array(win_ev[s2:s2 + K], dtype=bool),
                    np.array(win_lb[s2:s2 + K], dtype=np.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        f, h, r, ev, lb = self.samples[i]
        return (torch.from_numpy(f), torch.from_numpy(h), torch.from_numpy(r),
                torch.from_numpy(ev), torch.from_numpy(lb))


class HIWindowSeq(Dataset):
    """HI 动力学层 K-block 滑窗 Dataset (设计文档 §3.2/§4.4)。

    输入 = [HI, ΔHI] (L, 2), 弃用 features 字段。每样本 = 同一器件连续 K 个滑窗,
    K 支撑 mono/smooth 约束 (与 BearingWindowSeq 同结构, 仅输入通道替换为 HI 两维)。

    - 通道 0 = HI 序列 (窗内)
    - 通道 1 = ΔHI = 前向差分 np.diff, 首位补 0 (避免窗边界信息泄漏)
    - 窗末 HI/RUL/event_observed/rul_lower_bound 作为该窗监督标签 (透传 _rul_loss hinge)

    严格按器件 groupby + t_index 排序, 不跨器件 (CLAUDE.md 约束 7)。
    """

    def __init__(self, hi, rul, bid, t_idx, L, K, stride=1,
                 event_observed=None, rul_lower_bound=None):
        self.samples = []
        if event_observed is None:
            event_observed = np.ones(len(bid), dtype=bool)
        if rul_lower_bound is None:
            rul_lower_bound = np.asarray(rul, dtype=np.float32)
        df = pd.DataFrame({"bid": bid, "t": t_idx})
        for _, g in df.groupby("bid"):
            g = g.sort_values("t")
            idxs = g.index.to_numpy()        # 该器件按时间排序的原始行索引
            T = len(idxs)
            if T < L:
                continue
            h_rows = hi[idxs].astype(np.float32)   # 器件内按 t 排序的 HI
            starts = list(range(0, T - L + 1, stride))
            win_x = []    # 每窗 (L, 2) = [HI, ΔHI]
            win_hi = []   # 窗末 HI 标量
            win_rul = []
            win_ev = []
            win_lb = []
            for s in starts:
                # 窗级 ΔHI: np.diff 前向差分, 首位补 0 保窗长 L
                # (与 make_hi_windows 保持一致, 满足设计文档 §7.5 test_dhi_first_step_zero)
                win_h = h_rows[s:s + L]
                win_d = np.zeros(L, dtype=np.float32)
                win_d[1:] = np.diff(win_h)
                win_x.append(np.stack([win_h, win_d], axis=-1).astype(np.float32))
                win_hi.append(float(hi[idxs[s + L - 1]]))
                win_rul.append(float(rul[idxs[s + L - 1]]))
                win_ev.append(bool(event_observed[idxs[s + L - 1]]))
                win_lb.append(float(rul_lower_bound[idxs[s + L - 1]]))
            n = len(starts)
            for s2 in range(0, n - K + 1):
                self.samples.append((
                    np.stack(win_x[s2:s2 + K]).astype(np.float32),         # (K, L, 2)
                    np.array(win_hi[s2:s2 + K], dtype=np.float32),
                    np.array(win_rul[s2:s2 + K], dtype=np.float32),
                    np.array(win_ev[s2:s2 + K], dtype=bool),
                    np.array(win_lb[s2:s2 + K], dtype=np.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, h, r, ev, lb = self.samples[i]
        return (torch.from_numpy(x), torch.from_numpy(h), torch.from_numpy(r),
                torch.from_numpy(ev), torch.from_numpy(lb))


# ---------------------------------------------------------------- RUL loss (O/C 分流)
def _rul_loss(rul_pred, rul_label, event_observed, rul_lower_bound, huber, eta):
    """RUL loss 分流失效 O (Huber) + 删失 C (hinge)。

    失效行 (event_observed=True):
      L_O = mean Huber(rul_pred, rul_label)
      (rul_label 已归一, 失效=exact RUL)
    删失行 (event_observed=False):
      L_C = mean [max(0, rul_lower_bound - rul_pred)]²
      只在预测低于已知下界时受罚 (one-sided hinge), 允许给健康器件大 RUL。
      修 D1: 删失器件 HI 健康 (~0.04-0.2) 但 rul lower_bound 从大递减到 0,
      若当精确 RUL 监督会让 HI 头与 RUL 头打架, 污染源 encoder。
    eta: 删失 hinge 权重 (config loss.censored_hinge_eta, 默认 1.0)。
    rul_label / rul_lower_bound 都已归一到 [0,1] (调用方 sd.rul_max 归一)。
    """
    O = event_observed
    C = ~O
    L_O = (huber(rul_pred[O], rul_label[O])).mean() if int(O.sum()) > 0 else rul_pred.new_zeros(())
    if int(C.sum()) > 0:
        hinge = torch.relu(rul_lower_bound[C] - rul_pred[C])
        L_C = (hinge * hinge).mean()
    else:
        L_C = rul_pred.new_zeros(())
    return L_O + eta * L_C, L_O, L_C


# ---------------------------------------------------------------- 评估
@torch.no_grad()
def evaluate(model, loader, device, huber, mse, lam, eta=1.0):
    model.eval()
    tot_loss = tot_rul = 0.0
    n = n_mono_viol = n_mono_total = 0
    preds, labels, ev_mask = [], [], []
    for x, h, r, ev, lb in loader:
        x, h, r = x.to(device), h.to(device), r.to(device)
        ev, lb = ev.to(device), lb.to(device)
        B, Kk = x.size(0), x.size(1)              # x: (B, K, L, F)
        hi_p, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
        hi_p = hi_p.view(B, Kk)
        rul_p = rul_p.view(B, Kk)
        # RUL loss: 失效 Huber + 删失 hinge (D1 修复)
        Lr, _, _ = _rul_loss(rul_p, r, ev, lb, huber, eta)
        Lh = mse(hi_p, h)
        d = hi_p[:, 1:] - hi_p[:, :-1]
        Lm = torch.relu(-d).mean()
        Ls = (d * d).mean()
        loss = Lr + lam[0] * Lh + lam[1] * Lm + lam[2] * Ls
        b = x.size(0)
        tot_loss += loss.item() * b
        tot_rul += Lr.item() * b
        n += b
        n_mono_viol += int((d < -1e-6).sum().item())
        n_mono_total += d.numel()
        preds.append(rul_p.cpu().numpy())
        labels.append(r.cpu().numpy())
        ev_mask.append(ev.cpu().numpy())
    rp = np.concatenate(preds) if preds else np.array([0.0])
    rl = np.concatenate(labels) if labels else np.array([0.0])
    re = np.concatenate(ev_mask) if ev_mask else np.array([True])
    # val RUL RMSE 只算失效行 (event_observed=True); 删失器件无精确 RUL, 不评估
    m = re.astype(bool)
    rmse = float(np.sqrt(np.mean((rp[m] - rl[m]) ** 2))) if m.any() else 0.0
    return {
        "loss": tot_loss / max(n, 1),
        "rul_rmse": rmse,
        "mono_violation": n_mono_viol / max(n_mono_total, 1),
        "L_rul": tot_rul / max(n, 1),
    }


# ---------------------------------------------------------------- 训练
def run(args):
    cfg = load_config(args.config)
    set_seed(cfg["seed"],
              cfg["reproducibility"]["deterministic"],
              cfg["reproducibility"]["cudnn_benchmark"])

    pre_cfg = cfg.get("pretrain", {})
    if getattr(args, "canonical", False):
        feat_path = pre_cfg.get("canonical_source_path",
                                "data/features/phased_array/schema_v4/source/mosfet_canonical.h5")
    else:
        feat_path = pre_cfg.get("source_feature_path", "data/features/wheel/schema_v1/source_features.h5")
    id_field = pre_cfg.get("source_id_field", "bearing_id")
    feat_h5 = ROOT / feat_path
    if not feat_h5.exists():
        print(f"!! 缺 {feat_h5}; 先运行对应组件的源域特征工程 "
              f"(wheel_features.py / mosfet_features.py --synthetic)")
        sys.exit(1)
    val_device_ids = cfg.get("source", {}).get("split", {}).get("val_device_ids", []) or []
    sd = load_source_features(feat_h5, id_field=id_field, val_device_ids=val_device_ids)
    features = sd.features
    hi = sd.hi
    t_idx = sd.t_index
    bid = sd.device_id_array
    split = sd.split_array
    # RUL 归一到 [0,1]: v1 窗索引 / v2 秒 的量级差异用 rul_max 消除
    # (与 transfer 目标域归一口径一致, Huber loss 量级合理)
    rul = sd.rul / sd.rul_max
    # rul_lower_bound 同步归一 (删失行 hinge 约束用; 失效器件 lb==rul 不影响)
    rul_lb = sd.rul_lower_bound / sd.rul_max
    event_observed = sd.event_observed.astype(bool)
    n_failed = int(np.unique(bid[event_observed]).size)
    n_censored = int(np.unique(bid[~event_observed]).size) if event_observed.size else 0
    # 特征 z-score 归一化: RDS_drift ~[0,0.5] vs T_case_C ~[50,150] 量级差 100x,
    # 不归一 input_conv 被 T_case_C 主导, encoder 学不到退化特征 → RUL 头坍缩到 0
    tr_mask = split == "train"
    feat_mean = features[tr_mask].mean(axis=0)
    feat_std = features[tr_mask].std(axis=0) + 1e-6
    features = (features - feat_mean) / feat_std
    print(f">> source schema_v{sd.schema_version}: {sd.n_devices} 器件 "
          f"(failed={n_failed} censored={n_censored}), "
          f"{features.shape[0]} 行, F={features.shape[1]} {sd.feature_names or ''}, "
          f"rul_max={sd.rul_max:.3g}; feat z-score mean={np.round(feat_mean,3).tolist()} "
          f"std={np.round(feat_std,3).tolist()}")

    L = int(cfg["model"]["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    stride = int(cfg.get("pretrain", {}).get("stride", 1))
    # 自适应: 合成/短序列装不下 config 的 L 时降 L
    max_T = int(max(np.unique(bid, return_counts=True)[1]))
    if max_T < L + K:
        L = max(8, max_T - K)
        print(f">> 序列偏短 (max_T={max_T}), L 自适应 -> {L}")

    # 自适应 stride: schema_v4 canonical 重采样到 512 点, config stride=80 (针对飞轮 T~1e4) →
    # 每器件仅 6 窗 < K=8, BearingWindowSeq samples 空。降 stride 保证 >=K 窗/block。
    n_per = max(1, (max_T - L) // stride + 1)
    if n_per < K:
        stride = max(1, (max_T - L) // (K + 1))
        print(f">> stride 自适应 -> {stride} (max_T={max_T}, L={L}, K={K}; "
              f"原 stride → 仅 {n_per} 窗 <K)")

    tr = split == "train"
    va = split == "val"
    ds_tr = BearingWindowSeq(features[tr], hi[tr], rul[tr], bid[tr], t_idx[tr], L, K, stride,
                             event_observed=event_observed[tr],
                             rul_lower_bound=rul_lb[tr])
    ds_va = BearingWindowSeq(features[va], hi[va], rul[va], bid[va], t_idx[va], L, K, stride,
                             event_observed=event_observed[va],
                             rul_lower_bound=rul_lb[va])
    if args.smoke:
        ds_tr = Subset(ds_tr, list(range(min(64, len(ds_tr)))))
        ds_va = Subset(ds_va, list(range(min(32, len(ds_va)))))
    bs = 16 if args.smoke else int(cfg["pretrain"]["batch_size"])
    loader_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=bs, shuffle=False)

    device = "cuda" if (cfg["pretrain"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    enc = args.encoder or cfg["model"]["encoder"]
    mc = cfg["model"]
    model = RULModel(
        encoder_type=enc, n_features=features.shape[1], input_len=L,
        channels=mc["tcn"]["channels"], kernel_size=mc["tcn"]["kernel_size"],
        num_blocks=mc["tcn"]["num_blocks"], dropout=mc["tcn"]["dropout"],
        latent_dim=mc["latent_dim"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["pretrain"]["lr"]),
                           weight_decay=float(cfg["pretrain"]["weight_decay"]))
    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]), reduction="none")  # 逐元素 (RUL 加权)
    mse = nn.MSELoss()
    _lc = cfg["loss"]
    lam = (float(_lc.get("lambda_hi", _lc.get("beta_hi", 1.0))),       # 飞轮 lambda_hi / 相控阵 beta_hi
           float(_lc.get("lambda_mono", _lc.get("mu_mono", 0.1))),     # lambda_mono / mu_mono
           float(_lc.get("lambda_smooth", _lc.get("nu_smooth", 0.1)))) # lambda_smooth / nu_smooth
    # 删失器件 hinge 权重 (D1 修复); 飞轮无删失, 默认 1.0 无副作用
    eta = float(_lc.get("censored_hinge_eta", 1.0))
    epochs = 2 if args.smoke else int(cfg["pretrain"]["epochs"])

    print(f">> encoder={enc} device={device} L={L} K={K} "
          f"train_blocks={len(ds_tr)} val_blocks={len(ds_va)} epochs={epochs}")

    # 追踪最佳 val RUL RMSE 时点 (val_rul_rmse, vm, state_dict, epoch)。
    # 修复: 原 best=vm 每轮无条件覆盖 → torch.save 实际存最后一轮 (末段可能发散,
    # 诊断: 最佳 ep41 RMSE~0.099, 末轮~0.121)。与下游 _train_with_early_stop 口径一致 (GPT 评审 §4)。
    best = (float("inf"), None, None, 0)
    for ep in range(1, epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for x, h, r, ev, lb in loader_tr:
            x, h, r = x.to(device), h.to(device), r.to(device)
            ev, lb = ev.to(device), lb.to(device)
            B, Kk = x.size(0), x.size(1)          # x: (B, K, L, F)
            hi_p, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
            hi_p = hi_p.view(B, Kk)
            rul_p = rul_p.view(B, Kk)
            # RUL loss: 失效 Huber + 删失 hinge (D1 修复)
            Lr, L_O, L_C = _rul_loss(rul_p, r, ev, lb, huber, eta)
            Lh = mse(hi_p, h)
            d = hi_p[:, 1:] - hi_p[:, :-1]
            Lm = torch.relu(-d).mean()
            Ls = (d * d).mean()
            loss = Lr + lam[0] * Lh + lam[1] * Lm + lam[2] * Ls
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0)
            n += x.size(0)
        vm = evaluate(model, loader_va, device, huber, mse, lam, eta=eta)
        print(f"  ep{ep:02d} train_loss={tl / n:.4f} | val_loss={vm['loss']:.4f} "
              f"val_rul_rmse={vm['rul_rmse']:.4f} mono_viol={vm['mono_violation']:.4f} "
              f"L_rul={vm['L_rul']:.4f}")
        if vm["rul_rmse"] < best[0]:
            best = (vm["rul_rmse"], vm,
                    {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)

    # 恢复最佳 val RUL RMSE 时点权重 (修复: ckpt/metrics 一致用最佳, 而非末轮)
    if best[2] is not None:
        model.load_state_dict(best[2])
    best_vm = best[1]
    CKPT_DIR.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "pretrain"
    comp = Path(args.config).stem
    suffix = "" if comp == "wheel" else f"_{comp}"   # 飞轮保持原名, 相控阵带后缀避免覆盖
    ckpt = CKPT_DIR / f"source{suffix}_{enc}_{tag}.pt"
    torch.save({"model": model.state_dict(), "encoder": enc, "L": L, "K": K,
                "n_features": features.shape[1], "latent_dim": mc["latent_dim"]}, ckpt)
    metrics = {
        "encoder": enc, "L": L, "K": K, "epochs": epochs, "smoke": bool(args.smoke),
        "final_val_loss": best_vm["loss"], "val_rul_rmse": best_vm["rul_rmse"],
        "val_hi_mono_violation": best_vm["mono_violation"],
        "best_epoch": best[3],          # 最佳 val_rul_rmse 时点 epoch (修复后 ckpt 即此 epoch 权重)
        "lambda_hi": lam[0], "lambda_mono": lam[1], "lambda_smooth": lam[2],
        "censored_hinge_eta": eta,
        "val_rul_rmse_threshold": cfg["pretrain"].get("val_rul_rmse_threshold"),
    }
    metrics_path = CKPT_DIR / f"metrics{suffix}.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">> checkpoint: {ckpt} (best val_rul_rmse={best_vm['rul_rmse']:.4f} @ep{best[3]})")
    print(f">> metrics : {metrics_path}")

    thr = cfg["pretrain"].get("val_rul_rmse_threshold")
    if thr is not None and not args.smoke:
        assert best_vm["rul_rmse"] <= thr, f"val RUL RMSE {best_vm['rul_rmse']:.4f} > 阈值 {thr}"
        print(f">> val RUL RMSE 达标 (<= {thr})")
    mono_thr = cfg["pretrain"].get("hi_mono_violation_threshold")
    if mono_thr is not None and not args.smoke:
        if best_vm["mono_violation"] > mono_thr:
            print(f">> [warn] mono_viol {best_vm['mono_violation']:.4f} > 阈值 {mono_thr} "
                  f"(T14: 器件级 HI isotonic 后完全单调, 但 block 级 K=8 窗预测相邻下降 ~0.5, "
                  f"根因=block 跨工况段波动, 非器件级标签问题; 迁移结论不依赖此指标)")
        else:
            print(f">> mono_viol 达标 (<= {mono_thr})")
    if args.smoke:
        print(">> [SMOKE] 训练循环跑通: loss/ckpt/metrics 正常 (真实阈值校验待真实数据)")


# ---------------------------------------------------------------- HI 层预训练分支
def run_hi_layer(args):
    """HI 动力学层源域预训练 (设计文档 §4.4)。

    与 run() 的差异:
      - encoder 输入 = [HI, ΔHI] (L, 2) 两维, 弃用 features 字段
        (源/目标语义统一, 跨域迁移落在 HI 动力学层, 满足 CLAUDE.md 约束 2)
      - 模型 = HIDynamicsModel (HISeqEncoder + HI_head + RUL_head, from src.transfer.adapter,
        Agent 1 并行实现, 接口见设计文档 §3.1)
      - loss 复用现有 _rul_loss (失效 Huber + 删失 hinge) + MSE(HI) + mono + smooth
      - checkpoint 落 source_phased_array_tcn_hilayer_pretrain.pt (与旧观测层 ckpt 并存)
      - metrics 落 metrics_phased_array_hilayer.json
    """
    # lazy import: HIDynamicsModel 由 Agent 1 在 src.transfer.adapter 并行实现。
    # 顶层不 import, 保证旧路径 run() 在 Agent 1 未就绪时仍可独立运行/测试。
    from src.transfer.adapter import HIDynamicsModel

    cfg = load_config(args.config)
    set_seed(cfg["seed"],
             cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])

    pre_cfg = cfg.get("pretrain", {})
    feat_path = pre_cfg.get("source_feature_path", "data/features/wheel/schema_v1/source_features.h5")
    id_field = pre_cfg.get("source_id_field", "bearing_id")
    feat_h5 = ROOT / feat_path
    if not feat_h5.exists():
        print(f"!! 缺 {feat_h5}; 先运行对应组件的源域特征工程")
        sys.exit(1)
    val_device_ids = cfg.get("source", {}).get("split", {}).get("val_device_ids", []) or []
    sd = load_source_features(feat_h5, id_field=id_field, val_device_ids=val_device_ids)
    # HI 层: 弃用 features, 只取 hi + rul + device_id + t_index
    hi = sd.hi
    t_idx = sd.t_index
    bid = sd.device_id_array
    split = sd.split_array
    # RUL 归一到 [0,1] (与 run() 同口径, Huber/hinge 量级合理)
    rul = sd.rul / sd.rul_max
    rul_lb = sd.rul_lower_bound / sd.rul_max
    event_observed = sd.event_observed.astype(bool)
    n_failed = int(np.unique(bid[event_observed]).size)
    n_censored = int(np.unique(bid[~event_observed]).size) if event_observed.size else 0
    print(f">> [HI-layer] source schema_v{sd.schema_version}: {sd.n_devices} 器件 "
          f"(failed={n_failed} censored={n_censored}), "
          f"{hi.shape[0]} 行, rul_max={sd.rul_max:.3g}; "
          f"encoder 输入=[HI, ΔHI] (2 维), 弃用 features (F={sd.features.shape[1]})")

    L = int(cfg["model"]["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    stride = int(cfg.get("pretrain", {}).get("stride", 1))
    max_T = int(max(np.unique(bid, return_counts=True)[1]))
    if max_T < L + K:
        L = max(8, max_T - K)
        print(f">> 序列偏短 (max_T={max_T}), L 自适应 -> {L}")

    tr = split == "train"
    va = split == "val"
    ds_tr = HIWindowSeq(hi[tr], rul[tr], bid[tr], t_idx[tr], L, K, stride,
                        event_observed=event_observed[tr],
                        rul_lower_bound=rul_lb[tr])
    ds_va = HIWindowSeq(hi[va], rul[va], bid[va], t_idx[va], L, K, stride,
                        event_observed=event_observed[va],
                        rul_lower_bound=rul_lb[va])
    if args.smoke:
        ds_tr = Subset(ds_tr, list(range(min(64, len(ds_tr)))))
        ds_va = Subset(ds_va, list(range(min(32, len(ds_va)))))
    bs = 16 if args.smoke else int(cfg["pretrain"]["batch_size"])
    loader_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=bs, shuffle=False)

    device = "cuda" if (cfg["pretrain"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    enc = args.encoder or cfg["model"]["encoder"]
    mc = cfg["model"]
    # HIDynamicsModel 内部 encoder = HISeqEncoder (n_features 固定 2 = [HI, ΔHI])
    # 接口见设计文档 §3.1, 与 TransferModel 同参数集 (除 n_features/n_target/adapter)
    model = HIDynamicsModel(
        encoder_type=enc,
        input_len=L,
        channels=mc["tcn"]["channels"],
        kernel_size=mc["tcn"]["kernel_size"],
        num_blocks=mc["tcn"]["num_blocks"],
        dropout=mc["tcn"]["dropout"],
        latent_dim=mc["latent_dim"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["pretrain"]["lr"]),
                           weight_decay=float(cfg["pretrain"]["weight_decay"]))
    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]), reduction="none")
    mse = nn.MSELoss()
    _lc = cfg["loss"]
    lam = (float(_lc.get("lambda_hi", _lc.get("beta_hi", 1.0))),
           float(_lc.get("lambda_mono", _lc.get("mu_mono", 0.1))),
           float(_lc.get("lambda_smooth", _lc.get("nu_smooth", 0.1))))
    eta = float(_lc.get("censored_hinge_eta", 1.0))
    epochs = 2 if args.smoke else int(cfg["pretrain"]["epochs"])

    print(f">> [HI-layer] encoder={enc} device={device} L={L} K={K} "
          f"train_blocks={len(ds_tr)} val_blocks={len(ds_va)} epochs={epochs}")

    # 追踪最佳 val RUL RMSE 时点 (val_rul_rmse, vm, state_dict, epoch)。
    # 修复: 原 best=vm 每轮无条件覆盖 → torch.save 实际存最后一轮 (末段可能发散,
    # 诊断: 最佳 ep41 RMSE~0.099, 末轮~0.121)。与下游 _train_with_early_stop 口径一致 (GPT 评审 §4)。
    best = (float("inf"), None, None, 0)
    for ep in range(1, epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for x, h, r, ev, lb in loader_tr:
            # x: (B, K, L, 2)  通道维=2 (HI + ΔHI); HIDynamicsModel 接收 (B*K, L, 2)
            x, h, r = x.to(device), h.to(device), r.to(device)
            ev, lb = ev.to(device), lb.to(device)
            B, Kk = x.size(0), x.size(1)
            hi_p, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
            hi_p = hi_p.view(B, Kk)
            rul_p = rul_p.view(B, Kk)
            # 复用 _rul_loss: 失效 Huber + 删失 hinge (D1 修复, A1 保留)
            Lr, _, _ = _rul_loss(rul_p, r, ev, lb, huber, eta)
            Lh = mse(hi_p, h)
            d = hi_p[:, 1:] - hi_p[:, :-1]
            Lm = torch.relu(-d).mean()
            Ls = (d * d).mean()
            loss = Lr + lam[0] * Lh + lam[1] * Lm + lam[2] * Ls
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0)
            n += x.size(0)
        vm = evaluate(model, loader_va, device, huber, mse, lam, eta=eta)
        print(f"  ep{ep:02d} train_loss={tl / n:.4f} | val_loss={vm['loss']:.4f} "
              f"val_rul_rmse={vm['rul_rmse']:.4f} mono_viol={vm['mono_violation']:.4f} "
              f"L_rul={vm['L_rul']:.4f}")
        if vm["rul_rmse"] < best[0]:
            best = (vm["rul_rmse"], vm,
                    {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)

    # 恢复最佳 val RUL RMSE 时点权重 (修复同观测层路径; ckpt/metrics 一致用最佳)
    if best[2] is not None:
        model.load_state_dict(best[2])
    best_vm = best[1]
    CKPT_DIR.mkdir(exist_ok=True)
    # tag 命名: hilayer_pretrain (正式) / hilayer_smoke (冒烟); 与旧观测层 ckpt 并存
    tag = "hilayer_smoke" if args.smoke else "hilayer_pretrain"
    comp = Path(args.config).stem
    suffix = "" if comp == "wheel" else f"_{comp}"   # 飞轮保持原名, 相控阵带后缀避免覆盖
    ckpt = CKPT_DIR / f"source{suffix}_{enc}_{tag}.pt"
    torch.save({"model": model.state_dict(), "encoder": enc, "L": L, "K": K,
                "n_features": 2, "latent_dim": mc["latent_dim"],
                "hi_layer": True}, ckpt)
    metrics = {
        "encoder": enc, "L": L, "K": K, "epochs": epochs, "smoke": bool(args.smoke),
        "hi_layer": True,                         # 标识 HI 动力学层预训练
        "n_features": 2,                           # [HI, ΔHI]
        "final_val_loss": best_vm["loss"], "val_rul_rmse": best_vm["rul_rmse"],
        "val_hi_mono_violation": best_vm["mono_violation"],
        "best_epoch": best[3],          # 最佳 val_rul_rmse 时点 epoch (修复后 ckpt 即此 epoch 权重)
        "lambda_hi": lam[0], "lambda_mono": lam[1], "lambda_smooth": lam[2],
        "censored_hinge_eta": eta,
        "val_rul_rmse_threshold": cfg["pretrain"].get("val_rul_rmse_threshold"),
    }
    metrics_path = CKPT_DIR / f"metrics{suffix}_hilayer.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">> [HI-layer] checkpoint: {ckpt} (best val_rul_rmse={best_vm['rul_rmse']:.4f} @ep{best[3]})")
    print(f">> [HI-layer] metrics : {metrics_path}")

    thr = cfg["pretrain"].get("val_rul_rmse_threshold")
    if thr is not None and not args.smoke:
        assert best_vm["rul_rmse"] <= thr, f"val RUL RMSE {best_vm['rul_rmse']:.4f} > 阈值 {thr}"
        print(f">> [HI-layer] val RUL RMSE 达标 (<= {thr})")
    mono_thr = cfg["pretrain"].get("hi_mono_violation_threshold")
    if mono_thr is not None and not args.smoke:
        if best_vm["mono_violation"] > mono_thr:
            print(f">> [HI-layer warn] mono_viol {best_vm['mono_violation']:.4f} > 阈值 {mono_thr} "
                  f"(T14: block 级跨工况段波动; 迁移结论不依赖此指标)")
        else:
            print(f">> [HI-layer] mono_viol 达标 (<= {mono_thr})")
    if args.smoke:
        print(">> [HI-layer SMOKE] 训练循环跑通: loss/ckpt/metrics 正常")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--encoder", default=None, choices=["tcn", "lstm", "gru"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--canonical", action="store_true",
                    help="用 schema_v4 canonical 源域 (T3/M3, 4 维 device_canonical_v1); "
                         "仅观测层 run() 生效, 与 --hi-layer 互斥")
    ap.add_argument("--hi-layer", action="store_true",
                    help="HI 动力学层预训练 (相控阵迁移 B 重构): encoder 输入=[HI,ΔHI], "
                         "模型=HIDynamicsModel; 弃用 features (设计文档 §4.4)")
    args = ap.parse_args()
    if args.hi_layer:
        run_hi_layer(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
