"""transfer/train_transfer.py

双迁移路径:
  - run()             : 旧观测层迁移 (adapter + 共享 encoder, 飞轮 + ablation)
  - run_hi_layer(args): 新 HI 动力学层迁移 (路径 A, docs/开发推进计划/hi_layer_refactor_design.md §4.2)

旧路径三阶段 (plan §四 Phase 5):
  S1 复用源域预训练编码器 (冻结)
  S2 目标域 x_T 经 adapter g_φ → 冻结 E_θ → 训 adapter + 头 (防小样本冲掉源域知识)
  S3 MMD (按 HI 健康阶段分箱, 非按时间) + 小样本微调

新 HI 层路径三阶段 (设计文档 §4.2):
  S1 复用源域 HISeqEncoder 预训练权重 (encoder 输入 = [HI, ΔHI])
  S2 冻结 encoder, 训 target HI_head + RUL_head (输入 = 目标域 hi_array 派生的 x_HI 窗)
  S3 解冻 + MMD 按 HI bin 对齐 z_S / z_T (两者都由共享 HISeqEncoder 编码 x_HI 得到)

目标域按完整轨迹划分 train15 / val20 / test65 (禁止时间窗打散)。

用法:
  python -m src.transfer.train_transfer --config configs/phased_array.yaml --smoke
  python -m src.transfer.train_transfer --config configs/phased_array.yaml --hi-layer
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed                          # noqa: E402
from src.transfer.adapter import TransferModel                       # noqa: E402
from src.transfer.mmd import mmd_by_hi_bins, reset_global_memory_bank  # noqa: E402
from src.data.preprocess.source_io import load_source_features       # noqa: E402

# HI 层路径依赖 (Agent 1/2 接口, 集成时验证)
from src.transfer.adapter import HIDynamicsModel                     # noqa: E402  (Agent 1, §3.1)
from src.data.preprocess.source_io import make_hi_windows            # noqa: E402  (Agent 2, §3.2)

CKPT_DIR = ROOT / "checkpoints"


# ---------------------------------------------------------------- 数据
class TargetSeqDataset(Dataset):
    """目标域: 同一 traj 连续 K 窗块, 带 hi_b / rul / event_observed / rul_lower_bound。

    P0-2 (复核第三条): 承载 event_observed + rul_lower_bound (删失行 hinge 用),
    仿 pretrain BearingWindowSeq。删失轨迹 rul 不再当精确标签 (旧裸 huber 污染指标)。
    """

    def __init__(self, x_T, hi_b, rul, traj_ids, L, K, stride=1,
                 event_observed=None, rul_lower_bound=None, damage_b=None):
        self.samples = []
        if event_observed is None:
            event_observed = np.ones(len(traj_ids), dtype=bool)
        if rul_lower_bound is None:
            rul_lower_bound = np.asarray(rul, dtype=np.float32)
        df = pd.DataFrame({"tid": traj_ids})
        for _, g in df.groupby("tid"):
            idxs = g.index.to_numpy()
            T = len(idxs)
            if T < L:
                continue
            starts = list(range(0, T - L + 1, stride))
            wf = [x_T[idxs[s:s + L]] for s in starts]
            wh = [float(hi_b[idxs[s + L - 1]]) for s in starts]
            wr = [float(rul[idxs[s + L - 1]]) for s in starts]
            we = [bool(event_observed[idxs[s + L - 1]]) for s in starts]
            wl = [float(rul_lower_bound[idxs[s + L - 1]]) for s in starts]
            # T13 rho_phys: window 末 damage_norm (Arrhenius+Coffin-Manson 物理损伤), 供 ρ·L_phys 监督
            wd = ([float(damage_b[idxs[s + L - 1]]) for s in starts]
                  if damage_b is not None else [0.0] * len(starts))
            n = len(starts)
            for s2 in range(0, n - K + 1):
                self.samples.append((
                    np.stack(wf[s2:s2 + K]).astype(np.float32),
                    np.array(wh[s2:s2 + K], dtype=np.float32),
                    np.array(wr[s2:s2 + K], dtype=np.float32),
                    np.array(we[s2:s2 + K], dtype=np.bool_),
                    np.array(wl[s2:s2 + K], dtype=np.float32),
                    np.array(wd[s2:s2 + K], dtype=np.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        f, h, r, ev, lb, dmg = self.samples[i]
        return (torch.from_numpy(f), torch.from_numpy(h), torch.from_numpy(r),
                torch.from_numpy(ev), torch.from_numpy(lb), torch.from_numpy(dmg))


class SourceWindowDataset(Dataset):
    """源域: 单窗 (L, 12) 带 hi (供 S3 MMD 的 z_S)。"""

    def __init__(self, features, hi, bearing_ids, t_idx, L, stride=1):
        self.samples = []
        df = pd.DataFrame({"bid": bearing_ids})
        for _, g in df.groupby("bid"):
            idxs = g.index.to_numpy()
            T = len(idxs)
            if T < L:
                continue
            for s in range(0, T - L + 1, stride):
                self.samples.append((features[idxs[s:s + L]].astype(np.float32),
                                     float(hi[idxs[s + L - 1]])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        f, h = self.samples[i]
        return torch.from_numpy(f), torch.tensor(h, dtype=torch.float32)


def split_trajectories(n_traj, ratios, seed):
    """确定性轨迹级划分 (完整轨迹, 不打散窗口)。返回 (train, val, test) 轨迹 id。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_traj)
    n_tr = int(round(ratios[0] * n_traj))
    n_va = int(round(ratios[1] * n_traj))
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def load_target(h5_path, has_nodes=False, drop_features=None, return_damage=False):
    """目标域加载。has_nodes=True (相控阵): x_global + x_nodes mean-pool → concat;
    否则 (飞轮) 读 x_T。HI 列: hi_array(相控阵) / hi_b(飞轮)。

    P0-2 (复核第三条): 同时读 event_observed (h5 attrs, 逐行广播), 供训练端
    _rul_loss 失效/删失分流 (删失行当精确标签会污染指标)。飞轮 v1 无 attrs → 默认 True。

    GPT §6 消融 (drop_features): 按 build_array_hi.X_GLOBAL_COLS 列名从 x_global 删除
    服务寿命判据量 (M_link_dB/SLL_dB/theta_err_deg), 验证 target-only 是否真学退化
    (而非读取判据量离阈值的距离)。x_nodes 不变; 模型 n_target=xT.shape[1] 数据驱动, 自动适应。

    T13 rho_phys (return_damage): True 时额外读 damage_norm (Arrhenius+Coffin-Manson 物理损伤
    [0,1], latent 但 h5 已存, build_array_hi 设计供物理损失), 供 source_mmd_physics 的
    ρ·L_phys=MSE(hi_pred, damage) 物理一致性监督。返回 7 元组 (末位 damageT); False 返回 6 元组。
    """
    drop_features = drop_features or []
    keep_idx = None
    if has_nodes and drop_features:
        from src.sim.build_array_hi import X_GLOBAL_COLS
        keep_idx = [i for i, c in enumerate(X_GLOBAL_COLS) if c not in drop_features]
    xT_list, hi_list, rul_list, tid_list, ev_list, dmg_list = [], [], [], [], [], []
    hi_key = "hi_array" if has_nodes else "hi_b"
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys())
        for ti, k in enumerate(keys):
            g = f[k]
            if has_nodes:
                xg = g["x_global"][:].astype(np.float32)
                if keep_idx is not None:
                    xg = xg[:, keep_idx]
                xn = g["x_nodes"][:].mean(axis=1).astype(np.float32)  # mean-pool 16 子阵节点
                xT = np.concatenate([xg, xn], axis=1)
            else:
                xT = g["x_T"][:]
            ev = bool(g.attrs.get("event_observed", 1))   # 默认失效 (飞轮 v1 无 attrs)
            xT_list.append(xT)
            hi_list.append(g[hi_key][:])
            rul_list.append(g["rul"][:])
            ev_list.append(np.full(len(xT), ev, dtype=bool))
            tid_list.append(np.full(len(xT), ti))
            if return_damage:
                dmg_list.append(g["damage_norm"][:].astype(np.float32) if "damage_norm" in g
                                else np.zeros(len(xT), dtype=np.float32))
    if return_damage:
        return (np.concatenate(xT_list), np.concatenate(hi_list),
                np.concatenate(rul_list), np.concatenate(tid_list), len(keys),
                np.concatenate(ev_list), np.concatenate(dmg_list))
    return (np.concatenate(xT_list), np.concatenate(hi_list),
            np.concatenate(rul_list), np.concatenate(tid_list), len(keys),
            np.concatenate(ev_list))


def load_source(h5_path, id_field="bearing_id", val_device_ids=None):
    """源域加载 (v1 扁平 / v2 分组自动适配; 复用 source_io 统一 reader)。

    返回扁平 5 元组 (features, hi, device_ids, t_index, split_array), 供 SourceWindowDataset
    构造 MMD 对齐用的源域 z_S + HI 分箱。split_array (dtype=object) 元素 ∈ {'train','val'},
    调用方可按 'train' 过滤, 避免 val 器件参与 MMD 对齐 (任务 3: 防源域 val 泄漏到迁移对齐)。
    """
    sd = load_source_features(h5_path, id_field=id_field, val_device_ids=val_device_ids)
    # 特征 z-score 归一 (source train 集 mean/std, 与 pretrain 一致) — encoder 加载预训练权重
    # 期望归一输入, 否则 S3 MMD 的 z_S 失准 (诊断: 未归一导致 source 组变差/爆炸)
    split = np.array(sd.split)
    tr = split == "train"
    if tr.any():
        fm = sd.features[tr].mean(axis=0)
        fs = sd.features[tr].std(axis=0) + 1e-6
        feats = ((sd.features - fm) / fs).astype(np.float32)
    else:
        feats = sd.features
    return feats, sd.hi, sd.device_id_array, sd.t_index, sd.split_array


# ============================================================ HI 动力学层 (路径 A)
class HIWindowDataset(Dataset):
    """HI 序列窗 + (hi_end, rul_end) 标签 (设计文档 §3.2, HI 动力学层迁移)。

    源 / 目标域共用; 按 id (源=device_id / 目标=traj_id) groupby 后组内滑窗,
    严格不跨个体。features 字段不再使用, 只用 hi + rul + id。

    每窗输出:
      x_HI   : (L, 2) float32  channel 0 = HI, channel 1 = ΔHI (前向差分首位补 0)
      hi_end : float32         窗末 HI 值 (供 MMD 按 HI bin 分箱)
      rul_end: float32         窗末 RUL 值 (监督标签)
    """

    def __init__(self, hi: np.ndarray, rul: np.ndarray,
                 ids: np.ndarray, L: int, stride: int = 1):
        self.samples: list[tuple[np.ndarray, float, float]] = []
        df = pd.DataFrame({"id": np.asarray(ids, dtype=object)})
        for _, g in df.groupby("id"):
            idxs = g.index.to_numpy()
            T = len(idxs)
            if T < L:
                continue
            for s in range(0, T - L + 1, stride):
                seg = np.asarray(hi[idxs[s:s + L]], dtype=np.float32)
                # ΔHI = 前向差分首位补 0, 保证 x_HI[..., 1] 与 seg 同长且不引入边界信息
                dhi = np.concatenate(
                    [np.zeros(1, dtype=np.float32), np.diff(seg)]).astype(np.float32)
                x_HI = np.stack([seg, dhi], axis=-1)               # (L, 2)
                hi_end = float(seg[-1])
                rul_end = float(rul[idxs[s + L - 1]])
                self.samples.append((x_HI, hi_end, rul_end))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, h, r = self.samples[i]
        return (torch.from_numpy(x),
                torch.tensor(h, dtype=torch.float32),
                torch.tensor(r, dtype=torch.float32))


def _load_target_hi(h5_path):
    """读取目标域 hi / rul / traj_id (路径 A 默认不读 x_global/x_nodes)。

    字段兼容: 相控阵=hi_array, 飞轮=hi_b。
    返回 (hi, rul, traj_ids, n_traj) — 供 HIWindowDataset 构造 x_HI_T 窗。
    """
    hi_list, rul_list, tid_list = [], [], []
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys())
        for ti, k in enumerate(keys):
            g = f[k]
            hi_key = "hi_array" if "hi_array" in g else "hi_b"
            hi_seg = np.asarray(g[hi_key][:]).astype(np.float32)
            rul_seg = np.asarray(g["rul"][:]).astype(np.float32)
            hi_list.append(hi_seg)
            rul_list.append(rul_seg)
            tid_list.append(np.full(hi_seg.shape[0], ti, dtype=np.int64))
    return (np.concatenate(hi_list), np.concatenate(rul_list),
            np.concatenate(tid_list), len(keys))


# ---------------------------------------------------------------- 评估
@torch.no_grad()
def evaluate_T(model, loader, device, huber, mse, lam):
    model.eval()
    tl = n = 0
    nm_v = nm_t = 0
    preds, labels = [], []
    for x, h, r, ev, lb, _dmg in loader:
        x, h, r = x.to(device), h.to(device), r.to(device)
        B, Kk = x.size(0), x.size(1)
        hi_p, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
        hi_p = hi_p.view(B, Kk)
        rul_p = rul_p.view(B, Kk)
        Lr = huber(rul_p, r)
        Lh = mse(hi_p, h)
        d = hi_p[:, 1:] - hi_p[:, :-1]
        loss = Lr + lam[0] * Lh + lam[1] * torch.relu(-d).mean() + lam[2] * (d * d).mean()
        tl += loss.item() * B
        n += B
        nm_v += int((d < -1e-6).sum())
        nm_t += d.numel()
        preds.append(rul_p.cpu().numpy())
        labels.append(r.cpu().numpy())
    rp = np.concatenate(preds) if preds else np.array([0.0])
    rl = np.concatenate(labels) if labels else np.array([0.0])
    return {"loss": tl / max(n, 1),
            "rul_rmse": float(np.sqrt(np.mean((rp - rl) ** 2))),
            "mono_viol": nm_v / max(nm_t, 1)}


# ---------------------------------------------------------------- 训练
def run(args):
    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])
    reset_global_memory_bank()          # 任务 1: 每 run 清全局 MMD bank, 防跨 run 累积污染
    L = int(cfg["model"]["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    tcfg = cfg["transfer"]
    mc = cfg["model"]
    bins = [tuple(b) for b in tcfg["hi_bins"]]
    _lc = cfg["loss"]
    lam = (float(_lc.get("lambda_hi", _lc.get("beta_hi", 1.0))),
           float(_lc.get("lambda_mono", _lc.get("mu_mono", 0.1))),
           float(_lc.get("lambda_smooth", _lc.get("nu_smooth", 0.1))))
    mmd_lambda = float(tcfg["mmd_lambda"])

    has_nodes = bool(tcfg.get("target_has_nodes", False))
    target_h5 = ROOT / tcfg.get("target_feature_path", "data/features/wheel/schema_v1/target_features.h5")
    source_h5 = ROOT / cfg["pretrain"].get("source_feature_path", "data/features/wheel/schema_v1/source_features.h5")
    id_field = cfg["pretrain"].get("source_id_field", "bearing_id")
    if not target_h5.exists():
        print(f"!! 缺 {target_h5}; 先运行目标域特征工程 (build_hi.py / build_array_hi.py)"); sys.exit(1)
    if not source_h5.exists():
        print(f"!! 缺 {source_h5}; 先运行源域特征工程 (wheel_features.py / mosfet_features.py --synthetic)"); sys.exit(1)

    xT, hiT, rulT, tidT, n_traj, evT = load_target(target_h5, has_nodes)
    tr_ids, va_ids, te_ids = split_trajectories(
        n_traj, [tcfg["split"]["train"], tcfg["split"]["val"], tcfg["split"]["test"]], cfg["seed"])

    def mask(ids):
        return np.isin(tidT, ids)

    # P1-2: RUL 归一因子用 config 固定物理上限 (跨 seed 可比); 缺失回退 train-only max
    if "rul_max_norm" in tcfg:
        rul_max = float(tcfg["rul_max_norm"])
    else:
        rul_max_train = float(rulT[mask(tr_ids)].max())     # 回退 (跨 seed 不可比, 仅兼容旧 config)
        rul_max = rul_max_train
        print(f"  [warning] config 未设 transfer.rul_max_norm, 回退 train-only max={rul_max:.1f}")
    rulT = rulT / max(rul_max, 1.0)

    # 目标域 xT z-score 归一 (train 集) — encoder 预训练权重期望归一输入, adapter 处理域差异
    _tr_mask = np.isin(tidT, tr_ids)
    _fm = xT[_tr_mask].mean(axis=0)
    _fs = xT[_tr_mask].std(axis=0) + 1e-6
    xT = (xT - _fm) / _fs

    tstride = int(tcfg.get("target_stride", 50))     # 目标域序列长, 用大 stride 控窗数
    if args.smoke:
        tstride = max(tstride, 1000)
    ds_tr = TargetSeqDataset(xT[mask(tr_ids)], hiT[mask(tr_ids)], rulT[mask(tr_ids)],
                             tidT[mask(tr_ids)], L, K, stride=tstride)
    ds_va = TargetSeqDataset(xT[mask(va_ids)], hiT[mask(va_ids)], rulT[mask(va_ids)],
                             tidT[mask(va_ids)], L, K, stride=tstride)
    if args.smoke:
        ds_tr = Subset(ds_tr, list(range(min(48, len(ds_tr)))))
        ds_va = Subset(ds_va, list(range(min(24, len(ds_va)))))
    bs = 16 if args.smoke else int(cfg["pretrain"]["batch_size"])
    loader_T_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    loader_T_va = DataLoader(ds_va, batch_size=bs, shuffle=False)

    featsS, hiS, bidS, tidxS, splitS = load_source(
        source_h5, id_field,
        cfg.get("source", {}).get("split", {}).get("val_device_ids", []) or [])
    # 任务 3: 源 MMD 对齐窗只用 source train 器件 (val 器件不参与迁移对齐)
    _src_tr = np.asarray(splitS) == "train"
    if not _src_tr.any():
        raise ValueError("source train split empty; refusing val source into MMD")
    featsS = featsS[_src_tr]; hiS = hiS[_src_tr]
    bidS = np.asarray(bidS)[_src_tr]; tidxS = tidxS[_src_tr]
    ds_S = SourceWindowDataset(featsS, hiS, bidS, tidxS, L,
                               stride=max(1, len(featsS) // 2000))
    if args.smoke:
        ds_S = Subset(ds_S, list(range(min(64, len(ds_S)))))
    loader_S = DataLoader(ds_S, batch_size=bs, shuffle=True)

    device = "cuda" if (cfg["pretrain"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    # F1-A: 复用唯一构造函数 (与 run_groups/run_service_eval/导出/推理同架构, 防漂移)
    from src.models.factory import build_transfer_model
    model = build_transfer_model(
        cfg, n_features=featsS.shape[1], n_target=xT.shape[1],
        encoder_type=cfg["model"]["encoder"], device=device)

    # ---- S1: 加载源域预训练编码器 ----
    comp = Path(args.config).stem
    suffix = "" if comp == "wheel" else f"_{comp}"      # 飞轮保持原名, 相控阵带后缀
    ckpt = args.ckpt or str(CKPT_DIR / f"source{suffix}_{cfg['model']['encoder']}_pretrain.pt")
    if not Path(ckpt).exists():
        ckpt = str(CKPT_DIR / f"source{suffix}_{cfg['model']['encoder']}_smoke.pt")
    if Path(ckpt).exists():
        miss, unexp = model.load_pretrained(ckpt, device)
        print(f">> S1 加载源域编码器: {ckpt}  (missing={len(miss)} keys, adapter 随机初始化)")
    else:
        print(">> [警告] 无源域 checkpoint, 编码器随机初始化 (仅供 smoke)")

    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]))
    mse = nn.MSELoss()
    e2 = 2 if args.smoke else int(tcfg.get("epochs_s2", 20))
    e3 = 2 if args.smoke else int(tcfg.get("epochs_s3", 20))

    # ---- S2: 训 adapter + 头 (encoder 冻结) ----
    model.freeze_encoder(True)
    opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=float(tcfg["finetune_lr"]))
    print(f">> S2 adapter+头训练 (encoder 冻结, {e2} epochs)")
    for ep in range(1, e2 + 1):
        model.train()
        tl, n = 0.0, 0
        for x, h, r, ev, lb, _dmg in loader_T_tr:
            x, h, r = x.to(device), h.to(device), r.to(device)
            B, Kk = x.size(0), x.size(1)
            hi_p, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
            hi_p = hi_p.view(B, Kk)
            rul_p = rul_p.view(B, Kk)
            Lr = huber(rul_p, r)
            Lh = mse(hi_p, h)
            d = hi_p[:, 1:] - hi_p[:, :-1]
            loss = Lr + lam[0] * Lh + lam[1] * torch.relu(-d).mean() + lam[2] * (d * d).mean()
            opt2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            tl += loss.item() * B
            n += B
        vm = evaluate_T(model, loader_T_va, device, huber, mse, lam)
        print(f"  S2 ep{ep:02d} loss={tl / n:.4f} val_rul_rmse={vm['rul_rmse']:.4f} "
              f"mono_viol={vm['mono_viol']:.4f}")

    # ---- S3: MMD (按 HI 分箱) + 微调 ----
    model.freeze_encoder(False)
    opt3 = torch.optim.Adam(model.parameters(), lr=float(tcfg["finetune_lr"]))
    print(f">> S3 MMD(按HI分箱)+微调 ({e3} epochs)")
    for ep in range(1, e3 + 1):
        model.train()
        tl, n = 0.0, 0
        src_iter = cycle(loader_S)
        for x, h, r, ev, lb, _dmg in loader_T_tr:
            xs, hs = next(src_iter)
            x, h, r = x.to(device), h.to(device), r.to(device)
            xs, hs = xs.to(device), hs.to(device)
            B, Kk = x.size(0), x.size(1)
            hi_p, rul_p, zT = model(x.reshape(B * Kk, x.size(2), x.size(3)))
            hi_p = hi_p.view(B, Kk)
            rul_p = rul_p.view(B, Kk)
            Lr = huber(rul_p, r)
            Lh = mse(hi_p, h)
            d = hi_p[:, 1:] - hi_p[:, :-1]
            Lmono = torch.relu(-d).mean()
            Ls = (d * d).mean()
            zS = model.encoder(xs)                                       # 源域 latent (无 adapter)
            mmd = mmd_by_hi_bins(zS, hs, zT, h.reshape(-1), bins)        # 按 HI 健康阶段对齐
            loss = Lr + lam[0] * Lh + lam[1] * Lmono + lam[2] * Ls + mmd_lambda * mmd
            opt3.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt3.step()
            tl += loss.item() * B
            n += B
        vm = evaluate_T(model, loader_T_va, device, huber, mse, lam)
        print(f"  S3 ep{ep:02d} loss={tl / n:.4f} val_rul_rmse={vm['rul_rmse']:.4f} "
              f"mono_viol={vm['mono_viol']:.4f}")

    CKPT_DIR.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "transfer"
    ckpt_out = CKPT_DIR / f"transfer{suffix}_{cfg['model']['encoder']}_{tag}.pt"
    torch.save({"model": model.state_dict(), "L": L, "K": K, "bins": bins,
                "n_target": xT.shape[1],
                "split_train_val_test": (len(tr_ids), len(va_ids), len(te_ids))}, ckpt_out)
    metrics = {
        "stage": "3-stage-transfer", "encoder": cfg["model"]["encoder"], "smoke": bool(args.smoke),
        "L": L, "K": K, "val_rul_rmse": vm["rul_rmse"], "val_mono_viol": vm["mono_viol"],
        "split_train_val_test": [len(tr_ids), len(va_ids), len(te_ids)],
        "mmd_aligned_by": "hi_health_bins (非时间)",
    }
    metrics_path = CKPT_DIR / f"transfer_metrics{suffix}.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">> transfer checkpoint: {ckpt_out}")
    print(f">> 划分: train {len(tr_ids)} / val {len(va_ids)} / test {len(te_ids)} 轨迹 (完整轨迹级)")


# ============================================================ HI 动力学层路径
@torch.no_grad()
def _eval_hi_layer(model, loader, device, huber, mse, lam):
    """HI 层模型评估: 输入 (x_HI, hi_end, rul_end)。返回 val loss + RUL RMSE。"""
    model.eval()
    tl = n = 0
    preds, labels = [], []
    for x_HI, hi_end, rul_end in loader:
        x_HI = x_HI.to(device)
        hi_end = hi_end.to(device)
        rul_end = rul_end.to(device)
        hi_p, rul_p, _ = model(x_HI)
        Lr = huber(rul_p, rul_end)
        Lh = mse(hi_p, hi_end)
        loss = Lr + lam[0] * Lh
        tl += loss.item() * x_HI.size(0)
        n += x_HI.size(0)
        preds.append(rul_p.cpu().numpy())
        labels.append(rul_end.cpu().numpy())
    rp = np.concatenate(preds) if preds else np.array([0.0])
    rl = np.concatenate(labels) if labels else np.array([0.0])
    return {"loss": tl / max(n, 1),
            "rul_rmse": float(np.sqrt(np.mean((rp - rl) ** 2)))}


def _train_hi_stage_es(model, ltr, lva, opt, device, huber, mse, lam, epochs, tag,
                       use_mmd=False, src_iter=None, bins=None, mmd_lambda=1.0):
    """HI 层训练公共函数 (P0-1): 每 epoch 训练 + val 评估, 记录并恢复最佳 val 模型。

    仿 _train_with_early_stop (run_groups.py), 确保 HI 层所有组 (target/source) 与
    旧观测层同等选择协议 (val early-stop, 不用末轮模型 test)。
    """
    best = (float("inf"), None)
    for ep in range(1, epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for x_HI, hi_end, rul_end in ltr:
            x_HI = x_HI.to(device)
            hi_end = hi_end.to(device)
            rul_end = rul_end.to(device)
            hi_p, rul_p, zT = model(x_HI)
            Lr = huber(rul_p, rul_end)
            Lh = mse(hi_p, hi_end)
            loss = Lr + lam[0] * Lh
            if use_mmd and src_iter is not None:
                xs_HI, xs_hi_end = next(src_iter)
                xs_HI = xs_HI.to(device)
                xs_hi_end = xs_hi_end.to(device)
                zS = model.encoder(xs_HI)
                loss = loss + mmd_lambda * mmd_by_hi_bins(
                    zS, xs_hi_end, zT, hi_end, bins)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * x_HI.size(0)
            n += x_HI.size(0)
        vm = _eval_hi_layer(model, lva, device, huber, mse, lam)
        if vm["rul_rmse"] < best[0]:
            best = (vm["rul_rmse"],
                    {k: v.detach().clone() for k, v in model.state_dict().items()})
        print(f"  {tag} ep{ep:02d} loss={tl / max(n,1):.4f} val_rul_rmse={vm['rul_rmse']:.4f}")
    if best[1] is not None:
        model.load_state_dict(best[1])
    print(f"  [{tag}] best val_rmse={best[0]:.4f} (已恢复)")
    return best[0]


def run_hi_layer(args):
    """HI 动力学层迁移训练 (路径 A, 设计文档 §4.2)。

    关键差异 (vs 旧 run()):
      - load_source 后只用 hi/device_id/t_index (弃用 features); 由 make_hi_windows 构造 x_HI_S
      - load_target 后只用 hi_array/rul/traj_id (默认不读 x_global/x_nodes)
      - encoder 输入 (B, L, 2) 固定, 不再依赖 n_features/n_target
      - 无 adapter, MMD 对齐的 z_S / z_T 都来自共享 HISeqEncoder 对 x_HI 的编码

    三阶段:
      S1 加载源域 HISeqEncoder 预训练权重 (`source_*_{encoder}_hilayer_pretrain.pt`, Agent 2 产出)
      S2 冻结 encoder, 训 target HI_head + RUL_head (输入 = 目标域 hi 派生的 x_HI 窗)
      S3 解冻 + MMD 按 HI bin 对齐 z_S / z_T (共享 encoder 编码源 / 目标 x_HI)
    """
    cfg = load_config(args.config)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed              # override (多种子消融)
    set_seed(cfg["seed"], cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])
    reset_global_memory_bank()          # 任务 1: 每 run 清全局 MMD bank, 防跨 run 累积污染
    L = int(cfg["model"]["input_len_L"])
    tcfg = cfg["transfer"]
    mc = cfg["model"]
    bins = [tuple(b) for b in tcfg["hi_bins"]]
    _lc = cfg["loss"]
    lam = (float(_lc.get("lambda_hi", _lc.get("beta_hi", 1.0))),
           float(_lc.get("lambda_mono", _lc.get("mu_mono", 0.1))),
           float(_lc.get("lambda_smooth", _lc.get("nu_smooth", 0.1))))
    mmd_lambda = float(tcfg["mmd_lambda"])

    hi_cfg = tcfg.get("hi_layer", {}) or {}
    hi_stride_target = int(hi_cfg.get("hi_window_stride", 50))

    target_h5 = ROOT / tcfg.get("target_feature_path",
                                 "data/features/wheel/schema_v1/target_features.h5")
    source_h5 = ROOT / cfg["pretrain"].get(
        "source_feature_path", "data/features/wheel/schema_v1/source_features.h5")
    id_field = cfg["pretrain"].get("source_id_field", "bearing_id")
    if not target_h5.exists():
        print(f"!! 缺 {target_h5}; 先运行目标域特征工程 (build_hi.py / build_array_hi.py)")
        sys.exit(1)
    if not source_h5.exists():
        print(f"!! 缺 {source_h5}; 先运行源域特征工程 (wheel_features.py / mosfet_features.py)")
        sys.exit(1)

    # ---- 目标域: 只读 hi / rul / traj_id (路径 A 默认不用 x_global/x_nodes) ----
    hiT, rulT, tidT, n_traj = _load_target_hi(target_h5)
    tr_ids, va_ids, te_ids = split_trajectories(
        n_traj, [tcfg["split"]["train"], tcfg["split"]["val"], tcfg["split"]["test"]],
        cfg["seed"])

    def mask(ids):
        return np.isin(tidT, ids)

    # P1-2: RUL 归一因子用 config 固定物理上限 (跨 seed 可比); 缺失回退 train-only max
    if "rul_max_norm" in tcfg:
        rul_max = float(tcfg["rul_max_norm"])
    else:
        rul_max_train = float(rulT[mask(tr_ids)].max())     # 回退 (跨 seed 不可比, 仅兼容旧 config)
        rul_max = rul_max_train
        print(f"  [warning] config 未设 transfer.rul_max_norm, 回退 train-only max={rul_max:.1f}")
    rulT = rulT / max(rul_max, 1.0)

    tstride = int(tcfg.get("target_stride", hi_stride_target))      # 序列长, 大 stride 控窗数
    if args.smoke:
        tstride = max(tstride, 1000)
    ds_tr = HIWindowDataset(hiT[mask(tr_ids)], rulT[mask(tr_ids)],
                            tidT[mask(tr_ids)], L, stride=tstride)
    ds_va = HIWindowDataset(hiT[mask(va_ids)], rulT[mask(va_ids)],
                            tidT[mask(va_ids)], L, stride=tstride)
    ds_te = HIWindowDataset(hiT[mask(te_ids)], rulT[mask(te_ids)],
                            tidT[mask(te_ids)], L, stride=tstride)
    if args.smoke:
        ds_tr = Subset(ds_tr, list(range(min(48, len(ds_tr)))))
        ds_va = Subset(ds_va, list(range(min(24, len(ds_va)))))
    bs = 16 if args.smoke else int(cfg["pretrain"]["batch_size"])
    loader_T_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    loader_T_va = DataLoader(ds_va, batch_size=bs, shuffle=False)
    loader_T_te = DataLoader(ds_te, batch_size=bs, shuffle=False)

    # ---- 源域: 只用 hi / device_id / t_index (弃用 features) ----
    # load_source 仍调 (复用 v1/v2 reader); 但下面 make_hi_windows 只吃 hi/device_id/t_index
    _, hiS, idS, tidxS, splitS = load_source(
        source_h5, id_field,
        cfg.get("source", {}).get("split", {}).get("val_device_ids", []) or [])
    # 任务 3: 源 MMD 对齐窗只用 source train 器件 (val 器件不参与迁移对齐)
    _src_tr = np.asarray(splitS) == "train"
    if not _src_tr.any():
        raise ValueError("source train split empty; refusing val source into MMD")
    hiS = hiS[_src_tr]; idS = np.asarray(idS)[_src_tr]; tidxS = tidxS[_src_tr]
    x_HI_S, hi_end_S = make_hi_windows(
        hiS, idS, tidxS, L, stride=max(1, len(hiS) // 2000))
    if args.smoke:
        cap = min(64, x_HI_S.shape[0])
        x_HI_S = x_HI_S[:cap]
        hi_end_S = hi_end_S[:cap]
    ds_S = TensorDataset(torch.from_numpy(x_HI_S).float(),
                         torch.from_numpy(hi_end_S).float())
    loader_S = DataLoader(ds_S, batch_size=bs, shuffle=True)

    device = "cuda" if (cfg["pretrain"]["device"] == "cuda"
                        and torch.cuda.is_available()) else "cpu"
    model = HIDynamicsModel(
        encoder_type=cfg["model"]["encoder"], input_len=L,
        channels=mc["tcn"]["channels"], kernel_size=mc["tcn"]["kernel_size"],
        num_blocks=mc["tcn"]["num_blocks"], dropout=mc["tcn"]["dropout"],
        latent_dim=mc["latent_dim"],
    ).to(device)

    # ---- S1: 加载源域 HISeqEncoder 预训练权重 (target_only 跳过) ----
    comp = Path(args.config).stem
    suffix = "" if comp == "wheel" else f"_{comp}"                  # 飞轮保持原名, 相控阵带后缀
    if getattr(args, "target_only", False):
        print(">> [target_only] 跳过 S1 源 ckpt 加载, encoder 随机初始化 + 后续全训 (迁移增益基线)")
    else:
        ckpt = (args.ckpt or
                str(CKPT_DIR / f"source{suffix}_{cfg['model']['encoder']}_hilayer_pretrain.pt"))
        if not Path(ckpt).exists():
            # fallback HI 层 smoke ckpt
            ckpt_smoke = str(CKPT_DIR / f"source{suffix}_{cfg['model']['encoder']}_hilayer_smoke.pt")
            if Path(ckpt_smoke).exists():
                ckpt = ckpt_smoke
        if Path(ckpt).exists():
            miss, unexp = model.load_pretrained(ckpt, device)
            print(f">> S1 加载源 HISeqEncoder: {ckpt}  (missing={len(miss)} keys)")
        else:
            print(">> [警告] 无源域 HI 层 checkpoint, encoder 随机初始化 (仅限 smoke)")
            print(f"   (期望 {ckpt}; 先跑 `python -m src.train.pretrain --config {args.config} --hi-layer --smoke` 落盘)")

    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]))
    mse = nn.MSELoss()
    e2 = 2 if args.smoke else int(tcfg.get("epochs_s2", 20))
    # P0-1: target_only 也跑 S3 等量 epoch (纯微调 use_mmd=False), 保 optimizer step 预算一致
    e3 = 2 if args.smoke else int(tcfg.get("epochs_s3", 20))
    is_target_only = getattr(args, "target_only", False)

    # ---- S2: 训 HI_head + RUL_head (target_only 解冻全训; 否则 encoder 冻结) ----
    model.freeze_encoder(not is_target_only)
    opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=float(tcfg["finetune_lr"]))
    print(f">> S2 HI/RUL 头训练 (encoder {'解冻' if is_target_only else '冻结'}, {e2} epochs, val early-stop)")
    _train_hi_stage_es(model, loader_T_tr, loader_T_va, opt2, device, huber, mse, lam,
                       e2, "S2")

    # ---- S3: 解冻 + (source: MMD 按 HI bin 对齐; target_only: 纯微调无 MMD) ----
    # P0-1: target_only 的 S3 设 use_mmd=False (纯微调), 但 epoch 数与 source 相同
    model.freeze_encoder(False)
    opt3 = torch.optim.Adam(model.parameters(), lr=float(tcfg["finetune_lr"]))
    if is_target_only:
        print(f">> S3 纯微调 (target_only, 无 MMD, {e3} epochs, val early-stop)")
        s3_val = _train_hi_stage_es(model, loader_T_tr, loader_T_va, opt3, device, huber, mse, lam,
                           e3, "S3")
    else:
        print(f">> S3 MMD(按HI分箱, HI 动力学层)+微调 ({e3} epochs, val early-stop)")
        s3_val = _train_hi_stage_es(model, loader_T_tr, loader_T_va, opt3, device, huber, mse, lam,
                           e3, "S3",
                           use_mmd=True, src_iter=cycle(loader_S), bins=bins,
                           mmd_lambda=mmd_lambda)

    # test 评估 (泛化, 对比旧观测层 run_groups 的 test rmse)
    te_m = _eval_hi_layer(model, loader_T_te, device, huber, mse, lam)
    print(f">> TEST rul_rmse={te_m['rul_rmse']:.4f} (test {len(te_ids)} 轨迹)")

    CKPT_DIR.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "transfer"
    ckpt_out = CKPT_DIR / f"transfer{suffix}_{cfg['model']['encoder']}_hilayer_{tag}.pt"
    torch.save({"model": model.state_dict(), "L": L, "bins": bins,
                "transfer_mode": "hi_dynamics",
                "split_train_val_test": (len(tr_ids), len(va_ids), len(te_ids))},
               ckpt_out)
    metrics = {
        "stage": "hi-layer-transfer", "transfer_mode": "hi_dynamics",
        "encoder": cfg["model"]["encoder"], "smoke": bool(args.smoke),
        "L": L, "val_rul_rmse": s3_val, "test_rul_rmse": te_m["rul_rmse"],
        "split_train_val_test": [len(tr_ids), len(va_ids), len(te_ids)],
        "mmd_aligned_by": "hi_health_bins on HISeqEncoder latent (HI dynamics)",
    }
    metrics_path = CKPT_DIR / f"transfer_metrics{suffix}_hilayer.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">> HI 层 transfer checkpoint: {ckpt_out}")
    print(f">> 划分: train {len(tr_ids)} / val {len(va_ids)} / test {len(te_ids)} 轨迹 (完整轨迹级)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hi-layer", action="store_true",
                    help="走 HI 动力学层迁移 (HIDynamicsModel); 默认走旧观测层 run()。"
                         "也可经 config transfer.mode=hi_dynamics 触发。")
    ap.add_argument("--target-only", action="store_true",
                    help="HI 层 target_only 基线: 随机初始化 + 全训 (不加载源 ckpt, 不 MMD), 迁移增益对比用")
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 config seed (多种子消融; 默认用 config seed)")
    args = ap.parse_args()
    # config transfer.mode == 'hi_dynamics' 时自动切到 HI 层路径 (除非用户显式 --no-hi-layer)
    cfg = load_config(args.config)
    tcfg = cfg.get("transfer", {}) or {}
    if args.hi_layer or tcfg.get("mode") == "hi_dynamics":
        run_hi_layer(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
