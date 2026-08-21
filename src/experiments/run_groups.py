"""experiments/run_groups.py

对比实验编排 (plan §四 Phase 6 飞轮 / §8 相控阵八组), 每组 n 种子,
聚合 mean±std -> docs/results_<component>[_smoke].md (组件级 + smoke 分文件, 防互相覆盖)。

组件级参数化 (2026-07-21 PA6 阶段1): 从 config experiments.groups 读组名,
内部 mode 映射 + checkpoint 组件后缀, 飞轮 (4 组) / 相控阵 (8 组) 共用。

组 (相控阵 config §8):
  target_only_tcn / target_only_gru    仅目标域少样本 (TCN / GRU 基线)
  source_pretrain_finetune             源域预训练 + 微调
  source_mmd_physics                   源域 + 阶段 MMD + 物理一致性 (主迁移模型)
  timesfm_zeroshot / xreg / lora_xreg  TimesFM 辅助分支 (PA7; 未启用则 skip 占位)
  main_timesfm_fusion                  主模型 + TimesFM 概率融合 (PA7)
飞轮 config §P6: target_only / source_only / source_finetune / source_mmd_finetune / physical_extrap
(旧名保留兼容, 经 _GROUP_MAP 映射到内部 mode)

用法:
  python -m src.experiments.run_groups --config configs/phased_array.yaml --seeds 5
  python -m src.experiments.run_groups --config configs/phased_array.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed                                    # noqa: E402
from src.transfer.adapter import TransferModel                                 # noqa: E402
from src.transfer.mmd import mmd_by_hi_bins, reset_global_memory_bank          # noqa: E402
from src.transfer.train_transfer import (                                      # noqa: E402
    TargetSeqDataset, SourceWindowDataset, split_trajectories,
    load_target, load_source)
from src.transfer.channel_dataset import (                                     # noqa: E402  (T6.3 channel level)
    ChannelSeqDataset, load_target_channel, apply_kshot_mask,
    sample_kshot_trajectories, assert_split_by_trajectory)
from src.train.pretrain import _rul_loss                                       # noqa: E402  (P0-2 失效/删失分流)
from src.baselines.physical_extrap import evaluate_physical, phm_score, mae as mae_fn   # noqa: E402
from src.baselines.phased_array_baselines import evaluate_phased_array_baselines        # noqa: E402

CKPT_DIR = ROOT / "checkpoints"

# 依赖 PA7 (TimesFM 分支未建): timesfm.enabled=False 时 skip + 占位
TIMESFM_GROUPS = {"timesfm_zeroshot", "timesfm_xreg", "timesfm_lora_xreg",
                  "main_timesfm_fusion"}

# config 组名 → (内部 mode, encoder_override)。None encoder = 用 config 默认 (tcn)
# 架构纪律 (迁移结论清零重审): 迁移归因组统一显式 GRU, 与主模型 target_only_gru 同架构,
# 消除 "GRU target-only vs TCN transfer" 混架构归因; *_tcn 组名回归真 TCN 作架构消融 (名实一致, 不进迁移归因)
_GROUP_MAP = {
    # 相控阵 (config §8)
    "target_only_tcn":            ("target_only", "tcn"),    # 架构消融 (真 TCN)
    "target_only_gru":            ("target_only", "gru"),    # 主基线
    "source_pretrain_finetune":   ("source_finetune", "gru"),
    "source_mmd_physics":         ("source_mmd_finetune", "gru"),
    "random_frozen":              ("random_frozen", "gru"),
    "random_full_finetune":       ("random_full_mmd", "gru"),
    "random_nommd":               ("random_full_nommd", "gru"),
    # 通道级 ch_* 组 (T6.3/M7, level=channel 路径专用; 内部 mode 与旧组同, level 决定走哪个 run_one_group)
    "ch_target_only_tcn":         ("target_only", "tcn"),    # 架构消融 (真 TCN)
    "ch_target_only_gru":         ("target_only", "gru"),
    "ch_source_pretrain_frozen":  ("source_finetune", "gru"),
    "ch_source_mmd_physics":      ("source_mmd_finetune", "gru"),
    "ch_random_frozen":           ("random_frozen", "gru"),
    "ch_random_full_finetune":    ("random_full_mmd", "gru"),
    "ch_random_nommd":            ("random_full_nommd", "gru"),
    "ch_source_igbt":             ("source_mmd_finetune", "gru"),   # §4c k-shot 源域臂: IGBT ckpt
    "ch_source_multi":            ("source_mmd_finetune", "gru"),   # §4c k-shot 源域臂: MOSFET+IGBT 多源 ckpt
    "ch_simv1_source":            ("source_mmd_finetune", "gru"),   # §4e B 路线: sim_v1 (legacy) 源初始化, MMD 窗=v1 canonical
    # A1 α-soft (§4d 四审方案): θ₀ = θ_rand + α·(θ_src − θ_rand), 源=MOSFET canonical;
    # 训练协议与 ch_random_full_finetune/ch_source_mmd_physics 同 (S2 冻结→S3 全微调+MMD),
    # 唯一变量 = encoder 初始化插值系数 α; a000 = α=0 校验臂 (应逐位复现 random 臂)
    "ch_alpha_soft_a000":         ("alpha_soft_finetune", "gru"),
    "ch_alpha_soft_a005":         ("alpha_soft_finetune", "gru"),
    "ch_alpha_soft_a010":         ("alpha_soft_finetune", "gru"),
    "ch_alpha_soft_a025":         ("alpha_soft_finetune", "gru"),
    "ch_alpha_soft_a050":         ("alpha_soft_finetune", "gru"),
    "ch_layerwise_gru_p1":        ("layerwise_finetune", "gru"),
    "ch_layerwise_gru_p2":        ("layerwise_finetune", "gru"),
    "cross_level_transfer":       ("cross_level", "gru"),    # T6.3 层级消融 (level=service; 同 GRU 架构, level_control 纯归因层级)
    # 飞轮旧名 (兼容)
    "target_only":                ("target_only", None),
    "source_only":                ("source_only", None),
    "source_finetune":            ("source_finetune", None),
    "source_mmd_finetune":        ("source_mmd_finetune", None),
}

# results.md 表格标签
LABELS = {
    "target_only":              "Target-only",
    "target_only_tcn":          "Target-only (TCN)",
    "target_only_gru":          "Target-only (GRU)",
    "source_only":              "Source-only",
    "source_finetune":          "Source+Finetune",
    "source_pretrain_finetune": "Source+Finetune",
    "source_mmd_finetune":      "**Source+MMD+Finetune**",
    "source_mmd_physics":       "**Source+MMD+Physics**",
    "random_frozen":            "Random+Frozen *(P0-3 对照)*",
    "random_full_finetune":     "Random+Full+MMD *(GPT §3 对照)*",
    "random_nommd":             "Random+Full+NoMMD *(GPT §3 P0-2)*",
    "ch_target_only_tcn":       "CH Target-only (TCN) *(M7)*",
    "ch_target_only_gru":       "CH Target-only (GRU) *(M7)*",
    "ch_source_pretrain_frozen":"CH Source+Frozen *(M7)*",
    "ch_source_mmd_physics":    "**CH Source+MMD+Physics** *(M7)*",
    "ch_random_frozen":         "CH Random+Frozen *(M7)*",
    "ch_random_full_finetune":  "CH Random+Full+MMD *(M7)*",
    "ch_random_nommd":          "CH Random+Full+NoMMD *(M7)*",
    "ch_source_igbt":            "CH Source IGBT *(§4c k-shot 臂)*",
    "ch_source_multi":           "CH Source Multi *(§4c k-shot 臂)*",
    "ch_simv1_source":           "**CH Source sim_v1** *(§4e B 路线)*",
    "ch_alpha_soft_a000":        "CH α-soft α=0.00 *(A1 校验臂)*",
    "ch_alpha_soft_a005":        "CH α-soft α=0.05 *(A1)*",
    "ch_alpha_soft_a010":        "CH α-soft α=0.10 *(A1)*",
    "ch_alpha_soft_a025":        "CH α-soft α=0.25 *(A1)*",
    "ch_alpha_soft_a050":        "CH α-soft α=0.50 *(A1)*",
    "ch_layerwise_gru_p1":       "CH Layer-wise GRU P1 *(A3)*",
    "ch_layerwise_gru_p2":       "CH Layer-wise GRU P2 *(A3)*",
    "cross_level_transfer":     "Cross-level (旧服务级) *(层级消融)*",
    "timesfm_zeroshot":         "TimesFM zero-shot *(PA7)*",
    "timesfm_xreg":             "TimesFM + XReg *(PA7)*",
    "timesfm_lora_xreg":        "TimesFM + LoRA + XReg *(PA7)*",
    "main_timesfm_fusion":      "主模型 + TimesFM 融合 *(PA7)*",
}


def _alpha_from_group(group_name):
    """A1 α-soft 组名 → α 值。

    ch_alpha_soft_aXXX[_kN] → XXX/100 (如 a005→0.05, a050→0.5); 非 α 组返回 None。
    k-shot 后缀用 split 截断, 与 _src_tag 的 startswith 约定互补。
    """
    if group_name and group_name.startswith("ch_alpha_soft_a"):
        tag = group_name[len("ch_alpha_soft_a"):].split("_")[0]
        return int(tag) / 100.0
    return None


def _source_ckpt_name(group_name, component, enc):
    """组名 → 源域 ckpt 文件名 (source 臂与 alpha_soft 臂共用, 保证 α=1 端点同 ckpt)。

    §4c 约定: 组名前缀决定源域 ckpt tag (startswith 防 _k{shot} 后缀 miss);
    默认 MOSFET canonical — alpha_soft 臂也走此默认 (A1 主臂源域 = MOSFET)。
    """
    suffix = "" if component == "wheel" else f"_{component}"
    _src_tag = ("simv1" if group_name and group_name.startswith("ch_simv1_source")
                else "igbt" if group_name and group_name.startswith("ch_source_igbt")
                else "mosfet_igbt" if group_name and group_name.startswith("ch_source_multi")
                else "")
    return (f"source{suffix}_{_src_tag}_{enc}_pretrain.pt" if _src_tag
            else f"source{suffix}_{enc}_pretrain.pt")


def _interpolate_encoder(model, sd_enc, alpha):
    """A1 α-soft 插值: θ₀ = θ_rand + α·(θ_src − θ_rand), 就地写回 model。

    只插值 encoder.* 浮点张量 — 与 α=1 端点 (source 臂 load_pretrained) 的加载范围
    严格一致, 其余参数 (adapter/heads) 保持 θ_rand, 端点可比性不受插值范围污染。
    不消耗 RNG (torch.load/load_state_dict/张量算术均不触碰随机流)。
    返回实际插值的张量数。
    """
    cur = model.state_dict()
    mixed = {}
    n_mixed = 0
    for k, v in cur.items():
        if k in sd_enc and v.dtype.is_floating_point:
            mixed[k] = v + alpha * (sd_enc[k].to(device=v.device, dtype=v.dtype) - v)
            n_mixed += 1
        else:
            mixed[k] = v
    model.load_state_dict(mixed)
    return n_mixed


# A3 分层迁移：GRU 前缀按嵌套深度严格加载，projection 始终保留目标域随机初始化。
_GRU_L0_KEYS = frozenset({
    "encoder.gru.weight_ih_l0", "encoder.gru.weight_hh_l0",
    "encoder.gru.bias_ih_l0", "encoder.gru.bias_hh_l0",
})
_GRU_L1_KEYS = frozenset({
    "encoder.gru.weight_ih_l1", "encoder.gru.weight_hh_l1",
    "encoder.gru.bias_ih_l1", "encoder.gru.bias_hh_l1",
})
_GRU_PROJ_KEYS = frozenset({"encoder.proj.weight", "encoder.proj.bias"})
_GRU_ENCODER_KEYS = _GRU_L0_KEYS | _GRU_L1_KEYS | _GRU_PROJ_KEYS


def _layerwise_depth_from_group(group_name: str | None) -> int | None:
    """从 A3 组名解析要迁移的 GRU 前缀深度。"""
    if not group_name:
        return None
    for depth in (1, 2):
        prefix = f"ch_layerwise_gru_p{depth}"
        if group_name == prefix or group_name.startswith(prefix + "_k"):
            return depth
    return None


def _load_layerwise_encoder(
        model: TransferModel, sd_enc: dict[str, torch.Tensor], depth: int) -> tuple[str, ...]:
    """严格加载 GRU 的连续前缀层，不加载 projection 且不消耗随机数。"""
    if depth not in (1, 2):
        raise ValueError(f"layer-wise depth 仅支持 1/2, 收到 {depth}")
    actual = set(sd_enc)
    if actual != set(_GRU_ENCODER_KEYS):
        missing = sorted(_GRU_ENCODER_KEYS - actual)
        extra = sorted(actual - _GRU_ENCODER_KEYS)
        raise ValueError(f"GRU encoder checkpoint 键不匹配: missing={missing}, extra={extra}")
    selected = _GRU_L0_KEYS if depth == 1 else (_GRU_L0_KEYS | _GRU_L1_KEYS)
    target = model.state_dict()
    # 先完整预检，拒绝无效 checkpoint 时模型必须保持逐位不变。
    for key in sorted(selected):
        src, dst = sd_enc[key], target[key]
        if src.shape != dst.shape or src.dtype != dst.dtype:
            raise ValueError(
                f"layer-wise 张量不兼容 {key}: source={src.shape}/{src.dtype}, "
                f"target={dst.shape}/{dst.dtype}")
    with torch.no_grad():
        for key in sorted(selected):
            target[key].copy_(sd_enc[key])
    return tuple(sorted(selected))


# ---- CPU 并行 + 资源限制 + 增量落盘 (2026-08-18, 外部实验无资源限制挤死长跑矩阵事故后加固) ----
_WORKER_THREADS = 4


def _worker_init(threads=None):
    """ProcessPoolExecutor 子进程资源限制。

    事故背景: 外部实验未限资源时, 多进程 × PyTorch 默认全核 OMP 线程 的乘积
    会线程/内存爆炸, 把本仓长跑矩阵挤死。每 worker 限制 torch CPU 线程数,
    使资源占用 ≈ workers × threads (有界), 与外部实验共存时可控。
    """
    torch.set_num_threads(threads or _WORKER_THREADS)


def _run_one_task(task):
    """并行执行单元: 单 组×seed 的 run_one_group。

    run_one_group 自带 set_seed + reset_global_memory_bank, 多进程下 RNG 与
    MMD bank 均为进程私有 → 与串行行为一致 (顺序无关, 逐位等价)。
    """
    (mode, seed, cfg, smoke, enc_ov, component, tag_name, level, k_shot) = task
    return run_one_group(mode, seed, cfg, smoke=smoke, encoder_override=enc_ov,
                         component=component, group_name=tag_name, level=level,
                         k_shot=k_shot)


def _load_jsonl_records(jsonl_path):
    """增量落盘恢复: 读 results_partial.jsonl, 返回 {(group, seed): 完整指标 dict}。

    逐行 try 解析, 坏行 (中断时写一半) 跳过 — 对应组-seed 会重跑, 幂等安全。
    记录含 run_one_group 产出的全部字段 (rmse/phm/mae/val_rmse/...), 调用方
    须把旧记录重载入聚合器 by — 只跳过不重载会让恢复跑的 aggregate 缺臂。
    """
    records = {}
    if not Path(jsonl_path).exists():
        return records
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (rec.get("group"), rec.get("seed"))
        if key[0] is not None and key[1] is not None:
            records[key] = rec
    return records


def _append_jsonl(jsonl_path, record):
    """单组-seed 完成即落盘 (append + flush): 中断只损失正在跑的那一个组-seed。"""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")
        f.flush()


def _group_level(group_name):
    """组名 → level ('channel' / 'service')。

    ch_* 前缀 → channel (T6.3 通道级, ChannelSeqDataset + canonical schema)
    cross_level_transfer / 旧组 → service (TargetSeqDataset + x_global+x_nodes)
    """
    if group_name.startswith("ch_"):
        return "channel"
    return "service"


def _resolve_group(group_name, cfg):
    """config 组名 → (内部 mode, encoder_override, enabled, reason)。

    TimesFM 组依赖 PA7 (timesfm.enabled); 物理基线单独 evaluate; 其他查 _GROUP_MAP。
    level 由 _group_level(group_name) 决定, 调用方据此选 run_one_group (service)
    或 run_one_group_channel (channel)。
    """
    if group_name in TIMESFM_GROUPS:
        enabled = bool(cfg.get("timesfm", {}).get("enabled", False))
        reason = "TimesFM 已启用" if enabled else "TimesFM 未启用 (PA7 待推进)"
        return None, None, enabled, reason
    if group_name == "physical_extrap":
        return "__physical__", None, True, "物理基线 (单独 evaluate)"
    if group_name in _GROUP_MAP:
        mode, enc = _GROUP_MAP[group_name]
        # cross_level_transfer: mode 标记 "cross_level" → 旧 source_mmd_finetune 路径 (service level)
        if mode == "cross_level":
            return "source_mmd_finetune", enc, True, ""
        return mode, enc, True, ""
    return None, None, False, "未知组名 (跳过)"


def _build_model(cfg, n_features, n_target, device, encoder_override=None):
    mc = cfg["model"]
    tc = cfg["transfer"]
    enc = encoder_override or mc["encoder"]
    return TransferModel(
        encoder_type=enc, n_features=n_features, n_target=n_target,
        channels=mc["tcn"]["channels"], kernel_size=mc["tcn"]["kernel_size"],
        num_blocks=mc["tcn"]["num_blocks"], dropout=mc["tcn"]["dropout"],
        latent_dim=mc["latent_dim"], adapter_hidden=tc["adapter_hidden"]).to(device)


def _train_epoch(model, loader, opt, device, huber, mse, lam,
                 use_mmd=False, src_iter=None, bins=None, mmd_lambda=1.0, eta=1.0,
                 use_phys=False, rho_phys=0.0):
    """P0-2: RUL loss 用 _rul_loss 失效 Huber + 删失 hinge (不再裸 huber 把删失当下界当精确标签)。
    T13 rho_phys: use_phys 时加 ρ·MSE(hi_pred, damage_norm) 物理一致性 (source_mmd_physics S3)。"""
    model.train()
    tl, n = 0.0, 0
    for batch in loader:
        if use_mmd:
            x, h, r, ev, lb, dmg = batch
            xs, hs = next(src_iter)
            xs, hs = xs.to(device), hs.to(device)
        else:
            x, h, r, ev, lb, dmg = batch
        x, h, r = x.to(device), h.to(device), r.to(device)
        ev, lb = ev.to(device), lb.to(device)
        dmg = dmg.to(device)
        B, Kk = x.size(0), x.size(1)
        hi_p, rul_p, zT = model(x.reshape(B * Kk, x.size(2), x.size(3)))
        hi_p = hi_p.view(B, Kk)
        rul_p = rul_p.view(B, Kk)
        Lr, _, _ = _rul_loss(rul_p, r, ev, lb, huber, eta)
        Lh = mse(hi_p, h)
        d = hi_p[:, 1:] - hi_p[:, :-1]
        loss = Lr + lam[0] * Lh + lam[1] * torch.relu(-d).mean() + lam[2] * (d * d).mean()
        if use_phys:   # T13: 预测 HI 向 Arrhenius+Coffin-Manson 物理损伤靠拢
            loss = loss + rho_phys * mse(hi_p, dmg)
        if use_mmd:
            zS = model.encoder(xs)
            loss = loss + mmd_lambda * mmd_by_hi_bins(zS, hs, zT, h.reshape(-1), bins)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item() * B
        n += B
    return tl / max(n, 1)


@torch.no_grad()
def _bias_diag(p, t, ev, hi):
    """§4h 误差方向/风险偏置诊断: e = RUL̂ − RUL 按 HI 真值三分箱 (label 非预测, 无泄漏)。

    全部在归一化 RUL 空间; 失效样本精确真值, 删失样本仅下界 (bias 相对 lb)。
    """
    m = ev.astype(bool)
    out = {}
    if m.any():
        err = p[m] - t[m]
        hm = hi[m]
        out["failed_all"] = {"n": int(m.sum()),
                             "mean_bias": float(err.mean()),
                             "median_bias": float(np.median(err)),
                             "over_rate": float((err > 0).mean())}
        for tag, sel in [("early", hm < 1.0 / 3.0),
                         ("middle", (hm >= 1.0 / 3.0) & (hm < 2.0 / 3.0)),
                         ("late", hm >= 2.0 / 3.0)]:
            if sel.any():
                ee = err[sel]
                out[tag] = {"n": int(sel.sum()),
                            "mean_bias": float(ee.mean()),
                            "median_bias": float(np.median(ee)),
                            "over_rate": float((ee > 0).mean())}
    if (~m).any():
        pc, lc = p[~m], t[~m]            # t = rul_lower_bound (删失下界)
        out["censored"] = {"n": int((~m).sum()),
                           "lb_violation_rate": float((pc < lc).mean()),   # 激进违反
                           "mean_bias_vs_lb": float((pc - lc).mean()),
                           "over_rate": float((pc > lc).mean())}
    return out


def eval_test(model, loader, device, bias_diag=False):
    """P0-2: 分开报失效轨迹 (精确 RUL 的 RMSE/PHM/MAE) + 删失轨迹 (下界违反率)。
    删失无精确 RUL, 混算 RMSE 会把"距仿真截止时刻"当退化标签 (复核第三条)。
    bias_diag=True 额外落 §4h 误差方向诊断 (early-stop 调用不传, 零开销)。"""
    model.eval()
    preds, labels, evs = [], [], []
    his = [] if bias_diag else None
    with torch.no_grad():
        for x, h, r, ev, lb, dmg in loader:
            x = x.to(device)
            B, Kk = x.size(0), x.size(1)
            _, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
            preds.append(rul_p.cpu().numpy())
            labels.append(r.reshape(-1).numpy())
            evs.append(ev.reshape(-1).numpy())
            if bias_diag:
                his.append(h.reshape(-1).numpy())
    p = np.concatenate(preds) if preds else np.array([0.0])
    t = np.concatenate(labels) if labels else np.array([0.0])
    e = np.concatenate(evs) if evs else np.array([True])
    m = e.astype(bool)            # 失效 (精确 RUL)
    cm = ~m                       # 删失 (仅下界)
    rmse_f = float(np.sqrt(np.mean((p[m] - t[m]) ** 2))) if m.any() else 0.0
    mae_f = float(np.mean(np.abs(p[m] - t[m]))) if m.any() else 0.0
    censor_viol = float(np.mean(p[cm] < t[cm])) if cm.any() else 0.0   # pred<lb 比例 (越低越坏)
    out = {"rmse": rmse_f, "mae": mae_f,                                # 兼容下游聚合 (失效轨迹)
           "rmse_failed": rmse_f, "mae_failed": mae_f,
           "phm": phm_score(p[m], t[m]) if m.any() else 0.0,
           "censor_violation_rate": censor_viol,
           "n_failed": int(m.sum()), "n_censored": int(cm.sum())}
    if bias_diag:
        h_arr = np.concatenate(his) if his else np.array([0.0])
        out["bias_diag"] = _bias_diag(p, t, e, h_arr)
    return out


def _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, tag,
                           use_mmd=False, src_iter=None, bins=None, mmd_lambda=1.0,
                           use_phys=False, rho_phys=0.0):
    """训练 e epoch, 基于 val RMSE early-stop (恢复最佳模型)。

    防末段发散/过拟合 — 如 S3 MMD 某些 seed 末段单调上升→坍缩 (诊断 seed43: 0.17→0.34)。
    T13: use_phys/rho_phys 透传 ρ·L_phys (source_mmd_physics S3 物理一致性)。
    """
    best = (float("inf"), None)
    for ep in range(e):
        _train_epoch(model, ltr, opt, device, huber, mse, lam,
                     use_mmd=use_mmd, src_iter=src_iter, bins=bins, mmd_lambda=mmd_lambda,
                     use_phys=use_phys, rho_phys=rho_phys)
        vm = eval_test(model, lva, device)
        if vm["rmse"] < best[0]:
            best = (vm["rmse"], {k: v.detach().clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
    print(f"    [{tag}] best val_rmse={best[0]:.4f}")
    return best[0]


def run_one_group(mode, seed, cfg, smoke=False, encoder_override=None,
                  component="wheel", group_name=None, level="service", k_shot=None):
    set_seed(seed, cfg["reproducibility"]["deterministic"], cfg["reproducibility"]["cudnn_benchmark"])
    reset_global_memory_bank()          # 任务 1: 每 run 清全局 MMD bank, 防跨 seed/group 累积污染
    L = int(cfg["model"]["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    tc = cfg["transfer"]
    mc = cfg["model"]
    bins = [tuple(b) for b in tc["hi_bins"]]
    _lc = cfg["loss"]
    lam = (float(_lc.get("lambda_hi", _lc.get("beta_hi", 1.0))),       # 飞轮 lambda_hi / 相控阵 beta_hi
           float(_lc.get("lambda_mono", _lc.get("mu_mono", 0.1))),     # lambda_mono / mu_mono
           float(_lc.get("lambda_smooth", _lc.get("nu_smooth", 0.1)))) # lambda_smooth / nu_smooth
    mmd_lambda = float(tc["mmd_lambda"])
    rho_phys_loss = float(_lc.get("rho_phys", 0.0))   # T13: ρ·L_phys 物理一致性权重 (rho_phys>0 时 S3 生效)
    enc = encoder_override or mc["encoder"]

    # ---- 数据加载 (level 分流; T6.3 channel level 新增) ----
    if level == "channel":
        # 通道级: ChannelSeqDataset + canonical source (4 维); ckT = traj*16+sub 分组键
        ch_cfg = cfg["channel_level"]
        target_h5 = ROOT / ch_cfg["feature_path"]
        if not target_h5.exists():
            raise FileNotFoundError(f"channel level 缺 {target_h5}; 先 build_channel_hi")
        xT, hiT, rulT, ckT, tidT, evT, lbT, n_traj, sidT = load_target_channel(
            target_h5, drop_features=tc.get("ablation_drop_features"))
        canonical_path = cfg["pretrain"].get(
            "canonical_source_path",
            "data/features/phased_array/schema_v4/source/mosfet_canonical.h5")
        featsS, hiS, bidS, tidxS, splitS = load_source(
            ROOT / canonical_path,
            cfg.get("pretrain", {}).get("source_id_field", "device_id"),
            cfg.get("source", {}).get("split", {}).get("val_device_ids", []) or [])
        tr, va, te = split_trajectories(
            n_traj, [tc["split"]["train"], tc["split"]["val"], tc["split"]["test"]], seed)
        assert_split_by_trajectory(tidT, tr, va, te, sidT)  # channel_level 铁律: 同 traj 16 sub 同 split
        # k-shot mask (T6.2): train 内采 k 条保留标签, 其余 mask (MMD 无监督对齐)
        k_ids = sample_kshot_trajectories(tr, k_shot, seed=seed)
        rulT, evT, lbT = apply_kshot_mask(rulT, evT, lbT, tidT, tr, k_ids)
        # 通道级无 damage_norm 真值 (build_channel_hi 未存): damageT=None → S3 阶段 ρ·L_phys 条件禁用
        # (旧版全零占位 + rho_phys>0 会把 HI 预测持续拉向 0, 属实验污染, 已根除)
        damageT = None
        if rho_phys_loss > 0:
            print("  [L_phys] channel level 无 damage 真值: S3 阶段 ρ·L_phys 已禁用 (use_phys=False)")
        n_label_traj = len(k_ids) if k_ids is not None else len(tr)
        if k_shot is not None and k_shot != "all":
            print(f"  [k-shot] k={k_shot}: {n_label_traj}/{len(tr)} train 轨迹带标签")
        # 按 ckT 分组 (ChannelSeqDataset = TargetSeqDataset 子类, 行为同; ckT 唯一标识每条通道)
        group_keys_T = ckT
    else:
        # 旧服务级 (service): TargetSeqDataset + x_global+x_nodes (cross_level_transfer / 旧 PA6)
        has_nodes = bool(tc.get("target_has_nodes", False))
        xT, hiT, rulT, tidT, n_traj, evT, damageT = load_target(
            ROOT / tc.get("target_feature_path", "data/features/wheel/schema_v1/target_features.h5"),
            has_nodes, drop_features=tc.get("ablation_drop_features"), return_damage=True)
        # source: 相控阵用 canonical (4 维, M3 主源域; 旧 5 维 schema_v3 已废弃); 飞轮用 source_feature_path
        if component == "phased_array":
            src_path = cfg["pretrain"].get(
                "canonical_source_path",
                "data/features/phased_array/schema_v4/source/mosfet_canonical.h5")
        else:
            src_path = cfg["pretrain"].get("source_feature_path",
                                           "data/features/wheel/schema_v1/source_features.h5")
        featsS, hiS, bidS, tidxS, splitS = load_source(
            ROOT / src_path,
            cfg.get("pretrain", {}).get("source_id_field", "bearing_id"),
            cfg.get("source", {}).get("split", {}).get("val_device_ids", []) or [])
        tr, va, te = split_trajectories(
            n_traj, [tc["split"]["train"], tc["split"]["val"], tc["split"]["test"]], seed)
        lbT = rulT.copy()
        group_keys_T = tidT

    # 目标域 xT z-score 归一 (train 集) — encoder 预训练权重期望归一输入 (与 pretrain/transfer 一致)
    _tr_mask = np.isin(tidT, tr)
    _fm = xT[_tr_mask].mean(axis=0)
    _fs = xT[_tr_mask].std(axis=0) + 1e-6
    xT = (xT - _fm) / _fs

    def mask(ids):
        return np.isin(tidT, ids)

    # P1-2: RUL 归一因子用 config 固定物理上限 (跨 seed 可比); 缺失回退 train-only max
    if "rul_max_norm" in tc:
        rul_max = float(tc["rul_max_norm"])
    else:
        rul_max_train = float(rulT[mask(tr)].max())     # 回退 (跨 seed 不可比, 仅兼容旧 config)
        rul_max = rul_max_train
        print(f"  [warning] config 未设 transfer.rul_max_norm, 回退 train-only max={rul_max:.1f}")
    rulT = rulT / max(rul_max, 1.0)
    lbT = lbT / max(rul_max, 1.0)

    # 任务 3: 源 MMD 对齐窗只用 source train 器件 (val 器件不参与迁移对齐)
    _src_tr = np.asarray(splitS) == "train"
    if not _src_tr.any():
        raise ValueError("source train split empty; refusing val source into MMD")
    featsS = featsS[_src_tr]; hiS = hiS[_src_tr]
    bidS = np.asarray(bidS)[_src_tr]; tidxS = tidxS[_src_tr]

    # smoke 轨迹序列长 730 (P3a --fast); stride=50 保证每条轨迹产生 ~14 个窗口
    tstride = 50 if smoke else int(tc.get("target_stride", 50))

    def mkDS(ids):
        m = mask(ids)
        ds = TargetSeqDataset(xT[m], hiT[m], rulT[m], group_keys_T[m], L, K, stride=tstride,
                              event_observed=evT[m], rul_lower_bound=lbT[m],
                              damage_b=damageT[m] if damageT is not None else None)  # T13: damage_norm 供 ρ·L_phys; channel level None → dataset 占位 0 且 S3 use_phys=False
        if smoke:
            ds = Subset(ds, list(range(min(32, len(ds)))))
        return ds

    bs = 16 if smoke else int(cfg["pretrain"]["batch_size"])
    ltr = DataLoader(mkDS(tr), batch_size=bs, shuffle=True)
    lva = DataLoader(mkDS(va), batch_size=bs, shuffle=False)   # val (early stopping 用, 防 test 泄漏)
    lte = DataLoader(mkDS(te), batch_size=bs, shuffle=False)
    dsS = SourceWindowDataset(featsS, hiS, bidS, tidxS, L, stride=max(1, len(featsS) // 2000))
    if smoke:
        dsS = Subset(dsS, list(range(min(64, len(dsS)))))
    lS = DataLoader(dsS, batch_size=bs, shuffle=True)
    device = "cuda" if (cfg["pretrain"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    model = _build_model(cfg, featsS.shape[1], xT.shape[1], device, encoder_override=encoder_override)
    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]))
    mse = nn.MSELoss()
    e = 2 if smoke else int(tc.get("epochs_s2", 20))

    # target_only / random_frozen / random_full_mmd / random_full_nommd 不加载 checkpoint (随机初始化)
    # random_frozen: 冻结判 P0-3; random_full_mmd: 随机+S3+MMD; random_full_nommd: 随机+S3 无MMD (GPT §3 P0-2 MMD 归因)
    if mode not in ("target_only", "random_frozen", "random_full_mmd", "random_full_nommd",
                    "alpha_soft_finetune", "layerwise_finetune"):
        # source ckpt: M3 canonical 4 维重预训练产出 (覆盖旧 5 维); 飞轮用旧 source_*.pt
        # channel + service level 共用同一 ckpt (source schema 统一为 canonical 4 维)
        # 用 enc (encoder_override 优先) 而非 mc['encoder'], 支持 GRU/TCN 架构对齐实验
        # §4c k-shot 源域臂: 组名前缀决定源域 ckpt tag (startswith 防 _k{shot} 后缀 miss);
        # MMD 窗口仍统一 MOSFET canonical — 臂间唯一差异 = 初始化 ckpt, 归因纯净
        ckpt = str(CKPT_DIR / _source_ckpt_name(group_name, component, enc))
        if not Path(ckpt).exists():
            ckpt = str(CKPT_DIR / Path(ckpt).name.replace("_pretrain.pt", "_smoke.pt"))
        if Path(ckpt).exists():
            model.load_pretrained(ckpt, device)
        elif not smoke:
            # P2 #17: 正式实验 source_* 组缺 checkpoint 必须显式报错 (非静默跳过用随机权重冒充迁移)
            raise FileNotFoundError(
                f"source_* 组需要预训练 checkpoint 但未找到: {ckpt}; 先跑 "
                f"`python -m src.train.pretrain --config configs/{component}.yaml "
                f"{'--canonical' if component == 'phased_array' else ''}`")
        # smoke 模式允许无 checkpoint (encoder 随机初始化, 仅调试管线连通性)

    # A1 α-soft (§4d 四审方案): θ₀ = θ_rand + α·(θ_src − θ_rand) — 剂量控制源先验。
    # θ_rand = 当前构建权重: _build_model 调用点在 mode 分支之前且 alpha/random 两臂
    # 此前代码路径完全一致 (数据加载/split/k-shot 采样相同), 本分支内 torch.load/
    # load_state_dict/插值均不消耗 RNG → 同 seed 下 θ_rand 与 ch_random_full_finetune
    # 严格同源 (α=0 校验臂应逐位复现 random 臂); ckpt 名与 source 臂共用 helper,
    # α=1 端点 = ch_source_mmd_physics 同一 ckpt 文件。
    _alpha = _alpha_from_group(group_name)
    if _alpha is not None:
        assert mode == "alpha_soft_finetune", f"α 组 {group_name} 的 mode 应为 alpha_soft_finetune"
        ckpt_a = str(CKPT_DIR / _source_ckpt_name(group_name, component, enc))
        if not Path(ckpt_a).exists():
            ckpt_a = str(CKPT_DIR / Path(ckpt_a).name.replace("_pretrain.pt", "_smoke.pt"))
        if Path(ckpt_a).exists():
            sd = torch.load(ckpt_a, map_location=device)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            sd_enc = {k: v for k, v in sd.items() if k.startswith("encoder.")}
            if not sd_enc:
                raise ValueError(f"alpha_soft: ckpt 无 encoder.* 权重: {ckpt_a}")
            n_mixed = _interpolate_encoder(model, sd_enc, _alpha)
            print(f"  [α-soft] α={_alpha:.2f}: encoder 插值 {n_mixed}/{len(sd_enc)} 张量 "
                  f"(θ₀ = θ_rand + α·(θ_src − θ_rand), 源=MOSFET canonical)")
        elif not smoke:
            raise FileNotFoundError(f"alpha_soft 组需要源 checkpoint: {ckpt_a}")
        # smoke 无 ckpt: 保持随机初始化 (与 source 臂 smoke 纪律一致, 仅调试连通性)

    # A3 分层迁移：完整随机模型构建后，只覆盖严格 GRU 前缀；其余 θ_rand 保持不变。
    _layerwise_depth = _layerwise_depth_from_group(group_name)
    if _layerwise_depth is not None:
        assert mode == "layerwise_finetune", (
            f"分层组 {group_name} 的 mode 应为 layerwise_finetune")
        ckpt_lw = CKPT_DIR / _source_ckpt_name(group_name, component, enc)
        if not ckpt_lw.exists():
            ckpt_lw = CKPT_DIR / ckpt_lw.name.replace("_pretrain.pt", "_smoke.pt")
        if ckpt_lw.exists():
            sd = torch.load(ckpt_lw, map_location=device)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            sd_enc = {key: value for key, value in sd.items() if key.startswith("encoder.")}
            loaded = _load_layerwise_encoder(model, sd_enc, _layerwise_depth)
            print(f"  [layerwise] P{_layerwise_depth}: 加载 {len(loaded)}/10 GRU encoder 张量 "
                  "(其余参数保留 θ_rand)")
        elif not smoke:
            raise FileNotFoundError(f"layerwise 组需要源 checkpoint: {ckpt_lw}")
        # smoke 无 checkpoint 时保留随机初始化，仅用于验证训练编排连通性。

    tag = f"{group_name or mode} seed{seed}"
    if mode == "source_only":
        pass
    elif mode == "target_only":
        model.freeze_encoder(False)
        opt = torch.optim.Adam(model.parameters(), lr=float(tc["finetune_lr"]))
        _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, tag)
    elif mode == "source_finetune":
        model.freeze_encoder(True)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(tc["finetune_lr"]))
        _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, tag)
    elif mode == "random_frozen":
        # P0-3: 随机 encoder (未加载源 ckpt) + 冻结, 仅训 adapter+头。判 source_pretrain 负迁移
        # 是"冻结少参数限制"还是"源域权重干扰": 若 ≈ source_pretrain → 冻结所致 (非源域)
        model.freeze_encoder(True)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(tc["finetune_lr"]))
        _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, tag)
    elif mode == "random_full_nommd":
        # GPT §3 P0-2: 随机 encoder + S3 全微调 + L_phys + 无 MMD
        # (与 random_full_finetune 唯一差异 = MMD, 隔离源域窗口 MMD 贡献)
        model.freeze_encoder(True)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(tc["finetune_lr"]))
        _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, f"{tag} S2")
        model.freeze_encoder(False)
        opt = torch.optim.Adam(model.parameters(), lr=float(tc["finetune_lr"]))
        _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, f"{tag} S3",
                               use_phys=(rho_phys_loss > 0 and damageT is not None),
                               rho_phys=rho_phys_loss)   # 无 MMD; L_phys 仅在有 damage 真值时启用
    else:  # source_mmd_finetune (源 ckpt+S3) 或 random_full_mmd (随机+S3, GPT §3)
        model.freeze_encoder(True)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(tc["finetune_lr"]))
        _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, f"{tag} S2")
        model.freeze_encoder(False)
        opt = torch.optim.Adam(model.parameters(), lr=float(tc["finetune_lr"]))
        _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, f"{tag} S3",
                               use_mmd=True, src_iter=cycle(lS), bins=bins, mmd_lambda=mmd_lambda,
                               use_phys=(rho_phys_loss > 0 and damageT is not None),
                               rho_phys=rho_phys_loss)   # T13: ρ·L_phys (channel level 无真值时禁用)

    m = eval_test(model, lte, device, bias_diag=True)   # §4h: 误差方向诊断随 jsonl 落盘
    m["mode"] = mode
    m["group"] = group_name or mode
    m["seed"] = seed
    m["encoder"] = enc
    m["level"] = level
    # A1 预注册纪律: α 只在 train/val 上选 — 所有组记录 val_rmse (eval_test 同口径,
    # 仅失效轨迹), 端点臂 (random/source) 重跑时同样带上, α 选择集 {0,0.05,0.1,0.25,0.5,1} 全覆盖
    m["val_rmse"] = eval_test(model, lva, device)["rmse"]
    if _alpha is not None:
        m["alpha"] = _alpha
    if _layerwise_depth is not None:
        m["layerwise_depth"] = _layerwise_depth
    if k_shot is not None:
        m["k_shot"] = k_shot
    return m


def _paired_delta_ci(tgt_seeds, src_seeds):
    """计算 paired ΔRMSE = target − source (同 seed 配对), 返回统计字典。

    P1-1: 用样本 std (statistics.stdev) + t(n-1) 临界值 (小样本校正);
          n<10 时不 claim 确认性显著性 (exploratory_only=True)。
    """
    import math
    from scipy import stats as sp_stats
    common = sorted(set(tgt_seeds) & set(src_seeds))
    deltas = [tgt_seeds[s] - src_seeds[s] for s in common]
    if not deltas:
        return None
    n = len(deltas)
    mean_d = statistics.mean(deltas)
    std_d = statistics.stdev(deltas) if n > 1 else 0.0
    n_pos = sum(1 for d in deltas if d > 0)   # 正迁移 seed (迁移组 RMSE 更低)
    crit = float(sp_stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    se = std_d / math.sqrt(max(n, 1))
    ci_lo = mean_d - crit * se
    ci_hi = mean_d + crit * se
    return {
        "delta_mean": mean_d,
        "delta_std": std_d,
        "n_positive_seeds": n_pos,
        "n_seeds": n,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "ci_crosses_zero": (ci_lo <= 0.0 <= ci_hi),
        "exploratory_only": (n < 10),      # P1-1: n<10 不 claim 确认性显著性
    }


def aggregate(by, primary_source_group=None):
    """聚合各组跨种子指标 (mean ± 样本 std), 并计算 paired per-seed ΔRMSE。

    P0-2: 主比较对象 = config 指定的 primary_source_group (不从 test argmin 选);
          对所有 source_* 组分别算 paired ΔRMSE+CI, 存 agg["_paired"]["all_pairs"] (嵌套 {target:{source:stats}})。
    P1-1: 用样本 std + t(n-1) 临界值; n<10 标 exploratory_only。
    """
    agg = {}
    for group, ms in by.items():
        if not ms:
            continue
        agg[group] = {
            "rmse_mean": statistics.mean(m["rmse"] for m in ms),
            "rmse_std": statistics.stdev(m["rmse"] for m in ms) if len(ms) > 1 else 0.0,
            "phm_mean": statistics.mean(m["phm"] for m in ms),
            "mae_mean": statistics.mean(m["mae"] for m in ms),
            "censor_violation_mean": statistics.mean(m.get("censor_violation_rate", 0.0) for m in ms),
            "n": len(ms),
        }

    # P0-4 (复核第一条): 对每个 target_only* 组都配对 (不只 target_only_tcn),
    # 主基准 = rmse_mean 最优 target (避免"只对弱 TCN 配对"的选择性汇报)
    # T6.3: 同时识别旧组 (source_*/target_only_*) 和 ch_* 组 (ch_source_*/ch_target_only_*)
    src_groups = [g for g in agg if g.startswith("source_") or g.startswith("ch_source_")]
    tgt_groups = [g for g in agg if g.startswith("target_only") or g.startswith("ch_target_only")]
    tgt_g = min(tgt_groups, key=lambda g: agg[g]["rmse_mean"]) if tgt_groups else None

    paired_all = {}     # {target_group: {source_group: paired_stats}}
    for tg in tgt_groups:
        by_seed_tgt = {m["seed"]: m["rmse"] for m in by[tg]}
        paired_all[tg] = {}
        for sg in src_groups:
            by_seed_src = {m["seed"]: m["rmse"] for m in by[sg]}
            ps = _paired_delta_ci(by_seed_tgt, by_seed_src)
            if ps is not None:
                paired_all[tg][sg] = ps

    # 主比较对象 = config 指定 source 组; 主基准 = 最优 target
    main_src = primary_source_group if primary_source_group in src_groups else None
    # 初始化对照 (GPT §2): source_pretrain_finetune (源 ckpt+冻结) vs random_frozen (随机+冻结)
    # 两组唯一差别 = encoder 初始化; delta = source − random, CI 跨0 → 源初始化无显著收益
    # T6.3: 优先 ch_* 版本 (M7 通道级), 回退旧组 (PA6 服务级)
    init_ctrl = None
    for _src_fg, _rf in [("ch_source_pretrain_frozen_kall", "ch_random_frozen_kall"),
                         ("source_pretrain_finetune", "random_frozen")]:
        if _src_fg in by and _rf in by:
            _by_s = {m["seed"]: m["rmse"] for m in by[_src_fg]}
            _by_r = {m["seed"]: m["rmse"] for m in by[_rf]}
            init_ctrl = _paired_delta_ci(_by_s, _by_r)
            break
    # GPT §3 归因: source_mmd_physics (源ckpt+S3全微调) vs random_full_finetune (随机+S3全微调), 唯一差别=源 ckpt
    full_ctrl = None
    for _sm, _rf in [("ch_source_mmd_physics_kall", "ch_random_full_finetune_kall"),
                     ("source_mmd_physics", "random_full_finetune")]:
        if _sm in by and _rf in by:
            _by_m = {m["seed"]: m["rmse"] for m in by[_sm]}
            _by_rf = {m["seed"]: m["rmse"] for m in by[_rf]}
            full_ctrl = _paired_delta_ci(_by_m, _by_rf)
            break
    # GPT §3 P0-2: random_full_finetune (随机+S3+MMD) vs random_nommd (随机+S3 无MMD), 唯一差别=MMD
    mmd_ctrl = None
    for _mmd, _nomd in [("ch_random_full_finetune_kall", "ch_random_nommd_kall"),
                        ("random_full_finetune", "random_nommd")]:
        if _mmd in by and _nomd in by:
            _by_mmd = {m["seed"]: m["rmse"] for m in by[_mmd]}
            _by_nomd = {m["seed"]: m["rmse"] for m in by[_nomd]}
            mmd_ctrl = _paired_delta_ci(_by_mmd, _by_nomd)
            break
    agg["_paired"] = {
        "primary_target": tgt_g,           # 最优 target (P0-4 主基准)
        "target_groups": tgt_groups,
        "primary_source_group": main_src,
        "all_pairs": paired_all,           # {target: {source: stats}} 每 target 配对
        "init_control": init_ctrl,         # source_pretrain_finetune vs random_frozen (源初始化对照, 冻结)
        "full_control": full_ctrl,         # source_mmd_physics vs random_full_finetune (S3 全微调下源 ckpt 归因, GPT §3)
        "mmd_control": mmd_ctrl,           # random_full vs random_nommd (MMD 贡献归因, GPT §3 P0-2)
    }
    # T7: per-seed raw RMSE (供 model vs baseline per-seed paired; main 传 physical_by_seed 配合)
    agg["_raw_by"] = {g: {m["seed"]: m["rmse"] for m in ms} for g, ms in by.items()}
    # T6.4 level_control (M7 通道级矩阵核心论证量): 同 source ckpt + 同统计口径,
    # 唯一变量 = 迁移接口层级 (channel 器件层 vs service 服务层)。
    # level_control = service_rmse − channel_rmse (per-seed paired)
    #   正 → channel level 更优 (RMSE 更低, 预期: 器件层迁移接口更匹配 source 层级)
    #   负 → service level 更优
    level_ctrl = None
    ch_main = "ch_source_mmd_physics_kall"
    svc_main = "cross_level_transfer"
    if ch_main in by and svc_main in by:
        level_ctrl = _paired_delta_ci(
            {m["seed"]: m["rmse"] for m in by[svc_main]},   # target = service
            {m["seed"]: m["rmse"] for m in by[ch_main]})    # source = channel
        # _paired_delta_ci 返回 delta = target − source = service − channel
    agg["_level_control"] = {"channel_group": ch_main, "service_group": svc_main,
                             "paired": level_ctrl}
    # T6.2 k-shot 维度: 对每个 k 算 transfer_gain = ch_target − ch_source_mmd_physics
    k_shot_stats = {}
    for g in agg:
        if g.startswith("ch_") and g.endswith("_k1"):
            k_tag = "k1"
        elif g.startswith("ch_") and g.endswith("_k3"):
            k_tag = "k3"
        elif g.startswith("ch_") and g.endswith("_k5"):
            k_tag = "k5"
        elif g.startswith("ch_") and g.endswith("_kall"):
            k_tag = "kall"
        else:
            continue
        k_shot_stats.setdefault(k_tag, {})[g] = agg[g]["rmse_mean"]
    agg["_k_shot_rmse"] = k_shot_stats
    return agg


def write_results(agg, physical, path, smoke, n_seeds, group_names, component,
                  primary_model_group=None, physical_by_seed=None):
    comp_label = "飞轮" if component == "wheel" else "相控阵天线"
    note = ("**注意: 基于 smoke/合成数据, 非最终数字; 正式迁移增益待真实数据预训练**\n"
            if smoke else "")
    lines = [
        f"# 对比实验结果 — {comp_label} (plan Phase 6 / §8)\n\n",
        note,
        f"每组 {n_seeds} 个随机种子, 报 mean±std。RMSE/PHM/MAE **仅失效轨迹** (P0-2: 删失无精确 RUL); 删失轨迹报下界违反率。\n\n",
        "| 实验组 | RMSE (mean±std) | PHM Score | MAE | 删失违反率 |\n",
        "|--------|-----------------|-----------|-----|-----------|\n",
    ]
    for g in group_names:
        label = LABELS.get(g, g)
        if g in agg:
            a = agg[g]
            lines.append(f"| {label} | {a['rmse_mean']:.4f}±{a['rmse_std']:.4f} | "
                         f"{a['phm_mean']:.2f} | {a['mae_mean']:.4f} | {a['censor_violation_mean']:.3f} |\n")
        elif g in TIMESFM_GROUPS:
            lines.append(f"| {label} | — | — | — | *(PA7 待启用)* |\n")
        elif g == "physical_extrap":
            if physical and "constant_zero" in physical:   # 相控阵四基线嵌套 (P0 实验 C)
                for bn in ["constant_zero", "constant", "hi_extrap", "arrhenius"]:
                    m = physical[bn]
                    lines.append(f"| {bn} (非学习) | {m['rmse']:.4f} | {m['phm']:.2f} | {m['mae']:.4f} |\n")
            elif physical:
                lines.append(f"| 物理外推基线 | {physical['rmse']:.4f} | {physical['phm']:.2f} | "
                             f"{physical['mae']:.4f} |\n")
            else:
                lines.append(f"| 物理外推基线 | — | — | — | *(待实现)* |\n")
    lines.append("\n**验收**:\n")
    # 验收: 迁移组 vs 同架构 target_only (迁移组均 TCN 编码器)
    src_in_agg = [g for g in group_names if g.startswith("source_") and g in agg]
    tgt_g = agg.get("_paired", {}).get("primary_target") or \
            next((g for g in group_names if g.startswith("target_only")), None)
    # P0-2: 主迁移组 = config 指定 (不从 test argmin 选); 回退到 source_* 均值最小者 (仅展示用)
    paired_info = agg.get("_paired", {})
    main_g = paired_info.get("primary_source_group")
    if main_g is None:
        main_g = min(src_in_agg, key=lambda g: agg[g]["rmse_mean"]) if src_in_agg else None
    if main_g and tgt_g in agg:
        gain = agg[tgt_g]["rmse_mean"] - agg[main_g]["rmse_mean"]
        lines.append(f"- 迁移增益 ({tgt_g} − 主迁移 {main_g} RMSE, 组均值差): {gain:+.4f}\n")
    # P0-2/P1-1: paired per-seed ΔRMSE + CI (主比较对象 = config 指定组)
    all_sources = paired_info.get("all_pairs", {}).get(tgt_g, {})   # P0-4: 主(最优)target 的 source 配对
    if main_g and main_g in all_sources:
        ps = all_sources[main_g]
        # delta = target − 主迁移: >0 → 迁移组 RMSE 更低 (正向迁移); <0 → 迁移组 RMSE 更高 (负迁移)
        # 修复: 原三元逻辑把"CI 不跨0 但负向"误报为"CI 跨0" (GPT 评审 §8); 改四象限准确描述
        if ps["ci_crosses_zero"]:
            sig_label = "不显著 (CI 跨 0)"
        elif ps["delta_mean"] > 0:
            sig_label = "方向正向 (CI 不跨 0, 全正侧: 迁移组 RMSE 更低)"
        else:
            sig_label = "方向负向 (CI 不跨 0, 全负侧: 迁移组 RMSE 更高 → 负迁移)"
        expl_note = " [探索性分析, n<10 不作确认性结论]" if ps["exploratory_only"] else ""
        lines.append(
            f"- 配对 ΔRMSE (target − 主迁移 {main_g}, per-seed){expl_note}: "
            f"{ps['delta_mean']:+.4f} ± {ps['delta_std']:.4f}, "
            f"正向 seed {ps['n_positive_seeds']}/{ps['n_seeds']}, "
            f"95% CI [t({max(ps['n_seeds']-1,1)})] [{ps['ci95_lo']:+.4f}, {ps['ci95_hi']:+.4f}] → {sig_label}\n"
        )
        lines.append(
            "  - 注: CI 用样本 std + t(n-1) 临界值 (小样本校正); "
            "n<10 时仅供方向性参考, 不作严格假设检验结论。\n"
        )
    elif main_g and tgt_g in agg:
        lines.append("  - 配对统计 TODO: seed 数不足或组缺失, 无法计算 CI。\n")
    # 附表: 所有 source 组各自的 Δ±CI (探索性分析, P0-2)
    if len(all_sources) > 1:
        lines.append("  - 各迁移组配对统计 (探索性分析):\n")
        for g in src_in_agg:
            if g in all_sources:
                ps = all_sources[g]
                lines.append(
                    f"    · {g}: Δ={ps['delta_mean']:+.4f} ± {ps['delta_std']:.4f}, "
                    f"CI [{ps['ci95_lo']:+.4f}, {ps['ci95_hi']:+.4f}], "
                    f"正向 {ps['n_positive_seeds']}/{ps['n_seeds']}\n"
                )
    # 初始化对照 (GPT §2): source_pretrain_finetune vs random_frozen 唯一差别=encoder 初始化
    init_ctrl = paired_info.get("init_control")
    if init_ctrl is not None:
        _ic_expl = " [探索性分析, n<10 不作确认性结论]" if init_ctrl["exploratory_only"] else ""
        if init_ctrl["ci_crosses_zero"]:
            _ic_sig = "无显著差异 (CI 跨 0) → 源初始化无可度量收益"
        elif init_ctrl["delta_mean"] < 0:
            _ic_sig = "源初始化显著更优 (CI 全负侧)"
        else:
            _ic_sig = "源初始化显著更差 (CI 全正侧 → 负迁移)"
        lines.append(
            f"- 初始化对照 (source_pretrain_finetune − random_frozen, per-seed){_ic_expl}: "
            f"{init_ctrl['delta_mean']:+.4f} ± {init_ctrl['delta_std']:.4f}, "
            f"95% CI [{init_ctrl['ci95_lo']:+.4f}, {init_ctrl['ci95_hi']:+.4f}] → {_ic_sig}\n"
        )
    # GPT §3 归因: source_mmd_physics vs random_full_finetune (S3 全微调下源 ckpt 归因)
    full_ctrl = paired_info.get("full_control")
    if full_ctrl is not None:
        _fc_expl = " [探索性分析, n<10 不作确认性结论]" if full_ctrl["exploratory_only"] else ""
        if full_ctrl["ci_crosses_zero"]:
            _fc_sig = "无显著差异 (CI 跨 0) → source_mmd 追平来自 S3 全微调, 非源 ckpt"
        elif full_ctrl["delta_mean"] < 0:
            _fc_sig = "源 ckpt 显著更优 (CI 全负侧) → 追平有源 ckpt 贡献"
        else:
            _fc_sig = "源 ckpt 显著更差 (CI 全正侧 → 源 ckpt 有害)"
        lines.append(
            f"- S3 全微调归因 (source_mmd_physics − random_full_finetune, per-seed){_fc_expl}: "
            f"{full_ctrl['delta_mean']:+.4f} ± {full_ctrl['delta_std']:.4f}, "
            f"95% CI [{full_ctrl['ci95_lo']:+.4f}, {full_ctrl['ci95_hi']:+.4f}] → {_fc_sig}\n"
        )
    # GPT §3 P0-2: MMD 贡献归因 (random_full_finetune − random_nommd, 唯一差别=MMD)
    mmd_ctrl = paired_info.get("mmd_control")
    if mmd_ctrl is not None:
        _mc_expl = " [探索性分析, n<10 不作确认性结论]" if mmd_ctrl["exploratory_only"] else ""
        if mmd_ctrl["ci_crosses_zero"]:
            _mc_sig = "无显著差异 (CI 跨 0) → MMD 无可度量贡献"
        elif mmd_ctrl["delta_mean"] < 0:
            _mc_sig = "MMD 显著有益 (CI 全负侧, random_full 优于 random_nommd)"
        else:
            _mc_sig = "MMD 显著有害 (CI 全正侧)"
        lines.append(
            f"- MMD 归因 (random_full_finetune − random_nommd, per-seed){_mc_expl}: "
            f"{mmd_ctrl['delta_mean']:+.4f} ± {mmd_ctrl['delta_std']:.4f}, "
            f"95% CI [{mmd_ctrl['ci95_lo']:+.4f}, {mmd_ctrl['ci95_hi']:+.4f}] → {_mc_sig}\n"
        )
    # P2-2: std 说明区分 MMD / 非 MMD 组
    for g in src_in_agg:
        a = agg[g]
        if a["rmse_std"] > 0.05:
            if g.startswith("source_mmd_"):
                note = " ⚠ std 大 (MMD 对齐不稳定, 待调 mmd_lambda/数值稳定性)"
            else:
                note = " ⚠ std 大 (seed 间方差大, 训练稳定性待查)"
        else:
            note = ""
        lines.append(f"  - {g}: {a['rmse_mean']:.4f}±{a['rmse_std']:.4f}{note}\n")
    # T19 (GPT 审阅): 主模型 = config primary_model_group 固定 (target_only_gru), 消除"机械选 min →
    # test selection bias" (旧版曾机械选 random_full 作主模型, 不当); 回退 min 仅当 config 未指定。
    _model_groups = [g for g in group_names if g in agg and g != "physical_extrap"]
    if primary_model_group and primary_model_group in agg:
        main_model_g = primary_model_group
    else:
        main_model_g = min(_model_groups, key=lambda g: agg[g]["rmse_mean"]) if _model_groups else None
    if main_model_g and physical:
        if "constant_zero" in physical:   # 相控阵: 主模型 vs 最强非学习基线
            base_names = ["constant_zero", "constant", "hi_extrap", "arrhenius"]
            # T19: 基线报 per-seed mean (五划分, GPT 审阅 arrhenius 0.3446 非 0.3203 单 seed42)
            if physical_by_seed:
                _sds = list(physical_by_seed.values())
                base_mean = {bn: float(np.mean([sd[bn]["rmse"] for sd in _sds]))
                             for bn in base_names if bn in _sds[0]}
            else:
                base_mean = {bn: physical[bn]["rmse"] for bn in base_names if bn in physical}
            best_name = min(base_mean, key=lambda bn: base_mean[bn])
            best_base = base_mean[best_name]
            better = agg[main_model_g]["rmse_mean"] < best_base
            lines.append(f"- 主模型 ({main_model_g}) vs 最强非学习基线 ({best_name}, 5-seed mean RMSE={best_base:.4f}): "
                         f"{'优于 ✓' if better else '⚠ 未优于'} "
                         f"({agg[main_model_g]['rmse_mean']:.4f} vs {best_base:.4f})\n")
        else:
            better = agg[main_model_g]["rmse_mean"] < physical["rmse"]
            lines.append(f"- 主模型 ({main_model_g}) vs 物理基线 RMSE: {'优于' if better else '待正式数据'} "
                         f"({agg[main_model_g]['rmse_mean']:.4f} vs {physical['rmse']:.4f})\n")
    # T7 (GPT §7): 主模型 vs 最强基线 per-seed paired CI (同 split 配对, 单点比较的严谨补充)
    _mvb = agg.get("_model_vs_baseline")
    if _mvb is not None:
        _ps = _mvb["paired"]
        _expl = " [探索性分析, n<10 不作确认性结论]" if _ps["exploratory_only"] else ""
        # delta = model − baseline: <0 model 更优 (RMSE 更低)
        if _ps["ci_crosses_zero"]:
            _sig = "无显著差异 (CI 跨 0)"
        else:
            _sig = "显著优于基线 (CI 全负侧)" if _ps["delta_mean"] < 0 else "显著劣于基线 (CI 全正侧)"
        lines.append(
            f"- 主模型 ({_mvb['model_group']}) vs 最强基线 ({_mvb['baseline']}, per-seed paired){_expl}: "
            f"Δ={_ps['delta_mean']:+.4f} ± {_ps['delta_std']:.4f}, "
            f"95% CI [{_ps['ci95_lo']:+.4f}, {_ps['ci95_hi']:+.4f}] → {_sig}\n"
        )
    # P1 修正 (GPT 评审 §8): 组名 ↔ 训练协议实现说明, 消除"名实不副"误导 (仅相控阵)
    if component == "phased_array":
        lines.append(
            "- 实现说明 (组名 ↔ 训练协议):\n"
            "  · source_pretrain_finetune: 加载源 ckpt + **encoder 冻结** (仅训 adapter+头; 名含 finetune 实为 frozen)\n"
            "  · source_mmd_physics: S2 冻结 → S3 **全量微调 + MMD 对齐 + ρ·L_phys 物理一致性** (T13: ρ·MSE(hi_pred, damage_norm) Arrhenius+CM 损伤监督)\n"
            "  · random_frozen: 随机 encoder + 冻结 (P0-3 初始化对照, 判源负迁移归属)\n"
            "  · ch_* (M7 通道级): ChannelSeqDataset + canonical 4 维; level=channel\n"
            "  · cross_level_transfer (M7 层级消融): 旧服务级口径 (build_array_hi 12 维); level=service\n"
        )
    # T6.4 level_control (M7 核心论证量): channel level vs service level RMSE 配对
    _lc = agg.get("_level_control")
    if _lc and _lc.get("paired") is not None:
        _ps = _lc["paired"]
        _expl = " [探索性分析, n<10 不作确认性结论]" if _ps["exploratory_only"] else ""
        # delta = service − channel: 正 → channel 更优 (RMSE 更低); 负 → service 更优
        if _ps["ci_crosses_zero"]:
            _sig = "无显著差异 (CI 跨 0)"
        elif _ps["delta_mean"] > 0:
            _sig = "channel level 显著更优 (CI 全正侧, service RMSE 更高)"
        else:
            _sig = "service level 显著更优 (CI 全负侧)"
        lines.append(
            f"- **level_control** ({_lc['service_group']} − {_lc['channel_group']}, per-seed paired){_expl}: "
            f"Δ={_ps['delta_mean']:+.4f} ± {_ps['delta_std']:.4f}, "
            f"95% CI [{_ps['ci95_lo']:+.4f}, {_ps['ci95_hi']:+.4f}] → {_sig}\n"
        )
        lines.append(
            "  - 语义: 唯一变量=迁移接口层级 (channel 器件层 vs service 服务层), "
            "同 source ckpt + 同统计口径; 正向=channel 层级选对了 (器件层迁移接口匹配 source)\n"
        )
    # T6.2 k-shot 维度: 各 k 的 RMSE 表 (迁移随少样本变化)
    _ks = agg.get("_k_shot_rmse", {})
    if _ks:
        lines.append("- k-shot 维度 (通道级 RMSE 随 k 变化):\n")
        lines.append("  | k | ch_target_only_gru | ch_source_mmd_physics | transfer_gain |\n")
        lines.append("  |---|---|---|---|\n")
        for k_tag in ["k1", "k3", "k5", "kall"]:
            if k_tag not in _ks:
                continue
            row = _ks[k_tag]
            tgt = row.get(f"ch_target_only_gru_{k_tag}", "—")
            src = row.get(f"ch_source_mmd_physics_{k_tag}", "—")
            gain = (tgt - src) if isinstance(tgt, float) and isinstance(src, float) else "—"
            gain_str = f"{gain:+.4f}" if isinstance(gain, float) else str(gain)
            tgt_str = f"{tgt:.4f}" if isinstance(tgt, float) else str(tgt)
            src_str = f"{src:.4f}" if isinstance(src, float) else str(src)
            lines.append(f"  | {k_tag} | {tgt_str} | {src_str} | {gain_str} |\n")
    # TODO (可选增强): 轨迹级 bootstrap CI (按 test 轨迹重采样算 RMSE 置信区间),
    # 当前未实现 — seed 数足够时 paired CI 已够审查, bootstrap 留作正式阶段补充。
    Path(path).write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--drop-features", default=None,
                    help="GPT §6 消融: 从 x_global 删除的列名 (逗号分隔, 如 M_link_dB,SLL_dB,theta_err_deg); 覆盖 config ablation_drop_features")
    ap.add_argument("--level", default=None, choices=["channel", "service", "auto"],
                    help="T6.3/M7: 强制 level (channel=ChannelSeqDataset+canonical; service=旧 TargetSeqDataset); "
                         "auto=按组名前缀 ch_ 决定 (默认)")
    ap.add_argument("--k-shot", default=None,
                    help="T6.2/M7 k-shot 协议: 逗号分隔 k 值 (如 1,3,5,all); 每个跑一组; "
                         "仅 channel level 生效 (service level 无 k-shot 概念)")
    ap.add_argument("--groups", default=None,
                    help="显式组列表 (逗号分隔, 覆盖 config; §4c k-shot 源域臂实验用)")
    ap.add_argument("--output-dir", default=None,
                    help="P0-2: 产物输出目录 (results_*.md 与 all_metrics_*.json 写入指定目录, "
                         "供 Docker volume 挂载回收); 不指定则写 checkpoints/ 和 docs/")
    ap.add_argument("--workers", type=int, default=1,
                    help="组-seed 级 CPU 并行进程数 (2026-08-18 加固): >1 时 ProcessPoolExecutor spawn 并行, "
                         "每 worker 限 torch CPU 线程; GPU 显存 ≈ workers × 单进程占用, "
                         "与外部实验共存时按显存余量选 (如 --workers 2)")
    ap.add_argument("--threads-per-worker", type=int, default=4,
                    help="每 worker 进程 torch CPU 线程上限 (默认 4); 防 多进程×全核线程 资源爆炸")
    args = ap.parse_args()
    # P0-2: 产物持久化 — --output-dir 指定产物目录 (Docker 挂载可回收); None=旧位置 (docs/ + checkpoints/)
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    if args.drop_features is not None:
        cfg.setdefault("transfer", {})["ablation_drop_features"] = [
            x.strip() for x in args.drop_features.split(",") if x.strip()]
    component = Path(args.config).stem
    n_seed = 1 if args.smoke else args.seeds
    exp_cfg = cfg.get("experiments", {})
    # group 选择优先级: --level 显式 → 对应 *_groups; 否则默认 groups (PA6 旧组)
    if args.groups:
        group_names = [g.strip() for g in args.groups.split(",") if g.strip()]
    elif args.level == "channel" or (args.k_shot and args.level != "service"):
        # T6.4: channel level 自动合并 service_groups (cross_level_transfer) 作层级消融对照,
        # 让 level_control 在同一次跑算出 (by 字典同时有 ch_* + cross_level_transfer)
        ch_groups = list(exp_cfg.get("channel_groups", exp_cfg.get("groups", [])))
        svc_groups = list(exp_cfg.get("service_groups", []))
        group_names = ch_groups + svc_groups
    elif args.level == "service":
        group_names = exp_cfg.get("service_groups", exp_cfg.get("groups", []))
    else:
        group_names = exp_cfg.get("groups",
                        ["target_only", "source_only", "source_finetune", "source_mmd_finetune"])
    if not group_names:
        print("!! 无 group 可跑 (config experiments.groups/channel_groups/service_groups 都空)")
        return

    # T6.2 k-shot 展开: k_shot_list=[None] (单跑, 不带 k-shot 维度) 或 [1,3,5,'all'] (每个 k 一组)
    if args.k_shot:
        k_shot_list = [("all" if x.strip().lower() == "all" else int(x.strip()))
                       for x in args.k_shot.split(",") if x.strip()]
    else:
        k_shot_list = [None]

    by = {}   # group_name (或 group_name+k=..) -> [metrics]
    _failures = []   # P0-2: 失败传播 — 记录所有失败臂, main 末尾任一失败则 sys.exit(1)

    # 任务收集: (mode, seed, cfg, smoke, enc_ov, component, tag_name, level, k_shot)
    tasks = []
    for gname in group_names:
        if gname == "physical_extrap":
            continue                              # 物理基线单独 evaluate (非学习组)
        mode, enc_ov, enabled, reason = _resolve_group(gname, cfg)
        if not enabled:
            print(f">> {gname:28s}: skip ({reason})")
            continue
        # level 决定: cross_level_transfer 固定 service (层级消融, 不受 --level 覆盖;
        # 否则 --level channel 会把 cross_level_transfer 也跑成 channel level → level_control 失效);
        # 其余: --level 强制, 否则按组名前缀 (ch_* → channel, 旧组 → service)
        if gname == "cross_level_transfer":
            level = "service"
        elif args.level and args.level != "auto":
            level = args.level
        else:
            level = _group_level(gname)
        for k_shot in k_shot_list:
            # k-shot 仅对 channel level 生效; service level 忽略 k_shot (单跑)
            if level == "service" and k_shot is not None and len(k_shot_list) > 1:
                continue
            tag_name = gname if k_shot is None else f"{gname}_k{k_shot}"
            by[tag_name] = []
            for s in range(n_seed):
                seed = cfg["seed"] + s
                tasks.append((mode, seed, cfg, args.smoke, enc_ov, component,
                              tag_name, level, k_shot))

    # 增量落盘 + 恢复: jsonl 记录已完成 组-seed, 中断重启只补缺口 (2026-08-18 事故加固);
    # 旧记录须重载入 by 聚合器 (GPT 四审+1 修正) — 只跳过不重载会让恢复跑的
    # aggregate/report 缺已完成臂, 产出看似完整实则缺数据的报告
    jsonl_path = (out_dir if out_dir is not None else CKPT_DIR) / "results_partial.jsonl"
    prior_records = _load_jsonl_records(jsonl_path)
    if prior_records:
        _n_total = len(tasks)
        _task_keys = {(t[6], t[1]) for t in tasks}
        tasks = [t for t in tasks if (t[6], t[1]) not in prior_records]
        _n_reloaded = 0
        for (_g, _s), _m in prior_records.items():
            if _g in by and (_g, _s) in _task_keys:
                by[_g].append(_m)
                _n_reloaded += 1
        print(f">> [resume] {jsonl_path}: 旧记录 {_n_reloaded} 条重载入聚合器, "
              f"剩余重跑 {len(tasks)}/{_n_total}")

    def _record(m):
        by[m["group"]].append(m)
        _append_jsonl(jsonl_path, m)          # m 含 group/seed (run_one_group 已写入)
        print(f">> {m['group']:30s} seed{m['seed']} ({m['encoder']}, {m['level']}): "
              f"RMSE={m['rmse']:.4f} PHM={m['phm']:.2f} MAE={m['mae']:.4f}"
              + (f" k={m['k_shot']}" if m.get("k_shot") is not None else ""))

    if args.workers > 1:
        # CPU 并行: 组-seed 级多进程 (spawn), 每 worker 限 torch 线程;
        # GPU 显存 ≈ workers × 单进程占用, 与外部实验共存时按余量选 workers
        import concurrent.futures
        torch.set_num_threads(args.threads_per_worker)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, initializer=_worker_init,
                initargs=(args.threads_per_worker,)) as ex:
            futs = {ex.submit(_run_one_task, t): t for t in tasks}
            for fut in concurrent.futures.as_completed(futs):
                t = futs[fut]
                try:
                    _record(fut.result())
                except Exception as exc:    # noqa: BLE001
                    print(f"!! {t[6]} seed{t[1]} 失败: {exc}")
                    _failures.append(f"{t[6]}/seed{t[1]}: {exc}")   # P0-2: 计数, main 末尾非零退出
    else:
        for mode, seed, _cfg, smoke, enc_ov, comp, tag_name, level, k_shot in tasks:
            try:
                m = run_one_group(mode, seed, cfg, smoke=smoke,
                                  encoder_override=enc_ov, component=comp,
                                  group_name=tag_name, level=level, k_shot=k_shot)
                _record(m)
            except Exception as exc:    # noqa: BLE001
                print(f"!! {tag_name} seed{seed} 失败: {exc}")
                _failures.append(f"{tag_name}/seed{seed}: {exc}")   # P0-2: 计数, main 末尾非零退出

    physical_by_seed = None
    if "physical_extrap" in group_names:
        if component == "phased_array":
            # T7 (GPT §7): baseline per-seed (arrhenius 纯物理, per-seed 只差 split; 与 model 同 split 配对)
            physical_by_seed = {}
            for s in range(n_seed):
                sd = cfg["seed"] + s
                physical_by_seed[sd] = evaluate_phased_array_baselines(cfg, sd)
            physical = physical_by_seed.get(cfg["seed"]) or next(iter(physical_by_seed.values()))
        else:
            physical = evaluate_physical(cfg, cfg["seed"])
    else:
        physical = None
    primary_src = cfg.get("experiments", {}).get("primary_source_group")
    agg = aggregate(by, primary_source_group=primary_src)
    # T7 (GPT §7): 主模型 vs 最强基线 per-seed paired CI (model[seed] − baseline[seed], 同 split 公平配对)
    if physical_by_seed and component == "phased_array":
        _model_groups = [g for g in group_names if g in agg and g != "physical_extrap"]
        _pm = cfg.get("experiments", {}).get("primary_model_group")   # T19 固定主模型 (消除 test selection bias)
        _mm = (_pm if _pm and _pm in agg
               else (min(_model_groups, key=lambda g: agg[g]["rmse_mean"]) if _model_groups else None))
        _raw = agg.get("_raw_by", {})
        if _mm and _mm in _raw:
            _base_names = ["constant_zero", "constant", "hi_extrap", "arrhenius"]
            _ph0 = next(iter(physical_by_seed.values()))
            if "constant_zero" in _ph0:
                _best_bn = min(_base_names, key=lambda bn: sum(
                    physical_by_seed[s][bn]["rmse"] for s in physical_by_seed) / len(physical_by_seed))
                _common = sorted(set(_raw[_mm]) & set(physical_by_seed))
                if len(_common) >= 2:
                    _dM = {s: _raw[_mm][s] for s in _common}
                    _dB = {s: physical_by_seed[s][_best_bn]["rmse"] for s in _common}
                    agg["_model_vs_baseline"] = {
                        "model_group": _mm,
                        "baseline": _best_bn,
                        "paired": _paired_delta_ci(_dM, _dB),
                    }
    # 组件级 + smoke 分文件, 防飞轮/相控阵互相覆盖 + smoke 覆盖正式 (修 progress PA6 待办)
    out_name = f"results_{component}{'_smoke' if args.smoke else ''}.md"
    _primary_model = cfg.get("experiments", {}).get("primary_model_group")
    # T6.2 k-shot 后缀: by 字典 key 是 group_name_k{shot}, write_results 需用匹配的 key
    if args.k_shot and len(k_shot_list) > 1:
        report_group_names = []
        for gname in group_names:
            for ks in k_shot_list:
                ks_tag = "all" if ks == "all" else str(ks)
                report_group_names.append(f"{gname}_k{ks_tag}")
    elif args.k_shot:
        # 单 k (如 --k-shot all): 全部 group_name 加 _k{shot} 后缀
        ks_tag = "all" if k_shot_list[0] == "all" else str(k_shot_list[0])
        report_group_names = [f"{gname}_k{ks_tag}" for gname in group_names]
    else:
        report_group_names = group_names
    # 主模型/源组 config 指定也要加 _k{shot} 后缀 (channel level)
    if args.k_shot:
        ks_tag = "all" if k_shot_list[0] == "all" else str(k_shot_list[0])
        if _primary_model and _primary_model.startswith("ch_"):
            _primary_model = f"{_primary_model}_k{ks_tag}"
        if primary_src and primary_src.startswith("ch_"):
            primary_src = f"{primary_src}_k{ks_tag}"
    # P0-2: 失败传播 — 任一实验臂失败则非零退出 (Docker/CI 据此判定复现失败),
    # 且不写结果文件 (避免产出看似完整实则缺臂的报告)
    if _failures:
        print(f"\n!! 共 {len(_failures)} 个实验臂失败:")
        for f in _failures:
            print(f"   - {f}")
        sys.exit(1)

    # P0-2: 产物持久化 — --output-dir 时 md+json 都写入指定目录 (Docker 挂载可回收);
    # 不指定则保持旧行为 (results→docs/, all_metrics→checkpoints/)
    results_dir = out_dir if out_dir is not None else ROOT / "docs"
    out_path = results_dir / out_name
    write_results(agg, physical, out_path, args.smoke, n_seed,
                  report_group_names, component, primary_model_group=_primary_model,
                  physical_by_seed=physical_by_seed)
    if out_dir is not None:
        metrics_path = out_dir / f"all_metrics_{component}{'_smoke' if args.smoke else ''}.json"
    else:
        CKPT_DIR.mkdir(exist_ok=True)
        metrics_path = CKPT_DIR / f"all_metrics_{component}.json"
    metrics_path.write_text(
        json.dumps({"agg": agg, "physical": physical, "physical_by_seed": physical_by_seed,
                    "groups": report_group_names},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f">> results -> {out_path}")
    print(f">> all_metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
