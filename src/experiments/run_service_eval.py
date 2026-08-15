"""experiments/run_service_eval.py

T8 服务层评估 (M8 加分项): 由通道级模型预测前推服务寿命分布。

对 test 划分的每条轨迹, 在评估时点 t∈{0.3,0.5,0.7}×EOL_svc 取模型预测:
  1. 16 子阵各自 x_ch 窗 → 模型预测 hi 序列 → 提取 z_hat (当前损伤) + rate_hat (速率)
  2. dose 由遥测 Tj/duty 按 Arrhenius 算 (标称 Ea, 非真值)
  3. service_rollout.rollout_mc 前推 → 服务 EOL 分布 (P10/P50/P90)
  4. 报告: 服务 EOL 绝对误差 / P10-P90 覆盖率 / 瓶颈子阵 top-3 命中率 / 降额分析

诚实性 (plan §T8): 服务事件 MAE 为 null (无有效事件) 时写 null 非 0;
样本数为 0 时标注"服务层未验证"。

用法:
  python -m src.experiments.run_service_eval --config configs/phased_array.yaml \
      --ckpt checkpoints/source_phased_array_tcn_pretrain.pt --seeds 3
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
from src.transfer.adapter import TransferModel                               # noqa: E402
from src.transfer.train_transfer import split_trajectories                   # noqa: E402
from src.physics.service_rollout import ArrayTwin, rollout_mc                # noqa: E402

K_BOLTZMANN_eVperK = 8.617333262e-5
COL_T_DEV = 1     # canonical x_ch 第 1 维 = T_dev_C (Arrhenius 输入)
COL_DUTY = 2      # 第 2 维 = duty


def _build_model(cfg, n_features, device):
    mc = cfg["model"]
    tc = cfg["transfer"]
    return TransferModel(
        encoder_type=mc["encoder"], n_features=n_features, n_target=n_features,
        channels=mc["tcn"]["channels"], kernel_size=mc["tcn"]["kernel_size"],
        num_blocks=mc["tcn"]["num_blocks"], dropout=mc["tcn"]["dropout"],
        latent_dim=mc["latent_dim"], adapter_hidden=tc["adapter_hidden"]).to(device)


def _arrhenius_dose(Tj_K: np.ndarray, duty: np.ndarray, Ea_eV: float) -> np.ndarray:
    """归一化剂量率 r = duty · exp(-Ea/(k_B·Tj)) (标称 Ea, 非真值)。"""
    return duty * np.exp(-Ea_eV / (K_BOLTZMANN_eVperK * Tj_K))


def _extract_z_rate_sub(model, x_ch_sub: np.ndarray, L: int, device) -> tuple[float, float]:
    """对单子阵 x_ch (T_seg, 4) 在窗末提取 z_hat + 滑窗速率。

    返回 (z_hat, rate); T_seg < 2 时返回 (0.0, 1e-9)
    """
    T_seg = x_ch_sub.shape[0]
    if T_seg < 2:
        return 0.0, 1e-9
    model.eval()
    with torch.no_grad():
        stride = max(1, (T_seg - L) // 8) if T_seg > L else 1
        hi_preds = []
        for start in range(0, max(1, T_seg - L + 1), stride):
            win = x_ch_sub[start:start + L] if start + L <= T_seg else x_ch_sub[-L:]
            if win.shape[0] < L:
                win = np.pad(win, ((L - win.shape[0], 0), (0, 0)))
            x = torch.from_numpy(win.astype(np.float32))[None].to(device)   # (1,L,4)
            hi_p, _, _ = model(x)
            hi_preds.append(float(hi_p.cpu().numpy().ravel()[-1]))
    hi_arr = np.asarray(hi_preds)
    if len(hi_arr) < 2:
        return 0.0, 1e-9
    z_hat = float(hi_arr[-1])
    rate = 1e-9
    if len(hi_arr) >= 3 and np.ptp(hi_arr) > 1e-9:
        slope = float(np.polyfit(np.arange(len(hi_arr)), hi_arr, 1)[0])
        rate = max(slope, 1e-9)
    return z_hat, rate


def evaluate_service_level(cfg: dict, ckpt_path: Path, seed: int,
                           n_eval_traj: int = 30, horizon: int = 4088,
                           n_mc: int = 200, verbose: bool = True) -> dict:
    """对 test 每条轨迹在 t∈{0.3,0.5,0.7}×EOL_svc 评估服务 EOL 前推。

    返回 {per_traj: [...], aggregate: {...}}; 样本数 0 时指标写 null。
    """
    set_seed(seed, cfg["reproducibility"]["deterministic"])
    ch_cfg = cfg["channel_level"]
    tc = cfg["transfer"]
    mc = cfg["model"]
    L = int(mc["input_len_L"])
    Ea_eV = float(np.mean(cfg["sim"]["physics"]["Ea_eV_range"]))
    service_limits = {
        "SLL_max_dB": float(cfg["sim"]["service_limits"]["SLL_max_dB"]),
        "theta_err_max_deg": float(cfg["sim"]["service_limits"]["theta_err_max_deg"]),
        "consecutive_windows": int(cfg["sim"]["service_limits"]["consecutive_windows"]),
    }
    array_cfg = cfg["sim"]["array"]
    target_h5 = ROOT / ch_cfg["feature_path"]
    if not target_h5.exists():
        raise FileNotFoundError(f"缺 {target_h5}; 先 build_channel_hi")

    # 加载模型
    device = "cuda" if (cfg["pretrain"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    n_features = int(ch_cfg["n_features"])
    model = _build_model(cfg, n_features, device)
    if ckpt_path.exists():
        model.load_pretrained(str(ckpt_path), device)
        if verbose:
            print(f">> 加载 ckpt: {ckpt_path}", flush=True)
    else:
        print(f"!! 缺 ckpt {ckpt_path}; 用随机权重 (T8 评估无意义, 仅管线验证)", flush=True)

    # target_only 微调 (在 IID train 上, 复用 T9 模式; pretrain 零样本 rate 估不准)
    from src.transfer.channel_dataset import load_target_channel, ChannelSeqDataset
    from src.experiments.run_groups import _train_with_early_stop
    from torch.utils.data import DataLoader
    import torch.nn as nn
    xT, hiT, rulT, ckT, tidT, evT, lbT, n_traj, sidT = load_target_channel(target_h5)
    tr, va, te = split_trajectories(
        n_traj, [tc["split"]["train"], tc["split"]["val"], tc["split"]["test"]], seed)
    _fm = xT[np.isin(tidT, tr)].mean(axis=0)
    _fs = xT[np.isin(tidT, tr)].std(axis=0) + 1e-6
    xT_norm = (xT - _fm) / _fs
    rul_max = float(tc["rul_max_norm"])
    rulT_norm = rulT / rul_max
    lbT_norm = lbT / rul_max
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    tstride = int(tc.get("target_stride", 50))
    def mkDS(ids):
        m = np.isin(tidT, ids)
        return ChannelSeqDataset(xT_norm[m], hiT[m], rulT_norm[m], ckT[m], L, K,
                                 stride=tstride, event_observed=evT[m], rul_lower_bound=lbT_norm[m])
    bs = int(cfg["pretrain"]["batch_size"])
    ltr = DataLoader(mkDS(tr), batch_size=bs, shuffle=True)
    lva = DataLoader(mkDS(va), batch_size=bs, shuffle=False)
    model.freeze_encoder(False)
    opt = torch.optim.Adam(model.parameters(), lr=float(tc["finetune_lr"]))
    lam = (float(cfg["loss"].get("beta_hi", 1.0)),
           float(cfg["loss"].get("mu_mono", 0.1)),
           float(cfg["loss"].get("nu_smooth", 0.1)))
    e = int(tc.get("epochs_s2", 20))
    _train_with_early_stop(model, ltr, lva, opt, device,
                           nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"])),
                           nn.MSELoss(), lam, e, f"T8-target_only seed{seed}")
    if verbose:
        print(f">> target_only 微调完成 (IID train {len(tr)} traj)", flush=True)

    # test 评估轨迹
    traj_keys = sorted(str(k) for k in range(n_traj))
    te_keys = [f"traj_{i:03d}" for i in te[:n_eval_traj]]   # 限制评估轨迹数 (T8 耗时)

    per_traj = []
    with h5py.File(target_h5, "r") as f:
        for tk in te_keys:
            g = f[tk]
            eol_svc_true = int(g.attrs.get("eol_svc", 0))
            failed_svc = bool(g.attrs.get("failed_svc", 0))
            if not failed_svc or eol_svc_true <= 0:
                continue   # 删失轨迹无服务 EOL 真值, 跳过
            # 读 16 子阵 x_ch 序列 (每子阵独立长度: 失效截断 / 删失全长)
            sub_keys = sorted(k for k in g.keys() if k.startswith("sub_"))
            n_sub = len(sub_keys)
            x_ch_subs = [g[sk]["x_ch"][:].astype(np.float32) for sk in sub_keys]
            T_max = max(x.shape[0] for x in x_ch_subs)
            # 构建 twin
            twin = ArrayTwin.from_h5_group(g, service_limits=service_limits,
                                           array_cfg=array_cfg,
                                           rth_a5=float(cfg["sim"]["physics"]["damage_model"]["rth_feedback"]["a5"]))
            traj_result = {"traj": tk, "eol_svc_true": eol_svc_true,
                           "eval_points": []}
            for t_frac in [0.3, 0.5, 0.7]:
                t_eval = int(t_frac * eol_svc_true)
                if t_eval >= T_max:
                    continue
                # 每子阵独立提取 z + rate (用 [0, min(t_eval, len-1)] 历史)
                zs = np.zeros(n_sub)
                rates = np.zeros(n_sub)
                for s, x_sub in enumerate(x_ch_subs):
                    t_s = min(t_eval, x_sub.shape[0] - 1)
                    x_hist = x_sub[:t_s + 1]
                    # 归一 (轨迹内 train scaler; 简化用子阵内均值)
                    fm = x_hist.mean(axis=0)
                    fs = x_hist.std(axis=0) + 1e-6
                    x_norm = (x_hist - fm) / fs
                    z_s, r_s = _extract_z_rate_sub(model, x_norm, L, device)
                    zs[s] = z_s
                    rates[s] = r_s
                z_hat = zs
                rate = rates
                # rate 分位 (固定 ±20% 不确定性, 对数正态近似)
                rate_p50 = np.maximum(rate, 1e-9)
                rate_p10 = rate_p50 * 0.8
                rate_p90 = rate_p50 * 1.25
                rate_hat = {"p10": rate_p10, "p50": rate_p50, "p90": rate_p90}
                # dose: 从 t_eval 后的遥测 Tj/duty (用子阵 0 的历史延伸 dose, 标称 Ea)
                x0 = x_ch_subs[0][:min(t_eval, x_ch_subs[0].shape[0] - 1) + 1]
                Tj_future = x0[-horizon:, COL_T_DEV] + 273.15
                duty_future = x0[-horizon:, COL_DUTY]
                if len(Tj_future) < horizon:
                    pad = horizon - len(Tj_future)
                    Tj_future = np.concatenate([Tj_future, np.full(pad, Tj_future[-1] if len(Tj_future) else 300.0)])
                    duty_future = np.concatenate([duty_future, np.full(pad, duty_future[-1] if len(duty_future) else 0.5)])
                dose = _arrhenius_dose(Tj_future, duty_future, Ea_eV)
                # rollout_mc
                rng = np.random.default_rng(seed + abs(hash(tk)) % 100000)
                rollout = rollout_mc(z_hat, rate_hat, dose, twin, horizon,
                                     n_mc=n_mc, rng=rng)
                # 真值剩余服务寿命
                rul_svc_true = eol_svc_true - t_eval
                pred_p50 = rollout["eol_p50"]
                abs_err = abs(pred_p50 - rul_svc_true)
                # P10-P90 覆盖率 (真值是否落在 [P10, P90])
                covered = int(rollout["eol_p10"] <= rul_svc_true <= rollout["eol_p90"])
                # 瓶颈子阵
                bottleneck_pred = rollout["bottleneck_sub"]
                z_ch_at_t = np.array([float(g[sk]["z_ch"][min(t_eval, len(g[sk]["z_ch"]) - 1)])
                                      for sk in sub_keys])
                bottleneck_true = int(np.argmax(z_ch_at_t))
                traj_result["eval_points"].append({
                    "t_frac": t_frac, "t_eval": t_eval,
                    "rul_svc_true": rul_svc_true,
                    "eol_p10": rollout["eol_p10"], "eol_p50": rollout["eol_p50"],
                    "eol_p90": rollout["eol_p90"],
                    "abs_err_windows": abs_err,
                    "abs_err_days": abs_err * 6 / 24,   # 6h/窗 → 天
                    "fail_prob": rollout["fail_prob"],
                    "p10_p90_covered": covered,
                    "bottleneck_pred": bottleneck_pred,
                    "bottleneck_true": bottleneck_true,
                    "bottleneck_hit": int(bottleneck_pred == bottleneck_true),
                })
            if traj_result["eval_points"]:
                per_traj.append(traj_result)
            if verbose:
                print(f"  [{tk}] eol_svc={eol_svc_true}, "
                      f"{len(traj_result['eval_points'])} eval points", flush=True)

    # 聚合
    all_points = [p for tr in per_traj for p in tr["eval_points"]]
    n = len(all_points)
    if n == 0:
        return {"per_traj": per_traj, "aggregate": {
            "n_eval_points": 0, "abs_err_days_mean": None, "p10_p90_coverage": None,
            "bottleneck_top3_hit_rate": None, "note": "无有效事件样本, 服务层未验证"}}
    abs_err_days = np.array([p["abs_err_days"] for p in all_points])
    covered = np.array([p["p10_p90_covered"] for p in all_points])
    bottleneck_hit = np.array([p["bottleneck_hit"] for p in all_points])
    fail_probs = np.array([p["fail_prob"] for p in all_points])
    abs_err_windows = np.array([p["abs_err_windows"] for p in all_points])
    # bottleneck 1/16 随机基线 (16 子阵均匀随机 = 6.25%)
    bottleneck_random = 1.0 / 16.0
    aggregate = {
        "n_eval_points": n,
        "n_traj": len(per_traj),
        "abs_err_days_mean": float(np.mean(abs_err_days)),
        "abs_err_days_median": float(np.median(abs_err_days)),
        "abs_err_days_p25": float(np.percentile(abs_err_days, 25)),
        "abs_err_days_p75": float(np.percentile(abs_err_days, 75)),
        "abs_err_windows_median": float(np.median(abs_err_windows)),
        "p10_p90_coverage": float(np.mean(covered)),
        "bottleneck_top1_hit_rate": float(np.mean(bottleneck_hit)),
        "bottleneck_random_baseline": bottleneck_random,
        "bottleneck_lift": float(np.mean(bottleneck_hit)) - bottleneck_random,
        "fail_prob_mean": float(np.mean(fail_probs)),
        "fail_prob_positive_rate": float(np.mean(fail_probs > 0)),
        "fail_prob_median": float(np.median(fail_probs)),
    }
    return {"per_traj": per_traj, "aggregate": aggregate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--ckpt", default="checkpoints/source_phased_array_tcn_pretrain.pt",
                    help="channel level 模型 ckpt (M7 训练产出; 实际应传 ch_target_only_gru 的 ckpt)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--n-eval-traj", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=2000)
    ap.add_argument("--n-mc", type=int, default=200)
    ap.add_argument("--out", default="docs/results_service_eval.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ckpt = ROOT / args.ckpt
    all_results = []
    for s in range(args.seeds):
        seed = cfg["seed"] + s
        res = evaluate_service_level(cfg, ckpt, seed, args.n_eval_traj,
                                     args.horizon, args.n_mc, verbose=True)
        res["seed"] = seed
        all_results.append(res)
        agg = res["aggregate"]
        print(f"\n===== seed {seed} 服务层评估 =====")
        if agg["n_eval_points"] == 0:
            print(f"  无有效事件样本 (n=0), 服务层未验证")
        else:
            print(f"  评估点数: {agg['n_eval_points']} ({agg['n_traj']} 轨迹)")
            print(f"  服务 EOL 绝对误差: mean={agg['abs_err_days_mean']:.1f} 天, "
                  f"median={agg['abs_err_days_median']:.1f} 天")
            print(f"  P10-P90 覆盖率: {agg['p10_p90_coverage']:.2%} (应接近 80%)")
            print(f"  瓶颈子阵 top-1 命中率: {agg['bottleneck_top1_hit_rate']:.2%}")
            print(f"  fail_prob mean: {agg['fail_prob_mean']:.3f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n>> {out}")


if __name__ == "__main__":
    main()
