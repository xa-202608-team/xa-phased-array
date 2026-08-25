# -*- coding: utf-8 -*-
"""真实过站观测协议适配层 (P1, LJ3 P0 审计落地, 2026-08-25)。

把连续窗口的通道级 canonical 时序投影到 LJ3 龙江三号实测的过站稀疏观测协议下,
生成同构变体数据集 (channel_features_pass.h5), 供 sim-to-real 观测协议对齐实验
(P2 预注册) 与答辩演示 "真实采样协议下任务长什么样" 使用。

协议模型 (全部参数来自 docs/数据集下载/LJ3_相控阵遥测审计报告.md §3, 父仓库):
- 过站调度: 站时刻 ~ 周期 pass_interval_h + 相对抖动; 站内时长 (7 min) << 任务
  窗口 (6 h), 故窗口级语义为 "窗口命中/未命中" (期望命中率 ≈ 窗口/间隔)。
- 未校准占位: 命中窗口按 uncal_fraction 概率整窗无效 (LJ3 29.3% 帧未校准,
  通道测量字段全零为占位语义, 非真实 0)。
- 观测量化: canonical 4 维各自等效观测阶梯 (1 dB 幅度阶梯 -> 比值 20log10 换算,
  温度 0.019 °C/LSB)。

不修改源数据集与任何现有链路; 默认 config enabled=false, 现有复现 bit-exact 不变。

obs_status 语义 (uint8): 0 = 无站 (窗口无过站覆盖), 1 = 正常观测, 2 = 未校准占位。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# obs_status 常量 (单文件唯一定义处, 下游按名引用)
OBS_NO_PASS = np.uint8(0)
OBS_NORMAL = np.uint8(1)
OBS_UNCAL_PLACEHOLDER = np.uint8(2)

# 默认协议参数 (LJ3 审计实测值; 完整来源见模块 docstring)
DEFAULTS = {
    "pass_interval_h": 8.16,        # 73.41 h / 9 站
    "pass_jitter": 0.25,            # 站间隔相对抖动
    "uncal_fraction": 0.293,        # 命中窗口内未校准占位比例
    "quant_steps": [0.059, 0.02, 0.05, 0.059],  # canonical 4 维等效阶梯
    "seed": 20260825,
}


def expected_hit_rate(pass_interval_h: float, window_h: float = 6.0) -> float:
    """名义窗口命中率 (jitter=0 极限): min(1, window/interval)。"""
    return min(1.0, window_h / pass_interval_h) if pass_interval_h > 0 else 1.0


def build_pass_schedule(n_windows: int, window_s: float, pass_interval_h: float,
                        pass_jitter: float, rng: np.random.Generator) -> np.ndarray:
    """生成过站时刻序列并投影到窗口 -> 每窗布尔命中掩码 (n_windows,)。

    站时刻: t_{k+1} = t_k + interval_s * (1 + jitter * U(-1,1)), 首站相位均匀随机。
    站时长 << 窗口粒度, 视为瞬时事件落入所属窗口。
    """
    total_s = n_windows * window_s
    interval_s = pass_interval_h * 3600.0
    t = rng.uniform(0.0, interval_s)
    hits = np.zeros(n_windows, dtype=bool)
    while t < total_s:
        hits[int(t // window_s)] = True
        t += interval_s * (1.0 + pass_jitter * rng.uniform(-1.0, 1.0))
    return hits


def apply_pass_protocol(x_ch: np.ndarray, window_s: float, params: dict,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """对单条 canonical 序列 (T,4) 施加过站协议 -> (失真后 x_ch, obs_status)。

    - 无站窗口: x_ch 行置 NaN, status=0
    - 未校准占位窗口 (命中内按比例): 置 NaN, status=2
    - 正常窗口: 各维按 quant_steps 阶梯量化, status=1
    标签 (hi/rul/z) 与时间轴保持不变 — 观测协议改变的是可观测性, 不是物理真值。
    """
    x = x_ch.astype(np.float32).copy()
    hits = build_pass_schedule(x.shape[0], window_s, float(params["pass_interval_h"]),
                               float(params["pass_jitter"]), rng)
    status = np.where(hits, OBS_NORMAL, OBS_NO_PASS).astype(np.uint8)
    uncal = hits & (rng.random(hits.shape[0]) < float(params["uncal_fraction"]))
    status[uncal] = OBS_UNCAL_PLACEHOLDER
    invalid = status != OBS_NORMAL
    x[invalid] = np.nan
    steps = np.asarray(params["quant_steps"], dtype=np.float32)
    x[~invalid] = np.round(x[~invalid] / steps) * steps
    return x, status


def _copy_group_deep(src: h5py.Group, dst: h5py.Group) -> None:
    """递归深拷贝 (用于 twin 等非 sub_ 成员: L2 物理孪生参数必须原样保留)。"""
    for key in src.keys():
        item = src[key]
        if isinstance(item, h5py.Group):
            _copy_group_deep(item, dst.create_group(key))
        else:
            dst.create_dataset(key, data=item[()])


def adapt_dataset(src: Path, dst: Path, cfg: dict) -> dict:
    """读 channel_features.h5 -> 写同构 pass 变体; 返回统计摘要。

    逐 traj: sub_* 成员复制全部字段并将 x_ch 替换为协议失真版 + obs_status;
    非 sub_ 成员 (twin 物理孪生 L2 参数) 与顶层 attrs 原样保留 — 观测协议改变
    可观测性, 不改变物理真值与 schema 识别元数据。
    """
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg.get("seed", DEFAULTS["seed"])))
    window_s = float(cfg.get("frame_period_s", 21600.0))
    stats = {"n_traj": 0, "n_sub": 0, "n_windows": 0, "n_normal": 0,
             "n_no_pass": 0, "n_uncal": 0}
    with h5py.File(src, "r") as fs, h5py.File(dst, "w") as fd:
        for traj in sorted(k for k in fs.keys() if k.startswith("traj_")):
            gt = fs[traj]
            for sub in sorted(gt.keys()):
                gd_sub = fd.create_group(f"{traj}/{sub}")
                if sub.startswith("sub_"):
                    gs = gt[sub]
                    for key in gs.keys():
                        if key == "x_ch":
                            continue
                        gd_sub.create_dataset(key, data=gs[key][()])
                    x_pass, status = apply_pass_protocol(gs["x_ch"][()], window_s, cfg, rng)
                    gd_sub.create_dataset("x_ch", data=x_pass)
                    gd_sub.create_dataset("obs_status", data=status)
                    stats["n_sub"] += 1
                    stats["n_windows"] += int(status.shape[0])
                    for code, k in [(OBS_NORMAL, "n_normal"), (OBS_NO_PASS, "n_no_pass"),
                                    (OBS_UNCAL_PLACEHOLDER, "n_uncal")]:
                        stats[k] += int((status == code).sum())
                else:
                    _copy_group_deep(gt[sub], gd_sub)   # twin 等 L2 参数原样保留
            stats["n_traj"] += 1
        for k, v in fs.attrs.items():
            fd.attrs[k] = v                            # schema 识别元数据原样保留
        fd.attrs["adapter"] = "pass_observation_v1 (LJ3 real protocol, P1)"
        fd.attrs["source_dataset"] = str(src)
        # h5py attrs 不支持嵌套 dict, 协议参数以 JSON 字符串落盘 (可审计可复现)
        fd.attrs["protocol_json"] = json.dumps(
            {k: v for k, v in cfg.items() if k != "enabled"}, ensure_ascii=False)
        fd.attrs["expected_hit_rate"] = expected_hit_rate(float(cfg["pass_interval_h"]),
                                                          window_s / 3600.0)
    n_hit = stats["n_normal"] + stats["n_uncal"]
    stats["hit_rate_actual"] = round(n_hit / max(1, stats["n_windows"]), 4)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="generate pass-observation variant dataset")
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--src", default=None, help="override source h5")
    ap.add_argument("--dst", default=None, help="override output h5")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils import load_config  # noqa: E402

    cfg_full = load_config(args.config)
    cfg = dict(cfg_full.get("observation_adapter", {}))
    if not cfg.get("enabled", False):
        raise SystemExit("observation_adapter.enabled=false (默认) — 变体生成需显式开启,"
                         " 现有复现链保持 bit-exact 不变")
    src = Path(args.src or cfg["source_dataset"])
    dst = Path(args.dst or cfg["output_dataset"])
    stats = adapt_dataset(src, dst, cfg)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"[out] {dst}")


if __name__ == "__main__":
    main()
