"""src/data/preprocess/mosfet_features.py

源域 (NASA MOSFET/IGBT) 电参数老化 -> 2 维退化特征 + HI/RUL 标签 (schema_v2)。

2026-07-21 Task 5: 与真实 loader mosfet_real_loader.py 对齐 (5 维 -> 2 维降级)。
  特征 2 维 = [RDS_drift, T_case_C]
    RDS_drift : (R_DS_ON - R0)/R0   导通电阻相对漂移 (主退化信号)
    T_case_C  : 封装表面温度 (工况上下文; 真实 = packageTemperature)
  Vth / g_m / I_leak 物理不可提取 (NASA 固定栅压方波驱动无 Vgs 扫描),
    依据 docs/数据集下载/NASA电子器件退化数据字段清单与loader设计.md §0 第 3 条。

标签:
  HI = clip(isotonic(RDS_drift)/threshold, 0, 1)   isotonic 去噪 + 固定尺度
  RUL = rul_s (秒单位; pretrain 按 rul_max 归一消除 v1/v2 量级差异)
  event_observed: 达 ΔR_DS(ON)=threshold → 1, 否则右删失 0

--synthetic 模式: 受控终点漂移 (final_drift∈[0.02,0.15], 阈值 0.05) → 部分失效/部分
  右删失。产出写到 *_synthetic.h5 (独立路径, 不覆盖真实 mosfet_real_loader 产出的 h5)。

用法:
  python -m src.data.preprocess.mosfet_features --config configs/phased_array.yaml --synthetic --report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                           # noqa: E402
from .nasa_mat_common import write_feature_h5               # noqa: E402

# 与真实 loader mosfet_real_loader.py 完全一致 (字段清单 §0 第 3 条)
FEATURE_NAMES = ["RDS_drift", "T_case_C"]
SAMPLE_INTERVAL_S = 3600.0   # synthetic 老化步虚拟采样间隔 (1h; 真实由 elapsed_time_s 给)


def extract_features(raw: pd.DataFrame) -> pd.DataFrame:
    """原始电参数 (每老化步) -> 2 维特征。raw 列: R_DS_ON, T_j (K)。"""
    R0 = float(raw["R_DS_ON"].iloc[0])
    return pd.DataFrame({
        "RDS_drift": (raw["R_DS_ON"].values - R0) / (R0 + 1e-12),
        "T_case_C": raw["T_j"].values - 273.15,    # 结温 K -> case 温度 C (工况上下文)
    })


def construct_labels(feat: pd.DataFrame, delta_threshold: float,
                     sample_interval_s: float = SAMPLE_INTERVAL_S) -> pd.DataFrame:
    """派生 HI (固定尺度) + RUL(秒) + label_fail + event_observed (右删失)。

    delta_threshold: NASA ΔR_DS(ON) 归一化失效阈值 (默认 0.05)。
    sample_interval_s: 老化步采样间隔 (synthetic 虚拟; 真实由 elapsed_time_s 提供)。
    未达阈值 -> event_observed=0, RUL 为下界 (观测截止时至少剩余)。
    """
    T = len(feat)
    t_idx = np.arange(T)
    elapsed_s = t_idx * float(sample_interval_s)
    drift = feat["RDS_drift"].values
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    drift_iso = iso.fit_transform(t_idx, drift)               # 去噪 (潜在单调健康状态)
    hi = np.clip(drift_iso / delta_threshold, 0.0, 1.0)       # isotonic + 固定尺度, HI=1 即失效
    fail_mask = drift >= delta_threshold
    label_fail = np.zeros(T, dtype=np.int8)
    event_observed = bool(fail_mask.any())
    if event_observed:
        eol = int(np.argmax(fail_mask))
        label_fail[eol:] = 1
        rul_s = np.maximum((eol - t_idx) * float(sample_interval_s), 0.0)
        rul_s[eol:] = 0.0
    else:
        rul_s = (T - 1 - t_idx) * float(sample_interval_s)    # 右删失下界
    feat = feat.copy()
    feat["hi"] = hi
    feat["rul_s"] = rul_s
    feat["rul_lower_bound_s"] = rul_s                          # 失效=精确, 删失=下界
    feat["elapsed_time_s"] = elapsed_s
    feat["label_fail"] = label_fail
    feat["event_observed"] = np.full(T, int(event_observed), dtype=np.int8)
    feat["t_idx"] = t_idx
    return feat


def _mono_violation_rate(devices: dict) -> float:
    total, viol = 0, 0
    for _, g in devices.items():
        hi = g.sort_values("t_idx")["hi"].values
        d = np.diff(hi)
        total += len(d)
        viol += int(np.sum(d < -1e-9))
    return viol / max(total, 1)


def make_synthetic(n_devices: int = 6, n_pts: int = 128, seed: int = 0):
    """合成 MOSFET/IGBT 老化器件 — 受控终点漂移 (修复 GPT 二 标签塌缩)。

    每器件: drift = final_drift · u^m, final_drift ∈ [0.02,0.15], 阈值 0.05
    → 部分器件达阈值失效, 部分接近但 survive (右删失, 体现真实器件散布)。
    温度/应力改变 m (曲率) 与 final_drift, 而非让幅值膨胀百万倍。
    只生成 R_DS_ON + T_j (Vth/gm/Ileak 真实不可提取, 不生成不进特征)。
    """
    rng = np.random.default_rng(seed)
    out = {}
    u = np.linspace(0.0, 1.0, n_pts)
    for d in range(n_devices):
        final_drift = rng.uniform(0.02, 0.15)              # 终点 ΔR_DS/R0 (合理量级)
        m = rng.uniform(0.7, 1.8)                          # 曲率 (应力影响)
        drift = final_drift * np.power(u, m)
        Tj_K = rng.uniform(350.0, 430.0)
        R0 = rng.uniform(0.05, 0.15)
        R_DS = R0 * (1.0 + drift) + rng.normal(0, 0.002 * R0, n_pts)
        raw = pd.DataFrame({"R_DS_ON": R_DS, "T_j": np.full(n_pts, Tj_K)})
        out[f"MOSFET_dev{d + 1}"] = raw
    return out


def run(args) -> None:
    cfg = load_config(args.config)
    delta_thr = float(cfg["source"]["failure"]["RDS_delta_threshold"])
    out_path = ROOT / cfg["pretrain"]["source_feature_path"]
    if args.synthetic:
        # 独立后缀路径, 不覆盖真实 mosfet_real_loader.py 产出的 h5
        out_path = out_path.with_name(out_path.stem + "_synthetic.h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    devices: dict[str, pd.DataFrame] = {}
    if args.synthetic:
        print(">> 合成数据模式 (受控终点漂移; 真实 NASA loader 见 mosfet_real_loader.py)")
        synth = make_synthetic(seed=cfg.get("seed", 42))
        for did, raw in synth.items():
            devices[did] = construct_labels(extract_features(raw), delta_thr)
    else:
        src = ROOT / "data" / "source" / "NASA_MOSFET"
        if not src.exists():
            print(f"!! 未找到 {src}; 真实加载请用 mosfet_real_loader.py, 或 --synthetic 验证逻辑")
            sys.exit(1)
        raise NotImplementedError("真实 MOSFET 加载已迁移到 mosfet_real_loader.py (schema_v2)")

    # 特征矩阵完整性校验 (维数 + 无 NaN/Inf)
    target_dim = int(cfg["source"]["features"]["target_dim"])
    for did, df in devices.items():
        assert df[FEATURE_NAMES].shape[1] == target_dim, \
            f"{did}: 特征维 {df[FEATURE_NAMES].shape[1]} != config target_dim {target_dim}"
        assert not np.any(np.isnan(df[FEATURE_NAMES].values)) and \
               not np.any(np.isinf(df[FEATURE_NAMES].values))

    metadata = {
        "dataset_id": "NASA_MOSFET_synthetic" if args.synthetic else "NASA_MOSFET_Thermal_Overstress",
        "schema_version": "2.0",
        "feature_names": FEATURE_NAMES,
        "feature_dim": len(FEATURE_NAMES),
        "failure_feature": "RDS_drift",
        "failure_threshold": delta_thr,
        "group_split_key": "device_id",
        "right_censoring": True,
        "temperature_semantics": "case temperature (synthetic: Tj_K - 273.15)",
        "excluded_features": ["Vth", "gm", "I_leak"],
        "excluded_reason": "物理不可提取 (NASA 固定栅压方波驱动无 Vgs 扫描)",
        "synthetic": bool(args.synthetic),
        "sample_interval_s": SAMPLE_INTERVAL_S,
        "n_devices": len(devices),
    }
    write_feature_h5(out_path, devices, feature_columns=FEATURE_NAMES, metadata=metadata)
    print(f">> 已写出 (schema_v2 分组): {out_path}  "
          f"({sum(len(d) for d in devices.values())} 行, {len(devices)} devices)")

    if args.report:
        all_hi = np.concatenate([df["hi"].values for df in devices.values()])
        all_rul = np.concatenate([df["rul_s"].values for df in devices.values()])
        ev = {did: bool(df["event_observed"].iloc[0]) for did, df in devices.items()}
        print("\n===== PA1 源域特征工程报告 (schema_v2, 2 维) =====")
        print(f"特征 = {FEATURE_NAMES}  (Vth/gm/Ileak 已剔除: 物理不可提取)")
        print(f"HI 范围 = [{all_hi.min():.4f}, {all_hi.max():.4f}]  (固定尺度 clip(drift/{delta_thr}))")
        print(f"RUL(秒) 范围 = [{all_rul.min():.0f}, {all_rul.max():.0f}]  (pretrain 按 rul_max 归一)")
        print(f"失效器件 {sum(ev.values())}/{len(ev)} (event_observed); "
              f"右删失 {sum(1 for v in ev.values() if not v)}")
        print(f"HI 单调违例率 = {_mono_violation_rate(devices):.4f}")
        print(f"LOO: val_device 由 config source.split.val_device_ids 决定 (缺省 = 最后器件)")
        print("=================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
