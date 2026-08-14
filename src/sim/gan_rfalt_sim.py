"""独立的 GaN RFALT 源域仿真器。

该模块只建模单个台架器件的恒定/阶梯 RF 应力和集总热路径，
与 LEO 阵列仿真器没有代码或参数共享。内部三状态仅用于仿真审计，
观测头输出的 RF 量才是后续训练特征的来源。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

from src.data.preprocess.gan_rfalt_schema import SCHEMA_VERSION, validate_rfalt_frame
from src.utils import load_config


ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_ID = "rfalt_lumped_v1"
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def sample_rfalt_profile(
    duration_s: float,
    sample_period_s: float,
    stress_modes: Iterable[str] = ("rfalt_cw", "rfalt_modulated", "recovery"),
    *,
    base_temperature_c: float | None = None,
    compression_dB: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """采样单器件 RFALT 工况剖面（仅恒定或阶梯应力）。"""
    if duration_s <= 0 or sample_period_s <= 0:
        raise ValueError("duration_s 与 sample_period_s 必须为正")
    modes = tuple(stress_modes)
    if not modes:
        raise ValueError("stress_modes 不能为空")
    allowed = {"rfalt_cw", "rfalt_modulated", "recovery"}
    unknown = set(modes) - allowed
    if unknown:
        raise ValueError(f"不支持的 RFALT 应力模式: {sorted(unknown)}")

    rng = np.random.default_rng(seed)
    time_s = np.arange(0.0, float(duration_s), sample_period_s)
    if time_s.size == 0 or not np.isclose(time_s[-1], float(duration_s)):
        time_s = np.append(time_s, float(duration_s))
    n = len(time_s)
    segment = np.minimum((np.arange(n) * len(modes)) // n, len(modes) - 1)
    selected = np.asarray([modes[index] for index in segment], dtype=object)
    base = float(base_temperature_c if base_temperature_c is not None else rng.uniform(70.0, 100.0))

    duty = np.select(
        [selected == "rfalt_cw", selected == "rfalt_modulated"], [0.78, 0.62], default=0.02,
    )
    pin = np.select(
        [selected == "rfalt_cw", selected == "rfalt_modulated"],
        [35.0 + compression_dB, 34.0 + compression_dB], default=0.0,
    )
    papr = np.select([selected == "rfalt_modulated"], [7.0], default=0.0)
    vswr = np.select(
        [selected == "rfalt_cw", selected == "rfalt_modulated"], [1.30, 1.55], default=1.05,
    )
    return pd.DataFrame({
        "time_s": time_s,
        "T_base_C": np.full(n, base),
        "VDS": np.select([selected == "recovery"], [4.0], default=28.0),
        "VGS": np.full(n, -2.7),
        "ID": 0.75 * duty,
        "IG": 1.0e-6 + 1.5e-6 * duty,
        "duty_cycle": duty,
        "waveform": np.where(selected == "rfalt_modulated", "ofdm", "cw"),
        "PAPR_dB": papr,
        "VSWR": vswr,
        "Pin_dBm": pin,
        "stress_mode": selected,
    })


def integrate_damage_states(profile: pd.DataFrame) -> pd.DataFrame:
    """以集总热模型积分 ``[d_perm, q_trap, r_th]`` 三个损伤状态。

    ``d_perm`` 与 ``r_th`` 由非负增量累计；恢复段不产生新永久损伤，
    并以一阶释放项降低 ``q_trap``。
    """
    required = {"time_s", "T_base_C", "duty_cycle", "Pin_dBm", "VSWR", "stress_mode"}
    missing = sorted(required - set(profile.columns))
    if missing:
        raise ValueError(f"RFALT 剖面缺少字段: {', '.join(missing)}")
    if len(profile) == 0:
        raise ValueError("RFALT 剖面不能为空")

    n = len(profile)
    d_perm = np.zeros(n, dtype=np.float64)
    q_trap = np.zeros(n, dtype=np.float64)
    r_th = np.zeros(n, dtype=np.float64)
    t_j = np.zeros(n, dtype=np.float64)
    time_s = profile["time_s"].to_numpy(dtype=float)
    base = profile["T_base_C"].to_numpy(dtype=float)
    duty = profile["duty_cycle"].to_numpy(dtype=float)
    pin = profile["Pin_dBm"].to_numpy(dtype=float)
    vswr = profile["VSWR"].to_numpy(dtype=float)
    recovery = profile["stress_mode"].to_numpy(dtype=object) == "recovery"

    for i in range(n):
        dt_h = 0.0 if i == 0 else max(time_s[i] - time_s[i - 1], 0.0) / 3600.0
        prev_d = d_perm[i - 1] if i else 0.0
        prev_q = q_trap[i - 1] if i else 0.0
        prev_r = r_th[i - 1] if i else 0.0
        rf_drive = max(pin[i] - 32.0, 0.0) / 3.0
        thermal_rise = 8.0 + 24.0 * duty[i] * rf_drive * (1.0 + 0.35 * max(vswr[i] - 1.0, 0.0))
        t_j[i] = base[i] + thermal_rise * (1.0 + prev_r)
        temp_factor = float(np.clip(np.exp((t_j[i] - 95.0) / 42.0), 0.20, 8.0))
        compression = max(pin[i] - 35.0, 0.0)
        stress_factor = duty[i] * (1.0 + 0.18 * compression + 0.25 * max(vswr[i] - 1.0, 0.0))

        if recovery[i]:
            q_next = prev_q * np.exp(-0.85 * dt_h)
            d_increment = 0.0
        else:
            # dq/dt = capture - release*q 的解析解；大采样步长不发生 Euler 翻转。
            capture_rate = 0.050 * stress_factor
            release_rate = 0.090
            q_steady = capture_rate / release_rate
            q_next = q_steady + (prev_q - q_steady) * np.exp(-release_rate * dt_h)
            d_increment = dt_h * 1.35e-5 * temp_factor * stress_factor * (1.0 + 0.40 * prev_q)
        d_perm[i] = prev_d + max(d_increment, 0.0)
        q_trap[i] = np.clip(q_next, 0.0, 1.0)
        r_th[i] = prev_r + max(d_increment, 0.0) * 14.0

    return pd.DataFrame({"d_perm": d_perm, "q_trap": q_trap, "r_th": r_th, "T_j_C": t_j})


def simulate_rfalt_trajectory(
    device_id: str,
    duration_s: float,
    sample_period_s: float,
    stress_modes: Iterable[str] = ("rfalt_cw", "rfalt_modulated", "recovery"),
    *,
    seed: int = 42,
    lot_id: str | None = None,
) -> pd.DataFrame:
    """生成一条可复现的单器件 RFALT 轨迹及其台架 RF 观测。"""
    rng = np.random.default_rng(seed)
    profile = sample_rfalt_profile(
        duration_s, sample_period_s, stress_modes, seed=seed,
        base_temperature_c=float(rng.uniform(70.0, 105.0)),
        compression_dB=float(rng.uniform(0.0, 2.5)),
    )
    states = integrate_damage_states(profile)
    d_perm = states["d_perm"].to_numpy()
    q_trap = states["q_trap"].to_numpy()
    r_th = states["r_th"].to_numpy()
    n = len(profile)
    active = profile["stress_mode"].to_numpy() != "recovery"
    noise = lambda scale: rng.normal(0.0, scale, n)

    # P2-5 观测方程重校准（仿真假设）：放大 d_perm 系数至 EOL（d_perm≈0.006）可辨识——
    # gain 降 ~0.6 dB（> 噪声 0.025）、PAE 降 ~5pp、RDS 升 ~18%、gm/vth/EVM/ACPR 同向显著。
    # 目的：让 memoryless encoder 能从观测识别永久损伤；d_perm 绝对范围不变（与目标域 EOL 0.006 同量级）。
    gain = 10.8 - 100.0 * d_perm - 1.15 * q_trap - 3.0 * r_th + noise(0.025)
    pout = profile["Pin_dBm"].to_numpy() + gain - 0.08 * profile["PAPR_dB"].to_numpy() + noise(0.03)
    pae = np.clip(0.59 - 8.0 * d_perm - 0.11 * q_trap - 0.04 * r_th + noise(0.002), 0.01, 0.85)
    rds = 0.115 * (1.0 + 30.0 * d_perm + 0.75 * q_trap + 0.7 * r_th) + noise(0.00015)
    gm = 0.42 * (1.0 - 10.0 * d_perm - 0.18 * q_trap) + noise(0.001)
    vth = -2.10 + 5.0 * d_perm + 0.05 * q_trap + noise(0.001)
    fail_mask = d_perm >= 0.0035
    event_observed = bool(fail_mask.any())
    time_s = profile["time_s"].to_numpy(dtype=float)
    if event_observed:
        eol_time_s = float(time_s[int(np.argmax(fail_mask))])
        rul = np.maximum(eol_time_s - time_s, 0.0)
    else:
        rul = time_s[-1] - time_s

    frame = profile.copy()
    frame.insert(1, "device_id", str(device_id))
    frame.insert(2, "lot_id", lot_id or f"lot-{seed % 4:02d}")
    frame["T_j_C"] = states["T_j_C"]
    frame["Pout_dBm"] = pout
    frame["gain_dB"] = gain
    frame["PAE"] = pae
    frame["AM_AM_dB"] = -0.16 * q_trap - 10.0 * d_perm + noise(0.004)
    frame["AM_PM_deg"] = 0.22 * q_trap + 10.0 * d_perm + noise(0.008)
    frame["EVM_pct"] = np.clip(0.65 + 4.2 * q_trap + 100.0 * d_perm + noise(0.025), 0.0, None)
    frame["ACPR_dBc"] = -47.0 + 7.2 * q_trap + 100.0 * d_perm + noise(0.035)
    frame["RDS_dynamic_ohm"] = np.clip(rds, 1e-4, None)
    frame["gm_S"] = np.clip(gm, 1e-4, None)
    frame["Vth_V"] = vth
    frame["event_observed"] = event_observed
    frame["rul_lower_bound_s"] = rul
    frame["d_perm"] = d_perm
    frame["q_trap"] = q_trap
    frame["r_th"] = r_th
    frame.loc[~active, "Pout_dBm"] = frame.loc[~active, "Pin_dBm"] + frame.loc[~active, "gain_dB"]
    return validate_rfalt_frame(frame, require_latent=True)


def write_rfalt_raw_h5(trajectories: Iterable[pd.DataFrame], output_path: str | Path) -> None:
    """写出 RFALT 原始轨迹；每个器件一个 ``trajectories/{device_id}`` 分组。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["dynamics_id"] = DYNAMICS_ID
        root = h5.create_group("trajectories")
        for frame in trajectories:
            validated = validate_rfalt_frame(frame, require_latent=True)
            device_ids = validated["device_id"].unique()
            if len(device_ids) != 1:
                raise ValueError("每个原始轨迹 DataFrame 只能包含一个 device_id")
            group = root.create_group(str(device_ids[0]))
            for column in validated.columns:
                values = validated[column].to_numpy()
                if values.dtype.kind in {"O", "U"}:
                    group.create_dataset(column, data=values.astype(object), dtype=_STRING_DTYPE)
                else:
                    group.create_dataset(column, data=values, compression="gzip", chunks=True)


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="生成独立 GaN RFALT 源域原始 HDF5")
    parser.add_argument("--config", default="configs/phased_array_gan.yaml")
    parser.add_argument("--smoke", action="store_true", help="生成小规模 24 小时 smoke 数据")
    parser.add_argument("--output", default=None, help="原始 HDF5 输出路径")
    args = parser.parse_args()
    cfg = load_config(args.config)
    source = cfg["source"]
    n_devices = int(source["n_devices"])
    duration_s = float(source["duration_s"])
    if args.smoke:
        n_devices = min(n_devices, 4)
        duration_s = min(duration_s, 24 * 3600.0)
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    trajectories = [
        simulate_rfalt_trajectory(
            device_id=f"rfalt-{index:03d}", duration_s=duration_s,
            sample_period_s=float(source["sample_period_s"]),
            stress_modes=source["stress_modes"], seed=int(rng.integers(0, 2**31)),
        )
        for index in range(n_devices)
    ]
    output_path = _resolve_path(args.output or source["raw_path"])
    write_rfalt_raw_h5(trajectories, output_path)
    print(f">> 已写出 {n_devices} 条 RFALT 原始轨迹: {output_path}")


if __name__ == "__main__":
    main()
