"""GaN RFALT 原始观测到可训练分组 HDF5 的特征写入。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

from .gan_rfalt_schema import LATENT_COLUMNS, SCHEMA_VERSION, validate_rfalt_frame
from src.utils import load_config


ROOT = Path(__file__).resolve().parents[3]
FEATURE_NAMES = (
    "T_base_C", "T_j_C", "VDS", "VGS", "ID", "IG", "duty_cycle", "PAPR_dB",
    "VSWR", "Pin_dBm", "Pout_dBm", "gain_dB", "PAE", "AM_AM_dB", "AM_PM_deg",
    "EVM_pct", "ACPR_dBc", "RDS_dynamic_ohm", "gm_S", "Vth_V",
)
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def _as_trajectories(trajectories: Iterable[pd.DataFrame] | dict[str, pd.DataFrame]) -> list[pd.DataFrame]:
    items = trajectories.values() if isinstance(trajectories, dict) else trajectories
    frames = [validate_rfalt_frame(frame, require_latent=False) for frame in items]
    if not frames:
        raise ValueError("至少需要一条 RFALT 轨迹")
    return frames


def _observed_hi(frame: pd.DataFrame) -> np.ndarray:
    """仅由可观测 RF 量派生辅助 HI；不对原始 RF 观测作 isotonic。"""
    gain_drop = float(frame["gain_dB"].iloc[0]) - frame["gain_dB"].to_numpy(dtype=float)
    rds0 = max(float(frame["RDS_dynamic_ohm"].iloc[0]), 1e-8)
    rds_drift = frame["RDS_dynamic_ohm"].to_numpy(dtype=float) / rds0 - 1.0
    return np.clip(0.45 * gain_drop + 0.55 * rds_drift, 0.0, 1.0).astype(np.float32)


def write_rfalt_h5(trajectories: Iterable[pd.DataFrame] | dict[str, pd.DataFrame], output_path: str | Path) -> None:
    """写入 schema_v2 兼容的分组训练特征，latent 永远不进入 ``x``。"""
    frames = _as_trajectories(trajectories)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["dynamics_id"] = "rfalt_lumped_v1"
        h5.attrs["feature_names"] = np.asarray(FEATURE_NAMES, dtype=_STRING_DTYPE)
        h5.attrs["feature_dim"] = len(FEATURE_NAMES)
        h5.attrs["group_split_key"] = "device_id"
        root = h5.create_group("devices")
        written: set[str] = set()
        for frame in frames:
            device_ids = frame["device_id"].unique()
            if len(device_ids) != 1:
                raise ValueError("每个特征轨迹 DataFrame 只能包含一个 device_id")
            device_id = str(device_ids[0])
            if device_id in written:
                raise ValueError(f"重复的 device_id: {device_id}")
            written.add(device_id)
            group = root.create_group(device_id)
            group.create_dataset("x", data=frame[list(FEATURE_NAMES)].to_numpy(np.float32), compression="gzip")
            group.create_dataset("hi", data=_observed_hi(frame), compression="gzip")
            group.create_dataset("elapsed_time_s", data=frame["time_s"].to_numpy(np.float64), compression="gzip")
            # 统一 transition 训练接口使用 time_s；elapsed_time_s 保留为兼容字段。
            group.create_dataset("time_s", data=frame["time_s"].to_numpy(np.float64), compression="gzip")
            rul = frame["rul_lower_bound_s"].to_numpy(np.float64)
            group.create_dataset("rul_s", data=rul, compression="gzip")
            group.create_dataset("rul_lower_bound_s", data=rul, compression="gzip")
            event = bool(frame["event_observed"].iloc[0])
            group.attrs["event_observed"] = event
            group.create_dataset("label_fail", data=(rul <= 0.0).astype(np.int8), compression="gzip")
            for latent in LATENT_COLUMNS:
                if latent in frame:
                    group.create_dataset(f"latent_{latent}", data=frame[latent].to_numpy(np.float32), compression="gzip")


def read_rfalt_raw_h5(input_path: str | Path) -> list[pd.DataFrame]:
    """读取 ``gan_rfalt_sim`` 写出的 ``trajectories`` 分组原始 HDF5。"""
    frames: list[pd.DataFrame] = []
    with h5py.File(input_path, "r") as h5:
        if h5.attrs.get("schema_version", "") != SCHEMA_VERSION:
            raise ValueError("输入不是 gan_rfalt_v1 原始 HDF5")
        if "trajectories" not in h5:
            raise ValueError("原始 HDF5 缺少 trajectories 分组")
        for device_id in sorted(h5["trajectories"].keys()):
            group = h5["trajectories"][device_id]
            data = {}
            for name, dataset in group.items():
                values = dataset[:]
                if values.dtype.kind == "S":
                    values = np.asarray([value.decode("utf-8") for value in values], dtype=object)
                data[name] = values
            frames.append(validate_rfalt_frame(pd.DataFrame(data), require_latent=True))
    return frames


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="将 GaN RFALT 原始 HDF5 写为训练特征 HDF5")
    parser.add_argument("--config", default="configs/phased_array_gan.yaml")
    parser.add_argument("--input", default=None, help="RFALT 原始 HDF5 路径")
    parser.add_argument("--output", default=None, help="训练特征 HDF5 路径")
    args = parser.parse_args()
    cfg = load_config(args.config)
    source = cfg["source"]
    input_path = _resolve_path(args.input or source["raw_path"])
    output_path = _resolve_path(args.output or source["feature_path"])
    frames = read_rfalt_raw_h5(input_path)
    write_rfalt_h5(frames, output_path)
    print(f">> 已写出 {len(frames)} 个 RFALT 训练器件: {output_path}")


if __name__ == "__main__":
    main()
