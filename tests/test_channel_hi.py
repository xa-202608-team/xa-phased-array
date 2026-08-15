"""通道级 HI 构建测试 (T2/M2)。

验证 src/sim/build_channel_hi.py 输出正确性 (不依赖 EOL_ch vs EOL_svc 物理关系):
  - hi_ch ∈ [0,1]
  - z_ch 单调非减 (f_s 单调 → z 单调)
  - 失效通道截断到 EOL (rul[eol]=0, 无 EOL 后零标签窗 — P0-1 铁律)
  - 删失通道 rul 封顶 cap_ratio*T (不反推仿真截止时刻)
  - 同 traj_id 的 16 sub_id 可同属一个 split (划分按轨迹)
  - δ 阈值只来自 config (不随实验数据变)
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                              # noqa: E402

PA_CONFIG = ROOT / "configs" / "phased_array.yaml"


def _load_channels():
    cfg = load_config(PA_CONFIG)
    feature_path = ROOT / cfg["channel_level"]["feature_path"]
    if not feature_path.exists():
        pytest.skip(f"先跑 build_channel_hi: 缺 {feature_path}")
    return feature_path, cfg


def _iter_subs(f, n_traj=5):
    """迭代前 n_traj 轨迹的所有子阵组。"""
    for key in sorted(f.keys())[:n_traj]:
        traj = f[key]
        for sub_key in sorted(k for k in traj.keys() if k.startswith("sub_")):
            yield key, sub_key, traj[sub_key]


def test_hi_in_unit_interval():
    feature_path, _ = _load_channels()
    with h5py.File(feature_path, "r") as f:
        for _key, sub_key, sub in _iter_subs(f, n_traj=5):
            hi = sub["hi_ch"][:]
            assert hi.min() >= 0.0 and hi.max() <= 1.0, \
                f"{_key}/{sub_key} hi_ch 超界 [{hi.min()}, {hi.max()}]"


def test_z_monotone_low_violation():
    """z_ch 应单调非减 (f_s=eff_age/life_scale 累积非减 → dR/dI/dg 非减 → z=max 非减)。"""
    feature_path, _ = _load_channels()
    viols = []
    with h5py.File(feature_path, "r") as f:
        for _key, _sub_key, sub in _iter_subs(f, n_traj=5):
            z = sub["z_ch"][:]
            if len(z) > 1:
                viols.append(float(np.mean(np.diff(z) < -1e-9)))
    med_viol = float(np.median(viols)) if viols else 0.0
    assert med_viol < 0.05, f"z 单调违例率中位 {med_viol:.3f} 应 < 0.05"


def test_failed_channel_truncated_to_eol():
    """失效通道: rul 末端=0, 无 EOL 后零标签窗 (P0-1 铁律根除泄漏)。"""
    feature_path, _ = _load_channels()
    seen = False
    with h5py.File(feature_path, "r") as f:
        for _key, sub_key, sub in _iter_subs(f, n_traj=10):
            if bool(sub.attrs["event_observed"]):
                seen = True
                rul = sub["rul_ch"][:]
                assert abs(rul[-1]) < 1e-6, f"{_key}/{sub_key} 失效通道末端 rul 应=0, 实={rul[-1]}"
    assert seen, "应至少一个失效通道"


def test_censored_rul_capped():
    """删失通道 rul 封顶 cap_ratio*T (防模型从 t 反推仿真截止时刻, P0 复核第三条)。"""
    feature_path, cfg = _load_channels()
    cap_ratio = float(cfg["channel_level"]["rul_cap_ratio"])
    seen = False
    with h5py.File(feature_path, "r") as f:
        for _key, sub_key, sub in _iter_subs(f, n_traj=10):
            if not bool(sub.attrs["event_observed"]):
                seen = True
                rul = sub["rul_ch"][:]
                assert rul.max() <= cap_ratio * len(rul) + 1e-6, \
                    f"{_key}/{sub_key} 删失 rul max={rul.max()} 未封顶 {cap_ratio}*T"
    assert seen, "应至少一个删失通道"


def test_split_by_trajectory():
    """同 traj_id 的 16 sub_id 必须一致 (划分严格按轨迹, 不打散)。"""
    feature_path, _ = _load_channels()
    with h5py.File(feature_path, "r") as f:
        for key in sorted(f.keys())[:3]:
            traj = f[key]
            ids = {int(traj[sk].attrs["traj_id"])
                   for sk in traj.keys() if sk.startswith("sub_")}
            assert len(ids) == 1, f"{key} 子阵 traj_id 不一致: {ids}"
            # sub_id 覆盖 0..15
            sids = {int(traj[sk].attrs["sub_id"])
                    for sk in traj.keys() if sk.startswith("sub_")}
            assert sids == set(range(16)), f"{key} sub_id 应为 0..15, 实={sorted(sids)}"


def test_delta_only_from_config():
    """δ 阈值只来自 config (标定后写死, 不随实验数据变; 防阈值泄漏)。"""
    cfg = load_config(PA_CONFIG)
    deltas_cfg = cfg["channel_level"]["delta_thresholds"]
    feature_path, _ = _load_channels()
    with h5py.File(feature_path, "r") as f:
        raw = str(f.attrs.get("delta_thresholds", ""))
    stored = {}
    for kv in raw.split(","):
        if "=" in kv:
            k, v = kv.split("=")
            stored[k.strip()] = float(v)
    for k in ["R_DS", "I_DSS", "g_m", "P_out"]:
        assert k in stored, f"h5 缺 δ {k}"
        assert abs(stored[k] - float(deltas_cfg[k])) < 1e-9, \
            f"δ {k}: h5={stored[k]} vs config={deltas_cfg[k]} 不一致"
