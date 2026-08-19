"""B 路线 (§4e) sim_v1 canonical 转换契约测试.

scripts/data/build_simv1_canonical.py 的核心语义:
  ① Test 编号 = traj*16+sub+1, val = 最后 N traj 整组 (同 traj 16 sub 同 split);
  ② rul_s = rul_ch(步) × sample_period_s; rul_lower_bound_s ≡ rul_s (mosfet 同口径);
  ③ 输出与 mosfet_canonical.h5 结构同构 (devices/Test_N + x/hi/rul_s/lb/elapsed).
"""
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "build_simv1_canonical.py"


def _mk_v1_channel(path: Path, n_traj=3, n_sub=4, T=50):
    """构造最小 v1 channel_features.h5 (legacy 结构)。"""
    with h5py.File(path, "w") as f:
        f.attrs["dynamics_id"] = "leo_coupled_v1"
        for t in range(n_traj):
            g = f.create_group(f"traj_{t:03d}")
            for s in range(n_sub):
                sg = g.create_group(f"sub_{s:02d}")
                rul = (T - 1 - np.arange(T)).astype(np.float32)
                sg.create_dataset("x_ch", data=np.random.rand(T, 4).astype(np.float32))
                sg.create_dataset("hi_ch", data=np.linspace(0, 1, T, dtype=np.float32))
                sg.create_dataset("rul_ch", data=rul)
                sg.attrs["event_observed"] = 1
                sg.attrs["eol_idx"] = T - 1


def test_simv1_canonical_contract(tmp_path):
    src = tmp_path / "v1_ch.h5"
    out = tmp_path / "simv1_canonical.h5"
    _mk_v1_channel(src, n_traj=3, n_sub=4, T=50)

    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(ROOT / "configs" / "phased_array.yaml"),
         "--in", str(src), "--out", str(out), "--val-n-traj", "1"],
        capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]

    with h5py.File(out, "r") as f:
        assert f.attrs["schema"] == "device_canonical_v1"
        assert f.attrs["source_dynamics_id"] == "leo_coupled_v1"
        devs = f["devices"]
        assert len(devs.keys()) == 12                    # 3 traj × 4 sub
        # ① Test 编号映射: traj0→Test_1..4, traj1→Test_5..8, traj2(val)→Test_9..12
        assert set(devs.keys()) == {f"Test_{i}" for i in range(1, 13)}
        g = devs["Test_1"]
        # ② 结构同构 + rul 秒换算 (21600 s/步)
        assert set(g.keys()) == {"x", "hi", "rul_s", "rul_lower_bound_s", "elapsed_time_s"}
        assert g["rul_s"].shape == (50,)
        np.testing.assert_allclose(g["rul_s"][:], g["rul_lower_bound_s"][:])
        np.testing.assert_allclose(g["rul_s"][0], (50 - 1) * 21600.0)
        np.testing.assert_allclose(g["elapsed_time_s"][:3], [0, 21600, 43200])
        assert g.attrs["event_observed"] == 1
        assert g.attrs["original_T"] == 50

    # ① val 名单 = 最后 1 traj 整组
    val = (tmp_path / "simv1_canonical_val_ids.txt").read_text().strip()
    assert val == ",".join(f"Test_{i}" for i in range(9, 13))


def test_simv1_val_is_whole_trajectory(tmp_path):
    """val 划分必须整 traj: 同 traj 的全部 sub 都在 val 名单, 其他 traj 一个都不在。"""
    src = tmp_path / "v1_ch.h5"
    out = tmp_path / "simv1_canonical.h5"
    _mk_v1_channel(src, n_traj=4, n_sub=4, T=30)
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(ROOT / "configs" / "phased_array.yaml"),
         "--in", str(src), "--out", str(out), "--val-n-traj", "2"],
        capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    val = set((tmp_path / "simv1_canonical_val_ids.txt").read_text().strip().split(","))
    assert val == {f"Test_{i}" for i in range(9, 17)}   # traj 2,3 (各 4 sub)


@pytest.mark.parametrize("group,expect", [
    ("ch_simv1_source", ("source_mmd_finetune", "gru")),
])
def test_simv1_group_registered(group, expect):
    from src.experiments.run_groups import _GROUP_MAP, _source_ckpt_name
    assert _GROUP_MAP[group] == expect
    # ckpt tag 分支: simv1 → source_phased_array_simv1_simv1_gru_pretrain.pt
    # (component=phased_array_simv1 即 config stem; pretrain 同规则产出, 两侧自洽)
    name = _source_ckpt_name("ch_simv1_source_k3", "phased_array_simv1", "gru")
    assert name == "source_phased_array_simv1_simv1_gru_pretrain.pt"
