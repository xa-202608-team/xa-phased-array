# -*- coding: utf-8 -*-
"""过站观测协议适配层测试 (P1, LJ3 P0 审计落地)。"""
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.sim.pass_observation import (
    DEFAULTS, OBS_NO_PASS, OBS_NORMAL, OBS_UNCAL_PLACEHOLDER,
    apply_pass_protocol, build_pass_schedule, expected_hit_rate,
)


@pytest.fixture(scope="module")
def tiny_h5(tmp_path_factory):
    """3 traj × 2 sub × 800 窗的最小 channel_features 同构文件。"""
    p = tmp_path_factory.mktemp("passobs") / "channel_features.h5"
    rng = np.random.default_rng(0)
    with h5py.File(p, "w") as f:
        f.attrs["canonical_schema"] = "device_canonical_v1"   # 顶层 schema 元数据
        for t in range(3):
            # twin 组 (L2 物理孪生参数): 非 sub_ 成员须原样保留
            tw = f.create_group(f"traj_{t:03d}/twin")
            tw.create_dataset("twin_c_elem", data=rng.normal(size=16).astype(np.float32))
            for s in range(2):
                g = f.create_group(f"traj_{t:03d}/sub_{s:02d}")
                g.attrs["sub_id"] = t * 16 + s         # 训练/推理加载器要求的 attrs
                T = 800
                g.create_dataset("x_ch", data=rng.normal(size=(T, 4)).astype(np.float32))
                g.create_dataset("hi_ch", data=np.linspace(0, 1, T, dtype=np.float32))
                g.create_dataset("z_ch", data=np.linspace(0, 1, T, dtype=np.float32))
                g.create_dataset("rul_ch_norm", data=np.linspace(1, 0, T, dtype=np.float32))
                g.create_dataset("rul_ch_windows", data=np.linspace(800, 0, T, dtype=np.float32))
    return p


class TestPassSchedule:
    def test_hit_rate_near_nominal(self):
        """8.16h 间隔 vs 6h 窗 -> 名义命中率 ~0.735, 大样本偏差 < 0.03。"""
        rng = np.random.default_rng(1)
        hits = build_pass_schedule(20000, 21600.0, 8.16, 0.25, rng)
        assert abs(hits.mean() - expected_hit_rate(8.16)) < 0.03

    def test_seed_reproducible(self):
        a = build_pass_schedule(500, 21600.0, 8.16, 0.25, np.random.default_rng(7))
        b = build_pass_schedule(500, 21600.0, 8.16, 0.25, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_dense_interval_full_hit(self):
        """间隔 < 窗口 -> 命中率贴 1 (每窗必有站)。"""
        rng = np.random.default_rng(2)
        hits = build_pass_schedule(300, 21600.0, 3.0, 0.1, rng)
        assert hits.mean() > 0.98


class TestApplyProtocol:
    def test_status_semantics(self):
        """三态: 正常窗口量化有效; 无站/占位窗口 x_ch 全 NaN。"""
        rng = np.random.default_rng(3)
        T = 4000
        x = rng.normal(0.5, 0.2, size=(T, 4)).astype(np.float32)
        xq, st = apply_pass_protocol(x, 21600.0, DEFAULTS, rng)
        assert set(np.unique(st)) <= {0, 1, 2}
        invalid = st != OBS_NORMAL
        assert np.isnan(xq[invalid]).all()
        assert np.isfinite(xq[~invalid]).all()
        # 占位比例: 命中窗口内 ~29.3% (容差 3pp)
        hits = st != OBS_NO_PASS
        uncal_rate = (st == OBS_UNCAL_PLACEHOLDER).sum() / hits.sum()
        assert abs(uncal_rate - DEFAULTS["uncal_fraction"]) < 0.03

    def test_quantization_steps(self):
        """正常窗口各维取值落在 quant_steps 网格上。"""
        rng = np.random.default_rng(4)
        T = 4000
        x = rng.normal(0.5, 0.2, size=(T, 4)).astype(np.float32)
        xq, st = apply_pass_protocol(x, 21600.0, DEFAULTS, rng)
        ok = st == OBS_NORMAL
        steps = np.asarray(DEFAULTS["quant_steps"], dtype=np.float32)
        grid = np.round(xq[ok] / steps)
        np.testing.assert_allclose(xq[ok], grid * steps, rtol=0, atol=1e-5)

    def test_labels_semantics_untouched_by_caller(self):
        """适配只改 x_ch — 标签由调用方原样复制 (见 adapt_dataset), 本测试钉住该约定:
        apply_pass_protocol 输出形状与输入一致, 不额外动时间轴。"""
        rng = np.random.default_rng(5)
        x = rng.normal(size=(100, 4)).astype(np.float32)
        xq, st = apply_pass_protocol(x, 21600.0, DEFAULTS, rng)
        assert xq.shape == x.shape and st.shape == (100,)

    def test_input_not_mutated(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(200, 4)).astype(np.float32)
        x0 = x.copy()
        apply_pass_protocol(x, 21600.0, DEFAULTS, rng)
        np.testing.assert_array_equal(x, x0)


class TestAdaptDataset:
    def test_variant_structure_and_source_intact(self, tiny_h5, tmp_path):
        from src.sim.pass_observation import adapt_dataset
        dst = tmp_path / "channel_features_pass.h5"
        stats = adapt_dataset(tiny_h5, dst, DEFAULTS)
        assert stats["n_traj"] == 3 and stats["n_sub"] == 6
        assert stats["n_windows"] == stats["n_normal"] + stats["n_no_pass"] + stats["n_uncal"]
        with h5py.File(dst, "r") as fd, h5py.File(tiny_h5, "r") as fs:
            # 全部原有字段 + obs_status; 标签 bit-exact 复制
            for t in ("traj_000", "traj_001", "traj_002"):
                assert set(fd[f"{t}"].keys()) == set(fs[f"{t}"].keys())  # twin 也保留
                np.testing.assert_array_equal(fd[f"{t}/twin/twin_c_elem"][()],
                                              fs[f"{t}/twin/twin_c_elem"][()])
                for s in ("sub_00", "sub_01"):
                    gd, gs = fd[f"{t}/{s}"], fs[f"{t}/{s}"]
                    assert set(gd.keys()) == set(gs.keys()) | {"obs_status"}
                    assert gd.attrs["sub_id"] == gs.attrs["sub_id"]
                    for k in ("hi_ch", "z_ch", "rul_ch_norm", "rul_ch_windows"):
                        np.testing.assert_array_equal(gd[k][()], gs[k][()])
            assert fd.attrs["canonical_schema"] == fs.attrs["canonical_schema"]
            assert fd.attrs["adapter"].startswith("pass_observation_v1")
            assert "protocol_json" in fd.attrs
        # 源文件未被修改
        with h5py.File(tiny_h5, "r") as fs:
            assert "obs_status" not in fs["traj_000/sub_00"]
            assert np.isfinite(fs["traj_000/sub_00/x_ch"][()]).all()
