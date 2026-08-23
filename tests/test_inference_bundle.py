# -*- coding: utf-8 -*-
"""F1 批3: 推理 bundle 导出 + predict_gru 整链测试。

覆盖:
  - run_one_group(export_spec) → model.pt + bundle.json 落盘, 字段/尺度/统计量正确
  - export_inference_bundle 单元行为 (v1 拒绝 / strict 重建往返)
  - component.predict_gru 整链: 无标签读 → factory 重建 → 预测 → schema 校验 →
    rul_norm/windows/days 三口径换算一致
全部 tmp_path fixture, clean checkout 可跑 (不依赖真实大数据)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.run_groups import (                                  # noqa: E402
    INFERENCE_BUNDLE_SCHEMA, export_inference_bundle, run_one_group)
from src.sim.build_channel_hi import CANONICAL_COLS                       # noqa: E402
from src.transfer.channel_dataset import read_channel_label_meta          # noqa: E402

H = 11688.0
SP_S = 21600.0
N_SUB = 16


# ---------------------------------------------------------------- fixtures
def _write_channel_h5(path, n_traj=4, n_sub=2, T=512):
    """最小 v2 channel_features.h5: sub_00 失效 / sub_01 删失, x_ch 含轨迹信息。"""
    t = np.arange(T, dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.attrs["channel_label_schema"] = "channel_label_v2"
        f.attrs["rul_scale_windows"] = H
        f.attrs["sample_period_s"] = SP_S
        f.attrs["mission_horizon_windows"] = H
        f.attrs["rul_capped"] = "false"
        f.attrs["t_dev_unit"] = "degC"
        for ti in range(n_traj):
            traj = f.create_group(f"traj_{ti:03d}")
            for si in range(n_sub):
                sub = traj.create_group(f"sub_{si:02d}")
                sub.attrs["sub_id"] = si
                failed = (si == 0)
                sub.attrs["event_observed"] = failed
                eol = T - 64 if failed else T              # 失效通道截到 eol+1 行
                Te = eol + 1 if failed else T
                base = 0.1 * ti + 0.05 * si
                ramp = np.linspace(0.0, 1.0, Te, dtype=np.float32)
                x = np.stack([ramp * (1 + base),                            # p_drift
                              80.0 + 40.0 * ramp + 5.0 * base,              # T_dev_C
                              np.full(Te, 0.5, dtype=np.float32),           # duty
                              ramp * base], axis=-1).astype(np.float32)
                hi = np.clip(ramp * (1.0 if failed else 0.9), 0.0, 1.0)
                rul_w = np.maximum(eol - t[:Te], 0).astype(np.float32)
                sub.create_dataset("x_ch", data=x)
                sub.create_dataset("hi_ch", data=hi.astype(np.float32))
                sub.create_dataset("rul_ch_windows", data=rul_w)
                sub.create_dataset("rul_ch_norm", data=(rul_w / H).astype(np.float32))
    return path


def _write_source_h5(path, n_dev=2, T=256):
    """最小 v2 分组源域 h5 (devices/<id>: x/hi/rul_s/rul_lower_bound_s/elapsed_time_s)。"""
    with h5py.File(path, "w") as f:
        grp = f.create_group("devices")
        for di in range(n_dev):
            g = grp.create_group(f"dev_{di:02d}")
            t = np.arange(T, dtype=np.float64)
            ramp = np.linspace(0.0, 1.0, T, dtype=np.float32)
            x = np.stack([ramp, 90.0 + 30.0 * ramp, np.full(T, 0.5, np.float32),
                          ramp], axis=-1).astype(np.float32)
            g.create_dataset("x", data=x)
            g.create_dataset("hi", data=ramp.copy())
            g.create_dataset("rul_s", data=((T - t) * 3600.0).astype(np.float32))
            g.create_dataset("rul_lower_bound_s",
                             data=((T - t) * 3600.0).astype(np.float32))
            g.create_dataset("elapsed_time_s", data=(t * 3600.0))
            g.attrs["event_observed"] = True
    return path


def _make_cfg(tmp_path, ch_h5, src_h5):
    return {
        "seed": 42,
        "reproducibility": {"deterministic": True, "cudnn_benchmark": False},
        "model": {
            "encoder": "gru", "input_len_L": 64, "latent_dim": 8,
            "tcn": {"channels": 8, "kernel_size": 3, "num_blocks": 2, "dropout": 0.0},
            "gru": {"hidden": 8, "num_layers": 1, "dropout": 0.0},
        },
        "loss": {"huber_delta": 1.0},
        "pretrain": {"batch_size": 4, "device": "cpu", "seq_block_K": 8,
                     "canonical_source_path": str(src_h5),
                     "source_id_field": "device_id"},
        "transfer": {"hi_bins": [[0.0, 0.5], [0.5, 1.0]], "mmd_lambda": 1.0,
                     "split": {"train": 0.5, "val": 0.25, "test": 0.25},
                     "finetune_lr": 1e-3, "epochs_s2": 2, "rul_max_norm": 4088,
                     "target_stride": 50, "adapter_hidden": 8,
                     "ablation_drop_features": []},
        "channel_level": {"feature_path": str(ch_h5),
                          "rul_scale_policy": "mission_horizon"},
        "source": {"split": {"val_device_ids": []}},
        "experiments": {"primary_model_group": "ch_target_only_gru"},
    }


# ------------------------------------------------- run_one_group 导出钩子 (e2e)
def test_run_one_group_exports_bundle(tmp_path):
    ch = _write_channel_h5(tmp_path / "ch.h5")
    src = _write_source_h5(tmp_path / "src.h5")
    cfg = _make_cfg(tmp_path, ch, src)
    out = tmp_path / "bundle"
    m = run_one_group("target_only", 42, cfg, smoke=True, component="phased_array",
                      group_name="ch_target_only_gru", level="channel",
                      export_spec={"dir": out, "group": "ch_target_only_gru", "seed": 42})
    assert m["group"] == "ch_target_only_gru"
    assert (out / "model.pt").is_file()
    b = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    assert b["bundle_schema"] == INFERENCE_BUNDLE_SCHEMA
    assert b["group"] == "ch_target_only_gru" and b["seed"] == 42
    assert b["encoder"] == "gru" and b["input_len_L"] == 64
    assert b["feature_names"] == CANONICAL_COLS
    assert b["n_features"] == 4 and b["n_target"] == 4
    assert len(b["normalizer"]["mean"]) == 4 and len(b["normalizer"]["std"]) == 4
    assert b["rul"] == {"channel_label_schema": "channel_label_v2",
                        "rul_scale_windows": H, "sample_period_s": SP_S}
    assert b["metrics"]["val_rmse"] == m["val_rmse"]
    assert len(b["git_commit"]) == 40


def test_run_one_group_no_export_without_spec(tmp_path):
    """不传 export_spec 时零导出副作用 (正常训练跑不受影响)。"""
    ch = _write_channel_h5(tmp_path / "ch.h5")
    src = _write_source_h5(tmp_path / "src.h5")
    cfg = _make_cfg(tmp_path, ch, src)
    out = tmp_path / "bundle"
    run_one_group("target_only", 42, cfg, smoke=True, component="phased_array",
                  group_name="ch_target_only_gru", level="channel",
                  export_spec={"dir": out, "group": "ch_other", "seed": 42})
    assert not out.exists()


# ------------------------------------------------- export 单元行为
def test_export_rejects_v1_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("XA_GIT_COMMIT", "a" * 40)
    from src.models.factory import build_transfer_model
    ch = _write_channel_h5(tmp_path / "ch.h5")
    cfg = _make_cfg(tmp_path, ch, tmp_path / "src.h5")
    model = build_transfer_model(cfg, n_features=4, n_target=4, encoder_type="gru")
    v1_meta = {"channel_label_schema": "channel_label_v1"}
    with pytest.raises(ValueError, match="channel_label_v2"):
        export_inference_bundle(
            model, tmp_path / "b", cfg=cfg, component="phased_array",
            group_name="g", seed=42, encoder="gru", ch_meta=v1_meta,
            feature_mean=np.zeros(4), feature_std=np.ones(4),
            drop_features=[], metrics={"rmse": 0.1, "phm": 1.0, "val_rmse": 0.2},
            L=64, n_features=4, n_target=4)


# ------------------------------------------------- predict_gru 整链
@pytest.fixture(scope="module")
def _trained_bundle(tmp_path_factory):
    """module 级共享: 训练一次 (smoke tiny) + 导出 bundle + 返回路径。"""
    tmp = tmp_path_factory.mktemp("gru_bundle")
    ch = _write_channel_h5(tmp / "ch.h5")
    src = _write_source_h5(tmp / "src.h5")
    cfg = _make_cfg(tmp, ch, src)
    out = tmp / "bundle"
    run_one_group("target_only", 42, cfg, smoke=True, component="phased_array",
                  group_name="ch_target_only_gru", level="channel",
                  export_spec={"dir": out, "group": "ch_target_only_gru", "seed": 42})
    return ch, out


def test_predict_gru_end_to_end(_trained_bundle, tmp_path, monkeypatch):
    from component import predict_gru
    ch, bundle_dir = _trained_bundle
    monkeypatch.setenv("XA_GIT_COMMIT", "b" * 40)
    out = tmp_path / "pred"
    monkeypatch.setattr(sys, "argv", [
        "predict_gru", "--features", str(ch), "--bundle-dir", str(bundle_dir),
        "--output", str(out), "--stride", "100", "--limit-channels", "3"])
    assert predict_gru.main() == 0
    doc = json.loads((out / "rul_prediction.json").read_text(encoding="utf-8"))
    # schema 出厂校验已在 main 内做; 这里复核关键契约
    assert doc["method"] == "gru_channel_rul" and doc["status"] == "RUL_PREDICTION_OK"
    assert doc["model"]["group"] == "ch_target_only_gru"
    assert doc["model"]["feature_names"] == CANONICAL_COLS
    assert doc["rul_scale"]["rul_scale_windows"] == H
    assert len(doc["predictions"]) >= 1
    keys = {p["channel_key"] for p in doc["predictions"]}
    assert keys <= {k for k in range(0, 3 * N_SUB)}       # limit-channels=3 → ck 0..47 内
    for p in doc["predictions"]:
        # 三口径换算一致: days = norm × H × sp_s / 86400
        expect = p["rul_norm"] * H * SP_S / 86400.0
        assert abs(p["rul_days"] - expect) < 0.02
        assert p["rul_windows"] >= 0 and p["rul_days"] >= 0
        # channel_key 分解 = 训练侧约定 traj*16+sub
        assert p["traj_id"] * N_SUB + p["sub_id"] == p["channel_key"]
        assert "rul_ch" not in json.dumps(p)               # 无真值字段混入


def test_predict_gru_bundle_model_roundtrip(_trained_bundle):
    """factory 重建 + strict 加载往返: bundle 参数足以重建训练时架构。"""
    from component.predict_gru import load_bundle, rebuild_model
    ch, bundle_dir = _trained_bundle
    b = load_bundle(bundle_dir)
    model = rebuild_model(b, bundle_dir)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    # GRU encoder 在架构里 (gru_hidden=8 单层)
    has_gru = any("gru" in k for k, _ in model.named_parameters())
    assert has_gru


def test_predict_gru_rejects_mismatched_scale(_trained_bundle, tmp_path):
    """bundle 与遥测 h5 的 H 不一致 → 显式拒绝 (尺度错配不得静默推理)。"""
    from component import predict_gru
    ch, bundle_dir = _trained_bundle
    bad = tmp_path / "bad.h5"
    import shutil
    shutil.copy(ch, bad)
    with h5py.File(bad, "a") as f:
        f.attrs["rul_scale_windows"] = H * 2
        f.attrs["mission_horizon_windows"] = H * 2
    with pytest.raises(ValueError, match="不一致"):
        predict_gru.predict(bad, bundle_dir, stride=100)
