"""RUL 尺度策略解析 (F1 收口, channel_label_v2)。

run_groups 归一分流：channel_level.rul_scale_policy=mission_horizon 时
数据已由 build_channel_hi 除以全局 H，run_groups 不得再除以 transfer.rul_max_norm(4088)
(否则双重重归一)。服务级路径仍使用 transfer.rul_max_norm。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                              # noqa: E402
from src.experiments.run_groups import _resolve_rul_scale      # noqa: E402

PA_CONFIG = ROOT / "configs" / "phased_array.yaml"


def test_channel_mission_horizon_no_second_norm():
    """channel + mission_horizon → factor=1.0 (数据已 /H, 不再除 4088)。

    F1-A 定位注: 本路径是 legacy/v1 兼容回归 — _resolve_rul_scale 对 channel+mission_horizon
    已无 v2 生产调用方 (run_groups 对 channel v2 直接从 channel_features.h5 meta 读
    factor=1.0/H, 见 run_groups 中 channel 分支), 但函数未删 (service fallback / v1 legacy
    仍用), 该测试保留作回归保护。函数确实仍返回 (1.0, 11688)。
    """
    cfg = load_config(PA_CONFIG)
    ch = cfg["channel_level"]
    factor, scale_windows = _resolve_rul_scale(
        level="channel", channel_cfg=ch, transfer_cfg=cfg["transfer"])
    assert factor == 1.0, f"v2 数据不得二次归一, factor 应=1.0, 实={factor}"
    assert scale_windows == 11688, f"H 应来自解析, 实={scale_windows}"


def test_channel_legacy_cap_uses_transfer_max():
    """channel + legacy_cap → factor=rul_max_norm (旧 v1 路径兼容)。"""
    cfg = load_config(PA_CONFIG)
    ch = dict(cfg["channel_level"])
    ch["rul_scale_policy"] = "legacy_cap"
    factor, _ = _resolve_rul_scale(level="channel", channel_cfg=ch,
                                   transfer_cfg=cfg["transfer"])
    assert factor == pytest.approx(float(cfg["transfer"]["rul_max_norm"])), \
        f"legacy 路径应沿用 transfer.rul_max_norm, 实={factor}"


def test_service_level_keeps_transfer_max():
    """service (旧服务级) → factor=rul_max_norm，不受 channel v2 策略影响。"""
    cfg = load_config(PA_CONFIG)
    factor, _ = _resolve_rul_scale(level="service", channel_cfg=cfg["channel_level"],
                                   transfer_cfg=cfg["transfer"])
    assert factor == pytest.approx(float(cfg["transfer"]["rul_max_norm"]))

# ============================================================ F2 收口: level_control 跨尺度纪律

def _mk_arm(group, seed, rmse, scale):
    """构造 aggregate() 可用的最小指标 dict。"""
    d = dict(group=group, seed=seed, rmse=rmse, phm=1.0, mae=rmse * 0.9,
             mode="m", encoder="gru", level="channel" if group.startswith("ch_") else "service",
             val_rmse=rmse)
    if scale is not None:
        d["rul_scale_windows"] = float(scale)
    return d


def test_level_control_pairs_in_absolute_windows():
    """无 k-shot 裸名回退 + 跨层级配对在绝对窗口口径 (rmse×各自尺度)。

    通道 ÷11688 / 服务 ÷4088 归一口径不同, 直接比归一 RMSE 是尺度混用;
    期望 delta = svc_rmse×4088 − ch_rmse×11688。
    """
    from src.experiments.run_groups import aggregate
    ch = [_mk_arm("ch_source_mmd_physics", s, 0.10, 11688.0) for s in (42, 43)]
    sv = [_mk_arm("cross_level_transfer", s, 0.20, 4088.0) for s in (42, 43)]
    agg = aggregate({"ch_source_mmd_physics": ch, "cross_level_transfer": sv})
    lc = agg["_level_control"]
    assert lc["channel_group"] == "ch_source_mmd_physics"      # 裸名回退 (旧硬编码 _kall)
    assert lc["unit"] == "windows"
    assert lc["channel_scale"] == 11688.0 and lc["service_scale"] == 4088.0
    expect = 0.20 * 4088.0 - 0.10 * 11688.0                    # = -351.2 窗口
    assert lc["paired"]["delta_mean"] == pytest.approx(expect, abs=1e-6)


def test_level_control_kall_preferred_when_present():
    """k-shot 跑 (*_kall 后缀) 仍优先配 kall 组 (历史行为不变)。"""
    from src.experiments.run_groups import aggregate
    ch_k = [_mk_arm("ch_source_mmd_physics_kall", s, 0.10, 11688.0) for s in (42,)]
    sv = [_mk_arm("cross_level_transfer", s, 0.20, 4088.0) for s in (42,)]
    agg = aggregate({"ch_source_mmd_physics_kall": ch_k, "cross_level_transfer": sv})
    assert agg["_level_control"]["channel_group"] == "ch_source_mmd_physics_kall"


def test_level_control_refuses_without_scale_records():
    """旧记录缺 rul_scale_windows → 拒绝跨尺度配对 (paired=None + 显式原因), 不静默比较。"""
    from src.experiments.run_groups import aggregate
    ch = [_mk_arm("ch_source_mmd_physics", s, 0.10, None) for s in (42, 43)]
    sv = [_mk_arm("cross_level_transfer", s, 0.20, None) for s in (42, 43)]
    agg = aggregate({"ch_source_mmd_physics": ch, "cross_level_transfer": sv})
    lc = agg["_level_control"]
    assert lc["paired"] is None
    assert lc["unit"] == "normalized_incomparable"
    assert "rul_scale_windows" in lc["reason"]


def test_attribution_controls_fallback_to_plain_names():
    """init/full/mmd 三归因对照: 无 k-shot 时回退裸 ch_* 组名 (旧: 只认 _kall → 静默 None)。"""
    from src.experiments.run_groups import aggregate
    by = {
        "ch_source_pretrain_frozen": [_mk_arm("ch_source_pretrain_frozen", 42, 0.16, 11688.0)],
        "ch_random_frozen": [_mk_arm("ch_random_frozen", 42, 0.17, 11688.0)],
        "ch_source_mmd_physics": [_mk_arm("ch_source_mmd_physics", 42, 0.15, 11688.0)],
        "ch_random_full_finetune": [_mk_arm("ch_random_full_finetune", 42, 0.15, 11688.0)],
        "ch_random_nommd": [_mk_arm("ch_random_nommd", 42, 0.13, 11688.0)],
    }
    agg = aggregate(by)
    pr = agg["_paired"]
    assert pr["init_control"] is not None
    assert pr["full_control"] is not None
    assert pr["mmd_control"] is not None
    # delta = source − random (in-level 同尺度, 归一口径即有效)
    assert pr["init_control"]["delta_mean"] == pytest.approx(0.16 - 0.17, abs=1e-9)
    assert pr["mmd_control"]["delta_mean"] == pytest.approx(0.15 - 0.13, abs=1e-9)
