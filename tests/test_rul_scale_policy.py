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
    """channel + mission_horizon → factor=1.0 (数据已 /H, 不再除 4088)。"""
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