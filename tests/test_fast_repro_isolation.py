# -*- coding: utf-8 -*-
"""reproduce_full --fast 数据隔离与 predict_gru 合格通道选择回归。

背景 (2026-08-23 F6): fast 模式曾直写 canonical 数据槽位, 覆盖 200 轨迹冻结仿真
基线与 F2 canonical 特征 (Defect A); 且 predict_gru --limit-channels 取前 N 个
通道键不筛长度, 在含早失效短通道的数据上全池 T<L 直接抛错 (Defect B)。
"""
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---- Defect B: 合格通道选择 (T >= L) ----

def test_eligible_channel_keys_filters_short_and_keeps_sorted_order():
    from src.transfer.channel_inference import eligible_channel_keys

    ck = np.array([7] * 9 + [3] * 3 + [5] * 6 + [2] * 8)   # T: 7→9, 3→3, 5→6, 2→8
    assert list(eligible_channel_keys(ck, L=5)) == [2, 5, 7]   # 3 太短
    assert list(eligible_channel_keys(ck, L=6)) == [2, 5, 7]   # 5 为 T==L 边界, 合格
    assert list(eligible_channel_keys(ck, L=7)) == [2, 7]      # 5 出局
    assert list(eligible_channel_keys(ck, L=9)) == [7]         # T==L 边界合格
    assert list(eligible_channel_keys(ck, L=10)) == []


def test_eligible_channel_keys_boundary_includes_equal_length():
    from src.transfer.channel_inference import eligible_channel_keys

    ck = np.array([1] * 5 + [2] * 6)
    assert list(eligible_channel_keys(ck, L=5)) == [1, 2]   # T==L 合格
    assert list(eligible_channel_keys(ck, L=6)) == [2]
    assert list(eligible_channel_keys(ck, L=7)) == []       # 全不合格 -> 空池


def test_limit_keep_mask_prefers_eligible_channels():
    from component.predict_gru import limit_keep_mask

    ck = np.array([10] * 3 + [20] * 8 + [30] * 3)           # 10/30 短 (T=3), 20 长 (T=8)
    mask = limit_keep_mask(ck, limit_channels=2, L=5)
    assert set(ck[mask]) == {20}                            # 只从合格通道取


def test_limit_keep_mask_keeps_first_sorted_eligible_behavior():
    from component.predict_gru import limit_keep_mask

    # 全合格数据: 保持旧语义 (按序取前 N 个通道键)
    ck = np.array([10] * 8 + [20] * 6 + [30] * 7)
    mask = limit_keep_mask(ck, limit_channels=1, L=5)
    assert set(ck[mask]) == {10}
    mask = limit_keep_mask(ck, limit_channels=2, L=5)
    assert set(ck[mask]) == {10, 20}
    # 无 limit: 不裁 (None 语义由调用方处理)


# ---- Defect A: fast 数据隔离 ----

def test_build_fast_config_redirects_feature_paths_only(tmp_path):
    from scripts.reproduce_full import build_fast_config

    cfg = {
        "channel_level": {
            "feature_path": "data/features/phased_array/schema_ch_v1/target/channel_features.h5",
            "window_len": 64},
        "transfer": {
            "target_feature_path": "data/features/phased_array/schema_v1/target/target_features.h5",
            "mode": "hi_dynamics"},
        "sim": {"seed": 42},
    }
    original = copy.deepcopy(cfg)
    fast = build_fast_config(cfg, tmp_path)

    ch = Path(fast["channel_level"]["feature_path"]).as_posix()
    tg = Path(fast["transfer"]["target_feature_path"]).as_posix()
    assert ch.endswith("data/features/channel_features.h5") and tmp_path.name in ch
    assert tg.endswith("data/features/target_features.h5") and tmp_path.name in tg
    # 其余键与子键原样保留
    assert fast["channel_level"]["window_len"] == 64
    assert fast["transfer"]["mode"] == "hi_dynamics"
    assert fast["sim"] == {"seed": 42}
    # 原 cfg 不被就地修改
    assert cfg == original


def test_sim_dir_for_canonical_default_and_data_root_redirect():
    from scripts.generate_simulation import sim_dir_for

    assert sim_dir_for(None, "on", 42) == Path(
        "data/simulated/phased_array/sim_v2/seed_42")
    assert sim_dir_for(None, "off", 42) == Path(
        "data/simulated/phased_array/sim_v1/seed_42")
    root = Path("outputs/f6_full_fast/data/simulated/phased_array")
    assert sim_dir_for(root, "on", 42) == root / "sim_v2" / "seed_42"
    assert sim_dir_for(root, "off", 42) == root / "sim_v1" / "seed_42"
