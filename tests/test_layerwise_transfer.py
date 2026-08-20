"""A3 GRU 嵌套前缀加载器的严格契约测试。"""
from __future__ import annotations

import pytest
import torch
import numpy as np

import src.experiments.run_groups as run_groups
from src.experiments.run_groups import (
    _GRU_L0_KEYS,
    _GRU_L1_KEYS,
    _GRU_PROJ_KEYS,
    _GROUP_MAP,
    LABELS,
    _layerwise_depth_from_group,
    _load_layerwise_encoder,
    _resolve_group,
    _source_ckpt_name,
)
from src.transfer.adapter import TransferModel


def _model(seed: int) -> TransferModel:
    torch.manual_seed(seed)
    return TransferModel(
        encoder_type="gru", n_features=4, n_target=4,
        latent_dim=8, adapter_hidden=8,
    )


def _source_encoder_state() -> dict[str, torch.Tensor]:
    source = _model(seed=11)
    return {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if key.startswith("encoder.")
    }


def test_layerwise_depth_identifies_only_p1_and_p2_groups():
    """错误解析普通组或 k-shot 后缀会使此测试失败。"""
    assert _layerwise_depth_from_group("ch_layerwise_gru_p1") == 1
    assert _layerwise_depth_from_group("ch_layerwise_gru_p2_k3") == 2
    assert _layerwise_depth_from_group("ch_source_mmd_physics") is None
    assert _layerwise_depth_from_group(None) is None


def test_p1_copies_only_first_gru_layer_and_preserves_all_other_tensors():
    """若 P1 错选层或覆盖未选参数，此测试会失败。"""
    model = _model(seed=23)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    source = _source_encoder_state()

    loaded = _load_layerwise_encoder(model, source, depth=1)

    assert set(loaded) == _GRU_L0_KEYS
    for key, value in model.state_dict().items():
        expected = source[key] if key in _GRU_L0_KEYS else before[key]
        assert torch.equal(value, expected), key


def test_p2_copies_both_gru_layers_without_copying_projection():
    """若 P2 漏 l1 或加载 projection，此测试会失败。"""
    model = _model(seed=23)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    source = _source_encoder_state()

    loaded = _load_layerwise_encoder(model, source, depth=2)

    expected_keys = _GRU_L0_KEYS | _GRU_L1_KEYS
    assert set(loaded) == expected_keys
    assert not expected_keys & _GRU_PROJ_KEYS
    for key in expected_keys:
        assert torch.equal(model.state_dict()[key], source[key]), key
    for key in _GRU_PROJ_KEYS:
        assert torch.equal(model.state_dict()[key], before[key]), key


def test_incompatible_later_prefix_tensor_leaves_model_unchanged():
    """若先复制再发现后续 l0 张量不兼容，此测试会失败。"""
    model = _model(seed=23)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    source = _source_encoder_state()
    source["encoder.gru.weight_ih_l0"] = torch.zeros(1)

    with pytest.raises(ValueError, match="张量不兼容"):
        _load_layerwise_encoder(model, source, depth=1)

    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key


@pytest.mark.parametrize(
    ("state_mutation", "depth", "match"),
    [
        (lambda state: state.pop("encoder.gru.bias_hh_l0"), 1, "missing"),
        (lambda state: state.__setitem__("encoder.extra", torch.zeros(1)), 1, "extra"),
        (lambda state: state.__setitem__("encoder.gru.weight_ih_l0", torch.zeros(1)), 1, "张量不兼容"),
        (lambda state: None, 3, "仅支持"),
    ],
)
def test_loader_rejects_invalid_encoder_contract(state_mutation, depth, match):
    """若接受缺额键、形状错配或非法深度，此测试会失败。"""
    source = _source_encoder_state()
    state_mutation(source)

    with pytest.raises(ValueError, match=match):
        _load_layerwise_encoder(_model(seed=23), source, depth=depth)


def test_p1_p2_groups_resolve_to_gru_and_mosfet_k3_checkpoint():
    """若 P1/P2 未注册、深度错配或改走非 MOSFET 源 checkpoint，此测试会失败。"""
    source_ckpt = _source_ckpt_name("ch_source_mmd_physics_k3", "phased_array", "gru")
    for depth in (1, 2):
        group = f"ch_layerwise_gru_p{depth}_k3"
        assert _GROUP_MAP[f"ch_layerwise_gru_p{depth}"] == ("layerwise_finetune", "gru")
        assert _resolve_group(f"ch_layerwise_gru_p{depth}", {})[:2] == ("layerwise_finetune", "gru")
        assert _layerwise_depth_from_group(group) == depth
        assert _source_ckpt_name(group, "phased_array", "gru") == source_ckpt
        assert LABELS[f"ch_layerwise_gru_p{depth}"] == f"CH Layer-wise GRU P{depth} *(A3)*"


@pytest.mark.parametrize("depth", [1, 2])
def test_runner_builds_random_model_before_loading_prefix_and_records_depth(
        monkeypatch, tmp_path, depth):
    """若 loader 在完整随机模型构建前触发，或结果漏 layerwise_depth，此测试会失败。"""
    cfg = run_groups.load_config("configs/phased_array.yaml")
    cfg["pretrain"]["device"] = "cpu"
    cfg["model"]["input_len_L"] = 4
    cfg["pretrain"]["seq_block_K"] = 2
    cfg["transfer"]["split"] = {"train": 1, "val": 1, "test": 1}
    cfg["reproducibility"]["deterministic"] = True
    cfg["reproducibility"]["cudnn_benchmark"] = False

    n_per_traj = 60
    tid = np.repeat(np.arange(3), n_per_traj)
    x_target = np.arange(len(tid) * 4, dtype=np.float32).reshape(len(tid), 4)
    hi_target = np.linspace(1.0, 0.1, len(tid), dtype=np.float32)
    rul_target = np.linspace(20.0, 1.0, len(tid), dtype=np.float32)
    events = []

    monkeypatch.setattr(run_groups, "CKPT_DIR", tmp_path)
    source = run_groups._build_model(cfg, 4, 4, "cpu", encoder_override="gru").state_dict()
    torch.save({"model": source}, tmp_path / "source_phased_array_gru_pretrain.pt")
    monkeypatch.setattr(
        run_groups, "load_target_channel",
        lambda *_args, **_kwargs: (x_target, hi_target, rul_target, tid, tid, np.ones(len(tid), dtype=bool),
                                    rul_target.copy(), 3, np.zeros(len(tid), dtype=int)))
    monkeypatch.setattr(
        run_groups, "load_source",
        lambda *_args, **_kwargs: (x_target, hi_target, tid, np.arange(len(tid)), np.array(["train"] * len(tid))))
    monkeypatch.setattr(run_groups, "split_trajectories", lambda *_args, **_kwargs: ([0], [1], [2]))
    monkeypatch.setattr(run_groups, "assert_split_by_trajectory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_groups, "_train_with_early_stop", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_groups, "eval_test", lambda *_args, **_kwargs: {
        "rmse": 0.1, "phm": 0.2, "mae": 0.3, "censor_violation_rate": 0.0})

    original_build = run_groups._build_model
    original_loader = run_groups._load_layerwise_encoder

    def build_spy(*args, **kwargs):
        events.append("build")
        return original_build(*args, **kwargs)

    def loader_spy(*args, **kwargs):
        events.append("load")
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(run_groups, "_build_model", build_spy)
    monkeypatch.setattr(run_groups, "_load_layerwise_encoder", loader_spy)

    result = run_groups.run_one_group(
        "layerwise_finetune", seed=7, cfg=cfg, smoke=True, encoder_override="gru",
        component="phased_array", group_name=f"ch_layerwise_gru_p{depth}_k3",
        level="channel", k_shot=3)

    assert events == ["build", "load"]
    assert result["layerwise_depth"] == depth
