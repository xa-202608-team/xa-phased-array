"""A3 GRU 嵌套前缀加载器的严格契约测试。"""
from __future__ import annotations

import pytest
import torch

from src.experiments.run_groups import (
    _GRU_L0_KEYS,
    _GRU_L1_KEYS,
    _GRU_PROJ_KEYS,
    _layerwise_depth_from_group,
    _load_layerwise_encoder,
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
