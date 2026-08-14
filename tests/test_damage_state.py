"""GaN 三状态迁移模型的行为契约（先于实现）。"""
from __future__ import annotations

import torch


def _model():
    from src.transfer.damage_state import DamageStateModel
    return DamageStateModel(source_input_dim=20, target_input_dim=12, stress_dim=5, hidden_dim=16)


def test_transition_preserves_state_dimensions_and_permanent_monotonicity():
    model = _model()
    state = torch.tensor([[0.2, 0.8, 0.1], [0.4, 0.3, 0.5]])
    stress = torch.randn(2, 5)

    nxt = model.transition(state, stress, torch.ones(2, 1))

    assert nxt.shape == (2, 3)
    assert torch.all(nxt[:, 0] >= state[:, 0])
    assert torch.all(nxt[:, 2] >= state[:, 2])
    assert torch.all((0.0 <= nxt[:, 1]) & (nxt[:, 1] <= 1.0))


def test_transition_allows_trap_recovery_under_recovery_condition():
    model = _model()
    state = torch.tensor([[0.3, 0.9, 0.2]])
    stress = torch.zeros(1, 5)

    nxt = model.transition(state, stress, torch.ones(1, 1), recovery=torch.tensor([True]))

    assert nxt[0, 1] < state[0, 1]
    assert nxt[0, 0] >= state[0, 0]
    assert nxt[0, 2] >= state[0, 2]


def test_transition_output_changes_with_normalized_dt():
    model = _model()
    state = torch.tensor([[0.3, 0.5, 0.2]])
    stress = torch.ones(1, 5)

    immediate = model.transition(state, stress, torch.zeros(1, 1))
    delayed = model.transition(state, stress, torch.ones(1, 1))

    assert torch.allclose(immediate, state)
    assert not torch.allclose(delayed, immediate)


def test_source_and_target_observation_heads_are_separate():
    model = _model()
    state = torch.rand(3, 3)

    source = model.observe_source(state)
    target = model.observe_target(state)

    assert source.shape == (3, 4)
    assert target.shape == (3, 4)
    assert model.source_head is not model.target_head


def test_optional_spatial_node_interface_encodes_and_observes_each_subarray():
    from src.transfer.damage_state import DamageStateModel

    model = DamageStateModel(source_input_dim=20, target_input_dim=12, target_node_input_dim=6, hidden_dim=16)
    node_observations = torch.randn(5, 16, 6)

    node_states = model.encode_target_nodes(node_observations)
    node_channels = model.observe_target_nodes(node_states)

    assert node_states.shape == (5, 16, 3)
    assert node_channels.shape == (5, 16, 4)
    assert torch.all(node_states[..., 0] >= 0.0) and torch.all(node_states[..., 2] >= 0.0)
    assert torch.all((0.0 <= node_states[..., 1]) & (node_states[..., 1] <= 1.0))


def test_node_next_state_uses_shared_transition_with_batched_nodes():
    from src.transfer.damage_state import DamageStateModel

    model = DamageStateModel(source_input_dim=20, target_input_dim=12, target_node_input_dim=6, hidden_dim=16)
    node_observations = torch.randn(3, 16, 6)
    dt = torch.full((3, 16, 1), 0.5)

    next_state = model.next_target_node_state(node_observations, dt)
    expected = model.transition(model.encode_target_nodes(node_observations), model.target_node_stress(node_observations), dt)

    assert next_state.shape == (3, 16, 3)
    assert torch.allclose(next_state, expected)


def test_transition_checkpoint_load_does_not_copy_source_or_target_heads():
    torch.manual_seed(1)
    source_model = _model()
    torch.manual_seed(2)
    target_model = _model()
    target_head_before = {k: v.detach().clone() for k, v in target_model.target_head.state_dict().items()}
    source_head_before = {k: v.detach().clone() for k, v in target_model.source_head.state_dict().items()}

    loaded = target_model.load_transition_state_dict(source_model.transition_state_dict())

    assert loaded
    for key, value in source_model.transition.state_dict().items():
        assert torch.equal(target_model.transition.state_dict()[key], value)
    for key, value in target_head_before.items():
        assert torch.equal(target_model.target_head.state_dict()[key], value)
    for key, value in source_head_before.items():
        assert torch.equal(target_model.source_head.state_dict()[key], value)
