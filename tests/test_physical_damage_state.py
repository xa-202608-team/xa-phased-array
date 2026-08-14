"""物理化 damage_state 的行为契约（Gate 1.1 物理 Fθ 正式版，先于 Gate 2 冻结迁移）。

与 test_damage_state.py 的旧 DamageStateModel 契约并列；旧类保留以守 §6-7 预注册
口径，本文件锁物理化路径：物理 u 外部传入（无 stress MLP）、状态物理尺度归一、
d/r 耦合（Δr=14·Δd，呼应源仿真 r_th=14·d_perm）、与诊断脚本 Gate 1.1 已验证
PhysicalTransition 同构。
"""
from __future__ import annotations

import torch


def _model():
    from src.transfer.damage_state import PhysicalDamageStateModel
    return PhysicalDamageStateModel(source_input_dim=20, target_input_dim=12, hidden_dim=16)


# ---- PhysicalDamageTransition 契约 ----

def test_physical_transition_matches_gate1_1_diagnostic_implementation():
    """正式 transition 与 Gate 1.1 已验证的诊断 PhysicalTransition 同构等价。

    同架构 → state_dict 互载后 forward 逐元素相等，确保正式版未偏离已得 +38%
    skill 的实现，Gate 2 可直接复用诊断训练逻辑。
    """
    from src.experiments.run_gan_diagnostics import PhysicalTransition as DiagTransition
    from src.transfer.damage_state import PhysicalDamageTransition

    torch.manual_seed(7)
    formal = PhysicalDamageTransition()
    diag = DiagTransition()
    diag.load_state_dict(formal.state_dict())  # 同结构 → key 完全对齐
    z = torch.rand(6, 3); u = torch.rand(6, 3); dt = torch.full((6, 1), 0.5)
    assert torch.allclose(formal(z, u, dt), diag(z, u, dt), atol=1e-7)


def test_physical_transition_zero_dt_is_identity():
    from src.transfer.damage_state import PhysicalDamageTransition

    transition = PhysicalDamageTransition()
    z = torch.rand(4, 3); u = torch.rand(4, 3)
    out = transition(z, u, torch.zeros(4, 1))
    assert torch.allclose(out, z, atol=1e-7)


def test_physical_transition_dt_drives_change():
    from src.transfer.damage_state import PhysicalDamageTransition

    transition = PhysicalDamageTransition()
    z = torch.rand(4, 3); u = torch.rand(4, 3)
    immediate = transition(z, u, torch.zeros(4, 1))
    delayed = transition(z, u, torch.ones(4, 1))
    assert torch.allclose(immediate, z, atol=1e-7)
    assert not torch.allclose(delayed, immediate)


def test_physical_transition_couples_r_to_d_in_normalized_space():
    """d/r 耦合：归一空间 Δr̃=Δd̃（源自源仿真 r_th=14·d_perm，归一后同增量）。"""
    from src.transfer.damage_state import PhysicalDamageTransition

    transition = PhysicalDamageTransition()
    z = torch.rand(5, 3); u = torch.rand(5, 3)
    out = transition(z, u, torch.ones(5, 1))
    delta_d = out[:, 0] - z[:, 0]
    delta_r = out[:, 2] - z[:, 2]
    assert torch.allclose(delta_d, delta_r, atol=1e-7)


def test_physical_transition_permanent_monotone_and_trap_bounded():
    from src.transfer.damage_state import PhysicalDamageTransition

    transition = PhysicalDamageTransition()
    z = torch.rand(5, 3); u = torch.rand(5, 3)
    out = transition(z, u, torch.ones(5, 1))
    assert torch.all(out[:, 0] >= z[:, 0] - 1e-7)
    assert torch.all(out[:, 2] >= z[:, 2] - 1e-7)
    assert torch.all((0.0 <= out[:, 1]) & (out[:, 1] <= 1.0))


def test_physical_transition_rejects_wrong_last_dim():
    from src.transfer.damage_state import PhysicalDamageTransition

    transition = PhysicalDamageTransition()
    z = torch.rand(3, 3); u_bad = torch.rand(3, 2); dt = torch.ones(3, 1)
    raised = False
    try:
        transition(z, u_bad, dt)
    except ValueError:
        raised = True
    assert raised


# ---- PhysicalDamageStateModel 契约 ----

def test_physical_model_has_no_domain_stress_mlp():
    model = _model()
    for attr in ("source_stress", "target_stress", "target_node_stress"):
        assert not hasattr(model, attr), f"物理化模型不应含域专属 stress MLP: {attr}"


def test_physical_model_state_scale_is_fixed_physics_constant():
    model = _model()
    assert torch.allclose(model.state_scale, torch.tensor([0.006, 1.0, 0.084]), atol=1e-7)


def test_physical_model_encoder_outputs_nonneg_physical_units():
    model = _model()
    z = model.encode_source(torch.randn(4, 20))
    assert torch.all(z[..., 0] >= 0.0) and torch.all(z[..., 2] >= 0.0)
    assert torch.all((0.0 <= z[..., 1]) & (z[..., 1] <= 1.0))


def test_physical_model_next_state_takes_physical_u_and_returns_physical_units():
    model = _model()
    x = torch.randn(3, 20)
    u_phys = torch.rand(3, 3) * torch.tensor([4.0, 0.5, 1.0])  # [a_T, s, recovery] 物理量
    dt = torch.ones(3, 1)

    z0 = model.encode_source(x)
    z1 = model.next_source_state(x, u_phys, dt)

    assert z1.shape == (3, 3)
    # 转移在归一空间发生：手算归一→transition→反归一 与 next_source_state 等价
    expected = model.denormalize_state(
        model.transition(model.normalize_state(z0), model.normalize_stress(u_phys), dt))
    assert torch.allclose(z1, expected, atol=1e-6)
    # 物理量纲 d/r 耦合：Δr = 14·Δd（源仿真 r_th=14·d_perm）
    delta_d = z1[:, 0] - z0[:, 0]
    delta_r = z1[:, 2] - z0[:, 2]
    assert torch.allclose(delta_r, 14.0 * delta_d, atol=1e-6)


def test_physical_model_fit_stress_standardizer_sets_u_scale_from_train_std():
    model = _model()
    u_train = torch.rand(200, 3) * torch.tensor([4.0, 0.5, 1.0])
    model.fit_stress_standardizer(u_train)
    expected = u_train.std(dim=0) + 1e-6
    assert torch.allclose(model.u_scale, expected, atol=1e-6)
    # 归一后各维 std ≈ 1（无均值中心化，方差不变）
    u_norm = model.normalize_stress(u_train)
    assert torch.allclose(u_norm.std(dim=0), torch.ones(3), atol=1e-4)


def test_physical_model_target_next_state_uses_target_encoder():
    model = _model()
    x = torch.randn(2, 12); u = torch.rand(2, 3) * torch.tensor([4.0, 0.5, 1.0]); dt = torch.ones(2, 1)
    z0 = model.encode_target(x)
    z1 = model.next_target_state(x, u, dt)
    expected = model.denormalize_state(
        model.transition(model.normalize_state(z0), model.normalize_stress(u), dt))
    assert torch.allclose(z1, expected, atol=1e-6)


def test_physical_model_node_interface_takes_node_u_phys():
    from src.transfer.damage_state import PhysicalDamageStateModel

    model = PhysicalDamageStateModel(source_input_dim=20, target_input_dim=12,
                                     target_node_input_dim=6, hidden_dim=16)
    node_x = torch.randn(2, 16, 6)
    node_u = torch.rand(2, 16, 3) * torch.tensor([4.0, 0.5, 1.0])
    dt = torch.full((2, 16, 1), 0.5)

    nxt = model.next_target_node_state(node_x, node_u, dt)
    assert nxt.shape == (2, 16, 3)

    z = model.encode_target_nodes(node_x)
    expected = model.denormalize_state(
        model.transition(model.normalize_state(z), model.normalize_stress(node_u), dt))
    assert torch.allclose(nxt, expected, atol=1e-6)
    # 节点通道头独立可用
    assert model.observe_target_nodes(nxt).shape == (2, 16, 4)


def test_physical_model_transition_checkpoint_isolates_ftheta_only():
    torch.manual_seed(1)
    source_model = _model()
    torch.manual_seed(2)
    target_model = _model()
    target_encoder_before = {k: v.detach().clone() for k, v in target_model.target_encoder.state_dict().items()}
    target_head_before = {k: v.detach().clone() for k, v in target_model.target_head.state_dict().items()}
    state_scale_before = target_model.state_scale.detach().clone()
    u_scale_before = target_model.u_scale.detach().clone()

    loaded = target_model.load_transition_state_dict(source_model.transition_state_dict())

    assert loaded
    for key, value in source_model.transition.state_dict().items():
        assert torch.equal(target_model.transition.state_dict()[key], value)
    for key, value in target_encoder_before.items():
        assert torch.equal(target_model.target_encoder.state_dict()[key], value)
    for key, value in target_head_before.items():
        assert torch.equal(target_model.target_head.state_dict()[key], value)
    # 物理常数与 u 归一尺度绝不随 Fθ 跨域加载
    assert torch.equal(target_model.state_scale, state_scale_before)
    assert torch.equal(target_model.u_scale, u_scale_before)


def test_physical_model_transition_checkpoint_rejects_encoder_weights():
    model = _model()
    bogus = {**model.transition_state_dict(), "source_encoder.0.weight": torch.zeros(16, 20)}
    raised = False
    try:
        model.load_transition_state_dict(bogus)
    except ValueError:
        raised = True
    assert raised
