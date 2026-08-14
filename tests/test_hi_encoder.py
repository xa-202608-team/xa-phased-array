"""HI 动力学层迁移 (路径 A) 接口不变量测试。

验证 docs/开发推进计划/hi_layer_refactor_design.md §3.1 的 3 个新类:
  - HISeqEncoder       (B,L,2) → (B,latent_dim), 三种 backbone (tcn/lstm/gru)
  - HIDynamicsModel    forward 返回 hi∈[0,1] / rul≥0 / z; freeze / load_pretrained strict
  - TargetAuxHead      (B,n_target) → (B,) rul 校正量

不动 GPU, 不跑 pretrain/transfer, 纯 CPU 单元测试。
"""
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.transfer.adapter import (                                # noqa: E402
    HIDynamicsModel,
    HISeqEncoder,
    TargetAuxHead,
)


# ============================================================ HISeqEncoder 形状
@pytest.mark.parametrize("encoder_type", ["tcn", "lstm", "gru"])
def test_hiseq_encoder_forward_shape(encoder_type):
    """HISeqEncoder: (B,L,2) → (B,latent_dim), 三种 backbone 形状一致。"""
    B, L, latent_dim = 4, 32, 24
    enc = HISeqEncoder(encoder_type=encoder_type, input_len=L,
                       latent_dim=latent_dim, channels=32,
                       lstm_hidden=32, lstm_layers=2)
    x = torch.randn(B, L, 2)            # [HI, ΔHI]
    z = enc(x)
    assert z.shape == (B, latent_dim), (
        f"{encoder_type}: z.shape={tuple(z.shape)} 应为 {(B, latent_dim)}")
    assert torch.isfinite(z).all(), f"{encoder_type}: z 含 NaN/Inf"


def test_hiseq_encoder_rejects_wrong_input_dim():
    """HISeqEncoder 固定 n_features=2: 输入最后一维不是 2 时应报错 (Conv1d/LSTM 校验)。"""
    enc = HISeqEncoder(encoder_type="tcn", latent_dim=8, channels=8)
    x_bad = torch.randn(2, 16, 3)       # 最后一维 3, 与实例化的 n_features=2 不匹配
    with pytest.raises(RuntimeError):
        enc(x_bad)


def test_hiseq_encoder_state_dict_keys_align_with_backbone():
    """HISeqEncoder 内部直接包装 TCNEncoder, state_dict keys 就是 TCN 子模块名
    (input_proj/blocks/out_proj), 没有"HISeqEncoder 重写 TCN"造成的命名漂移。"""
    enc = HISeqEncoder(encoder_type="tcn", num_blocks=2, latent_dim=8, channels=8)
    keys = list(enc.state_dict().keys())
    # 至少包含 TCNEncoder 的标志性子模块
    assert any(k.startswith("encoder.input_proj") for k in keys), \
        f"缺 encoder.input_proj.*: {keys[:5]}"
    assert any(k.startswith("encoder.blocks.0") for k in keys), \
        f"缺 encoder.blocks.0.*: {keys[:5]}"
    assert any(k.startswith("encoder.out_proj") for k in keys), \
        f"缺 encoder.out_proj.*: {keys[:5]}"


# ============================================================ HIDynamicsModel forward
def test_hidynamics_forward_ranges():
    """HIDynamicsModel: hi∈[0,1], rul≥0 (softplus), z 形状正确。"""
    B, L, latent_dim = 6, 32, 16
    model = HIDynamicsModel(encoder_type="tcn", input_len=L,
                            latent_dim=latent_dim, channels=16, num_blocks=2)
    x = torch.randn(B, L, 2)
    hi, rul, z = model(x)
    assert hi.shape == (B,), f"hi.shape={tuple(hi.shape)} 应为 {(B,)}"
    assert rul.shape == (B,), f"rul.shape={tuple(rul.shape)} 应为 {(B,)}"
    assert z.shape == (B, latent_dim), f"z.shape={tuple(z.shape)} 应为 {(B, latent_dim)}"
    # hi 经 sigmoid → [0,1]
    assert torch.all((hi >= 0.0) & (hi <= 1.0)), f"hi 应 ∈[0,1], min={hi.min()}, max={hi.max()}"
    # rul 经 softplus → ≥0
    assert torch.all(rul >= 0.0), f"rul 应 ≥0, min={rul.min()}"
    assert torch.isfinite(z).all() and torch.isfinite(hi).all() and torch.isfinite(rul).all()


def test_hidynamics_forward_grad_flow():
    """forward 输出可反传到 encoder 参数 (确认 HISeqEncoder 是模型计算图的一部分)。"""
    model = HIDynamicsModel(encoder_type="lstm", latent_dim=8,
                            lstm_hidden=8, lstm_layers=1)
    x = torch.randn(4, 16, 2)
    hi, rul, z = model(x)
    loss = hi.sum() + rul.sum()
    loss.backward()
    # encoder 至少有一个参数拿到梯度
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.encoder.parameters())
    assert has_grad, "encoder 参数未收到梯度 (计算图断裂?)"


# ============================================================ freeze_encoder
def test_freeze_encoder_toggles_requires_grad():
    """freeze_encoder(True): encoder 全部 requires_grad=False; (False): 恢复 True。
    heads 不受影响, 始终可训练。"""
    model = HIDynamicsModel(encoder_type="tcn", latent_dim=8, channels=8, num_blocks=2)

    # 初始: 全部可训练
    for n, p in model.encoder.named_parameters():
        assert p.requires_grad, f"初始 encoder.{n} 应可训练"

    # freeze
    model.freeze_encoder(True)
    enc_frozen = all(not p.requires_grad for p in model.encoder.parameters())
    assert enc_frozen, "freeze_encoder(True) 后 encoder 仍有 requires_grad=True 的参数"
    # heads 必须仍可训练
    assert model.hi_head.weight.requires_grad, "hi_head.weight 应仍可训练"
    assert model.rul_head.weight.requires_grad, "rul_head.weight 应仍可训练"

    # unfreeze
    model.freeze_encoder(False)
    enc_unfrozen = all(p.requires_grad for p in model.encoder.parameters())
    assert enc_unfrozen, "freeze_encoder(False) 后 encoder 仍有 requires_grad=False 的参数"


def test_freeze_encoder_actually_blocks_grad_update():
    """freeze 时 encoder 参数 .step() 不更新; unfreeze 后可更新。"""
    torch.manual_seed(0)
    model = HIDynamicsModel(encoder_type="tcn", latent_dim=8, channels=8, num_blocks=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(4, 16, 2)

    # 记录初始 encoder 参数
    before = model.encoder.encoder.out_proj.weight.detach().clone()

    model.freeze_encoder(True)
    opt.zero_grad()
    hi, rul, _ = model(x)
    loss = (hi - 1.0).pow(2).mean() + rul.mean()
    loss.backward()
    opt.step()
    after_frozen = model.encoder.encoder.out_proj.weight.detach().clone()
    assert torch.allclose(before, after_frozen, atol=1e-8), \
        "freeze 状态下 encoder 参数被 .step() 改动 (不应发生)"

    # 解冻后再训一步, encoder 应该变了
    model.freeze_encoder(False)
    opt.zero_grad()
    hi, rul, _ = model(x)
    loss = (hi - 1.0).pow(2).mean() + rul.mean()
    loss.backward()
    opt.step()
    after_unfrozen = model.encoder.encoder.out_proj.weight.detach().clone()
    assert not torch.allclose(before, after_unfrozen, atol=1e-8), \
        "unfreeze 后 encoder 参数未变化 (梯度未生效?)"


# ============================================================ load_pretrained (strict)
def test_load_pretrained_strict_loads_encoder_only():
    """构造临时 HIDynamicsModel checkpoint, load_pretrained 应:
       - strict=True 加载 encoder.* (missing=0, unexpected=0)
       - hi_head / rul_head 不被覆盖 (heads 权重保持目标域初始化值)
    """
    torch.manual_seed(42)
    # 源域模型 (与目标域 shape 完全一致: F_in=2)
    src = HIDynamicsModel(encoder_type="tcn", latent_dim=12, channels=12, num_blocks=2)
    # 目标域模型 (同 shape)
    tgt = HIDynamicsModel(encoder_type="tcn", latent_dim=12, channels=12, num_blocks=2)

    # 记录目标域 heads 初始权重 (load_pretrained 后应不变)
    tgt_hi_w_before = tgt.hi_head.weight.detach().clone()
    tgt_rul_w_before = tgt.rul_head.weight.detach().clone()
    # encoder 权重不同 (随机初始化)
    enc_w_before = tgt.encoder.encoder.out_proj.weight.detach().clone()
    assert not torch.allclose(src.encoder.encoder.out_proj.weight.detach(),
                              enc_w_before), "两模型 encoder 初始化不应完全相同"

    # 保存源 ckpt (与 pretrain.py 同格式: {"model": sd, ...})
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "src.pt"
        torch.save({"model": src.state_dict(),
                    "encoder": "tcn", "L": 32,
                    "n_features": 2, "latent_dim": 12}, ckpt)
        missing, unexpected = tgt.load_pretrained(str(ckpt))

    # strict=True → 两边都应为空
    assert missing == [], f"strict 加载后 missing 应为空, 实际 {missing}"
    assert unexpected == [], f"strict 加载后 unexpected 应为空, 实际 {unexpected}"

    # encoder 权重应被源域覆盖
    enc_w_after = tgt.encoder.encoder.out_proj.weight.detach().clone()
    assert torch.allclose(src.encoder.encoder.out_proj.weight.detach(), enc_w_after), \
        "load_pretrained 后 encoder 权重应等于源 ckpt"

    # heads 权重应保持不变
    assert torch.allclose(tgt_hi_w_before, tgt.hi_head.weight.detach()), \
        "hi_head 权重被 load_pretrained 覆盖 (不应发生)"
    assert torch.allclose(tgt_rul_w_before, tgt.rul_head.weight.detach()), \
        "rul_head 权重被 load_pretrained 覆盖 (不应发生)"


def test_load_pretrained_accepts_bare_state_dict():
    """checkpoint 不是 {"model": sd} 包装, 而是裸 state_dict 时也能加载。"""
    src = HIDynamicsModel(encoder_type="tcn", latent_dim=8, channels=8, num_blocks=1)
    tgt = HIDynamicsModel(encoder_type="tcn", latent_dim=8, channels=8, num_blocks=1)
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "bare.pt"
        torch.save(src.state_dict(), ckpt)         # 裸 state_dict
        missing, unexpected = tgt.load_pretrained(str(ckpt))
    assert missing == [] and unexpected == [], \
        f"裸 state_dict 加载失败: missing={missing}, unexpected={unexpected}"


def test_load_pretrained_strict_fails_on_shape_mismatch():
    """encoder 形状不一致时 strict=True 应抛异常 (不会静默漏权重)。"""
    src = HIDynamicsModel(encoder_type="tcn", latent_dim=16, channels=16, num_blocks=2)
    # 目标域 latent_dim 不同 → out_proj 形状不同
    tgt = HIDynamicsModel(encoder_type="tcn", latent_dim=8, channels=16, num_blocks=2)
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "mismatch.pt"
        torch.save({"model": src.state_dict()}, ckpt)
        with pytest.raises(RuntimeError):
            tgt.load_pretrained(str(ckpt))


# ============================================================ TargetAuxHead
def test_target_aux_head_shape():
    """TargetAuxHead: (B, n_target) → (B,) 标量 rul_correction。"""
    head = TargetAuxHead(n_target=12, hidden=32)
    x = torch.randn(8, 12)              # 窗末 12 维 obs
    y = head(x)
    assert y.shape == (8,), f"y.shape={tuple(y.shape)} 应为 {(8,)}"
    assert torch.isfinite(y).all()


def test_target_aux_head_default_n_target_12():
    """默认 n_target=12 (与相控阵 x_global 6 + x_nodes.mean 6 对齐)。"""
    head = TargetAuxHead()
    x = torch.randn(4, 12)
    y = head(x)
    assert y.shape == (4,)
    # 第一层应该是 Linear(12, hidden)
    first_layer, = [m for m in head.net.modules() if isinstance(m, torch.nn.Linear)][:1]
    assert first_layer.in_features == 12, \
        f"默认首层 in_features={first_layer.in_features} 应为 12"


def test_target_aux_head_default_isolated_from_encoder():
    """alpha=0 时 TargetAuxHead 独立于 HIDynamicsModel forward (接口预留, 不进图)。"""
    # 单独实例化, 验证它是一个独立的 nn.Module, 不依赖 HIDynamicsModel
    head = TargetAuxHead(n_target=12)
    model = HIDynamicsModel(encoder_type="tcn", latent_dim=8, channels=8, num_blocks=1)
    # HIDynamicsModel forward 不调用 head
    x_HI = torch.randn(4, 16, 2)
    hi, rul, z = model(x_HI)
    # head 单独前向
    x_obs = torch.randn(4, 12)
    correction = head(x_obs)
    assert correction.shape == (4,)
    # 两者计算图独立: head 的反传不会触及 model.encoder
    (correction.sum() + rul.sum()).backward()
    enc_has_grad = any(p.grad is not None for p in model.encoder.parameters())
    assert enc_has_grad, "encoder 应从 rul.sum() 路径拿到梯度"


# ============================================================ 与文档 §3.1 签名一致性
def test_constructor_signatures_match_design_doc():
    """构造函数默认参数与设计文档 §3.1 完全一致。"""
    from src.models.tcn_encoder import TCNEncoder
    # HISeqEncoder 默认值
    enc = HISeqEncoder()
    assert enc.encoder_type == "tcn"
    assert isinstance(enc.encoder, TCNEncoder), \
        f"默认 backbone 应是 TCNEncoder, 实际 {type(enc.encoder)}"

    # HIDynamicsModel 默认值
    model = HIDynamicsModel()
    assert isinstance(model.encoder, HISeqEncoder)
    assert model.hi_head.in_features == 64     # latent_dim 默认 64
    assert model.hi_head.out_features == 1
    assert model.rul_head.in_features == 64
    assert model.rul_head.out_features == 1

    # TargetAuxHead 默认值
    head = TargetAuxHead()
    # 验证 n_target=12, hidden=64
    layers = [m for m in head.net.modules() if isinstance(m, torch.nn.Linear)]
    assert layers[0].in_features == 12 and layers[0].out_features == 64
    assert layers[1].in_features == 64 and layers[1].out_features == 1


def test_old_adapter_and_transfer_model_preserved():
    """旧 Adapter / TransferModel (飞轮仍在用) 仍可正常导入和实例化 (向后兼容)。"""
    from src.transfer.adapter import Adapter, TransferModel
    # Adapter
    ad = Adapter(n_in=8, n_out=12, hidden=32)
    out = ad(torch.randn(2, 10, 8))
    assert out.shape == (2, 10, 12)
    # TransferModel
    tm = TransferModel(encoder_type="tcn", n_features=12, n_target=8, latent_dim=16)
    hi, rul, z = tm(torch.randn(2, 10, 8))
    assert hi.shape == (2,) and rul.shape == (2,) and z.shape == (2, 16)
