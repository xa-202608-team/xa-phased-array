"""迁移不变量测试 (plan P5 验收):
  - S2 编码器冻结 / S3 解冻
  - 轨迹级划分不相交且覆盖
  - MMD 按 HI 健康阶段分箱 (非按时间)
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                                       # noqa: E402
from src.transfer.adapter import TransferModel                          # noqa: E402
from src.transfer.mmd import mmd_by_hi_bins                             # noqa: E402
from src.transfer.train_transfer import split_trajectories              # noqa: E402


def test_encoder_freeze_in_S2_unfreeze_in_S3():
    """S2 编码器参数 requires_grad=False (冻结); S3 解冻为 True。"""
    cfg = load_config()
    mc = cfg["model"]
    model = TransferModel(
        encoder_type="tcn", n_features=12, n_target=8,
        channels=mc["tcn"]["channels"], kernel_size=mc["tcn"]["kernel_size"],
        num_blocks=mc["tcn"]["num_blocks"], dropout=mc["tcn"]["dropout"],
        latent_dim=mc["latent_dim"])
    model.freeze_encoder(True)
    for name, p in model.encoder.named_parameters():
        assert not p.requires_grad, f"S2 编码器应冻结: {name} 仍 requires_grad"
    model.freeze_encoder(False)
    for name, p in model.encoder.named_parameters():
        assert p.requires_grad, f"S3 编码器应解冻: {name}"


def test_trajectory_split_disjoint_and_covering():
    """train/val/test 轨迹集合两两不相交且覆盖全部 (完整轨迹级)。"""
    cfg = load_config()
    s = cfg["transfer"]["split"]
    tr, va, te = split_trajectories(100, [s["train"], s["val"], s["test"]], cfg["seed"])
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == 100
    # 比例近似 15/20/65
    assert 0.10 <= len(tr) / 100 <= 0.20
    assert 0.15 <= len(va) / 100 <= 0.25
    assert len(te) / 100 >= 0.55


def test_mmd_uses_hi_bins():
    """MMD 按 HI 分箱计算 (对齐键为健康阶段)。"""
    torch.manual_seed(0)
    zS = torch.randn(100, 8)
    hiS = torch.rand(100)
    zT = torch.randn(80, 8)
    hiT = torch.rand(80)
    bins = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
    val = mmd_by_hi_bins(zS, hiS, zT, hiT, bins)
    assert torch.isfinite(val) and val >= 0


def test_mmd_smaller_for_aligned_distributions():
    """同分布 latent 的 MMD 应显著小于偏移分布 (验证度量有效)。"""
    torch.manual_seed(0)
    z = torch.randn(200, 8)
    hi = torch.rand(200)
    z_same = z + 0.01 * torch.randn(200, 8)
    z_shifted = torch.randn(200, 8) + 5.0
    bins = [(0.0, 1.0)]
    m_same = mmd_by_hi_bins(z, hi, z_same, hi, bins)
    m_shift = mmd_by_hi_bins(z, hi, z_shifted, hi, bins)
    assert m_same < m_shift


def test_transfer_model_forward_shapes():
    """adapter(x_T 8维) → encoder → heads 输出形状正确。"""
    model = TransferModel(encoder_type="tcn", n_features=12, n_target=8, latent_dim=16)
    x = torch.randn(4, 32, 8)            # (B, L, 8)
    hi, rul, z = model(x)
    assert hi.shape == (4,) and rul.shape == (4,) and z.shape == (4, 16)
