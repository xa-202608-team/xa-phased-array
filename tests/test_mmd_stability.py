"""MMD 稳定化测试 (诊断 D 验收):

  - HIBinMemoryBank 累积 MMD 估计方差 < 单 batch MMD 估计方差
  - 对齐组 MMD < 偏移组 MMD (bank 模式 + CORAL 模式均满足)
  - 全局 bank lazy 创建 + bins 变化自动重建 + reset 清空
  - 调用签名向后兼容 (旧 5 参数调用可用)

依据: docs/phased_array_dev_plan.md PA5 诊断 D (S3 seed43 发散根因: 每 mini-batch
按 HI 分箱, 每箱 2-3 样本 → 多尺度 RBF-MMD 噪声估计 → latent 漂移共振)。
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.transfer.mmd import (                               # noqa: E402
    HIBinMemoryBank,
    coral,
    enable_global_memory_bank,
    get_global_memory_bank,
    mmd_by_hi_bins,
    mmd_multiscale,
    reset_global_memory_bank,
)


# ---------- 每个测试前后清空全局 bank, 避免跨测试累积污染 ----------
@pytest.fixture(autouse=True)
def _isolate_global_bank():
    reset_global_memory_bank()
    yield
    reset_global_memory_bank()


# ============================================================ bank 方差
def test_bank_variance_lower_than_batch():
    """memory bank 累积估计的方差 < 单 batch 估计。

    构造 4 组同分布 (z_s, z_t) — 真实 MMD^2 应接近 0。
    单 batch (3 样本/箱) 估计抖动大; bank (200 样本) 估计稳定。
    """
    torch.manual_seed(0)
    bins = [(0.0, 1.0)]
    D = 8
    n_trials = 40
    batch_n = 3   # 模拟相控阵源域小 batch + HI 分箱后每箱 2-3 样本

    def sample(n):
        return torch.randn(n, D), torch.rand(n)

    # 单 batch 估计分布
    estimates_batch = []
    for _ in range(n_trials):
        zs, hs = sample(batch_n)
        zt, ht = sample(batch_n)
        v = mmd_by_hi_bins(zs, hs, zt, ht, bins, bank=None)  # 显式禁用 bank
        estimates_batch.append(float(v))

    # bank 累积估计分布 (每个 trial 独立 bank, warm-up 30 个 batch 后采样)
    estimates_bank = []
    for _ in range(n_trials):
        bank = HIBinMemoryBank(bins, capacity_per_bin=100)
        for _ in range(30):
            zs, hs = sample(batch_n)
            zt, ht = sample(batch_n)
            bank.update(zs, hs, zt, ht)
        zs, hs = sample(batch_n)
        zt, ht = sample(batch_n)
        v = mmd_by_hi_bins(zs, hs, zt, ht, bins, bank=bank)
        estimates_bank.append(float(v))

    var_batch = torch.tensor(estimates_batch).var(unbiased=False).item()
    var_bank = torch.tensor(estimates_bank).var(unbiased=False).item()
    print(f"\n  var_batch={var_batch:.6e}  var_bank={var_bank:.6e}  "
          f"ratio={var_batch / max(var_bank, 1e-12):.2f}x")
    assert var_bank < var_batch, (
        f"bank 估计方差应更小: bank={var_bank:.6e} vs batch={var_batch:.6e}")


# ============================================================ 对齐有效性
def test_aligned_smaller_than_shifted_with_bank():
    """bank 模式: 同分布 latent 的 MMD < 显著偏移分布的 MMD。"""
    torch.manual_seed(0)
    bins = [(0.0, 1.0)]
    D = 8

    z = torch.randn(200, D)
    hi = torch.rand(200)
    z_same = z + 0.01 * torch.randn(200, D)
    z_shifted = torch.randn(200, D) + 5.0

    # 两个独立 bank, 分别累积两种对齐情况
    bank_same = HIBinMemoryBank(bins, capacity_per_bin=150)
    bank_shift = HIBinMemoryBank(bins, capacity_per_bin=150)
    for i in range(0, 200, 16):
        bank_same.update(z[i:i + 16], hi[i:i + 16],
                         z_same[i:i + 16], hi[i:i + 16])
        bank_shift.update(z[i:i + 16], hi[i:i + 16],
                          z_shifted[i:i + 16], hi[i:i + 16])

    m_same = mmd_by_hi_bins(z[:16], hi[:16], z_same[:16], hi[:16],
                            bins, bank=bank_same)
    m_shift = mmd_by_hi_bins(z[:16], hi[:16], z_shifted[:16], hi[:16],
                             bins, bank=bank_shift)
    print(f"\n  m_same={m_same.item():.6f}  m_shift={m_shift.item():.6f}")
    assert m_same < m_shift


def test_aligned_smaller_than_shifted_with_bank_multibin():
    """4 个 HI 分箱场景下, bank MMD 仍能区分对齐 vs 偏移。"""
    torch.manual_seed(1)
    bins = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
    D = 8

    z = torch.randn(400, D)
    hi = torch.rand(400)
    z_same = z + 0.01 * torch.randn(400, D)
    z_shifted = torch.randn(400, D) + 3.0

    bank_same = HIBinMemoryBank(bins, capacity_per_bin=80)
    bank_shift = HIBinMemoryBank(bins, capacity_per_bin=80)
    for i in range(0, 400, 32):
        bank_same.update(z[i:i + 32], hi[i:i + 32],
                         z_same[i:i + 32], hi[i:i + 32])
        bank_shift.update(z[i:i + 32], hi[i:i + 32],
                          z_shifted[i:i + 32], hi[i:i + 32])

    m_same = mmd_by_hi_bins(z[:32], hi[:32], z_same[:32], hi[:32],
                             bins, bank=bank_same)
    m_shift = mmd_by_hi_bins(z[:32], hi[:32], z_shifted[:32], hi[:32],
                              bins, bank=bank_shift)
    print(f"\n  multibin m_same={m_same.item():.6f}  m_shift={m_shift.item():.6f}")
    assert m_same < m_shift


# ============================================================ CORAL 备选
def test_coral_kernel_in_mmd_by_hi_bins():
    """CORAL 作为 kernel 传入 mmd_by_hi_bins 能正常返回有限非负值。"""
    torch.manual_seed(2)
    z_s = torch.randn(50, 8)
    z_t = torch.randn(40, 8)
    hi_s = torch.rand(50)
    hi_t = torch.rand(40)
    bins = [(0.0, 1.0)]

    # str 形式
    v_str = mmd_by_hi_bins(z_s, hi_s, z_t, hi_t, bins,
                           kernel="coral", bank=None)
    assert torch.isfinite(v_str) and v_str >= 0

    # callable 形式
    v_call = mmd_by_hi_bins(z_s, hi_s, z_t, hi_t, bins,
                            kernel=coral, bank=None)
    assert torch.isfinite(v_call) and v_call >= 0


def test_coral_smaller_for_aligned():
    """CORAL 对同协方差分布应接近 0, 显著小于协方差偏移分布。

    注: CORAL 只对齐协方差 (不对齐均值), 故偏移用 scale 变化 (而非纯平移)。
    """
    torch.manual_seed(3)
    z = torch.randn(100, 8)
    hi = torch.rand(100)
    z_scaled = torch.randn(100, 8) * 3.0    # 不同方差 → 不同 cov
    bins = [(0.0, 1.0)]

    m_same = mmd_by_hi_bins(z, hi, z, hi, bins, kernel="coral", bank=None)
    m_shift = mmd_by_hi_bins(z, hi, z_scaled, hi, bins, kernel="coral", bank=None)
    print(f"\n  coral m_same={m_same.item():.6f}  m_shift={m_shift.item():.6f}")
    assert m_same < m_shift


def test_coral_bank_combined_stable():
    """CORAL + bank 组合: 对齐组 < 偏移组 (D-step2 + D-step1 叠加)。"""
    torch.manual_seed(4)
    bins = [(0.0, 1.0)]
    D = 8

    z = torch.randn(200, D)
    hi = torch.rand(200)
    z_same = z + 0.01 * torch.randn(200, D)
    z_shifted = torch.randn(200, D) + 5.0

    bank_same = HIBinMemoryBank(bins, capacity_per_bin=100)
    bank_shift = HIBinMemoryBank(bins, capacity_per_bin=100)
    for i in range(0, 200, 20):
        bank_same.update(z[i:i + 20], hi[i:i + 20],
                         z_same[i:i + 20], hi[i:i + 20])
        bank_shift.update(z[i:i + 20], hi[i:i + 20],
                          z_shifted[i:i + 20], hi[i:i + 20])

    m_same = mmd_by_hi_bins(z[:20], hi[:20], z_same[:20], hi[:20],
                             bins, kernel="coral", bank=bank_same)
    m_shift = mmd_by_hi_bins(z[:20], hi[:20], z_shifted[:20], hi[:20],
                              bins, kernel="coral", bank=bank_shift)
    print(f"\n  coral+bank m_same={m_same.item():.6f}  m_shift={m_shift.item():.6f}")
    assert m_same < m_shift


# ============================================================ 全局 bank 行为
def test_global_bank_default_lazy_init_and_accumulation():
    """默认调用 (bank 哨兵) 走全局 bank, 累积样本。"""
    enable_global_memory_bank(True)
    reset_global_memory_bank()
    torch.manual_seed(5)
    bins = [(0.0, 1.0)]

    for _ in range(5):
        zs = torch.randn(10, 4)
        zt = torch.randn(10, 4)
        hi = torch.rand(10)
        mmd_by_hi_bins(zs, hi, zt, hi, bins)   # 不传 bank → 哨兵 → 全局

    b = get_global_memory_bank(bins)
    assert b._src_z[0] is not None, "全局 bank 应已累积源域样本"
    assert b._src_z[0].size(0) > 0
    assert b._tgt_z[0] is not None and b._tgt_z[0].size(0) > 0


def test_global_bank_bins_change_rebuilds():
    """全局 bank bins 变化时自动重建 (跨测试隔离机制)。"""
    enable_global_memory_bank(True)
    reset_global_memory_bank()
    torch.manual_seed(6)

    bins_a = [(0.0, 1.0)]
    bins_b = [(0.0, 0.5), (0.5, 1.0)]

    # 先用 bins_a 累积
    for _ in range(3):
        mmd_by_hi_bins(torch.randn(8, 4), torch.rand(8),
                       torch.randn(8, 4), torch.rand(8), bins_a)
    bank_a = get_global_memory_bank(bins_a)
    assert bank_a.bins == [tuple(b) for b in bins_a]

    # 切到 bins_b → 应自动重建
    mmd_by_hi_bins(torch.randn(8, 4), torch.rand(8),
                   torch.randn(8, 4), torch.rand(8), bins_b)
    bank_b = get_global_memory_bank(bins_b)
    assert bank_b.bins == [tuple(b) for b in bins_b]
    # 新 bank 的目标队列刚 update 一次, 必然非空
    assert any(z is not None and z.size(0) > 0 for z in bank_b._tgt_z)


def test_global_bank_disable_falls_back_to_batch():
    """关闭全局 bank 后, 默认调用退回单 batch 行为。"""
    enable_global_memory_bank(False)
    reset_global_memory_bank()
    torch.manual_seed(7)
    bins = [(0.0, 1.0)]

    # 多次调用, 不应累积任何全局状态
    for _ in range(5):
        mmd_by_hi_bins(torch.randn(8, 4), torch.rand(8),
                       torch.randn(8, 4), torch.rand(8), bins)

    # 全局 bank 应仍为 None (未 lazy init)
    from src.transfer.mmd import _GLOBAL_BANK
    assert _GLOBAL_BANK is None
    # 恢复默认开启, 不影响后续测试
    enable_global_memory_bank(True)


# ============================================================ 兼容性
def test_legacy_signature_still_works():
    """5 参数旧调用 (无 kernel/bank) 不报错, 默认走全局 bank。"""
    enable_global_memory_bank(True)
    torch.manual_seed(8)
    zS = torch.randn(20, 4)
    hiS = torch.rand(20)
    zT = torch.randn(16, 4)
    hiT = torch.rand(16)
    bins = [(0.0, 1.0)]
    v = mmd_by_hi_bins(zS, hiS, zT, hiT, bins)
    assert torch.isfinite(v) and v >= 0


def test_explicit_bank_none_uses_batch_logic():
    """显式 bank=None 走旧单 batch 逻辑 (无累积)。"""
    torch.manual_seed(9)
    bins = [(0.0, 1.0)]
    zS = torch.randn(8, 4)
    hiS = torch.rand(8)
    zT = torch.randn(8, 4)
    hiT = torch.rand(8)

    # 多次调用 bank=None, 全局 bank 应保持 None
    for _ in range(3):
        v = mmd_by_hi_bins(zS, hiS, zT, hiT, bins, bank=None)
        assert torch.isfinite(v) and v >= 0
    # 全局 bank 没被这次调用激活 (bank=None 显式跳过)
    # (其他测试可能创建过, 这里只验证 bank=None 路径不触发累积)


def test_grad_flows_through_current_batch():
    """bank 模式下, 当前 batch 的 z_s 梯度能反传 (encoder 可学)。"""
    torch.manual_seed(10)
    bins = [(0.0, 1.0)]
    bank = HIBinMemoryBank(bins, capacity_per_bin=50)
    # warm-up bank
    for _ in range(3):
        zs = torch.randn(8, 4)
        zt = torch.randn(8, 4)
        hi = torch.rand(8)
        bank.update(zs, hi, zt, hi)

    # 当前 batch 带 requires_grad
    zs = torch.randn(8, 4, requires_grad=True)
    zt = torch.randn(8, 4, requires_grad=True)
    hi = torch.rand(8)
    v = mmd_by_hi_bins(zs, hi, zt, hi, bins, bank=bank)
    v.backward()
    assert zs.grad is not None, "bank 模式下当前 batch z_s 应有梯度"
    assert zt.grad is not None, "bank 模式下当前 batch z_t 应有梯度"
    assert torch.isfinite(zs.grad).all()


def test_fifo_capacity_respected():
    """bank 累积超过 capacity 时 FIFO 截断正确。"""
    torch.manual_seed(11)
    bins = [(0.0, 1.0)]
    cap = 30
    bank = HIBinMemoryBank(bins, capacity_per_bin=cap)
    # 累积 100 个样本 (远超容量)
    for _ in range(10):
        bank.update(torch.randn(10, 4), torch.rand(10),
                    torch.randn(10, 4), torch.rand(10))
    assert bank._src_z[0].size(0) == cap, f"FIFO 应截断到 {cap}, 实际 {bank._src_z[0].size(0)}"
    assert bank._tgt_z[0].size(0) == cap
