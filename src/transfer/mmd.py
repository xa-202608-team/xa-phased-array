"""transfer/mmd.py — MMD / CORAL 域对齐 (plan §5.8)。

迁移发生在 HI/latent 层 (非原始信号), 按 HI 健康阶段分箱对齐, 不按绝对时间。

S3 稳定化 (诊断 D):
  - HIBinMemoryBank: 跨 batch 累积 latent, 把每箱 2-3 样本的噪声 MMD 升级为
    100+ 样本的稳定估计 (FIFO 容量适应 latent 在 S3 微调期间的缓慢漂移)。
  - CORAL 备选对齐: 小样本下协方差对齐比多尺度 RBF 稳。
  - 全局 bank 默认开启, 调用方 (train_transfer.py / run_groups.py) 零改动即可启用;
    可经 env var MMD_USE_BANK=0 关闭, MMD_BANK_CAPACITY 调容量。
"""
from __future__ import annotations

import os
from typing import Optional, Union

import torch


# sentinel: "use module-level global bank" — 与 None(禁用) / 实例(显式) 三态区分
_USE_GLOBAL = object()


# ---------------------------------------------------------------- 点对距离
def _pairwise_sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(Na, Nb) 平方欧氏距离矩阵。"""
    a2 = (a * a).sum(1, keepdim=True)          # (Na, 1)
    b2 = (b * b).sum(1, keepdim=True)          # (Nb, 1)
    return a2 + b2.t() - 2.0 * (a @ b.t())


# ---------------------------------------------------------------- 核函数
def mmd_multiscale(z_s: torch.Tensor, z_t: torch.Tensor,
                   sigmas=(0.01, 0.1, 1.0, 10.0)) -> torch.Tensor:
    """多尺度高斯核 MMD^2。z_s / z_t: (N, D)。

    先 L2 归一 latent, 稳定核计算 (防大值 pairwise 距离数值爆炸)。
    MMD^2 = E[k(s,s)] + E[k(t,t)] - 2·E[k(s,t)]
    """
    z_s = z_s / (z_s.norm(dim=1, keepdim=True) + 1e-8)
    z_t = z_t / (z_t.norm(dim=1, keepdim=True) + 1e-8)
    d_ss = _pairwise_sqdist(z_s, z_s)
    d_tt = _pairwise_sqdist(z_t, z_t)
    d_st = _pairwise_sqdist(z_s, z_t)

    def k(d):
        return sum(torch.exp(-d / (2.0 * s * s)) for s in sigmas)

    return k(d_ss).mean() + k(d_tt).mean() - 2.0 * k(d_st).mean()


def coral(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """CORAL: 协方差对齐损失 (小样本下比多尺度 RBF 稳)。"""
    def cov(z):
        n = z.size(0)
        zc = z - z.mean(0, keepdim=True)
        return (zc.t() @ zc) / max(n - 1, 1)

    return torch.mean((cov(z_s) - cov(z_t)) ** 2)


def _resolve_kernel(kernel):
    """kernel 接受 callable / 'mmd' / 'multiscale' / 'coral'。"""
    if callable(kernel):
        return kernel
    if kernel in ("mmd", "multiscale"):
        return mmd_multiscale
    if kernel == "coral":
        return coral
    raise ValueError(f"unknown kernel: {kernel!r}")


# ---------------------------------------------------------------- memory bank
class HIBinMemoryBank:
    """按 HI 分箱的 latent memory bank (跨 batch FIFO 累积)。

    每箱分别维护源域 / 目标域的 latent 队列, FIFO 截断到 capacity_per_bin:
      - 累积样本升估计稳定性 (2-3 → 上百, 估计方差骤降)。
      - FIFO 限制适应 S3 微调期间 latent 的缓慢漂移 (encoder 解冻后会漂)。
      - bank 内样本 detached, 不污染反传图; 当前 batch 仍保留梯度。

    用法 (训练循环外创建, 每 batch 自动 update):
        bank = HIBinMemoryBank(bins, capacity_per_bin=200)
        for batch in loader:
            ...
            mmd = mmd_by_hi_bins(zS, hs, zT, ht, bins, bank=bank)
    """

    def __init__(self, bins, capacity_per_bin: int = 200):
        self.bins = [tuple(b) for b in bins]
        self.capacity = max(2, int(capacity_per_bin))
        # 每箱一对 (z) 队列, None 表示尚未累积
        self._src_z: list[Optional[torch.Tensor]] = [None] * len(self.bins)
        self._tgt_z: list[Optional[torch.Tensor]] = [None] * len(self.bins)

    def _bin_index(self, hi: torch.Tensor) -> torch.Tensor:
        """返回每个样本所属 bin 索引 (-1 = 不在任何 bin)。

        多个 bin 区间不重叠时, 落在第一个匹配的 bin。
        """
        h1d = hi.reshape(-1)
        idx = torch.full((h1d.numel(),), -1, dtype=torch.long, device=h1d.device)
        for i, (lo, hi_bound) in enumerate(self.bins):
            in_bin = (h1d >= lo) & (h1d < hi_bound) & (idx == -1)
            idx[in_bin] = i
        return idx

    @torch.no_grad()
    def update(self, z_s: torch.Tensor, hi_s, z_t: torch.Tensor, hi_t):
        """把当前 batch (detached) 按分箱追加到 bank 队列, FIFO 截断。

        detach 保证: (1) 不保留历史计算图 (防内存爆); (2) update 本身无梯度。
        """
        z_s = z_s.detach().clone()
        z_t = z_t.detach().clone()
        hi_s = torch.as_tensor(hi_s, dtype=z_s.dtype, device=z_s.device).reshape(-1)
        hi_t = torch.as_tensor(hi_t, dtype=z_t.dtype, device=z_t.device).reshape(-1)
        bs = self._bin_index(hi_s)
        bt = self._bin_index(hi_t)
        cap = self.capacity
        for i in range(len(self.bins)):
            ms = bs == i
            mt = bt == i
            if bool(ms.any()):
                self._src_z[i] = self._append_fifo(self._src_z[i], z_s[ms], cap)
            if bool(mt.any()):
                self._tgt_z[i] = self._append_fifo(self._tgt_z[i], z_t[mt], cap)

    @staticmethod
    def _append_fifo(prev: Optional[torch.Tensor], new: torch.Tensor,
                     capacity: int) -> torch.Tensor:
        """FIFO 追加: prev 为 None 时初始化; 否则 cat 后截断到末尾 capacity 个。"""
        cur = new if prev is None else torch.cat([prev, new], dim=0)
        if cur.size(0) > capacity:
            cur = cur[-capacity:]
        return cur

    def get_augmented(self, z_s_curr: torch.Tensor, hi_s_curr,
                      z_t_curr: torch.Tensor, hi_t_curr):
        """对每个 bin 返回 (z_s_aug, z_t_aug) = bank 历史 (detach) + 当前 batch (grad)。

        返回 list[(zs, zt)] 长度 = len(self.bins), 顺序与 bins 一致。
        bank 部分提供稳定样本数 (detached), 当前 batch 部分保留梯度 (encoder/adapter 可学)。
        """
        hi_s_t = torch.as_tensor(hi_s_curr).reshape(-1)
        hi_t_t = torch.as_tensor(hi_t_curr).reshape(-1)
        bs = self._bin_index(hi_s_t).to(z_s_curr.device)
        bt = self._bin_index(hi_t_t).to(z_t_curr.device)
        out = []
        for i in range(len(self.bins)):
            ms = bs == i
            mt = bt == i
            cur_s = z_s_curr[ms] if bool(ms.any()) else z_s_curr[:0]
            cur_t = z_t_curr[mt] if bool(mt.any()) else z_t_curr[:0]
            hist_s = self._src_z[i]
            hist_t = self._tgt_z[i]
            if hist_s is not None and hist_s.numel() > 0:
                zs = torch.cat([hist_s, cur_s], dim=0)
            else:
                zs = cur_s
            if hist_t is not None and hist_t.numel() > 0:
                zt = torch.cat([hist_t, cur_t], dim=0)
            else:
                zt = cur_t
            out.append((zs, zt))
        return out


# ---------------------------------------------------------------- 全局 bank
_GLOBAL_BANK: Optional[HIBinMemoryBank] = None
_GLOBAL_BANK_ENABLED = os.environ.get("MMD_USE_BANK", "1") == "1"   # 默认开启


def reset_global_memory_bank():
    """清空全局 bank (测试 / 训练新 run 开始时调用)。"""
    global _GLOBAL_BANK
    _GLOBAL_BANK = None


def enable_global_memory_bank(enabled: bool = True):
    """开启 / 关闭全局 bank (运行时切换; 关闭同时清空)。"""
    global _GLOBAL_BANK_ENABLED
    _GLOBAL_BANK_ENABLED = bool(enabled)
    if not enabled:
        reset_global_memory_bank()


def get_global_memory_bank(bins, capacity_per_bin: Optional[int] = None) -> HIBinMemoryBank:
    """获取 (或按当前 bins lazy 创建) 全局 bank。

    bins 变化或 capacity 变化时自动重建 (测试用不同 bins 自动隔离, 防跨测试污染)。
    capacity 默认 200, 可经 env var MMD_BANK_CAPACITY 覆盖。
    """
    global _GLOBAL_BANK
    bins_list = [tuple(b) for b in bins]
    cap = int(capacity_per_bin if capacity_per_bin is not None
              else os.environ.get("MMD_BANK_CAPACITY", "200"))
    if _GLOBAL_BANK is None or _GLOBAL_BANK.bins != bins_list \
            or _GLOBAL_BANK.capacity != cap:
        _GLOBAL_BANK = HIBinMemoryBank(bins_list, capacity_per_bin=cap)
    return _GLOBAL_BANK


# ---------------------------------------------------------------- 主 API
def _mmd_by_hi_bins_batch(z_s, hi_s, z_t, hi_t, bins, kernel):
    """单 batch 内分箱算 kernel (旧 mmd_by_hi_bins 行为)。"""
    hi_s = torch.as_tensor(hi_s, dtype=z_s.dtype, device=z_s.device).reshape(-1)
    hi_t = torch.as_tensor(hi_t, dtype=z_t.dtype, device=z_t.device).reshape(-1)
    total = None
    n_used = 0
    for lo, hi in bins:
        ms = (hi_s >= lo) & (hi_s < hi)
        mt = (hi_t >= lo) & (hi_t < hi)
        if int(ms.sum()) >= 2 and int(mt.sum()) >= 2:
            k = kernel(z_s[ms], z_t[mt])
            total = k if total is None else total + k
            n_used += 1
    if total is None:
        return z_s.new_zeros(())
    return total / max(n_used, 1)


def mmd_by_hi_bins(z_s, hi_s, z_t, hi_t, bins,
                   kernel=mmd_multiscale,
                   bank: Union[object, Optional[HIBinMemoryBank]] = _USE_GLOBAL,
                   ) -> torch.Tensor:
    """按 HI 健康阶段分箱对齐 latent (plan §5.8 核心: 不按绝对时间)。

    对每个 bin [lo, hi), 取源域与目标域落在该 bin 的 latent 算 kernel, 再平均。

    架构定位 (HI 层重构后, 设计文档 §4.3):
      新路径 (`src.transfer.train_transfer.run_hi_layer`) 下, `z_s` 与 `z_t`
      **都来自共享 HISeqEncoder 对 x_HI = [HI, ΔHI] 序列的编码**, 输入语义跨域统一
      (源域 [HI_S, ΔHI_S] vs 目标域 [HI_T, ΔHI_T], 均为归一化 HI 轨迹 + 一阶差分)。
      MMD 此时对齐的 latent 真正编码 HI 动力学层 (CLAUDE.md 约束 2 要求), 而非
      旧观测层路径下"adapter 投影的异构观测特征的 latent"。

      旧路径 (`run`, 飞轮 + ablation) 下: `z_s = TCNEncoder(x_S)`, `z_t = TCNEncoder(adapter(x_T))`,
      语义错位但保留作 ablation 对照 (`latent_legacy` 组)。

    稳定化 (诊断 D):
      - bank=_USE_GLOBAL (默认): 用模块级全局 bank, bins 变化时自动重建;
        调用方 (train_transfer.py / run_groups.py) 零改动即可启用。
      - bank=None: 走单 batch 分箱 (旧行为, 兼容旧测试 / 显式规避污染)。
      - bank=HIBinMemoryBank: 用显式 bank (训练循环外创建)。
      - 当 bank 还没攒够样本时, 自动回退到当前 batch (保证可微)。

    kernel 可接受: mmd_multiscale / coral / 'mmd' / 'multiscale' / 'coral' (str)。
    """
    align_kernel = _resolve_kernel(kernel)

    # 解析 bank 三态
    if bank is _USE_GLOBAL:
        b = get_global_memory_bank(bins) if _GLOBAL_BANK_ENABLED else None
    elif bank is None:
        b = None
    else:
        b = bank

    if b is None:
        # 旧逻辑 (单 batch)
        return _mmd_by_hi_bins_batch(z_s, hi_s, z_t, hi_t, bins, align_kernel)

    # 累积当前 batch 到 bank (detached, 不污染反传图)
    b.update(z_s, hi_s, z_t, hi_t)

    # 在 [bank ∪ 当前 batch] 上算 kernel — bank 部分稳定样本数, 当前 batch 保留梯度
    pairs = b.get_augmented(z_s, hi_s, z_t, hi_t)
    total = None
    n_used = 0
    for zs, zt in pairs:
        if int(zs.size(0)) >= 2 and int(zt.size(0)) >= 2:
            k = align_kernel(zs, zt)
            total = k if total is None else total + k
            n_used += 1
    if total is None:
        # 所有 bin 都不够 (罕见), fallback 单 batch 保证可微
        return _mmd_by_hi_bins_batch(z_s, hi_s, z_t, hi_t, bins, align_kernel)
    return total / max(n_used, 1)


# ---------------------------------------------------------------- HI 标量直对齐 (消融)
def mmd_on_hi_direct(hi_s: torch.Tensor, hi_t: torch.Tensor,
                     bins,
                     sigmas=(0.01, 0.1, 1.0, 10.0)) -> torch.Tensor:
    """直接在 HI 标量上做 1 维多尺度核 MMD (按 HI bin 分箱), 消融基线。

    设计文档 §4.3 可选新增 (路径 A 消融: mmd_on_hi_direct 组):
      - 不经过 HISeqEncoder latent, 直接对齐源 / 目标域 HI 标量分布。
      - 比 mmd_by_hi_bins(z_s, ..., z_t, ...) 更激进 — 完全绕过 encoder,
        对齐对象从"encoder 编码后的 HI 动力学表示"退化为"HI 数值分布本身"。
      - 用于消融: 验证对齐对象选 latent (动力学层) vs HI 标量 (数值层) 的差异。

    Args:
        hi_s / hi_t: (N,) 源 / 目标 HI 标量。
        bins: list[(lo, hi)] 分箱区间。
        sigmas: 多尺度高斯核带宽。

    Returns:
        mmd^2 标量 (>= 0, 与 mmd_by_hi_bins 同量纲级别)。
    """
    hi_s = hi_s.reshape(-1).float()
    hi_t = hi_t.reshape(-1).float()
    total = None
    n_used = 0
    for lo, hi in bins:
        ms = (hi_s >= lo) & (hi_s < hi)
        mt = (hi_t >= lo) & (hi_t < hi)
        if int(ms.sum()) >= 2 and int(mt.sum()) >= 2:
            xs = hi_s[ms].unsqueeze(1)               # (N, 1) — 当作 1 维特征
            xt = hi_t[mt].unsqueeze(1)
            # 复用多尺度核: 1 维 pairwise sq-dist 与高维一致
            d_ss = _pairwise_sqdist(xs, xs)
            d_tt = _pairwise_sqdist(xt, xt)
            d_st = _pairwise_sqdist(xs, xt)

            def k(d):
                return sum(torch.exp(-d / (2.0 * s * s)) for s in sigmas)

            v = k(d_ss).mean() + k(d_tt).mean() - 2.0 * k(d_st).mean()
            total = v if total is None else total + v
            n_used += 1
    if total is None:
        return hi_s.new_zeros(())
    return total / max(n_used, 1)
