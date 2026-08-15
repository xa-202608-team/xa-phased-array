"""可复现性测试 — 固定种子两次运行输出哈希完全一致。

P0 验收第 4 条 (plan §四 Phase 0): 同一脚本固定种子运行两次, 输出哈希一致。
覆盖 numpy 随机 / torch 随机 / torch 矩阵算子三类随机源。
"""
import hashlib

import numpy as np
import torch

from src.utils import set_seed

SEED = 42


def _run_once(seed: int) -> str:
    """跑一次确定性计算, 返回结果字节的 sha256。"""
    set_seed(seed, deterministic=True, cudnn_benchmark=False)
    a = np.random.rand(1000)
    b = torch.randn(1000).numpy()
    # 微型前向: 覆盖 cudnn 算子确定性
    w = torch.randn(8, 4)
    c = (torch.randn(16, 8) @ w).sum().item()
    payload = a.tobytes() + b.tobytes() + str(c).encode()
    return hashlib.sha256(payload).hexdigest()


def test_seed_reproducibility():
    """同一种子两次运行, 哈希必须一致。"""
    h1 = _run_once(SEED)
    h2 = _run_once(SEED)
    assert h1 == h2, f"同一种子两次运行哈希不一致: {h1} vs {h2}"


def test_seed_sensitivity():
    """不同种子应产生不同结果 (排除种子被忽略的退化)。"""
    h1 = _run_once(SEED)
    h2 = _run_once(SEED + 1)
    assert h1 != h2, "不同种子产生相同结果, 种子未生效"


if __name__ == "__main__":
    print(f"seed={SEED} repro hash: {_run_once(SEED)}")
