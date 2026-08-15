"""固定随机种子 — 项目可复现性的基石。

设置 python random / numpy / torch / cuda 的种子并切换确定性模式。
所有实验入口均调用 set_seed(cfg['seed'])。
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """固定全部随机源。

    Args:
        seed: 全局种子, 来自 configs/phased_array.yaml 的 seed 字段。
        deterministic: True 则启用 cudnn 确定性 (牺牲少量速度换可复现)。
        cudnn_benchmark: 是否启用 cudnn 自动调优 (复现时应 False)。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def worker_init_fn(worker_id: int) -> None:
    """DataLoader 多进程 worker 的种子初始化, 保证多进程可复现。"""
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
