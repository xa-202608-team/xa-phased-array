"""共享工具: 种子固定 / 配置加载。"""
from .seed import set_seed, worker_init_fn
from .config import load_config, PROJECT_ROOT, DEFAULT_CONFIG

__all__ = ["set_seed", "worker_init_fn", "load_config", "PROJECT_ROOT", "DEFAULT_CONFIG"]
