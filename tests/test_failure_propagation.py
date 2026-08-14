"""验证 run_groups 在实验臂失败时返回非零退出码 (P0-2 失败传播)。

Docker/CI 依据 run_groups 的退出码判定复现是否成功:
任一实验臂失败 (数据缺失 / 训练异常) 都必须让进程以非零码退出,
而不是静默继续并产出看似完整的结果文件。
"""
import subprocess
import sys
from pathlib import Path

import yaml


def test_run_groups_exits_nonzero_on_failure(tmp_path):
    """模拟一个实验臂失败时, run_groups.py 应以非零码退出。

    用故意损坏的配置 (指向不存在的数据文件) 触发失败, 验证退出码。
    注意: config 必须含 experiments.groups (至少一组), 否则 main() 在
    "无 group 可跑" 分支 return 0, 测不到失败传播路径。
    """
    bad_cfg = {
        "seed": 42,
        "reproducibility": {"deterministic": True, "cudnn_benchmark": False},
        "model": {"input_len_L": 64},
        "transfer": {"hi_bins": [[0.0, 1.0]], "mmd_lambda": 1.0,
                     "split": {"train": 0.7, "val": 0.1, "test": 0.2}},
        "loss": {"huber_delta": 1.0},
        # 至少一组实验臂; ch_ 前缀 → channel level (run_one_group 先查 channel_level.feature_path)
        "experiments": {"groups": ["ch_target_only_tcn"]},
        # 指向不存在的数据文件 → run_one_group 抛 FileNotFoundError → 失败臂
        "channel_level": {"feature_path": "/nonexistent/path.h5"},
    }
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.dump(bad_cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "src.experiments.run_groups",
         "--config", str(cfg_path), "--smoke", "--level", "channel"],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(__file__).resolve().parent.parent)
    )
    assert result.returncode != 0, (
        f"run_groups 应在实验臂失败时返回非零退出码, 但返回了 {result.returncode}\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
    # 失败臂应打印到 stdout (供 Docker 日志定位), 排除"因 import 等无关原因崩溃"的假通过
    assert "失败" in result.stdout, (
        f"失败臂信息应打印到 stdout 供日志定位\nstdout: {result.stdout[-500:]}"
    )
