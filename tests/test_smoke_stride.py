"""验证 smoke 模式下 tstride 不会导致 DataLoader 为空。

根因: run_groups.py:340 tstride=1000 if smoke, 但 smoke 轨迹序列长 730 < 1000,
导致 TargetSeqDataset 窗口化为 0 样本 -> DataLoader num_samples=0。
"""
import importlib


def test_smoke_tstride_not_larger_than_seq_len():
    """smoke 模式的 tstride 必须小于 smoke 轨迹序列长度(730)。"""
    # 直接检查源码中的 tstride 值
    cfg_module = importlib.import_module("src.experiments.run_groups")
    import inspect
    source = inspect.getsource(cfg_module.run_one_group)
    # 找到 tstride 赋值行
    for line in source.split("\n"):
        line = line.strip()
        if "tstride" in line and "if smoke" in line and "=" in line:
            # 提取 smoke 分支的数值
            # 形如: tstride = 1000 if smoke else ...
            parts = line.split("if smoke")
            if len(parts) >= 2:
                val_part = parts[0].split("=")[-1].strip()
                try:
                    smoke_stride = int(val_part)
                    assert smoke_stride <= 100, (
                        f"smoke tstride={smoke_stride} 过大; smoke 轨迹序列长 730, "
                        f"stride 应 ≤ 100 以保证每条轨迹产生 ≥7 个窗口"
                    )
                except ValueError:
                    assert False, f"非直接常量赋值, 无法静态检查: {val_part}"
            break
    else:
        # for 循环未 break（未找到 tstride + if smoke 赋值行）=> 测试应显式失败
        assert False, "未找到 tstride if smoke 赋值行; 测试需更新"
