"""实验卫生测试 (任务 1/2/3 验收):

  ① MMD 全局 bank reset 隔离 — 连续两次 run bank 不串
  ② RUL 归一因子只用 train 集估计 — test 集极值不进归一因子
  ③ 源 MMD 对齐窗排除 val 器件 — val device 不参与迁移对齐

这三项是审查会重点核查的实验卫生项 (test 泄漏 / 跨 run 污染),
对应 src/experiments/run_groups.py 和 src/transfer/train_transfer.py 的修改。
"""
import sys
from pathlib import Path

import numpy as np
import statistics
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.transfer.mmd import (                                        # noqa: E402
    _GLOBAL_BANK,
    get_global_memory_bank,
    mmd_by_hi_bins,
    reset_global_memory_bank,
)
from src.transfer.train_transfer import load_source, split_trajectories   # noqa: E402


# ============================================================ 任务 1: reset 隔离
def test_reset_clears_global_bank():
    """reset_global_memory_bank() 后 _GLOBAL_BANK 必须为 None。"""
    reset_global_memory_bank()
    # 累积一些样本触发 lazy init
    for _ in range(3):
        mmd_by_hi_bins(torch.randn(8, 4), torch.rand(8),
                       torch.randn(8, 4), torch.rand(8), [(0.0, 1.0)])
    bank = get_global_memory_bank([(0.0, 1.0)])
    assert bank._src_z[0] is not None, "累积后 bank 应非空"
    reset_global_memory_bank()
    from src.transfer.mmd import _GLOBAL_BANK as BANK_NOW
    assert BANK_NOW is None, "reset 后 _GLOBAL_BANK 应为 None"


def test_reset_isolates_consecutive_runs():
    """连续两次 (reset → 累积), 第二次 bank 不含第一次的样本 (不串)。"""
    bins = [(0.0, 1.0)]

    # Run A: 累积 src latent 均值 ≈ 0
    reset_global_memory_bank()
    torch.manual_seed(11)
    for _ in range(5):
        mmd_by_hi_bins(torch.randn(16, 4) * 0.1 + 0.0,    # mean ≈ 0
                       torch.rand(16),
                       torch.randn(16, 4), torch.rand(16), bins)
    bank_a = get_global_memory_bank(bins)
    # 拷贝出 run A 的源 latent 均值引用
    mean_a = float(bank_a._src_z[0].mean().item())

    # Run B: reset 后累积完全不同的 src latent (均值 ≈ 50)
    reset_global_memory_bank()
    torch.manual_seed(22)
    for _ in range(5):
        mmd_by_hi_bins(torch.randn(16, 4) * 0.1 + 50.0,   # mean ≈ 50
                       torch.rand(16),
                       torch.randn(16, 4), torch.rand(16), bins)
    bank_b = get_global_memory_bank(bins)
    mean_b = float(bank_b._src_z[0].mean().item())

    # 两次 run 的 bank 应处于完全不同的数值范围 (隔离)
    assert abs(mean_b - 50.0) < 5.0, f"run B bank 应反映 run B 的 latent, mean={mean_b}"
    assert abs(mean_b - mean_a) > 30.0, (
        f"reset 后 bank 不应串入前一次 run 的 latent: mean_a={mean_a:.2f} "
        f"mean_b={mean_b:.2f} (差应 > 30)")
    reset_global_memory_bank()


def test_run_one_group_and_run_call_reset_static():
    """静态校验: run_groups.run_one_group 与 train_transfer.run/run_hi_layer
    源码均包含 reset_global_memory_bank() 调用 (防回归)。

    run_one_group 是 main() 单进程 for-loop 跨 seed/group 的入口, 若漏 reset 则
    全局 bank 跨 run 累积 → source_mmd 组 std 数字混入别的 seed/group 的 latent。
    """
    rg_src = (ROOT / "src" / "experiments" / "run_groups.py").read_text(encoding="utf-8")
    tt_src = (ROOT / "src" / "transfer" / "train_transfer.py").read_text(encoding="utf-8")

    # import 检查
    assert "reset_global_memory_bank" in rg_src, \
        "run_groups.py 应 import reset_global_memory_bank"
    assert "reset_global_memory_bank" in tt_src, \
        "train_transfer.py 应 import reset_global_memory_bank"

    # run_one_group 内的调用 (在 set_seed 之后)
    assert "reset_global_memory_bank()" in rg_src, \
        "run_groups.run_one_group 应调用 reset_global_memory_bank()"
    # run / run_hi_layer 内的调用
    assert tt_src.count("reset_global_memory_bank()") >= 2, \
        "train_transfer.py 中 run() 和 run_hi_layer() 都应调用 reset_global_memory_bank()"


# ============================================================ 任务 2: train-only max
def test_train_only_rul_max_excludes_test():
    """RUL 归一因子只用 train 集估计, test 集的大 RUL 不进归一因子。

    构造 train rul ∈ [0, 1], test rul = 100 (远大于 train max),
    应用 run_one_group 同款公式 rulT[mask(tr)].max(), 确认归一因子 = 1.0
    (而非泄漏的 100.0)。
    """
    # 构造 10 条轨迹, 每条 5 步; 全部用 rul ∈ [0, 1]
    n_traj = 10
    n_per = 5
    base = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)
    big = np.array([100.0, 80.0, 60.0, 40.0, 20.0], dtype=np.float32)
    rul_per_traj = np.tile(base, (n_traj, 1)).astype(np.float32)
    rulT = rul_per_traj.reshape(-1)
    tidT = np.repeat(np.arange(n_traj), n_per)

    # 先 split (固定 seed 可复现), 再把大 RUL 赋给 te 中一条轨迹
    tr, va, te = split_trajectories(n_traj, [0.7, 0.1, 0.2], seed=42)
    assert len(te) >= 1
    big_traj = int(te[0])                      # te 中第一条轨迹赋大 RUL
    rul_per_traj[big_traj] = big
    rulT = rul_per_traj.reshape(-1)

    # 确认 tr 不含大 RUL 轨迹 (split 不相交)
    assert big_traj not in set(tr.tolist()), \
        f"测试构造: big_traj={big_traj} 不应在 tr={tr} (split 不相交)"

    # run_one_group 同款公式 (任务 2 修改后)
    def mask(ids):
        return np.isin(tidT, ids)

    rul_max_train = float(rulT[mask(tr)].max())
    # 旧 (泄漏) 公式
    rul_max_full = float(rulT.max())

    assert rul_max_train == pytest.approx(1.0, abs=1e-6), \
        f"train-only max 应 = 1.0 (train 集 rul 上界), 实际 {rul_max_train}"
    assert rul_max_full == pytest.approx(100.0, abs=1e-6), \
        "旧全量 max 应 = 100.0 (含 test 大值)"
    assert rul_max_train < rul_max_full, \
        "train-only max 必须严格小于全量 max (堵住 test 泄漏)"


def test_train_only_rul_max_run_groups_and_run_formula_consistent():
    """静态校验: run_groups.run_one_group / train_transfer.run / run_hi_layer
    均使用 train-only max (含 rul_max_train 标识符, 不含旧的 rulT.max() 归一)。
    """
    rg_src = (ROOT / "src" / "experiments" / "run_groups.py").read_text(encoding="utf-8")
    tt_src = (ROOT / "src" / "transfer" / "train_transfer.py").read_text(encoding="utf-8")

    # 新标识符
    assert "rul_max_train" in rg_src, \
        "run_groups.py 应使用 rul_max_train (train-only max)"
    assert tt_src.count("rul_max_train") >= 2, \
        "train_transfer.py 中 run() 和 run_hi_layer() 都应使用 rul_max_train"

    # 旧的 'rulT = rulT / max(float(rulT.max())' 不应再出现 (已替换)
    assert "rulT / max(float(rulT.max())" not in rg_src, \
        "run_groups.py 不应保留全量 max 归一 (旧泄漏公式)"
    assert "rulT / max(float(rulT.max())" not in tt_src, \
        "train_transfer.py 不应保留全量 max 归一 (旧泄漏公式)"


# ============================================================ 任务 3: 源窗过滤 val
def test_source_filter_excludes_val_devices():
    """源 MMD 对齐窗排除 val 器件: 按源 split_array=='train' 过滤后,
    val 器件的行不进 SourceWindowDataset / make_hi_windows。

    构造 6 个器件 (4 train + 2 val), 过滤后只保留 4 个 train 器件的行。
    """
    # 6 器件 × 5 行
    n_per = 5
    split = np.array(["d0", "d1", "d2", "d3", "d4", "d5"]).repeat(n_per)
    is_train = split == "train"    # 全部 train (默认)
    # 人为标记 d4 / d5 为 val
    split_array = np.where(
        np.isin(split, ["d4", "d5"]), "val", "train")
    hi = np.arange(30, dtype=np.float32)
    ids = split   # 用名字作 id
    t_idx = np.tile(np.arange(n_per), 6)

    # run_hi_layer / run 同款过滤逻辑
    src_tr = np.asarray(split_array) == "train"
    if src_tr.any():
        hi_f = hi[src_tr]
        ids_f = np.asarray(ids)[src_tr]
        t_idx_f = t_idx[src_tr]
    else:
        hi_f, ids_f, t_idx_f = hi, ids, t_idx

    # 过滤后应只含 d0-d3 的 20 行
    assert len(hi_f) == 20, f"过滤后应剩 20 行 (4 train 器件 × 5), 实际 {len(hi_f)}"
    assert set(np.unique(ids_f).tolist()) == {"d0", "d1", "d2", "d3"}, \
        "val 器件 d4/d5 不应在过滤后的源对齐窗中"
    assert "d4" not in set(np.unique(ids_f).tolist())
    assert "d5" not in set(np.unique(ids_f).tolist())


def test_load_source_returns_split_array(tmp_path):
    """load_source 返回 5 元组 (新增 split_array), 调用方可按 'train' 过滤。

    构造最小 v1 h5: 2 bearing × 5 行, bearing_id=b1 标 val。
    """
    import h5py
    h5_path = tmp_path / "src.h5"
    n = 5
    feat = np.random.RandomState(0).randn(2 * n, 3).astype(np.float32)
    hi = np.linspace(0, 1, 2 * n, dtype=np.float32)
    rul = np.linspace(1, 0, 2 * n, dtype=np.float32)
    t_idx = np.tile(np.arange(n, dtype=np.int64), 2)
    bid = np.array(["b0"] * n + ["b1"] * n, dtype="S8")
    split = np.array(["train"] * n + ["val"] * n, dtype="S8")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("features", data=feat)
        f.create_dataset("hi", data=hi)
        f.create_dataset("rul", data=rul)
        f.create_dataset("t_idx", data=t_idx)
        f.create_dataset("bearing_id", data=bid)
        f.create_dataset("split", data=split)

    out = load_source(h5_path, id_field="bearing_id")
    assert len(out) == 5, "load_source 应返回 5 元组 (新增 split_array)"
    feats, hi_s, dev_ids, t_index, split_arr = out
    # v1: split 数组 dtype=object, 含 'train'/'val'
    split_list = list(split_arr)
    assert split_list.count("train") == n
    assert split_list.count("val") == n

    # 过滤 train (run/run_hi_layer/run_one_group 同款逻辑)
    src_tr = np.asarray(split_arr) == "train"
    feats_tr = feats[src_tr]
    assert feats_tr.shape[0] == n, "过滤后应只剩 train 器件 (b0) 的 5 行"
    # 验证 b1 (val) 的特征均值与 b0 不同, 确认过滤有效
    feats_val = feats[~src_tr]
    assert not np.allclose(feats_tr.mean(0), feats_val.mean(0)), \
        "train/val 特征应有差异 (否则过滤效果不可验证)"


def test_run_groups_source_filter_applied_static():
    """静态校验: run_groups.run_one_group / train_transfer.run / run_hi_layer
    均按 split_array=='train' 过滤源 MMD 窗 (防 val 泄漏到迁移对齐)。
    """
    rg_src = (ROOT / "src" / "experiments" / "run_groups.py").read_text(encoding="utf-8")
    tt_src = (ROOT / "src" / "transfer" / "train_transfer.py").read_text(encoding="utf-8")

    # 3 个调用点都应含 splitS 解包 (5 元组) + _src_tr 过滤
    assert "featsS, hiS, bidS, tidxS, splitS = load_source(" in rg_src, \
        "run_groups.run_one_group 应解包 5 元组 (含 splitS)"
    assert "_src_tr = np.asarray(splitS) == \"train\"" in rg_src, \
        "run_groups.run_one_group 应按 split_array=='train' 过滤源窗"
    # train_transfer.py 有两处 (run 旧观测层 + run_hi_layer 新 HI 层)
    assert tt_src.count("splitS = load_source(") >= 1 or \
           tt_src.count("_, hiS, idS, tidxS, splitS = load_source(") >= 1, \
        "train_transfer.py 应解包 5 元组"
    assert tt_src.count("_src_tr = np.asarray(splitS) == \"train\"") >= 2, \
        "train_transfer.py 中 run() 和 run_hi_layer() 都应按 split_array=='train' 过滤源窗"


# ============================================================ P0-2: paired CI 不从 test 选
def test_aggregate_paired_all_sources_and_config_primary():
    """P0-2: aggregate 对所有 source_* 组算 paired; 主组=config 指定, 不从 test argmin 选。

    构造 1 target + 2 source 组, 验证:
      1) paired 对每个 source 组都算
      2) primary_source_group 即使不是 test 最优也生效
      3) 不再用 min(rmse_mean) 选最优迁移组
    """
    from src.experiments.run_groups import aggregate
    by = {
        "target_only_tcn": [
            {"rmse": 0.20, "phm": 1.0, "mae": 0.15, "seed": 42},
            {"rmse": 0.22, "phm": 1.0, "mae": 0.16, "seed": 43},
            {"rmse": 0.18, "phm": 1.0, "mae": 0.14, "seed": 44},
        ],
        "source_pretrain_finetune": [
            {"rmse": 0.19, "phm": 1.0, "mae": 0.14, "seed": 42},
            {"rmse": 0.21, "phm": 1.0, "mae": 0.15, "seed": 43},
            {"rmse": 0.17, "phm": 1.0, "mae": 0.13, "seed": 44},
        ],
        "source_mmd_physics": [
            {"rmse": 0.15, "phm": 1.0, "mae": 0.12, "seed": 42},   # 均值最低
            {"rmse": 0.16, "phm": 1.0, "mae": 0.13, "seed": 43},
            {"rmse": 0.14, "phm": 1.0, "mae": 0.11, "seed": 44},
        ],
    }
    # primary = source_pretrain_finetune (非 test 最优)
    agg = aggregate(by, primary_source_group="source_pretrain_finetune")
    paired = agg["_paired"]
    # all_pairs 是 {target: {source: stats}} (P0-4 每 target 都配对); 取 primary_target 那层
    tgt_g = paired["primary_target"]
    all_src = paired["all_pairs"][tgt_g]

    # 1) 对每个 source 组都算了 paired
    assert "source_pretrain_finetune" in all_src, "paired 应覆盖 source_pretrain_finetune"
    assert "source_mmd_physics" in all_src, "paired 应覆盖 source_mmd_physics"

    # 2) 主组 = config 指定 (不是 test argmin)
    assert paired["primary_source_group"] == "source_pretrain_finetune", \
        "主组应 = config primary_source_group, 而非 test argmin"

    # 3) delta = target - source (正值表示迁移有帮助)
    for sg, ps in all_src.items():
        assert ps["n_seeds"] == 3
        assert ps["delta_mean"] > 0, f"{sg}: target RMSE 应 > source (delta>0)"


def test_aggregate_no_test_argmin_selection():
    """P0-2: 确认 aggregate 不含 min(src_groups, key=rmse_mean) 的 test selection。"""
    rg_src = (ROOT / "src" / "experiments" / "run_groups.py").read_text(encoding="utf-8")
    # 旧: best_src = min(src_groups, key=lambda g: agg[g]["rmse_mean"])
    # 新: 不应再出现从 agg rmse_mean 选 best_src 的逻辑
    assert 'best_src = min' not in rg_src, \
        "aggregate 不应用 test rmse_mean argmin 选最优迁移组 (P0-2 test selection bias)"


# ============================================================ P1-1: CI 小样本校正
def test_paired_ci_uses_sample_stdev_and_t():
    """P1-1: CI 用样本 std (statistics.stdev) + t(n-1) 临界值, 不用 pstdev + 1.96。

    n=5 手算: deltas = [0.05, 0.06, 0.04, 0.06, 0.04]
      mean = 0.05, stdev(n-1) = 0.01, t(4, 0.975) ≈ 2.7764
      CI = 0.05 ± 2.7764 × 0.01/√5 = [0.03758, 0.06242]
    对比旧: pstdev + 1.96 => CI = [0.03758, 0.06242] 但 std 不同 (pstdev < stdev)
    """
    import math
    from scipy import stats as sp_stats
    from src.experiments.run_groups import _paired_delta_ci

    tgt = {0: 0.20, 1: 0.22, 2: 0.18, 3: 0.21, 4: 0.19}
    src = {0: 0.15, 1: 0.16, 2: 0.14, 3: 0.15, 4: 0.15}
    deltas = [0.05, 0.06, 0.04, 0.06, 0.04]

    result = _paired_delta_ci(tgt, src)

    # 手算验证
    mean_d = statistics.mean(deltas)
    std_sample = statistics.stdev(deltas)    # 分母 n-1
    std_pop = statistics.pstdev(deltas)       # 分母 n
    n = 5
    crit_t = float(sp_stats.t.ppf(0.975, n - 1))

    assert result["delta_mean"] == pytest.approx(mean_d, abs=1e-10)
    assert result["delta_std"] == pytest.approx(std_sample, abs=1e-10), \
        "应使用样本 std (statistics.stdev, 分母 n-1)"
    assert result["delta_std"] > std_pop, \
        "样本 std 应 > 总体 std (n-1 分母更小 → std 更大)"

    # CI 用 t 临界值而非 1.96
    se = std_sample / math.sqrt(n)
    expected_lo = mean_d - crit_t * se
    expected_hi = mean_d + crit_t * se
    assert result["ci95_lo"] == pytest.approx(expected_lo, abs=1e-6)
    assert result["ci95_hi"] == pytest.approx(expected_hi, abs=1e-6)

    # t(4) = 2.7764 > 1.96, 所以 CI 应比正态近似更宽
    ci_norm_lo = mean_d - 1.96 * se
    assert result["ci95_lo"] < ci_norm_lo, "t-CI 下界应低于正态近似 (t > z)"


def test_paired_ci_exploratory_only_for_small_n():
    """P1-1: n<10 标 exploratory_only=True; n>=10 标 False。"""
    from src.experiments.run_groups import _paired_delta_ci

    # n=5 → exploratory
    tgt5 = {i: 0.20 + i * 0.001 for i in range(5)}
    src5 = {i: 0.15 + i * 0.001 for i in range(5)}
    r5 = _paired_delta_ci(tgt5, src5)
    assert r5["exploratory_only"] is True, "n=5 应标 exploratory_only"

    # n=10 → 非 exploratory
    tgt10 = {i: 0.20 + i * 0.001 for i in range(10)}
    src10 = {i: 0.15 + i * 0.001 for i in range(10)}
    r10 = _paired_delta_ci(tgt10, src10)
    assert r10["exploratory_only"] is False, "n=10 不应标 exploratory_only"


def test_rmse_std_uses_sample_stdev():
    """P1-1: aggregate 的 rmse_std 用 statistics.stdev (样本), 非 pstdev。"""
    from src.experiments.run_groups import aggregate
    by = {
        "target_only": [
            {"rmse": 0.1, "phm": 1.0, "mae": 0.08, "seed": 1},
            {"rmse": 0.3, "phm": 1.0, "mae": 0.22, "seed": 2},
            {"rmse": 0.2, "phm": 1.0, "mae": 0.15, "seed": 3},
        ],
    }
    agg = aggregate(by)
    vals = [0.1, 0.3, 0.2]
    assert agg["target_only"]["rmse_std"] == pytest.approx(statistics.stdev(vals))
    assert agg["target_only"]["rmse_std"] != pytest.approx(statistics.pstdev(vals))


# ============================================================ P1-3: 源 split 失败保护
def test_empty_source_split_raises_static():
    """P1-3: 源 train split 为空时代码 raise ValueError, 不静默保留全量 (静态校验)。"""
    rg_src = (ROOT / "src" / "experiments" / "run_groups.py").read_text(encoding="utf-8")
    tt_src = (ROOT / "src" / "transfer" / "train_transfer.py").read_text(encoding="utf-8")

    assert 'raise ValueError("source train split empty' in rg_src, \
        "run_groups.py 应在 source train split 为空时 raise ValueError"
    # train_transfer.py run() 和 run_hi_layer() 两处
    assert tt_src.count('raise ValueError("source train split empty') >= 2, \
        "train_transfer.py 中 run() 和 run_hi_layer() 都应在空 split 时 raise ValueError"
    # 不应保留旧的 if _src_tr.any(): 静默保留模式
    assert "if _src_tr.any():" not in rg_src, \
        "run_groups.py 不应保留 if _src_tr.any(): 静默保留模式"
    assert "if _src_tr.any():" not in tt_src, \
        "train_transfer.py 不应保留 if _src_tr.any(): 静默保留模式"


def test_source_filter_train_val_no_intersection():
    """P1-3: 过滤后 train 器件集与 val 器件集无交集。"""
    # 6 器件 × 3 行: d0-d3 train, d4-d5 val
    n_per = 3
    ids = np.array([f"d{i}" for i in range(6)]).repeat(n_per)
    split_flags = np.where(np.isin(ids, ["d4", "d5"]), "val", "train")

    _src_tr = split_flags == "train"
    ids_train = set(ids[_src_tr].tolist())
    ids_val = set(ids[~_src_tr].tolist())

    assert ids_train.isdisjoint(ids_val), \
        f"train/val 器件集应无交集: train={ids_train}, val={ids_val}"
    assert ids_train == {"d0", "d1", "d2", "d3"}
    assert ids_val == {"d4", "d5"}


def test_source_filter_preserves_within_device_time_order():
    """P1-3: 过滤只删行不重排, 器件内时间序保持不变。"""
    # 3 器件 × 4 行: d2 标 val
    n_per = 4
    ids = np.array(["d0"] * n_per + ["d1"] * n_per + ["d2"] * n_per)
    split_flags = np.where(np.isin(ids, ["d2"]), "val", "train")
    t_idx = np.tile(np.arange(n_per), 3)
    hi = np.arange(12, dtype=np.float32)

    _src_tr = split_flags == "train"
    t_f = t_idx[_src_tr]
    ids_f = ids[_src_tr]

    # d0 / d1 的 4 行时间序 = [0,1,2,3]
    d0_mask = ids_f == "d0"
    assert np.array_equal(t_f[d0_mask], [0, 1, 2, 3]), "d0 过滤后时间序应不变"
    d1_mask = ids_f == "d1"
    assert np.array_equal(t_f[d1_mask], [0, 1, 2, 3]), "d1 过滤后时间序应不变"
    # d2 全删
    assert "d2" not in set(ids_f.tolist()), "val 器件 d2 应被删除"


# ============================================================ P0-1: HI 层 early-stop
def test_hi_layer_uses_early_stop_and_equal_budget_static():
    """P0-1 静态校验: run_hi_layer 使用 _train_hi_stage_es (val early-stop + 恢复最佳模型),
    target_only S3 设 use_mmd=False (纯微调, 但 epoch 等量)。
    """
    tt_src = (ROOT / "src" / "transfer" / "train_transfer.py").read_text(encoding="utf-8")
    # 存在公共训练函数
    assert "def _train_hi_stage_es(" in tt_src, \
        "应抽公共函数 _train_hi_stage_es (HI 层 val early-stop)"
    # run_hi_layer 中 S2/S3 调用它
    assert "_train_hi_stage_es(model, loader_T_tr, loader_T_va" in tt_src, \
        "run_hi_layer 的 S2/S3 应调用 _train_hi_stage_es"
    # target_only S3 也跑 (e3 不再为 0)
    assert "e3 = 0 if getattr" not in tt_src, \
        "不应再设 target_only 的 e3=0 (P0-1: target/source 同预算)"
    # target_only S3 use_mmd=False
    assert 'is_target_only' in tt_src, "应通过 is_target_only 分支控制 S3 的 use_mmd"


def test_phase3c_calls_run_hi_layer_static():
    """P0-1 静态校验: phase3c 脚本 import + 调用 run_hi_layer, 不再用 subprocess。"""
    script = (ROOT / "scripts" / "phase3c_hi_layer_seeds.py").read_text(encoding="utf-8")
    assert "from src.transfer.train_transfer import run_hi_layer" in script, \
        "phase3c 应 import run_hi_layer 统一入口"
    assert "import subprocess" not in script, \
        "phase3c 不应再 import subprocess (改为直接调用 run_hi_layer)"
    assert "subprocess.run" not in script, \
        "phase3c 不应再调用 subprocess.run (改为直接调用 run_hi_layer)"


# ============================================================ 清零重审: 架构纪律 + L_phys 污染根除

def test_group_map_architecture_discipline():
    """迁移归因组统一显式 GRU (与主模型 target_only_gru 同架构); *_tcn 组名真 TCN (名实一致)。

    清零重审: 旧版 GRU target-only vs TCN transfer 混架构比较无法归因源迁移; 修复中间版
    曾把 *_tcn 组强制成 GRU 造成名实不符。本测试锁死两条纪律。
    """
    from src.experiments.run_groups import _GROUP_MAP
    for name, (_mode, enc) in _GROUP_MAP.items():
        if name.endswith("_tcn"):
            assert enc == "tcn", f"{name} 组名含 _tcn 应为真 TCN 架构消融, 实际 {enc}"
    attribution = [
        "source_pretrain_finetune", "source_mmd_physics", "random_frozen",
        "random_full_finetune", "random_nommd",
        "ch_source_pretrain_frozen", "ch_source_mmd_physics", "ch_random_frozen",
        "ch_random_full_finetune", "ch_random_nommd", "cross_level_transfer",
    ]
    for name in attribution:
        assert _GROUP_MAP[name][1] == "gru",             f"{name} 迁移归因组应显式 GRU, 实际 {_GROUP_MAP[name][1]}"


def test_group_map_channel_transfer_groups_not_default_tcn():
    """通道级迁移组不允许 None (config 默认 tcn) — 防回归到混架构归因。"""
    from src.experiments.run_groups import _GROUP_MAP
    for name in ["ch_source_pretrain_frozen", "ch_source_mmd_physics",
                 "ch_random_frozen", "ch_random_full_finetune", "ch_random_nommd"]:
        assert _GROUP_MAP[name][1] is not None, f"{name} 不应回落 config 默认 encoder"


def test_channel_lphys_disabled_without_damage_truth():
    """清零重审 P0: 通道级无 damage 真值时 S3 必须禁用 ρ·L_phys (全零占位会把 HI 拉向 0)。"""
    rg_src = (ROOT / "src" / "experiments" / "run_groups.py").read_text(encoding="utf-8")
    assert "damageT = None" in rg_src, "channel 分支应设 damageT=None (非全零占位)"
    assert rg_src.count("damageT is not None") >= 3,         "mkDS damage_b + S3 两处 use_phys 应显式判 damageT is not None"
    assert "np.zeros_like(rulT)" not in rg_src, "不应再出现全零 damage 占位"


def test_target_seq_dataset_damage_none_placeholder_zero():
    """TargetSeqDataset damage_b=None → 第 6 元 (dmg) 全 0 占位; 有真值时透传 (服务级)。"""
    from src.transfer.train_transfer import TargetSeqDataset
    L, K, T = 4, 2, 10
    rng = np.random.default_rng(0)
    x = rng.normal(size=(T, 3)).astype(np.float32)
    hi = np.linspace(0.1, 0.9, T)
    rul = np.linspace(1.0, 0.0, T)
    tid = np.zeros(T, dtype=int)
    dmg = np.linspace(0.0, 0.8, T)
    ds_none = TargetSeqDataset(x, hi, rul, tid, L, K, damage_b=None)
    ds_real = TargetSeqDataset(x, hi, rul, tid, L, K, damage_b=dmg)
    assert len(ds_none) > 0 and len(ds_real) == len(ds_none)
    for i in range(len(ds_none)):
        dn = np.asarray(ds_none[i][-1], dtype=float)
        dv = np.asarray(ds_real[i][-1], dtype=float)
        assert (dn == 0.0).all(), "None 占位应为全 0"
        assert not (dv == 0.0).all(), "真值不应被置 0"
