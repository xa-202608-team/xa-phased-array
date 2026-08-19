"""run_groups 并行执行基建契约测试 (2026-08-18 加固).

① 增量落盘: _append_jsonl 逐组-seed append + flush
② 恢复: _load_done_tasks 容忍坏行 (中断写一半), 返回 (group, seed) 集合
③ 资源限制: _worker_init 限制 torch CPU 线程数
背景: 外部实验无资源限制时挤死长跑矩阵 (10:39-12:08 事故), 加固后中断只损失
正在跑的单个组-seed, 重启自动补缺口。
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.run_groups import (                          # noqa: E402
    _append_jsonl,
    _load_jsonl_records,
    _worker_init,
)


class TestJsonlResume:
    def test_append_and_load_roundtrip(self, tmp_path):
        p = tmp_path / "results_partial.jsonl"
        _append_jsonl(p, {"group": "ch_alpha_soft_a005_k3", "seed": 42, "rmse": 0.31})
        _append_jsonl(p, {"group": "ch_alpha_soft_a005_k3", "seed": 43, "rmse": 0.29})
        records = _load_jsonl_records(p)
        assert set(records) == {("ch_alpha_soft_a005_k3", 42), ("ch_alpha_soft_a005_k3", 43)}
        assert records[("ch_alpha_soft_a005_k3", 42)]["rmse"] == 0.31, "旧记录须保留完整指标 (重载入聚合器用)"

    def test_load_tolerates_corrupt_lines(self, tmp_path):
        # 中断时最后一行可能写一半 → 坏行跳过, 该组-seed 重跑 (幂等)
        p = tmp_path / "results_partial.jsonl"
        p.write_text(
            json.dumps({"group": "g_k3", "seed": 42, "rmse": 0.3}) + "\n"
            + '{"group": "g_k3", "seed": 43, "rmse": 0.2\n'      # 截断行
            + "\n", encoding="utf-8")
        records = _load_jsonl_records(p)
        assert ("g_k3", 42) in records
        assert ("g_k3", 43) not in records, "坏行对应的组-seed 必须可重跑"

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert _load_jsonl_records(tmp_path / "nope.jsonl") == {}

    def test_numpy_values_serializable(self, tmp_path):
        # eval_test 返回值若为 numpy 标量, default=float 兜底不崩
        p = tmp_path / "results_partial.jsonl"
        _append_jsonl(p, {"group": "g", "seed": 1, "rmse": float(0.5)})
        rec = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
        assert rec["rmse"] == 0.5


class TestResumeReloadsAggregator:
    def test_resume_reload_logic_static(self):
        """GPT 四审+1 修正的静态守护: main 的恢复段必须把旧记录 append 进 by。

        只跳过不重载会让恢复跑的 aggregate/report 缺已完成臂。
        """
        src = (ROOT / "src" / "experiments" / "run_groups.py").read_text(encoding="utf-8")
        fn = src[src.index("def main()"):]
        assert "prior_records = _load_jsonl_records(jsonl_path)" in fn, "恢复段应读全部旧记录"
        assert "by[_g].append(_m)" in fn, "旧记录必须重载入 by 聚合器"
        assert "not in prior_records" in fn, "已完成 (group, seed) 必须被跳过不重跑"
        # 跳过过滤必须发生在重载循环之前 (先过滤任务, 再重载旧记录)
        i_filter = fn.index("not in prior_records")
        i_reload = fn.index("by[_g].append(_m)")
        assert i_filter < i_reload


class TestWorkerResourceLimit:
    def test_worker_init_sets_threads(self):
        _worker_init(threads=2)
        assert torch.get_num_threads() == 2
        _worker_init(threads=4)
        assert torch.get_num_threads() == 4
