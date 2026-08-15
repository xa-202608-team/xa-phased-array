# -*- coding: utf-8 -*-
"""仓库边界守护测试：本地工件槽位不得被跟踪。

约束（与导入策略、公开泄漏扫描器语义一致）：
- 根级 ``data/``、``results/``、``checkpoints/`` 是本地工件槽位，除七个批准
  描述文件外不得出现任何受跟踪文件；
- 所有 ``.pt/.h5/.hdf5/.log`` 文件一律不受跟踪（无论目录）；
- 测试基于 ``git ls-files -z``（与扫描器同源），因此只做静态索引断言，
  不依赖本地工件是否存在。
"""
import subprocess
from pathlib import Path

# 设计批准的槽位描述文件（本地维护；若被强制跟踪亦不豁免扩展名规则）
APPROVED_SLOT_FILES = {
    "data/README.md",
    "data/data_manifest.json",
    "results/README.md",
    "results/public_summary.json",
    "results/expected_metrics.json",
    "checkpoints/README.md",
    "checkpoints/checkpoint_manifest.json",
}

# 工件类扩展名：任何位置都不得受跟踪
FORBIDDEN_SUFFIXES = (".pt", ".h5", ".hdf5", ".log")


def _tracked_files(root: Path) -> list[str]:
    out = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root
    ).decode("utf-8")
    return [p for p in out.split("\0") if p]


def test_no_forbidden_tracked_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = _tracked_files(root)
    forbidden = [
        path
        for path in tracked
        if path.startswith(("data/", "results/", "checkpoints/"))
        and path not in APPROVED_SLOT_FILES
    ]
    assert forbidden == [], f"工件槽位出现未批准受跟踪文件: {forbidden}"


def test_no_tracked_artifact_suffixes() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = _tracked_files(root)
    offenders = [p for p in tracked if p.lower().endswith(FORBIDDEN_SUFFIXES)]
    assert offenders == [], f"工件扩展名文件被跟踪: {offenders}"


def test_no_tracked_pycache_or_pytest_cache() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = _tracked_files(root)
    offenders = [
        p
        for p in tracked
        if "/__pycache__/" in f"/{p}" or "/.pytest_cache/" in f"/{p}"
        or p.startswith(("__pycache__/", ".pytest_cache/"))
        or p.endswith(".pyc")
    ]
    assert offenders == [], f"缓存产物被跟踪: {offenders}"
