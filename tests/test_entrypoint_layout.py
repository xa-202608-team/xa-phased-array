# -*- coding: utf-8 -*-
"""entrypoint.sh 挂载布局守护测试（设计 §7）。

层级优先契约：canonical H5 必须先按层级工件布局探测
/artifacts/data/features/phased_array/schema_v4/source/mosfet_canonical.h5
（新契约，设计 §7「不能继续假定文件直挂 /artifacts/data/ 根」），
不存在时回退 legacy 平挂 /artifacts/data/mosfet_canonical.h5 并输出显式警告。
纯文本断言 + bash -n 语法检查（Windows 下 bash -n 由 Git Bash 提供）。
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "scripts" / "entrypoint.sh"

# 设计 §7 层级工件布局（与新交付 04_数据/phased_array 内部相对层级一致）
HIER_MOUNT = "/artifacts/data/features/phased_array/schema_v4/source/mosfet_canonical.h5"
# legacy 平挂布局（旧交付顶层 Compose /data 直挂方式）
LEGACY_MOUNT = "/artifacts/data/mosfet_canonical.h5"

HIER_VAR = '"$CANONICAL_MOUNT_HIER"'
LEGACY_VAR = '"$CANONICAL_MOUNT_LEGACY"'


def _script_text() -> str:
    return ENTRYPOINT.read_text(encoding="utf-8")


def _find_git_bash():
    """返回可用的 Git Bash 路径；跳过 WSL 的 System32/WindowsApps bash
    （其不接受 Windows 路径参数）。找不到返回 None。"""
    bash = shutil.which("bash")
    if bash and "system32" not in bash.lower() and "windowsapps" not in bash.lower():
        return bash
    for cand in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"F:\Git\usr\bin\bash.exe",
    ):
        if Path(cand).is_file():
            return cand
    return None


def _mount_artifacts_body(text: str) -> str:
    """截取 mount_artifacts() 函数体（到首个闭合大括号）。"""
    start = text.index("mount_artifacts()")
    end = text.index("\n}", start)
    return text[start:end]


def test_hierarchical_mount_path_declared():
    """入口须声明设计 §7 层级工件挂载路径，且绑定到层级优先变量。"""
    text = _script_text()
    assert HIER_MOUNT in text, f"entrypoint.sh 须声明层级挂载路径 {HIER_MOUNT}（设计 §7）"
    assert f'CANONICAL_MOUNT_HIER="{HIER_MOUNT}"' in text


def test_legacy_fallback_path_retained():
    """legacy 平挂回退路径与变量保留（兼容旧布局）。"""
    text = _script_text()
    assert f'CANONICAL_MOUNT_LEGACY="{LEGACY_MOUNT}"' in text


def test_hierarchical_probe_precedes_legacy_fallback():
    """层级探测必须先于 legacy 平挂回退（if/elif 顺序，设计 §7 层级优先）。"""
    body = _mount_artifacts_body(_script_text())
    hier_idx = body.index(HIER_VAR)
    legacy_idx = body.index(LEGACY_VAR)
    assert hier_idx < legacy_idx, "mount_artifacts 须先探测层级布局，再回退 legacy 平挂"


def test_legacy_fallback_emits_explicit_warning():
    """回退平挂时必须输出显式迁移警告。"""
    body = _mount_artifacts_body(_script_text())
    assert "legacy 平挂布局" in body, "回退路径须含「legacy 平挂布局」警告文案"
    assert "请迁移到层级工件布局" in body


def test_missing_path_falls_to_synthetic_message_retained():
    """两处都不存在时保持既有缺失处理（synthetic 源域提示）不变。"""
    body = _mount_artifacts_body(_script_text())
    assert "synthetic 源域" in body


@pytest.mark.skipif(_find_git_bash() is None, reason="无 Git Bash（System32/WindowsApps 的 WSL bash 不适用）")
def test_bash_syntax():
    """bash -n 语法检查（Git Bash；Windows 路径转正斜杠，避开 MSYS 反斜杠吞噬）。"""
    proc = subprocess.run(
        [_find_git_bash(), "-n", str(ENTRYPOINT).replace("\\", "/")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"bash -n 失败: {proc.stderr}"
