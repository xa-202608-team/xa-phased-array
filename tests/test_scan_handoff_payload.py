# -*- coding: utf-8 -*-
"""staging 扫描测试：未声明文件/NASA 原始/绝对路径/符号链接/秘密/与代码快照重名。"""
import importlib.util
import json
import os
from pathlib import Path

import pytest


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "scan_handoff_payload",
        Path(__file__).resolve().parents[1] / "scripts" / "scan_handoff_payload.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_staging_passes(tmp_path):
    (tmp_path / "data" / "sim").mkdir(parents=True)
    (tmp_path / "data" / "sim" / "a.h5").write_bytes(b"x")
    violations = _load_module().scan(
        tmp_path, approved={"data/sim/a.h5"}, code_members=set()
    )
    assert violations == []


def test_detects_all_violation_classes(tmp_path):
    module = _load_module()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "undeclared.h5").write_bytes(b"x")          # 未声明（不在 approved）
    (tmp_path / "data" / "nasa_raw.dat").write_bytes(b"x")           # NASA 原始签名
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "README.md").write_text("dup")       # 与代码快照重名
    (tmp_path / "data" / "cfg.json").write_text('{"p": "D:\\legacy\\path"}')  # 绝对路径（单反斜杠）
    (tmp_path / "data" / "cred.txt").write_text("api_key = AKIA1234567890")   # 秘密信息

    violations = module.scan(
        tmp_path,
        approved={"data/nasa_raw.dat", "data/cfg.json", "data/cred.txt",
                  "checkpoints/README.md"},
        code_members={"checkpoints/README.md"},
    )
    joined = "\n".join(violations)
    assert "nasa_raw.dat" in joined          # NASA 原始命名签名
    assert "checkpoints/README.md" in joined  # 与 git archive 成员冲突
    assert "cfg.json" in joined               # 内嵌绝对路径
    assert "未在 manifest 声明" in joined       # undeclared.h5 未在 approved
    assert "疑似秘密信息" in joined and "cred.txt" in joined  # 秘密签名命中


def test_symlink_rejected(tmp_path):
    module = _load_module()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.h5").write_bytes(b"x")
    try:
        os.symlink(tmp_path / "data" / "a.h5", tmp_path / "data" / "link.h5")
    except OSError:
        pytest.skip("本环境无法创建符号链接")
    violations = module.scan(
        tmp_path, approved={"data/a.h5", "data/link.h5"}, code_members=set()
    )
    joined = "\n".join(violations)
    assert "符号链接禁止" in joined and "link.h5" in joined


# ---------------------------------------------------------------------------
# CLI：--repo-root 从组件仓三 manifest 派生 approved（消除自批准）
# ---------------------------------------------------------------------------

_MANIFEST_REL = {
    "data": "data/data_manifest.json",
    "checkpoints": "checkpoints/checkpoint_manifest.json",
    "results": "results/results_manifest.json",
}


def _write_manifest(repo: Path, slot: str, entries: list) -> None:
    target = repo / _MANIFEST_REL[slot]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"entries": entries}), encoding="utf-8")


def test_cli_derives_approved_from_repo_root_manifests(tmp_path, capsys):
    """--repo-root 时 approved 只含三 manifest 中 rc_payload=true 的条目。

    results manifest 条目 path 已含 reference/ 前缀，与 staging 布局
    handoff/payload/results/<path> 一致，槽位映射为 f"{slot}/{path}"。
    staging 中不在 approved 的文件必须如实报"未在 manifest 声明"。
    """
    module = _load_module()
    repo = tmp_path / "repo"
    _write_manifest(
        repo,
        "data",
        [
            {"path": "a.h5", "rc_payload": True},
            {"path": "local_only.h5", "rc_payload": False},
        ],
    )
    _write_manifest(
        repo,
        "results",
        [{"path": "reference/metrics_phased_array.json", "rc_payload": True}],
    )
    staging = tmp_path / "staging"
    (staging / "data").mkdir(parents=True)
    (staging / "data" / "a.h5").write_bytes(b"x")          # 已声明且 rc_payload=true
    (staging / "data" / "rogue.h5").write_bytes(b"x")      # manifest 外文件
    (staging / "data" / "local_only.h5").write_bytes(b"x")  # rc_payload=false 不属 approved
    (staging / "results" / "reference").mkdir(parents=True)
    (staging / "results" / "reference" / "metrics_phased_array.json").write_bytes(b"x")

    rc = module._main(["--staging", str(staging), "--repo-root", str(repo)])
    out = capsys.readouterr().out
    assert rc == 1, "manifest 外文件必须使扫描失败"
    assert "未在 manifest 声明" in out and "data/rogue.h5" in out
    assert "data/local_only.h5" in out, "rc_payload=false 条目不得被批准"
    assert "data/a.h5" not in out, "已声明条目不得被误报"
    assert "results/reference/metrics_phased_array.json" not in out


def test_cli_repo_root_clean_staging_passes(tmp_path, capsys):
    """staging 与 manifest 白名单完全一致时，--repo-root 扫描通过（退出码 0）。"""
    module = _load_module()
    repo = tmp_path / "repo"
    _write_manifest(repo, "data", [{"path": "a.h5", "rc_payload": True}])
    staging = tmp_path / "staging"
    (staging / "data").mkdir(parents=True)
    (staging / "data" / "a.h5").write_bytes(b"x")
    assert module._main(["--staging", str(staging), "--repo-root", str(repo)]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_without_repo_root_keeps_self_approved_behavior(tmp_path, capsys):
    """不带 --repo-root 时行为不变：staging 现存文件自批准（向后兼容）。"""
    module = _load_module()
    staging = tmp_path / "staging"
    (staging / "data").mkdir(parents=True)
    (staging / "data" / "rogue.h5").write_bytes(b"x")
    assert module._main(["--staging", str(staging)]) == 0
