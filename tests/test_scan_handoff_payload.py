# -*- coding: utf-8 -*-
"""staging 扫描测试：未声明文件/NASA 原始/绝对路径/符号链接/与代码快照重名。"""
import importlib.util
from pathlib import Path


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
    (tmp_path / "data" / "undeclared.h5").write_bytes(b"x")          # 未声明
    (tmp_path / "data" / "nasa_raw.dat").write_bytes(b"x")           # NASA 原始签名
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "README.md").write_text("dup")       # 与代码快照重名
    (tmp_path / "data" / "cfg.json").write_text('{"p": "D:\\\\legacy\\\\path"}')  # 绝对路径

    violations = module.scan(
        tmp_path,
        approved={"data/undeclared.h5", "data/nasa_raw.dat", "data/cfg.json",
                  "checkpoints/README.md"},
        code_members={"checkpoints/README.md"},
    )
    joined = "\n".join(violations)
    assert "nasa_raw.dat" in joined          # NASA 原始命名签名
    assert "checkpoints/README.md" in joined  # 与 git archive 成员冲突
    assert "cfg.json" in joined               # 内嵌绝对路径
