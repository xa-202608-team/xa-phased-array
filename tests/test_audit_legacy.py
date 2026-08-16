# -*- coding: utf-8 -*-
"""旧交付审计脚本测试：只读枚举 + SHA256 + 分类。"""
import hashlib
import importlib.util
from pathlib import Path


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_legacy_delivery",
        Path(__file__).resolve().parents[1] / "scripts" / "audit_legacy_delivery.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_categorizes_and_hashes(tmp_path):
    (tmp_path / "data" / "sim").mkdir(parents=True)
    (tmp_path / "data" / "sim" / "a.h5").write_bytes(b"x")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "m.pt").write_bytes(b"y")
    (tmp_path / "checkpoints" / "metrics.json").write_bytes(b"z")

    entries = _load_module().audit(tmp_path)

    by_path = {e["path"]: e for e in entries}
    assert by_path["data/sim/a.h5"]["category"] == "data"
    assert by_path["checkpoints/m.pt"]["category"] == "checkpoint"
    assert by_path["checkpoints/metrics.json"]["category"] == "metrics"
    assert by_path["checkpoints/m.pt"]["sha256"] == hashlib.sha256(b"y").hexdigest()
    assert by_path["data/sim/a.h5"]["size"] == 1
