# -*- coding: utf-8 -*-
"""staging 构建脚本测试：白名单复制 + SHA256 复核。"""
import hashlib
import importlib.util
from pathlib import Path


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "stage_handoff_payload",
        Path(__file__).resolve().parents[1] / "scripts" / "stage_handoff_payload.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_copies_approved_and_rejects_corrupt(tmp_path):
    module = _load_module()
    src = tmp_path / "data" / "a.h5"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"payload")
    unapproved = tmp_path / "data" / "raw_nasa.mat"
    unapproved.write_bytes(b"secret")

    staged = module.stage(
        tmp_path,
        slots={
            "data": [
                {
                    "path": "a.h5",
                    "sha256": _sha256(src),
                    "rc_payload": True,
                },
                {"path": "raw_nasa.mat", "sha256": _sha256(unapproved), "rc_payload": False},
            ]
        },
    )

    assert staged == ["handoff/payload/data/a.h5"]
    assert (tmp_path / "handoff/payload/data/a.h5").read_bytes() == b"payload"
    assert not (tmp_path / "handoff/payload/data/raw_nasa.mat").exists()


def test_stage_fails_on_sha256_mismatch(tmp_path):
    module = _load_module()
    src = tmp_path / "checkpoints" / "m.pt"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"weights")
    try:
        module.stage(
            tmp_path,
            slots={"checkpoints": [{"path": "m.pt", "sha256": "0" * 64, "rc_payload": True}]},
        )
    except ValueError as exc:
        assert "m.pt" in str(exc)
    else:
        raise AssertionError("SHA256 不符必须抛 ValueError")


def test_stage_rejects_path_escape_and_absolute(tmp_path):
    module = _load_module()
    for bad in ("../outside.h5", "C:/x.h5"):
        try:
            module.stage(
                tmp_path,
                slots={"data": [{"path": bad, "sha256": "0" * 64, "rc_payload": True}]},
            )
        except ValueError as exc:
            assert "仓库内相对路径" in str(exc), f"{bad}: 缺少 containment 守卫消息: {exc}"
            assert bad in str(exc)
        else:
            raise AssertionError(f"越权路径必须抛 ValueError: {bad}")
