# -*- coding: utf-8 -*-
"""staging 构建脚本测试：白名单复制 + SHA256 复核。"""
import hashlib
import importlib.util
import json
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


def test_stage_results_slot_via_manifest(tmp_path):
    """results 槽位经 results_manifest.json 白名单装配（dry-run 评审补机制）。"""
    module = _load_module()
    approved = tmp_path / "results" / "reference" / "metrics_phased_array.json"
    approved.parent.mkdir(parents=True)
    approved.write_bytes(b"metrics")
    local_only = tmp_path / "results" / "reference" / "local_scratch.jsonl"
    local_only.write_bytes(b"scratch")
    (tmp_path / "results" / "results_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.1.0",
                "component": "phased_array",
                "entries": [
                    {
                        "path": "reference/metrics_phased_array.json",
                        "sha256": _sha256(approved),
                        "rc_payload": True,
                    },
                    {
                        "path": "reference/local_scratch.jsonl",
                        "sha256": _sha256(local_only),
                        "rc_payload": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    slots = module._load_manifest_slots(tmp_path)
    assert "results" in slots, "_load_manifest_slots 必须消费 results/results_manifest.json"
    staged = module.stage(tmp_path, slots)
    assert staged == ["handoff/payload/results/reference/metrics_phased_array.json"]
    assert (
        tmp_path / "handoff/payload/results/reference/metrics_phased_array.json"
    ).read_bytes() == b"metrics"
    assert not (
        tmp_path / "handoff/payload/results/reference/local_scratch.jsonl"
    ).exists(), "rc_payload=false 的 results 条目不得进入 staging"


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


def test_stage_guard_source_rejects_windows_drive_letters():
    """守卫源码须含盘符形态拒绝（跨平台一致）。

    Path("C:/x.h5").is_absolute() 在 POSIX 上为 False，仅靠 is_absolute()
    会让盘符路径在 Linux 容器内（Dockerfile 构建跑全量 pytest）漏拒；
    守卫必须显式含 ``re.match(r"^[A-Za-z]:", rel)`` 形态的盘符正则。
    文本守护断言，风格同 test_entrypoint_layout.py。
    """
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "stage_handoff_payload.py"
    ).read_text(encoding="utf-8")
    assert 're.match(r"^[A-Za-z]:", rel)' in source, (
        "路径守卫须含盘符正则 re.match(r\"^[A-Za-z]:\", rel)（POSIX 上 "
        "is_absolute() 对 'C:/x.h5' 为 False，会漏拒）"
    )
