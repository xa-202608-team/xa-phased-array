# -*- coding: utf-8 -*-
"""handoff 契约守护：artifact-map 必填键与白名单 staging 映射（contract v1.1.0）。"""
from pathlib import Path

import yaml

EXPECTED_LOCALS = [
    "handoff/payload/data",
    "handoff/payload/checkpoints",
    "handoff/payload/results/reference",
]


def _load_map() -> dict:
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load(
        (root / "handoff" / "artifact-map.yaml").read_text(encoding="utf-8")
    )


def test_artifact_map_required_keys() -> None:
    data = _load_map()
    assert data["component"] == "phased_array"
    assert data["contract_version"] == "component-contract-v1.1.0"
    seeds = data["random_seeds"]
    assert isinstance(seeds, list) and seeds
    assert all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds)
    commands = data["commands"]
    assert isinstance(commands, list) and commands
    assert all(isinstance(c, str) and c for c in commands)
    for mapping in data["mappings"]:
        payload = mapping.get("payload")
        assert isinstance(payload, str) and payload, (
            f"mapping {mapping.get('local')!r} 必须有非空 payload 键（RC 交付目标路径）"
        )


def test_artifact_map_maps_whitelist_staging_only() -> None:
    locals_ = [m["local"] for m in _load_map()["mappings"]]
    assert locals_ == EXPECTED_LOCALS, "artifact-map 必须只映射 handoff/payload 白名单"
    for mapping in _load_map()["mappings"]:
        assert mapping["role"] in {"dataset", "checkpoint", "reference_result"}


def test_handoff_md_declares_v1_1_0() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "handoff" / "HANDOFF.md").read_text(encoding="utf-8")
    assert "component-contract-v1.1.0" in text
    assert "component-contract-v1.0.0" not in text, (
        "HANDOFF.md 不得残留旧契约版本 component-contract-v1.0.0（已升级 v1.1.0）"
    )
