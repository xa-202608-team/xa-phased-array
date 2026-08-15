# -*- coding: utf-8 -*-
"""三级物理链冻结文档与复现入口测试（TDD，先于实现）。

PHYSICS_CHAIN.yaml 是 GaN T/R→阵列→链路三级退化真值链的冻结描述，
必须与 configs/phased_array.yaml 及 src/sim 实际生成函数一致。
"""
from pathlib import Path

import yaml


def test_physics_chain_freezes_three_levels() -> None:
    path = Path(__file__).resolve().parents[1] / "docs/simulation/PHYSICS_CHAIN.yaml"
    chain = yaml.safe_load(path.read_text("utf-8"))
    assert chain["seed"] == 42
    assert list(chain["levels"]) == ["gan_tr", "array", "link"]
    for level in chain["levels"].values():
        assert level["inputs"]
        assert level["outputs"]
        assert level["truth_fields"]
        assert level["prediction_time_observable"] is not None


def test_physics_chain_documents_units_and_config_keys() -> None:
    path = Path(__file__).resolve().parents[1] / "docs/simulation/PHYSICS_CHAIN.yaml"
    chain = yaml.safe_load(path.read_text("utf-8"))
    for name, level in chain["levels"].items():
        assert level.get("units"), f"level {name} 缺单位说明"
        assert level.get("config_keys"), f"level {name} 缺配置键"
        assert level.get("random_sources") is not None, f"level {name} 缺随机源"
        assert level.get("generation_functions"), f"level {name} 缺生成函数路径"
        # 生成函数路径必须指向仓库内真实存在的模块文件
        for func_path in level["generation_functions"]:
            module = func_path.split("::")[0]
            assert (Path(__file__).resolve().parents[1] / module).is_file(), (
                f"level {name} 生成函数模块不存在: {module}"
            )


def test_physics_chain_config_key_matches_live_config() -> None:
    """PHYSICS_CHAIN 引用的配置键必须存在于 configs/phased_array.yaml。"""
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/phased_array.yaml").read_text("utf-8"))

    def _has_path(d: dict, dotted: str) -> bool:
        node: object = d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return True

    chain = yaml.safe_load((root / "docs/simulation/PHYSICS_CHAIN.yaml").read_text("utf-8"))
    for name, level in chain["levels"].items():
        for key in level["config_keys"]:
            assert _has_path(cfg, key), f"level {name} 引用的配置键不在 config 中: {key}"


def test_simulation_reproduce_doc_exists_with_frozen_commands() -> None:
    root = Path(__file__).resolve().parents[1]
    doc = root / "docs/SIMULATION_REPRODUCE.md"
    assert doc.is_file(), "缺少 docs/SIMULATION_REPRODUCE.md"
    text = doc.read_text("utf-8")
    assert "seed" in text and "42" in text
    assert "generate_simulation.py" in text
    assert "reproduce_judge" in text and "reproduce_full" in text


def test_unified_entry_scripts_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "scripts/prepare_source_data.py",
        "scripts/generate_simulation.py",
        "scripts/predict_telemetry.py",
        "scripts/reproduce_judge.py",
        "scripts/reproduce_judge.ps1",
        "scripts/reproduce_judge.sh",
        "scripts/reproduce_full.py",
        "scripts/reproduce_full.ps1",
        "scripts/reproduce_full.sh",
    ):
        assert (root / rel).is_file(), f"缺少统一入口脚本: {rel}"
