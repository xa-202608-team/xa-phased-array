# -*- coding: utf-8 -*-
"""component.predict 单指标遥测入口契约测试（TDD，先于实现）。

覆盖：
- 五个 v1.1 Schema（+ 三个 v1.0）本地快照与契约 Tag component-contract-v1.1.0 逐字节一致（SHA256）；
- fixture 元数据符合 dataset-metadata.schema.json；
- 只喂 array_gain_db 单指标即可产出未来趋势（prediction.schema.json 校验通过）；
- telemetry 中出现 true_rul / damage_truth（值级或列级）时拒绝（标签隔离铁律）；
- causal_linear_forecast / estimate_rul 的方向守卫与数值正确性。
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import jsonschema
import pytest
import yaml

from component.io import FORBIDDEN_INFERENCE_COLUMNS, load_request
from component.predictor import build_predictor, causal_linear_forecast, estimate_rul

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"
SCHEMAS = ROOT / "schemas"

# 契约仓 component-contract-v1.1.0 Tag 下各 schema 的 SHA256
# （由 `git show component-contract-v1.1.0:schemas/<name> | sha256sum` 固化；
#   若本机存在 ../xa-component-contract 仓库，test_schema_snapshot_matches_contract_repo
#   会再用 git show 动态复核，双保险。）
CONTRACT_TAG_SHA256 = {
    "prediction.schema.json": "aa91220dbc4b07c93bd6c0ca4ccb4f77888144628e395cf27b4bca1d0aa176c1",
    "telemetry.schema.json": "a2721506426230c823d225c802bcedad9e886f23749c891ce51598696e6ac596",
    "labels.schema.json": "08ffec023d28dcba88ed0186fe125dcb1d0eaca45256e77eb9d45b35674e62e0",
    "dataset-metadata.schema.json": "3f85a5cf6626833d1c3c710b7605a0ba5f636965c590b854bd88079d15950cda",
    "handoff-manifest.schema.json": "4e31282ab963eee7697980060cdd238a847e3f1d61e38e42a2d1f48c776d1fb2",
    # v1.0 三个（judge/full 复现输出校验所用，同 Tag 导出）
    "manifest.schema.json": "acabfa37f0af391129e35bab166f82f6648f0b776be0249d03a0c8631f2c4cad",
    "metrics.schema.json": "16b54a354a2d0e9bbcbbb83123acbf0b4e966610585056ff7fb004ab7c39c345",
    "expected_metrics.schema.json": "8f04b17287316009f1da7a13b59de950f497d15ac4a0122867cb5d1c1872e9a0",
}

CONTRACT_REPO = ROOT.parent / "xa-component-contract"
CONTRACT_TAG = "component-contract-v1.1.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_schema_snapshots_match_contract_tag() -> None:
    """本地 schemas/ 快照必须与契约 Tag 逐字节一致。"""
    for name, expected in CONTRACT_TAG_SHA256.items():
        local = SCHEMAS / name
        assert local.is_file(), f"缺少契约 schema 快照: {local}"
        assert _sha256(local) == expected, (
            f"{name} 与契约 Tag {CONTRACT_TAG} 不一致: "
            f"local={_sha256(local)} expected={expected}"
        )


def test_schema_snapshot_matches_contract_repo() -> None:
    """若本机存在契约仓库，则动态用 git show 复核 Tag 内容（CI 无仓库时跳过）。"""
    if not (CONTRACT_REPO / ".git").exists():
        pytest.skip("契约仓库 ../xa-component-contract 不在本机，跳过动态复核")
    for name, expected in CONTRACT_TAG_SHA256.items():
        blob = subprocess.check_output(
            ["git", "-C", str(CONTRACT_REPO), "show", f"{CONTRACT_TAG}:schemas/{name}"]
        )
        assert hashlib.sha256(blob).hexdigest() == expected, f"{name} 契约 Tag SHA256 漂移"
        assert hashlib.sha256(blob).hexdigest() == _sha256(SCHEMAS / name), (
            f"{name} 本地快照与契约 Tag 不一致"
        )


def test_fixture_metadata_conforms_schema() -> None:
    schema = yaml.safe_load((SCHEMAS / "dataset-metadata.schema.json").read_text("utf-8"))
    metadata = yaml.safe_load((FIXTURES / "dataset.yaml").read_text("utf-8"))
    jsonschema.validate(metadata, schema)


def test_forbidden_columns_cover_label_leak() -> None:
    assert {"true_rul", "damage_truth", "rul", "label"} <= FORBIDDEN_INFERENCE_COLUMNS


def test_load_request_selects_primary_telemetry() -> None:
    request = load_request(FIXTURES / "telemetry_long.csv",
                           FIXTURES / "dataset.yaml", "array_gain_db")
    assert request.telemetry_name == "array_gain_db"
    assert set(request.telemetry["telemetry_name"]) == {"array_gain_db"}
    # 三个阵列个体，各 96 窗
    counts = request.telemetry.groupby("component_id").size().to_dict()
    assert counts == {"pa-array-001": 96, "pa-array-002": 96, "pa-array-003": 96}


@pytest.mark.parametrize("leak_file", [
    "telemetry_leak_true_rul.csv",
    "telemetry_leak_damage_truth.csv",
])
def test_predict_rejects_label_leak_in_telemetry(leak_file: str) -> None:
    """遥测表混入 true_rul / damage_truth 必须拒绝，杜绝标签泄漏。"""
    with pytest.raises(ValueError):
        load_request(FIXTURES / leak_file, FIXTURES / "dataset.yaml", "array_gain_db")


def test_predict_generates_forecast_conforming_schema(monkeypatch) -> None:
    """正例：只喂 array_gain_db 即可产出未来趋势且通过 prediction.schema.json。

    显式注入 XA_GIT_COMMIT：predictor 的 git_commit 解析优先 git 仓库，
    无 .git 的容器/导出环境回退该环境变量（与 judge 运行时同机制）。
    """
    monkeypatch.setenv("XA_GIT_COMMIT", "0" * 39 + "1")
    request = load_request(FIXTURES / "telemetry_long.csv",
                           FIXTURES / "dataset.yaml", "array_gain_db")
    prediction = build_predictor().predict(request)

    schema = yaml.safe_load((SCHEMAS / "prediction.schema.json").read_text("utf-8"))
    jsonschema.validate(prediction, schema)

    assert prediction["component"] == "phased_array"
    assert prediction["telemetry_name"] == "array_gain_db"
    assert prediction["status"] == "PREDICTION_OK"
    assert len(prediction["forecasts"]) == 3 * 12        # 3 个体 × forecast_horizon=12
    units = {f["unit"] for f in prediction["forecasts"]}
    assert units == {"dB"}
    # decreasing 主遥测的因果外推必须整体下行（线性退化序列）
    for cid in ("pa-array-001", "pa-array-002", "pa-array-003"):
        steps = [f for f in prediction["forecasts"] if f["component_id"] == cid]
        assert [f["horizon_step"] for f in steps] == list(range(1, 13))
        assert steps[-1]["predicted_value"] < steps[0]["predicted_value"]
        rul = steps[0]["rul"]
        assert rul is None or rul > 0


def test_causal_linear_forecast_recovers_slope() -> None:
    import numpy as np

    values = np.array([0.5 - 0.01 * i for i in range(50)], dtype=float)
    forecast, slope = causal_linear_forecast(values, 5)
    assert forecast.shape == (5,)
    assert slope == pytest.approx(-0.01, abs=1e-9)
    assert forecast[0] == pytest.approx(0.5 - 0.01 * 50, abs=1e-9)
    assert forecast[4] == pytest.approx(0.5 - 0.01 * 54, abs=1e-9)


def test_estimate_rul_direction_guard() -> None:
    # decreasing：负斜率朝阈值运动 -> 有 RUL；正斜率背离 -> None
    assert estimate_rul(0.5, -0.01, 0.2, "decreasing") == pytest.approx(30.0)
    assert estimate_rul(0.5, 0.01, 0.2, "decreasing") is None
    # increasing：正斜率朝阈值运动 -> 有 RUL；负斜率背离 -> None
    assert estimate_rul(0.2, 0.01, 0.5, "increasing") == pytest.approx(30.0)
    assert estimate_rul(0.2, -0.01, 0.5, "increasing") is None
    # 已越过阈值：RUL=0 语义由调用方处理，斜率为零一律 None
    assert estimate_rul(0.1, 0.0, 0.2, "decreasing") is None
    assert estimate_rul(0.1, 0.0, 0.5, "increasing") is None
