# -*- coding: utf-8 -*-
"""生成 component.predict 契约测试 fixture（确定性，无随机源）。

产物：
- telemetry_long.csv            正例：三阵列个体 96 窗 array_gain_db (dB) 线性退化
- telemetry_leak_true_rul.csv   负例：telemetry_name=true_rul（标签泄漏，必须拒绝）
- telemetry_leak_damage_truth.csv 负例：telemetry_name=damage_truth（真值泄漏）

运行：python tests/fixtures/generate_fixtures.py
（重跑逐字节一致；dataset.yaml 中 sources[0].sha256 登记本脚本产物 telemetry_long.csv 的 SHA256。）
"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

HEADER = ["timestamp", "component_id", "component_type", "condition_id",
          "telemetry_name", "value", "unit"]
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

# 个体 -> (斜率 dB/窗, 截距 dB)；6h/窗 × 96 窗 = 24 天
SERIES = {
    "pa-array-001": (-0.0125, 0.02),
    "pa-array-002": (-0.0200, -0.05),
    "pa-array-003": (-0.0080, 0.01),
}


def _rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for cid, (slope, intercept) in SERIES.items():
        for i in range(96):
            value = intercept + slope * i + 0.005 * ((i % 7) - 3) / 3.0
            ts = T0 + timedelta(hours=6 * i)
            rows.append([ts.strftime("%Y-%m-%dT%H:%M:%SZ"), cid, "phased_array",
                         "leo_nominal", "array_gain_db", f"{value:.6f}", "dB"])
    return rows


def _write(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)


def main() -> None:
    out = Path(__file__).resolve().parent
    rows = _rows()
    _write(out / "telemetry_long.csv", rows)

    leak_rul = [[r[0], r[1], r[2], r[3], "true_rul", f"{(95 - i) * 6.0:.1f}", "hours"]
                for i, r in enumerate(rows[:8])]
    _write(out / "telemetry_leak_true_rul.csv", leak_rul)

    leak_damage = [[r[0], r[1], r[2], r[3], "damage_truth", f"{i / 95:.6f}", "ratio"]
                   for i, r in enumerate(rows[:8])]
    _write(out / "telemetry_leak_damage_truth.csv", leak_damage)

    digest = hashlib.sha256((out / "telemetry_long.csv").read_bytes()).hexdigest()
    print(f"telemetry_long.csv sha256: {digest}")


if __name__ == "__main__":
    main()
