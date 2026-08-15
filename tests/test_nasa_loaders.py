from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.preprocess.nasa_mat_common import (   # noqa: E402
    extract_on_state_ratio,
    extract_on_state_voltage,
    persistent_first_crossing,
)


def test_on_state_extractors() -> None:
    n = 1000
    gate = np.zeros(n)
    gate[300:700] = 15.0
    current = np.zeros(n)
    current[300:700] = 10.0
    voltage_mosfet = np.full(n, 7.0)
    voltage_mosfet[300:700] = 1.8
    voltage_igbt = np.full(n, 7.0)
    voltage_igbt[300:700] = 2.4

    rds = extract_on_state_ratio(gate, voltage_mosfet, current)
    vce = extract_on_state_voltage(gate, voltage_igbt, current)
    assert abs(rds - 0.18) < 1e-10
    assert abs(vce - 2.4) < 1e-10


def test_persistent_crossing() -> None:
    values = np.array([0.01, 0.06, 0.01, 0.06, 0.07, 0.08])
    assert persistent_first_crossing(values, 0.05, 3) == 3
    assert persistent_first_crossing(values, 0.10, 2) is None


if __name__ == "__main__":
    test_on_state_extractors()
    test_persistent_crossing()
    print("ok")
