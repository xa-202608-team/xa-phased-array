"""无可学习参数的 16 子阵→16×16 阵元确定性阵列孪生。

输入仅为模型预测的子阵通道 `[gain_dB, phase_deg, Pout_dBm, PAE]` 与显式
阵列元数据；score-only 评分真值不属于本模块接口，不能作为输入。
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _subarray_ids(grid: tuple[int, int], block: int) -> np.ndarray:
    nx, ny = grid
    ix, iy = np.arange(nx), np.arange(ny)
    gx, gy = np.meshgrid(ix, iy, indexing="xy")
    return (gx // block + (gy // block) * (ny // block)).ravel().astype(int)


def _metadata(metadata: Mapping[str, object]) -> tuple[float, float, tuple[int, int], float, int]:
    required = {"scan_az_deg", "margin0_dB", "array_grid", "element_spacing_lambda", "subarray_block"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"阵列孪生 metadata 缺少: {', '.join(sorted(missing))}")
    grid_raw = tuple(int(value) for value in metadata["array_grid"])
    if len(grid_raw) != 2 or any(value <= 0 for value in grid_raw):
        raise ValueError("array_grid 必须为两个正整数")
    scan, margin = float(metadata["scan_az_deg"]), float(metadata["margin0_dB"])
    spacing, block = float(metadata["element_spacing_lambda"]), int(metadata["subarray_block"])
    if not np.isfinite([scan, margin, spacing]).all() or spacing <= 0 or block <= 0:
        raise ValueError("阵列孪生 metadata 含非法数值")
    if grid_raw[0] % block or grid_raw[1] % block:
        raise ValueError("array_grid 必须可被 subarray_block 整除")
    return scan, margin, grid_raw, spacing, block


def evaluate_subarray_array(predicted_channels: np.ndarray, metadata: Mapping[str, object]) -> dict[str, np.ndarray]:
    """从预测子阵通道确定性计算 G/EIRP、SLL、指向误差和链路余量。

    不含任何可学习参数或用评分标签拟合的校准项。幅度由相对首时刻的 gain/Pout
    等权 dB 变化确定，phase 为相对首时刻的子阵相位误差；PAE 不影响辐射方向图。
    """
    scan_az_deg, margin0_dB, grid, spacing, block = _metadata(metadata)
    channels = np.asarray(predicted_channels, dtype=float)
    if channels.ndim != 3 or channels.shape[-1] != 4:
        raise ValueError("predicted_channels 必须为 (T, N_subarray, 4)")
    if not np.isfinite(channels).all():
        raise ValueError("predicted_channels 含非有限值")
    node_ids = _subarray_ids(grid, block)
    n_nodes = int(node_ids.max()) + 1
    if channels.shape[1] != n_nodes:
        raise ValueError(f"metadata 对应 {n_nodes} 个子阵，但输入为 {channels.shape[1]} 个")
    n_time, n_elements = channels.shape[0], len(node_ids)
    pos_x = (np.tile(np.arange(grid[0]), grid[1]) - (grid[0] - 1) / 2.0) * spacing
    u0 = np.sin(np.deg2rad(scan_az_deg))
    theta_grid = np.linspace(-75.0, 75.0, 601)
    u_grid = np.sin(np.deg2rad(theta_grid))
    steering = np.exp(1j * 2.0 * np.pi * np.outer(u_grid - u0, pos_x))

    relative_gain = channels[:, :, 0] - channels[0:1, :, 0]
    relative_pout = channels[:, :, 2] - channels[0:1, :, 2]
    amplitude_sa = np.power(10.0, 0.5 * (relative_gain + relative_pout) / 20.0)
    phase_sa_rad = np.deg2rad(channels[:, :, 1] - channels[0:1, :, 1])
    amplitude = amplitude_sa[:, node_ids]
    phase = phase_sa_rad[:, node_ids]
    pattern = np.abs((amplitude * np.exp(1j * phase)) @ steering.T) ** 2
    peak_index = np.argmax(pattern, axis=1)
    peak_power = pattern[np.arange(n_time), peak_index]
    theta_peak = theta_grid[peak_index]
    gain = 10.0 * np.log10(peak_power / (n_elements * n_elements) + 1e-30)
    eirp_norm = peak_power / max(float(peak_power[0]), 1e-30)
    du = 1.0 / (grid[0] * spacing)
    main = (u_grid >= u0 - du) & (u_grid <= u0 + du)
    sidelobes = pattern.copy()
    sidelobes[:, main] = 0.0
    sll = 10.0 * np.log10(np.max(sidelobes, axis=1) / (peak_power + 1e-30) + 1e-30)
    theta_err = theta_peak - scan_az_deg
    margin = margin0_dB - np.clip(gain[0] - gain, 0.0, 40.0)
    return {
        "G_array_dB": gain.astype(np.float32), "EIRP_norm": eirp_norm.astype(np.float32),
        "SLL_dB": np.clip(sll, -50.0, 0.0).astype(np.float32),
        "theta_err_deg": theta_err.astype(np.float32), "M_link_dB": margin.astype(np.float32),
    }
