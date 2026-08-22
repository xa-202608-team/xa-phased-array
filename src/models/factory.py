# -*- coding: utf-8 -*-
"""模型唯一构造函数 (F1-A 收口)。

运行三处共用, 消除训练/导出/推理间的架构参数漂移:
  - src/experiments/run_groups.py::_build_model (训练/评估)
  - src/experiments/run_service_eval.py::_build_model (T8 服务层评估)
  - bundle 导出器 / 推理入口 (后续 F1 批 3)

GRU hidden/layers/dropout 显式从 config model.gru 读 (F1-A: 不再吃 adapter 隐式默认 64/2)。
任何一侧改动必须同步本模块 + tests/test_models_factory.py 契约测试。
"""
from __future__ import annotations

from src.transfer.adapter import TransferModel  # noqa: E402


def build_transfer_model(cfg, n_features: int | None = None, n_target: int | None = None,
                         encoder_type: str | None = None, device: str = "cpu"):
    """由 config 构建 TransferModel (训练/导出/推理唯一入口)。

    cfg: 组件 config (含 model/transfer 段)
    n_features: 源域特征维 (默认 cfg.model 侧推理用 target 维, 调用方按需传入显式值)
    n_target: 目标域特征维 (默认 cfg.transfer 侧, 调用方显式传入最稳)
    encoder_type: 覆盖 config model.encoder (None 用 config 默认)
    """
    mc = cfg["model"]
    tc = cfg["transfer"]
    enc = encoder_type or mc["encoder"]
    nf = n_features if n_features is not None else int(mc.get("n_features", tc.get("n_target", 12)))
    nt = n_target if n_target is not None else int(tc.get("n_target", mc.get("n_features", 12)))
    gru_cfg = mc.get("gru", {})
    return TransferModel(
        encoder_type=enc, n_features=nf, n_target=nt,
        channels=mc["tcn"]["channels"], kernel_size=mc["tcn"]["kernel_size"],
        num_blocks=mc["tcn"]["num_blocks"], dropout=mc["tcn"]["dropout"],
        latent_dim=mc["latent_dim"], adapter_hidden=tc["adapter_hidden"],
        gru_hidden=int(gru_cfg.get("hidden", 64)),
        gru_layers=int(gru_cfg.get("num_layers", 2)),
        gru_dropout=gru_cfg.get("dropout", mc["tcn"]["dropout"])).to(device)