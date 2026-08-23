# -*- coding: utf-8 -*-
"""`python -m component.predict_gru` — 通道级 GRU RUL 推理入口（F1 批3）。

与 component.predict（因果单遥测基线）的关系：
- predict = 无模型因果线性外推（趋势基线，不依赖训练权重）；
- predict_gru = 消费 run_groups `--export-inference-dir` 导出的训练侧 bundle
  （val-best 主模型，正式口径 ch_target_only_gru seed42），对 channel_features.h5
  的 x_ch 特征做通道级 RUL 预测。

推理纪律：
- 输入只读 x_ch（src/transfer/channel_inference.py 无标签加载器），绝不触碰
  hi/rul 标签字段（契约 §9 标签隔离）；
- 模型经 src/models/factory.build_transfer_model 由 bundle.json 参数重建 +
  model.pt 严格加载（strict），架构与训练侧零漂移；
- rul_norm → rul_windows（×H=rul_scale_windows）→ rul_days（×sample_period_s/86400），
  相对寿命比例仍须除该通道自身真实 EOL（F1-A §9 例外口径）；
- 输出 rul_prediction.json 出厂前经 schemas/rul-prediction.schema.json 校验。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from component.predictor import current_git_commit
from src.models.factory import build_transfer_model
from src.sim.build_channel_hi import CANONICAL_COLS
from src.transfer.channel_dataset import CHANNEL_LABEL_SCHEMA_V2
from src.transfer.channel_inference import (
    ChannelInferenceDataset, eligible_channel_keys, load_channel_inference,
    normalize_with_stats)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0.0"
CONTRACT_VERSION = "component-contract-v1.1.0"
COMPONENT = "phased_array"


def limit_keep_mask(channel_keys: np.ndarray, limit_channels: int, L: int) -> np.ndarray:
    """--limit-channels 的行掩码: 只从 T>=L 的合格通道中按序取前 N 个。

    旧实现取 sorted(unique)[:N] 不筛长度, 在含早失效短通道 (如 --fast 数据) 上
    会取到全池 T<L 而无完整窗口 (2026-08-23 F6 回归)。
    """
    keys = eligible_channel_keys(channel_keys, L)
    return np.isin(channel_keys, keys[:int(limit_channels)])
METHOD = "gru_channel_rul"
BUNDLE_SCHEMA = "channel-inference-bundle-v1"
SECONDS_PER_DAY = 86400.0


def load_bundle(bundle_dir: Path) -> dict:
    """读 bundle.json 并做自洽校验（bundle_schema / v2 尺度 / 统计量维数一致）。"""
    path = Path(bundle_dir) / "bundle.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺 bundle.json: {path} (先 run_groups --export-inference-dir)")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("bundle_schema") != BUNDLE_SCHEMA:
        raise ValueError(f"bundle_schema {bundle.get('bundle_schema')!r} != {BUNDLE_SCHEMA!r}")
    if bundle["rul"]["channel_label_schema"] != CHANNEL_LABEL_SCHEMA_V2:
        raise ValueError("bundle 仅支持 channel_label_v2 尺度元数据")
    n = bundle["n_target"]
    if len(bundle["normalizer"]["mean"]) != n or len(bundle["normalizer"]["std"]) != n:
        raise ValueError(f"normalizer 维数与 n_target={n} 不一致 (bundle 损坏)")
    if not (Path(bundle_dir) / "model.pt").is_file():
        raise FileNotFoundError(f"缺 model.pt: {bundle_dir}")
    return bundle


def rebuild_model(bundle: dict, bundle_dir: Path, device: str = "cpu"):
    """按 bundle 参数经唯一 factory 重建 TransferModel 并严格加载 val-best 权重。"""
    b = bundle["model"]
    cfg_like = {
        "model": {"encoder": bundle["encoder"], "tcn": b["tcn"],
                  "latent_dim": b["latent_dim"], "gru": b["gru"],
                  "n_features": bundle["n_features"]},
        "transfer": {"adapter_hidden": b["adapter_hidden"],
                     "n_target": bundle["n_target"]},
    }
    model = build_transfer_model(
        cfg_like, n_features=bundle["n_features"], n_target=bundle["n_target"],
        encoder_type=bundle["encoder"], device=device)
    state = torch.load(Path(bundle_dir) / "model.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state)          # strict: 键不匹配即 fail (架构漂移拒载)
    model.eval()
    return model


def predict(features_path: Path, bundle_dir: Path, stride: int = 50,
            batch_size: int = 256, limit_channels: int | None = None,
            device: str = "cpu") -> dict:
    """整链推理: 无标签加载 → 校验 → 归一 → 窗口 → 模型 → 物理还原 → 契约文档。"""
    bundle = load_bundle(bundle_dir)
    model = rebuild_model(bundle, bundle_dir, device=device)

    x, ck, tid, sid, meta = load_channel_inference(features_path)
    # 特征校验: bundle 侧期望列 = CANONICAL_COLS − drop; h5 元数据尺度与 bundle 一致
    expected = bundle["feature_names"]
    dropped = [c for c in CANONICAL_COLS if c not in expected]
    if dropped:
        x = x[:, [i for i, c in enumerate(CANONICAL_COLS) if c in expected]]
    if x.shape[1] != bundle["n_target"]:
        raise ValueError(
            f"特征维不一致: 遥测 {x.shape[1]} != bundle n_target {bundle['n_target']}")
    H_bundle = float(bundle["rul"]["rul_scale_windows"])
    if (meta["channel_label_schema"] != CHANNEL_LABEL_SCHEMA_V2
            or abs(float(meta["rul_scale_windows"]) - H_bundle) > 1e-6
            or abs(float(meta["sample_period_s"])
                   - float(bundle["rul"]["sample_period_s"])) > 1e-6):
        raise ValueError(
            f"遥测 h5 尺度元数据 {meta} 与 bundle {bundle['rul']} 不一致 (数据/bundle 不配套)")

    if limit_channels is not None:
        keep = limit_keep_mask(ck, limit_channels, int(bundle["input_len_L"]))
        x, ck, tid, sid = x[keep], ck[keep], tid[keep], sid[keep]

    x = normalize_with_stats(x, np.array(bundle["normalizer"]["mean"]),
                             np.array(bundle["normalizer"]["std"]))
    ds = ChannelInferenceDataset(x, ck, L=int(bundle["input_len_L"]), stride=stride,
                                 expected_features=int(bundle["n_target"]))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    H = H_bundle
    sp_s = float(bundle["rul"]["sample_period_s"])
    n_sub_per_traj = 16
    preds = []
    with torch.no_grad():
        for xb, ckb, endb in loader:
            xb = xb.to(device)
            _, rul, _ = model(xb)                       # (B,) 归一口径 (÷H)
            rul = torch.clamp(rul, min=0.0).cpu().numpy().astype(np.float64)
            ends = endb.numpy().astype(np.int64)
            cks = ckb.numpy().astype(np.int64)
            for r_n, k, e in zip(rul, cks, ends):
                r_w = float(r_n) * H
                preds.append({
                    "channel_key": int(k),
                    "traj_id": int(k) // n_sub_per_traj,
                    "sub_id": int(k) % n_sub_per_traj,
                    "window_end_index": int(e),
                    "rul_norm": round(float(r_n), 6),
                    "rul_windows": round(r_w, 3),
                    "rul_days": round(r_w * sp_s / SECONDS_PER_DAY, 3),
                })
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "component": COMPONENT,
        "method": METHOD,
        "git_commit": current_git_commit(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "group": bundle["group"],
            "seed": int(bundle["seed"]),
            "encoder": bundle["encoder"],
            "input_len_L": int(bundle["input_len_L"]),
            "feature_names": list(bundle["feature_names"]),
            "val_rmse": float(bundle["metrics"]["val_rmse"]),
        },
        "rul_scale": {
            "channel_label_schema": bundle["rul"]["channel_label_schema"],
            "rul_scale_windows": H,
            "sample_period_s": sp_s,
        },
        "predictions": preds,
        "status": "RUL_PREDICTION_OK",
    }


def _validate_schema(doc: dict) -> None:
    import jsonschema
    schema = json.loads(
        (ROOT / "schemas" / "rul-prediction.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(doc, schema)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m component.predict_gru",
        description="通道级 GRU RUL 推理 (method=gru_channel_rul, 消费训练侧 bundle)")
    ap.add_argument("--features", type=Path, required=True,
                    help="channel_features.h5 (v2; 只读 x_ch, 无标签)")
    ap.add_argument("--bundle-dir", type=Path, required=True,
                    help="run_groups --export-inference-dir 导出目录 (model.pt + bundle.json)")
    ap.add_argument("--output", type=Path, required=True,
                    help="输出目录, 写入 rul_prediction.json")
    ap.add_argument("--stride", type=int, default=50,
                    help="推理滑窗步长 (默认 50, 与评估口径一致)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit-channels", type=int, default=None,
                    help="只取前 N 条通道 (通道键升序; 冒烟/演示用)")
    args = ap.parse_args()

    doc = predict(args.features, args.bundle_dir, stride=args.stride,
                  batch_size=args.batch_size, limit_channels=args.limit_channels)
    _validate_schema(doc)
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "rul_prediction.json"
    out_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f">> rul_prediction.json -> {out_path} "
          f"({len(doc['predictions'])} predictions, "
          f"model={doc['model']['group']} seed{doc['model']['seed']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
