# -*- coding: utf-8 -*-
"""三级退化链仿真统一入口：依次生成 sim_v2 (subdose on) 与 sim_v1 (subdose off)。

并把**实际使用的 config SHA256**、种子、输出路径与 H5 哈希写入 sim_manifest.json，
保证"数据由哪份配置、哪个种子生成"可追溯（契约 §6 数据治理）。

用法：
  python scripts/generate_simulation.py [--config configs/phased_array.yaml]
                                        [--n_traj 200] [--seed 42] [--fast]
                                        [--manifest outputs/sim_manifest.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DYNAMICS = {"on": "phased_array_subdose_v2", "off": "leo_coupled_v1"}
TAG = {"on": "sim_v2", "off": "sim_v1"}


def sim_dir_for(data_root: Path | str | None, subdose: str, seed: int) -> Path:
    """仿真输出目录 (ROOT 相对或绝对): 默认 canonical 槽位; data_root 给定时重定向。

    --fast 调试链必须用 --data-root 重定向, 避免小规模数据覆盖 canonical
    200 轨迹冻结基线 (2026-08-23 F6 事故回归)。
    """
    if data_root is None:
        return Path("data/simulated/phased_array") / TAG[subdose] / f"seed_{seed}"
    return Path(data_root) / TAG[subdose] / f"seed_{seed}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phased_array.yaml"))
    ap.add_argument("--n_traj", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="小规模 smoke (4 轨迹/1 年)")
    ap.add_argument("--data-root", default=None,
                    help="仿真输出根目录 (默认 data/simulated/phased_array; "
                         "--fast 隔离用, 避免覆盖 canonical 槽位)")
    ap.add_argument("--manifest", default=None, help="sim_manifest.json 输出路径")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    config_sha = _sha256(config_path)

    entries = []
    for subdose in ("on", "off"):
        cmd = [sys.executable, "-m", "src.sim.phased_array_sim",
               "--config", str(config_path), "--seed", str(args.seed),
               "--subdose", subdose, "--hash"]
        if args.fast:
            cmd.append("--fast")
        elif args.n_traj:
            cmd += ["--n_traj", str(args.n_traj)]
        out_subdir = sim_dir_for(args.data_root, subdose, args.seed)
        cmd += ["--out", str(out_subdir)]
        print(f">> 运行: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)

        sim_dir = ROOT / out_subdir
        h5 = sim_dir / "phased_array_all.h5"
        if not h5.is_file():
            print(f"!! 仿真输出缺失: {h5}")
            return 1
        try:
            dir_rel = sim_dir.relative_to(ROOT)
        except ValueError:                     # data_root 在 ROOT 外 (如绝对路径挂载)
            dir_rel = sim_dir
        entries.append({
            "subdose": subdose,
            "tag": TAG[subdose],
            "dynamics_id": DYNAMICS[subdose],
            "seed": args.seed,
            "output_dir": str(dir_rel),
            "h5": str(dir_rel / "phased_array_all.h5"),
            "h5_sha256": _sha256(h5),
            "h5_size_bytes": h5.stat().st_size,
        })

    manifest = {
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": config_sha,
        "seed": args.seed,
        "fast": bool(args.fast),
        "physics_chain_doc": "docs/simulation/PHYSICS_CHAIN.yaml",
        "levels": ["gan_tr", "array", "link"],
        "runs": entries,
    }
    out = Path(args.manifest) if args.manifest else ROOT / "data/simulated/phased_array/sim_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">> sim_manifest -> {out} (config_sha256={config_sha[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
