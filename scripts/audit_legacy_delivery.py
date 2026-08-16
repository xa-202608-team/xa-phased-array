# -*- coding: utf-8 -*-
"""旧交付相控阵工件审计：只读枚举并生成 路径+大小+SHA256+类别 基线清单。

用于设计 §9 迁移办法第 1 步（基线）与第 6 步（复制后逐项核对）。
"""
import argparse
import hashlib
import json
from pathlib import Path

CATEGORIES = {
    ".pt": "checkpoint",
    ".h5": "data",
    ".hdf5": "data",
    ".csv": "data",
    ".json": "metrics",
    ".jsonl": "metrics",
    ".log": "run_log",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> list:
    """枚举 root 下普通文件（跳过符号链接），返回按路径排序的清单条目。"""
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "category": CATEGORIES.get(path.suffix.lower(), "other"),
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True, help="旧交付相控阵目录（只读）")
    parser.add_argument("--output", required=True, help="基线 JSON 输出路径（建议写到本地被忽略目录）")
    args = parser.parse_args()
    entries = audit(Path(args.legacy_root))
    Path(args.output).write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{len(entries)} files audited -> {args.output}")


if __name__ == "__main__":
    main()
