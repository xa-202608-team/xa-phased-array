# -*- coding: utf-8 -*-
"""按 manifest 白名单装配 handoff/payload/ staging（设计 §5.3）。

只复制 entries 中 rc_payload=true 的文件；复制后逐文件复核 SHA256；
staging 是可删除、可重建的派生视图，绝不允许反向复制回槽位。
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

SLOT_ROOTS = {
    "data": Path("data"),
    "checkpoints": Path("checkpoints"),
    "results": Path("results"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(repo_root: Path, slots: dict) -> list:
    """按槽位白名单复制到 handoff/payload/，返回 staging 相对路径清单。"""
    repo_root = Path(repo_root)
    staging = repo_root / "handoff" / "payload"
    staged = []
    for slot, entries in slots.items():
        if slot not in SLOT_ROOTS:
            raise ValueError(f"未知槽位: {slot!r}")
        for entry in entries:
            if not entry.get("rc_payload"):
                continue
            rel = entry["path"]
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise ValueError(f"manifest 条目路径必须是仓库内相对路径（禁绝对路径与 ..）: {rel!r}")
            src = repo_root / SLOT_ROOTS[slot] / rel
            if not src.is_file() or src.is_symlink():
                raise ValueError(f"白名单文件缺失或为符号链接: {src}")
            digest = sha256_file(src)
            if digest != entry["sha256"]:
                raise ValueError(
                    f"SHA256 与 manifest 不符: {rel} (manifest={entry['sha256']}, actual={digest})"
                )
            dst = staging / slot / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if sha256_file(dst) != digest:
                raise ValueError(f"复制后校验失败: {dst}")
            staged.append(dst.relative_to(repo_root).as_posix())
    return sorted(staged)


def _load_manifest_slots(repo_root: Path) -> dict:
    slots = {}
    manifests = {
        "data": repo_root / "data" / "data_manifest.json",
        "checkpoints": repo_root / "checkpoints" / "checkpoint_manifest.json",
    }
    for slot, manifest_path in manifests.items():
        if manifest_path.is_file():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            slots[slot] = data.get("entries", [])
    return slots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="组件仓库根目录")
    parser.add_argument("--clean", action="store_true", help="装配前删除已有 staging")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    staging = repo_root / "handoff" / "payload"
    if args.clean and staging.exists():
        shutil.rmtree(staging)
    staged = stage(repo_root, _load_manifest_slots(repo_root))
    print(f"{len(staged)} files staged -> {staging}")
    for rel in staged:
        print(f"  {rel}")


if __name__ == "__main__":
    main()
