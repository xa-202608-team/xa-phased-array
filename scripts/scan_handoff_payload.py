# -*- coding: utf-8 -*-
"""handoff/payload/ staging 安装前扫描（设计 §5.3/§11）。

规则：未声明文件、NASA 原始数据命名签名、小文本文件内嵌绝对路径、
符号链接、秘密信息签名、与 git archive 代码快照重名的成员。
违规非空时 CLI 退出码 1，阻断 RC 构建。

approved 白名单来源：
- ``--repo-root <path>``：从组件仓三槽位 manifest（data/data_manifest.json、
  checkpoints/checkpoint_manifest.json、results/results_manifest.json）中
  rc_payload=true 条目派生（消除"从 staging 现存文件自批准"）；results 条目
  path 已含 reference/ 前缀，与 staging 布局 handoff/payload/results/<path> 一致。
- 不带 --repo-root：从 staging 现存文件构造（旧行为，向后兼容）。
"""
import argparse
import json
import re
from pathlib import Path

NASA_RAW_PATTERN = re.compile(r"(nasa|mosfet_raw|raw_.+\.(mat|txt)|original)", re.IGNORECASE)
ABSOLUTE_PATH_PATTERN = re.compile(r"([A-Za-z]:[/\\]{1,2}|/home/|/Users/)")
SECRET_PATTERN = re.compile(r"(api[_-]?key|token|password|BEGIN (RSA|OPENSSH) PRIVATE KEY)", re.IGNORECASE)
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".csv", ".log", ".py"}

# 组件仓三槽位 manifest 的相对位置（与 stage_handoff_payload 同源）
MANIFEST_SLOTS = {
    "data": Path("data") / "data_manifest.json",
    "checkpoints": Path("checkpoints") / "checkpoint_manifest.json",
    "results": Path("results") / "results_manifest.json",
}


def _approved_from_manifests(repo_root: Path) -> set:
    """从组件仓三 manifest 的 rc_payload=true 条目派生 staging 白名单。

    staging 相对路径形态为 "<槽位>/<条目 path>"（results 条目 path 已含
    reference/ 前缀，与 handoff/payload/results/<path> 布局一致）。
    """
    approved = set()
    for slot, rel_manifest in MANIFEST_SLOTS.items():
        manifest_path = Path(repo_root) / rel_manifest
        if not manifest_path.is_file():
            continue
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in data.get("entries", []):
            if entry.get("rc_payload"):
                approved.add(f"{slot}/{entry['path']}")
    return approved


def scan(staging: Path, approved: set, code_members: set) -> list:
    """返回违规描述清单；空清单表示通过。"""
    violations = []
    for path in sorted(staging.rglob("*")):
        rel = path.relative_to(staging).as_posix()
        if path.is_symlink():
            violations.append(f"符号链接禁止: {rel}")
            continue
        if not path.is_file():
            continue
        if rel not in approved:
            violations.append(f"未在 manifest 声明: {rel}")
        if rel in code_members:
            violations.append(f"与代码快照成员重名: {rel}")
        if NASA_RAW_PATTERN.search(path.name):
            violations.append(f"疑似 NASA 原始数据命名: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size < (1 << 20):
            text = path.read_text(encoding="utf-8", errors="ignore")
            match = ABSOLUTE_PATH_PATTERN.search(text)
            if match:
                violations.append(f"内嵌绝对路径 ({match.group(0)!r}): {rel}")
            secret = SECRET_PATTERN.search(text)
            if secret:
                violations.append(f"疑似秘密信息 ({secret.group(0)!r}): {rel}")
    return violations


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", required=True, help="handoff/payload 目录")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="组件仓库根目录：提供时 approved 从三槽位 manifest 的 "
        "rc_payload=true 条目派生（消除自批准）；缺省时保持从 staging "
        "现存文件构造的旧行为",
    )
    args = parser.parse_args(argv)
    staging = Path(args.staging)
    if args.repo_root:
        approved = _approved_from_manifests(Path(args.repo_root))
    else:
        approved = {
            p.relative_to(staging).as_posix()
            for p in staging.rglob("*")
            if p.is_file()
        }
    violations = scan(staging, approved=approved, code_members=set())
    for line in violations:
        print(line)
    print(f"{'FAIL' if violations else 'OK'}: {len(violations)} violations")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(_main())
