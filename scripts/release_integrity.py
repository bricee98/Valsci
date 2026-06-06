#!/usr/bin/env python3
"""Generate and verify integrity metadata for a Valsci release distribution.

The distribution bundles a Docker image tarball, the hash-pinned lockfile, the
compose file, the Dockerfile, and the curated corpus manifests. This module
produces two artifacts next to those files:

  * ``SHA256SUMS``        - ``sha256sum``-compatible lines (``<hex>  <relpath>``)
  * ``release_manifest.json`` - structured manifest with per-file sha256 + size

It can also verify a distribution: every file listed must exist and hash to the
recorded value, and the two artifacts must agree. Any mismatch (a tampered or
truncated artifact) makes verification fail with a nonzero exit code.

Usage:
    python -m scripts.release_integrity generate --dist DIR [--version V] FILE [FILE ...]
    python -m scripts.release_integrity verify --dist DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SHA256SUMS_NAME = "SHA256SUMS"
RELEASE_MANIFEST_NAME = "release_manifest.json"
# Artifacts that describe the distribution; never include them in their own digest.
GENERATED_NAMES = {SHA256SUMS_NAME, RELEASE_MANIFEST_NAME}


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_rel(dist_dir: Path, file_path: Path) -> str:
    """Return ``file_path`` relative to ``dist_dir`` using forward slashes."""
    abs_path = file_path if file_path.is_absolute() else (dist_dir / file_path)
    rel = abs_path.resolve().relative_to(dist_dir.resolve())
    return rel.as_posix()


def _iter_relpaths(dist_dir: Path, files: List[str]) -> List[str]:
    """Expand the requested files/dirs into a sorted, de-duplicated relpath list."""
    relpaths: List[str] = []
    seen = set()
    for item in files:
        target = dist_dir / item
        if target.is_dir():
            members = [p for p in sorted(target.rglob("*")) if p.is_file()]
        else:
            members = [target]
        for member in members:
            if not member.exists():
                raise FileNotFoundError(f"Cannot hash missing distribution file: {member}")
            rel = _normalize_rel(dist_dir, member)
            if rel in GENERATED_NAMES or rel in seen:
                continue
            seen.add(rel)
            relpaths.append(rel)
    return sorted(relpaths)


def generate(
    dist_dir: Path,
    files: List[str],
    *,
    version: Optional[str] = None,
    created: Optional[str] = None,
) -> Dict:
    """Write SHA256SUMS and release_manifest.json for ``files`` under ``dist_dir``.

    Returns the release manifest dict. ``created`` is accepted so build scripts
    can stamp a deterministic timestamp; it is otherwise left null.
    """
    dist_dir = Path(dist_dir)
    relpaths = _iter_relpaths(dist_dir, files)
    if not relpaths:
        raise ValueError("No files provided to hash for the release distribution.")

    file_entries = []
    sha_lines = []
    for rel in relpaths:
        abs_path = dist_dir / rel
        digest = sha256_file(abs_path)
        size = abs_path.stat().st_size
        file_entries.append({"path": rel, "sha256": digest, "size_bytes": size})
        sha_lines.append(f"{digest}  {rel}")

    (dist_dir / SHA256SUMS_NAME).write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema": "valsci-release-manifest/1",
        "version": version,
        "created": created,
        "algorithm": "sha256",
        "file_count": len(file_entries),
        "files": file_entries,
    }
    (dist_dir / RELEASE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_sha256sums(text: str) -> Dict[str, str]:
    digests: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        # Format: "<hex>  <relpath>" (two spaces, binary mode marker tolerated).
        parts = line.split("  ", 1)
        if len(parts) != 2:
            parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed SHA256SUMS line: {line!r}")
        digest, rel = parts[0].strip(), parts[1].strip().lstrip("*")
        digests[rel] = digest.lower()
    return digests


def verify(dist_dir: Path) -> Tuple[bool, List[str]]:
    """Verify a distribution against its SHA256SUMS and release_manifest.json.

    Returns ``(ok, errors)``. ``ok`` is False if any listed file is missing,
    hashes differ, or the two metadata artifacts disagree.
    """
    dist_dir = Path(dist_dir)
    errors: List[str] = []

    sums_path = dist_dir / SHA256SUMS_NAME
    manifest_path = dist_dir / RELEASE_MANIFEST_NAME
    if not sums_path.exists():
        errors.append(f"Missing {SHA256SUMS_NAME}")
    if not manifest_path.exists():
        errors.append(f"Missing {RELEASE_MANIFEST_NAME}")
    if errors:
        return False, errors

    sums = _parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, [f"{RELEASE_MANIFEST_NAME} is not valid JSON: {exc}"]

    manifest_files = {entry["path"]: entry for entry in manifest.get("files", [])}

    # The two artifacts must list the same files.
    only_in_sums = set(sums) - set(manifest_files)
    only_in_manifest = set(manifest_files) - set(sums)
    for rel in sorted(only_in_sums):
        errors.append(f"{rel}: listed in SHA256SUMS but not release_manifest.json")
    for rel in sorted(only_in_manifest):
        errors.append(f"{rel}: listed in release_manifest.json but not SHA256SUMS")

    for rel, entry in sorted(manifest_files.items()):
        abs_path = dist_dir / rel
        if not abs_path.exists():
            errors.append(f"{rel}: missing from distribution")
            continue
        actual = sha256_file(abs_path)
        if actual != str(entry.get("sha256", "")).lower():
            errors.append(f"{rel}: sha256 mismatch (manifest says {entry.get('sha256')}, actual {actual})")
            continue
        if rel in sums and sums[rel] != actual:
            errors.append(f"{rel}: sha256 mismatch vs SHA256SUMS ({sums[rel]} vs {actual})")
        expected_size = entry.get("size_bytes")
        if expected_size is not None and abs_path.stat().st_size != expected_size:
            errors.append(f"{rel}: size mismatch (manifest says {expected_size})")

    return (not errors), errors


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate/verify release integrity metadata.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Write SHA256SUMS and release_manifest.json.")
    gen.add_argument("--dist", required=True, help="Distribution directory.")
    gen.add_argument("--version", default=None, help="Release version label.")
    gen.add_argument("--created", default=None, help="Created timestamp (ISO 8601).")
    gen.add_argument("files", nargs="+", help="Files/dirs (relative to --dist) to hash.")

    ver = sub.add_parser("verify", help="Verify a distribution's integrity.")
    ver.add_argument("--dist", required=True, help="Distribution directory.")

    args = parser.parse_args(argv)

    if args.command == "generate":
        manifest = generate(
            Path(args.dist),
            args.files,
            version=args.version,
            created=args.created,
        )
        print(f"Wrote {SHA256SUMS_NAME} and {RELEASE_MANIFEST_NAME} for {manifest['file_count']} files.")
        return 0

    if args.command == "verify":
        ok, errors = verify(Path(args.dist))
        if ok:
            print("Release integrity OK.")
            return 0
        print("Release integrity FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
