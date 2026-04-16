#!/usr/bin/env python3
"""
Extract a downloaded BDD100K tarball safely into a target directory.

Usage examples:
    # Extract into the tar's parent directory (auto-discover tarball if not provided)
    python project/scripts/00_extract_bdd_tar.py \
        --tar-path project/bdd100k_download/bdd100k_images_100k.tar

    # Extract to a specific directory
    python project/scripts/00_extract_bdd_tar.py \
        --tar-path project/bdd100k_download/bdd100k_images_100k.tar \
        --out-dir project/data/ad/bdd100k_raw/bdd100k
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract BDD100K tarball safely.")
    parser.add_argument(
        "--tar-path",
        type=Path,
        help="Path to the downloaded BDD100K tarball. If omitted, the script will search common locations.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory to extract into. Defaults to the tar's parent directory.",
    )
    parser.add_argument("--force", action="store_true", help="Proceed even if target files already exist.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _is_safe_member(member: tarfile.TarInfo) -> bool:
    # Prevent absolute paths and path traversal.
    path = Path(member.name)
    return not path.is_absolute() and ".." not in path.parts


def safe_extract(tar: tarfile.TarFile, path: Path, force: bool) -> None:
    members = tar.getmembers()
    unsafe = [m.name for m in members if not _is_safe_member(m)]
    if unsafe:
        raise ValueError(f"Unsafe paths detected in tar: {unsafe[:3]} ...")

    path.mkdir(parents=True, exist_ok=True)
    for member in members:
        target_path = path / member.name
        if target_path.exists() and not force:
            raise FileExistsError(
                f"Target already exists: {target_path}. Use --force to overwrite or extract elsewhere."
            )

    tar.extractall(path=path)


def find_tarball() -> Path:
    search_roots = [
        Path.cwd(),
        Path.cwd() / "project",
        Path.cwd() / "project" / "bdd100k_download",
        Path.cwd() / "project" / "data" / "ad" / "bdd100k_raw" / "bdd100k",
    ]
    candidates = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in ["bdd100k*.tar", "bdd100k*.tar.gz", "*.tar", "*.tar.gz"]:
            candidates.extend(root.glob(pattern))
    unique = []
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        unique.append(cand)
    if not unique:
        raise FileNotFoundError("No tarball found. Pass --tar-path explicitly.")
    if len(unique) > 1:
        raise FileExistsError(f"Multiple tarballs found; specify one explicitly: {', '.join(str(u) for u in unique)}")
    return unique[0]


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    tar_path = args.tar_path or find_tarball()
    if not tar_path.is_file():
        raise FileNotFoundError(f"Tar file not found: {tar_path}")

    out_dir = args.out_dir or tar_path.parent
    logging.info("Extracting %s into %s", tar_path, out_dir)

    mode = "r:gz" if tar_path.suffixes[-2:] in [[".tar", ".gz"], [".tgz"]] or tar_path.suffix == ".gz" else "r"
    with tarfile.open(tar_path, mode) as tar:
        safe_extract(tar, out_dir, args.force)

    logging.info("Extraction complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("Extraction failed: %s", exc)
        sys.exit(1)
