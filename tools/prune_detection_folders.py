#!/usr/bin/env python3
"""Remove detection folders that do not have matching frame directories."""

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Set


def collect_dir_names(root: Path) -> Set[str]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    return {p.name for p in root.iterdir() if p.is_dir()}


def prune_one(frame_root: Path, det_root: Path, dry_run: bool) -> None:
    print(f"[step] Checking {det_root} against {frame_root}")

    frame_names = collect_dir_names(frame_root)
    det_names = collect_dir_names(det_root)

    extra = sorted(det_names - frame_names)
    missing = sorted(frame_names - det_names)

    if missing:
        print(f"[info] {len(missing)} frame folders have no detections (kept): {', '.join(missing[:10])}")

    if not extra:
        print("[info] No extra detection folders found. Nothing to prune.")
        return

    print(f"[info] Found {len(extra)} detection folders without matching frames.")
    for name in extra:
        path = det_root / name
        if dry_run:
            print(f"[dry-run] Would remove {path}")
        else:
            print(f"[delete] Removing {path}")
            shutil.rmtree(path)


def prune(args) -> None:
    frame_root = Path(args.frames_dir)
    for det_dir in args.detections_dirs:
        prune_one(frame_root, Path(det_dir), args.dry_run)


def parse_args():
    ap = argparse.ArgumentParser("Prune detection dirs that are absent in frames")
    ap.add_argument("--frames_dir", type=str, default="data/shanghaitech/testing/frames",
                    help="Reference frames directory")
    ap.add_argument("--detections_dirs", nargs="+", default=["artifacts/detections/testing"],
                    help="One or more detection directories to prune")
    ap.add_argument("--dry_run", action="store_true", help="List deletions without removing anything")
    return ap.parse_args()


def main():
    args = parse_args()
    prune(args)


if __name__ == "__main__":
    main()
