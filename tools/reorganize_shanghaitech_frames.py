#!/usr/bin/env python3
"""Rearrange Shanghaitech frame folders to match the official train/test split.

This script reads the split definition files (``*_split.txt``) and ensures that each
video folder containing extracted frames lives under the correct split directory
(`training/frames` or `testing/frames`). It can either move or copy the folders and
supports dry-run mode for safety.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


def _parse_split_file(path: Path, split_name: str) -> Dict[str, Tuple[str, int, int]]:
    """Return mapping video_id -> (split, label, frame_count)."""

    mapping: Dict[str, Tuple[str, int, int]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Split file missing: {path}")

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                raise ValueError(f"Malformed line in {path}: '{line}'")
            video_id, label_str, frame_count_str = parts[:3]
            if video_id in mapping:
                raise ValueError(f"Duplicate video id '{video_id}' found in {path}")
            mapping[video_id] = (split_name, int(label_str), int(frame_count_str))
    return mapping


def _collect_splits(train_path: Path, test_path: Path) -> Dict[str, Tuple[str, int, int]]:
    mapping = {}
    mapping.update(_parse_split_file(train_path, "training"))
    test_mapping = _parse_split_file(test_path, "testing")
    overlap = set(mapping).intersection(test_mapping)
    if overlap:
        dup = ", ".join(sorted(list(overlap)[:5]))
        raise ValueError(f"Video ids present in both split files: {dup}")
    mapping.update(test_mapping)
    return mapping


def _locate_video_dir(video_id: str, search_roots: Iterable[Path]) -> Optional[Path]:
    for root in search_roots:
        candidate = root / video_id
        if candidate.exists():
            return candidate
    return None


def _safe_remove(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] Would remove existing destination: {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def reorganize_frames(args) -> None:
    base_dir = Path(args.base_dir)
    train_split = Path(args.train_split)
    test_split = Path(args.test_split)

    splits = _collect_splits(train_split, test_split)

    training_root = Path(args.training_frames_dir or base_dir / "training" / "frames")
    testing_root = Path(args.testing_frames_dir or base_dir / "testing" / "frames")
    extra_roots = [Path(p) for p in args.extra_source_dirs]
    search_roots = []
    for root in (training_root, testing_root, *extra_roots):
        if root not in search_roots:
            search_roots.append(root)

    if not search_roots:
        raise ValueError("No frame directories provided to search.")

    moved, skipped, missing = 0, 0, 0
    for video_id, (target_split, label, frame_count) in splits.items():
        src_dir = _locate_video_dir(video_id, search_roots)
        if src_dir is None:
            print(f"[warn] Missing frames for {video_id}; expected ~{frame_count} frames (label={label}).")
            missing += 1
            continue

        target_root = training_root if target_split == "training" else testing_root
        dst_dir = target_root / video_id

        if src_dir.resolve() == dst_dir.resolve():
            skipped += 1
            continue

        if dst_dir.exists():
            if args.skip_existing:
                print(f"[skip] Destination already exists for {video_id}: {dst_dir}")
                skipped += 1
                continue
            if not args.overwrite:
                print(f"[warn] Destination exists for {video_id} (use --overwrite to replace): {dst_dir}")
                skipped += 1
                continue
            _safe_remove(dst_dir, args.dry_run)

        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        operation = "copy" if args.copy else "move"
        if args.dry_run:
            print(f"[dry-run] Would {operation} {src_dir} -> {dst_dir}")
        else:
            if args.copy:
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=False)
            else:
                shutil.move(str(src_dir), str(dst_dir))
        moved += 1

    print("=== Summary ===")
    print(f"Processed videos: {len(splits)}")
    print(f"Moved/copied: {moved}")
    print(f"Skipped (already correct or existing): {skipped}")
    print(f"Missing source folders: {missing}")


def parse_args(argv: Optional[Iterable[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_dir", type=str, default="data/cache/shanghaitech",
                    help="Root directory containing training/testing frame folders")
    ap.add_argument("--train_split", type=str, default="data/cache/shanghaitech/train_split.txt")
    ap.add_argument("--test_split", type=str, default="data/cache/shanghaitech/test_split.txt")
    ap.add_argument("--training_frames_dir", type=str, default=None,
                    help="Override training frames directory (default: <base>/training/frames)")
    ap.add_argument("--testing_frames_dir", type=str, default=None,
                    help="Override testing frames directory (default: <base>/testing/frames)")
    ap.add_argument("--extra_source_dirs", nargs="*", default=[],
                    help="Additional directories to search for misplaced video folders")
    ap.add_argument("--copy", action="store_true",
                    help="Copy instead of move (default: move)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print actions without modifying the filesystem")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite destination folders when they already exist")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip videos whose destination already exists (takes precedence over overwrite)")
    return ap.parse_args(args=argv)


def main(argv: Optional[Iterable[str]] = None):
    args = parse_args(argv)
    try:
        reorganize_frames(args)
    except Exception as exc:  # pragma: no cover - surface error with exit code
        print(f"[error] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
