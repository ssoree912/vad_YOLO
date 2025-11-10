#!/usr/bin/env python3
"""Reshuffle YOLO detection folders into training/testing subdirectories."""

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, Set


def _parse_split_file(split_path: Path) -> Set[str]:
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    names: Set[str] = set()
    with open(split_path, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            first = line.split(",", 1)[0].strip()
            if not first:
                continue
            names.add(first)
    if not names:
        raise ValueError(f"No entries parsed from {split_path}")
    return names


def _scan_detection_dirs(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not root.exists():
        raise FileNotFoundError(f"Detections root does not exist: {root}")
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name in ("training", "testing"):
            for video_dir in child.iterdir():
                if video_dir.is_dir():
                    mapping.setdefault(video_dir.name, video_dir)
        else:
            mapping.setdefault(child.name, child)
    return mapping


def _moves_for_split(
    videos: Iterable[str], det_map: Dict[str, Path], target_dir: Path
):
    for name in videos:
        src = det_map.get(name)
        if src is None:
            yield ("missing", name, None, target_dir / name)
            continue
        dst = target_dir / name
        if src.resolve() == dst.resolve():
            yield ("skip", name, src, dst)
            continue
        if dst.exists():
            yield ("conflict", name, src, dst)
            continue
        yield ("move", name, src, dst)


def reorganize(
    detections_root: Path,
    train_split: Path,
    test_split: Path,
    dry_run: bool = False,
):
    train_names = _parse_split_file(train_split)
    test_names = _parse_split_file(test_split)

    overlap = train_names & test_names
    if overlap:
        raise ValueError(f"Duplicate video ids between splits: {sorted(overlap)[:5]} ...")

    det_map = _scan_detection_dirs(detections_root)
    train_dir = detections_root / "training"
    test_dir = detections_root / "testing"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    stats = {"move": 0, "skip": 0, "missing": 0, "conflict": 0}

    for split_name, names in (
        ("training", sorted(train_names)),
        ("testing", sorted(test_names)),
    ):
        target_dir = train_dir if split_name == "training" else test_dir
        for status, vid, src, dst in _moves_for_split(names, det_map, target_dir):
            stats[status] += 1
            if status == "skip":
                continue
            if status == "missing":
                print(f"[warn] detections missing for {vid}, expected under {dst.parent}")
                continue
            if status == "conflict":
                print(f"[warn] destination already exists for {vid}: {dst}")
                continue
            # move
            if dry_run:
                print(f"[dry-run] mv {src} -> {dst}")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                det_map[vid] = dst

    print(
        "Summary:",
        ", ".join([f"{k}={v}" for k, v in stats.items()]),
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Organize YOLO detection folders into training/testing subdirectories"
    )
    ap.add_argument("--detections_root", type=str, default="./artifacts/detections")
    ap.add_argument("--train_split_file", type=str, default="./data/shanghaitech/train_split.txt")
    ap.add_argument("--test_split_file", type=str, default="./data/shanghaitech/test_split.txt")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    reorganize(
        detections_root=Path(args.detections_root),
        train_split=Path(args.train_split_file),
        test_split=Path(args.test_split_file),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

