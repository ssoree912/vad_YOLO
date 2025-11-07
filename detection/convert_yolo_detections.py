#!/usr/bin/env python3
"""
Flatten per-video YOLO detection dicts into the bbox arrays that
Attribute-based VAD expects (./data/<dataset>/<dataset>_bboxes_{train,test}.npy).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _frame_key(frame_path: Path, fallback_idx: int) -> int:
    """Match yolo_detection.py: prefer numeric stem, else enumeration index."""
    try:
        return int(frame_path.stem)
    except ValueError:
        return fallback_idx


def _frame_sort_key(frame_path: Path):
    stem = frame_path.stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def _ensure_boxes(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return arr.reshape(-1, 4)


def _ensure_classes(arr: np.ndarray, n_boxes: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int32).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int32)
    if arr.size != n_boxes:
        raise ValueError(f"class count ({arr.size}) != box count ({n_boxes})")
    return arr


def _collect_split(
    frames_root: Path, detections_root: Path
) -> Tuple[np.ndarray, np.ndarray]:
    all_boxes = []
    all_classes = []

    videos = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    if not videos:
        raise FileNotFoundError(f"No frame directories under {frames_root}")

    for video_dir in videos:
        frame_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            frame_files.extend(video_dir.glob(ext))
        frame_files = sorted(frame_files, key=_frame_sort_key)
        det_file = detections_root / video_dir.name / "detections.npy"
        if not det_file.exists():
            print(f"[warn] missing detections for {video_dir.name}, using empty boxes")
            det_map: Dict[int, Dict[str, np.ndarray]] = {}
        else:
            det_map = np.load(det_file, allow_pickle=True).item()
        for idx, frame_path in enumerate(frame_files):
            key = _frame_key(frame_path, idx)
            info = det_map.get(key)
            if info is None:
                boxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros((0,), dtype=np.int32)
            else:
                boxes = _ensure_boxes(info.get("boxes", np.zeros((0, 4), np.float32)))
                classes = _ensure_classes(info.get("classes", np.zeros((0,), np.int32)), len(boxes))
            all_boxes.append(boxes)
            all_classes.append(classes)

    return np.array(all_boxes, dtype=object), np.array(all_classes, dtype=object)


def main():
    ap = argparse.ArgumentParser(
        description="Convert YOLO detection pickles to Attribute-VAD bbox npy files"
    )
    ap.add_argument("--dataset_name", type=str, default="shanghaitech")
    ap.add_argument("--data_root", type=str, default="./data/shanghaitech")
    ap.add_argument("--detections_root", type=str, default="./artifacts/detections")
    ap.add_argument("--split", choices=["training", "testing"], required=True)
    ap.add_argument("--frames_subdir", type=str, default="frames",
                    help="Subdirectory under each split containing frames (default: frames)")
    ap.add_argument("--frames_root", type=str, default=None,
                    help="Optional absolute path override for the frames directory")
    ap.add_argument("--out_root", type=str, default=None,
                    help="Directory to save bbox npy files (default: data_root)")
    args = ap.parse_args()

    split_suffix = "train" if args.split == "training" else "test"
    out_root = Path(args.out_root) if args.out_root else Path(args.data_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_boxes = out_root / f"{args.dataset_name}_bboxes_{split_suffix}.npy"
    out_classes = out_root / f"{args.dataset_name}_bboxes_{split_suffix}_classes.npy"

    if args.frames_root:
        frames_root = Path(args.frames_root)
    else:
        frames_root = Path(args.data_root) / args.split / args.frames_subdir
    detections_root = Path(args.detections_root) / args.split

    boxes_arr, classes_arr = _collect_split(frames_root, detections_root)
    np.save(out_boxes, boxes_arr, allow_pickle=True)
    np.save(out_classes, classes_arr, allow_pickle=True)

    meta = {
        "dataset": args.dataset_name,
        "split": args.split,
        "frames_dir": str(frames_root),
        "detections_dir": str(detections_root),
        "frames_processed": int(len(boxes_arr)),
        "out_root": str(out_root),
    }
    print(json.dumps(meta, indent=2))
    print(f"[save] {out_boxes}")
    print(f"[save] {out_classes}")


if __name__ == "__main__":
    main()
