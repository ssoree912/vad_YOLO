#!/usr/bin/env python3
"""
Approximate per-box velocity features directly from cached YOLO detections.

This script walks the ShanghaiTech split files (train/test), loads each video's
`detections.npy`, and for every frame computes the delta between a box center
and its best IoU match in the previous frame. The displacements are stored as
an object-array (`velocity.npy`) compatible with Attribute-VAD pipelines.
"""

import argparse 
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np


class SplitEntry(NamedTuple):
    name: str
    label: int
    frames: int


def _normalize_video_key(name: str) -> Tuple[str, int]:
    parts = name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Video name does not match <split>_<id> pattern: {name}")
    prefix, idx = parts[0].strip(), parts[1].strip()
    if not prefix or not idx:
        raise ValueError(f"Video name missing components: {name}")
    return prefix, int(idx)


def _index_detection_dirs(detections_dir: Path) -> Dict[Tuple[str, int], Path]:
    index: Dict[Tuple[str, int], Path] = {}
    for item in detections_dir.iterdir():
        if not item.is_dir():
            continue
        try:
            key = _normalize_video_key(item.name)
        except ValueError:
            continue
        index[key] = item
    return index


def _resolve_detection_folder(
    detections_dir: Path, entry_name: str, det_index: Dict[Tuple[str, int], Path]
) -> Path:
    direct = detections_dir / entry_name
    if direct.exists():
        return direct
    try:
        key = _normalize_video_key(entry_name)
    except ValueError:
        return direct
    return det_index.get(key, direct)


def _load_split_entries(split_file: Path) -> List[SplitEntry]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    entries: List[SplitEntry] = []
    with open(split_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 3:
                raise ValueError(f"Malformed split line: {line}")
            entries.append(SplitEntry(name=parts[0], label=int(parts[1]), frames=int(parts[2])))
    if not entries:
        raise ValueError(f"No entries parsed from {split_file}")
    return entries


def _frame_sort_key(key) -> Tuple[int, str]:
    try:
        return (0, f"{int(key):010d}")
    except (TypeError, ValueError):
        return (1, str(key))


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return float(inter / max(1e-6, area_a + area_b - inter))


def _greedy_match(prev_boxes: np.ndarray, curr_boxes: np.ndarray, thr: float) -> List[int]:
    if prev_boxes.size == 0 or curr_boxes.size == 0:
        return [-1] * len(curr_boxes)
    assigned: List[int] = [-1] * len(curr_boxes)
    used_prev = set()
    for curr_idx, cbox in enumerate(curr_boxes):
        best_idx, best_val = -1, 0.0
        for prev_idx, pbox in enumerate(prev_boxes):
            if prev_idx in used_prev:
                continue
            val = _iou_xyxy(pbox, cbox)
            if val > best_val:
                best_val, best_idx = val, prev_idx
        if best_val >= thr:
            assigned[curr_idx] = best_idx
            used_prev.add(best_idx)
    return assigned


def _centers_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    centers = np.stack(
        ((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5),
        axis=1,
    )
    return centers.astype(np.float32, copy=False)


def parse_args():
    ap = argparse.ArgumentParser("Build velocity.npy from YOLO detections")
    ap.add_argument("--dataset_name", type=str, default="shanghaitech")
    ap.add_argument("--split", type=str, choices=["training", "testing"], required=True)
    ap.add_argument(
        "--data_root", type=str, default="./data/shanghaitech", help="Dataset root containing split text files"
    )
    ap.add_argument(
        "--detections_root",
        type=str,
        default="./artifacts/detections",
        help="Root directory holding detections/<split>/<video>/detections.npy",
    )
    ap.add_argument(
        "--output_root",
        type=str,
        default="./artifacts/features",
        help="Where to write artifacts/features/<dataset>/<split_tag>/velocity.npy",
    )
    ap.add_argument(
        "--train_split_file",
        type=str,
        default="./data/shanghaitech/train_split.txt",
        help="CSV listing training videos (name,label,frames)",
    )
    ap.add_argument(
        "--test_split_file",
        type=str,
        default="./data/shanghaitech/test_split.txt",
        help="CSV listing testing videos (name,label,frames)",
    )
    ap.add_argument("--track_iou", type=float, default=0.5, help="IoU threshold for linking boxes across frames")
    return ap.parse_args()


def main():
    args = parse_args()
    split_tag = "train" if args.split == "training" else "test"
    split_file = Path(args.train_split_file if split_tag == "train" else args.test_split_file)
    split_entries = _load_split_entries(split_file)

    detections_dir = Path(args.detections_root) / args.split
    if not detections_dir.exists():
        raise FileNotFoundError(f"Detections directory missing: {detections_dir}")
    det_index = _index_detection_dirs(detections_dir)

    out_dir = Path(args.output_root) / args.dataset_name / split_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    velocity_frames: List[np.ndarray] = []
    for entry in split_entries:
        det_folder = _resolve_detection_folder(detections_dir, entry.name, det_index)
        det_file = det_folder / "detections.npy"
        if not det_file.exists():
            base_msg = f"Missing detections for video {entry.name}: {det_file}"
            if det_folder != detections_dir / entry.name:
                base_msg += f" (resolved alias: {det_folder.name})"
            raise FileNotFoundError(base_msg)
        det_map: Dict[int, Dict[str, np.ndarray]] = np.load(det_file, allow_pickle=True).item()
        frame_items = sorted(det_map.items(), key=lambda kv: _frame_sort_key(kv[0]))
        if len(frame_items) != entry.frames:
            raise ValueError(
                f"Frame count mismatch for {entry.name}: split={entry.frames}, detections={len(frame_items)}"
            )

        prev_boxes = np.zeros((0, 4), dtype=np.float32)
        prev_centers = np.zeros((0, 2), dtype=np.float32)
        for _, payload in frame_items:
            boxes = np.asarray(payload.get("boxes", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32).reshape(
                -1, 4
            )
            centers = _centers_xyxy(boxes)
            match = _greedy_match(prev_boxes, boxes, args.track_iou)
            vel = np.zeros((len(boxes), 2), dtype=np.float32)
            for det_idx, prev_idx in enumerate(match):
                if prev_idx >= 0 and prev_idx < prev_centers.shape[0]:
                    vel[det_idx] = centers[det_idx] - prev_centers[prev_idx]
            velocity_frames.append(vel)
            prev_boxes = boxes
            prev_centers = centers

    expected_frames = sum(entry.frames for entry in split_entries)
    if len(velocity_frames) != expected_frames:
        raise RuntimeError(
            f"Total frame mismatch: collected={len(velocity_frames)}, expected={expected_frames}. "
            "Did some videos fail earlier?"
        )

    velocity_arr = np.array(velocity_frames, dtype=object)
    out_path = out_dir / "velocity.npy"
    np.save(out_path, velocity_arr, allow_pickle=True)
    print(f"[save] {out_path}  (frames={len(velocity_arr)})")


if __name__ == "__main__":
    main()
