#!/usr/bin/env python3
"""Build velocity.npy directly from YOLO detections and frame directories."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


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
    centers = np.stack(((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5), axis=1)
    return centers.astype(np.float32, copy=False)


def parse_args():
    ap = argparse.ArgumentParser("Build velocity.npy from YOLO detections")
    ap.add_argument("--dataset_name", type=str, default="shanghaitech")
    ap.add_argument("--split", choices=["training", "testing"], required=True)
    ap.add_argument("--data_root", type=str, default="./data/cache/shanghaitech",
                    help="Dataset root containing <split>/frames")
    ap.add_argument("--detections_root", type=str, default="./artifacts/detections",
                    help="Root holding detections/<split>/<video>/detections.npy")
    ap.add_argument("--output_root", type=str, default="./artifacts/features",
                    help="Directory to save features/<dataset>/<split_tag>/velocity.npy")
    ap.add_argument("--track_iou", type=float, default=0.5,
                    help="IoU threshold for linking boxes across frames")
    return ap.parse_args()


def main():
    args = parse_args()
    split_tag = "train" if args.split == "training" else "test"

    frames_root = Path(args.data_root) / args.split / "frames"
    detections_dir = Path(args.detections_root) / args.split
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames directory missing: {frames_root}")
    if not detections_dir.exists():
        raise FileNotFoundError(f"Detections directory missing: {detections_dir}")

    videos = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    if not videos:
        raise FileNotFoundError(f"No videos found under {frames_root}")

    velocity_frames: List[np.ndarray] = []
    for video_dir in videos:
        det_file = detections_dir / video_dir.name / "detections.npy"
        if not det_file.exists():
            raise FileNotFoundError(f"Missing detections for video {video_dir.name}: {det_file}")

        det_map: Dict[int, Dict[str, np.ndarray]] = np.load(det_file, allow_pickle=True).item()
        frame_items = sorted(det_map.items(), key=lambda kv: _frame_sort_key(kv[0]))

        prev_boxes = np.zeros((0, 4), dtype=np.float32)
        prev_centers = np.zeros((0, 2), dtype=np.float32)
        for _, payload in frame_items:
            boxes = np.asarray(payload.get("boxes", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32).reshape(
                -1, 4
            )
            centers = _centers_xyxy(boxes)
            match = _greedy_match(prev_boxes, boxes, args.track_iou)
            vel = np.zeros((len(boxes), 2), dtype=np.float32)
            for curr_idx, prev_idx in enumerate(match):
                if prev_idx >= 0 and prev_idx < prev_centers.shape[0]:
                    vel[curr_idx] = centers[curr_idx] - prev_centers[prev_idx]
            velocity_frames.append(vel)
            prev_boxes = boxes
            prev_centers = centers

    out_dir = Path(args.output_root) / args.dataset_name / split_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "velocity.npy"
    np.save(out_path, np.array(velocity_frames, dtype=object), allow_pickle=True)
    print(f"[save] {out_path}  (frames={len(velocity_frames)})")


if __name__ == "__main__":
    main()
