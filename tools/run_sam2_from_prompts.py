#!/usr/bin/env python3
"""
Run SAM2 (or any promptable segmentation model) from robust_prompts.json.

Python 3.7 호환. SAM2 호출부는 sam2_segment 스텁을 교체하면 되고,
--box-mask-fallback 옵션을 주면 SAM2 없이 박스 마스크만 저장하는 드라이런이 가능합니다.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def sam2_segment(image_rgb: np.ndarray, points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Placeholder for SAM2 inference.
    Replace this function with the API call you use in your SAM2 repo:

        return sam2_model.forward(image_rgb, points=points, boxes=boxes)

    The function must return a boolean or uint8 mask with shape (H, W).
    """
    raise NotImplementedError("Plug your SAM2 inference here or use --box-mask-fallback.")


def _guess_frame_path(frames_dir: Path, prompt: Dict[str, object]) -> Optional[Path]:
    frame_file = prompt.get("frame_file")
    if frame_file:
        candidate = frames_dir / frame_file
        if candidate.exists():
            return candidate
    key = str(prompt.get("frame_key"))
    candidates = []
    if key.isdigit():
        z = key.zfill(6)
        candidates.extend([f"{z}.jpg", f"{z}.png", f"{z}.jpeg"])
    candidates.extend([f"{key}.jpg", f"{key}.png", f"{key}.jpeg"])
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        candidate = frames_dir / name
        if candidate.exists():
            return candidate
    return None


def _box_mask(image_shape: Tuple[int, int], box: Sequence[float]) -> np.ndarray:
    h, w = image_shape
    x1, y1, x2, y2 = box
    x1 = max(0, min(w, int(np.floor(x1))))
    x2 = max(0, min(w, int(np.ceil(x2))))
    y1 = max(0, min(h, int(np.floor(y1))))
    y2 = max(0, min(h, int(np.ceil(y2))))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def parse_args():
    ap = argparse.ArgumentParser("SAM2 runner from robust prompts")
    ap.add_argument("--prompts_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--box-mask-fallback", action="store_true",
                    help="Skip SAM2 and save simple box masks (useful for dry-runs).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing mask files.")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.prompts_json, "r") as f:
        payload = json.load(f)
    frames_dir = Path(payload["frames_dir"])
    prompts = payload.get("prompts", [])
    if not prompts:
        print("[info] No prompts provided; nothing to do.")
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks_by_frame: Dict[str, np.ndarray] = {}
    cached_key: Optional[str] = None
    cached_image: Optional[np.ndarray] = None
    frame_file_map: Dict[str, str] = {}

    for prompt in tqdm(prompts, desc="Running SAM2"):
        frame_key = str(prompt.get("frame_key"))
        frame_path = _guess_frame_path(frames_dir, prompt)
        if frame_path is None:
            print(f"[warn] Frame path missing for key={frame_key}; skipping prompt.")
            continue
        frame_file_map.setdefault(frame_key, frame_path.name)
        if frame_key != cached_key:
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                print(f"[warn] Failed to read frame {frame_path}; skipping.")
                continue
            cached_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            cached_key = frame_key
        image_rgb = cached_image
        if image_rgb is None:
            continue

        if args.box_mask_fallback:
            mask_arr = _box_mask(image_rgb.shape[:2], prompt["bbox"])
        else:
            points = np.array([prompt["center"]], dtype=np.float32)
            boxes = np.array([prompt["bbox"]], dtype=np.float32)
            mask_arr = sam2_segment(image_rgb, points=points, boxes=boxes)
        mask_arr = np.asarray(mask_arr)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr.squeeze()
        mask_arr = (mask_arr > 0).astype(np.uint8)

        prev = masks_by_frame.get(frame_key)
        masks_by_frame[frame_key] = mask_arr if prev is None else np.maximum(prev, mask_arr)

    for frame_key, mask in masks_by_frame.items():
        fname = frame_file_map.get(frame_key, f"{frame_key}.png")
        out_path = out_dir / f"{Path(fname).stem}_mask.png"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path} exists (use --overwrite to replace).")
            continue
        cv2.imwrite(str(out_path), mask * 255)
    print(f"[save] Masks written to {out_dir}")


if __name__ == "__main__":
    main()
