#!/usr/bin/env python3
"""Generate frame+mask overlay images for every video/prompt automatically."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import cv2
import numpy as np


def _frame_candidates(frame_key: str) -> Sequence[str]:
    stem = frame_key.strip()
    if not stem:
        return []
    candidates = []
    if stem.isdigit():
        z = stem.zfill(6)
        candidates.extend([f"{z}.jpg", f"{z}.png", f"{z}.jpeg"])
    candidates.extend([f"{stem}.jpg", f"{stem}.png", f"{stem}.jpeg"])
    return candidates


def _build_frame_lookup(prompts: Iterable[Dict[str, object]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for prompt in prompts:
        fname = prompt.get("frame_file")
        if not fname:
            continue
        stem = Path(str(fname)).stem
        lookup.setdefault(stem, str(fname))
    return lookup


def _resolve_frame_path(frames_dir: Path, frame_key: str, lookup: Dict[str, str]) -> Optional[Path]:
    stem = lookup.get(frame_key)
    if stem:
        candidate = frames_dir / stem
        if candidate.exists():
            return candidate
    # fallback: try guesses derived from key/stem
    for name in _frame_candidates(frame_key):
        candidate = frames_dir / name
        if candidate.exists():
            return candidate
    return None


def _colorize_mask(mask: np.ndarray, colormap: int) -> np.ndarray:
    mask_norm = (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)
    mask_uint8 = (mask_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(mask_uint8, colormap)
    return colored


def _blend(frame: np.ndarray, colored_mask: np.ndarray, alpha: float) -> np.ndarray:
    beta = 1.0 - alpha
    overlay = cv2.addWeighted(frame, beta, colored_mask, alpha, 0)
    return overlay


def _iter_prompt_jsons(root: Path, split: Optional[str]) -> Iterable[Path]:
    if split:
        root = root / split
    return sorted(root.rglob("robust_prompts.json"))


def render_overlays(args) -> None:
    prompts_root = Path(args.prompts_root)
    masks_root = Path(args.masks_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    json_files = list(_iter_prompt_jsons(prompts_root, args.split))
    if not json_files:
        print(f"[warn] No robust_prompts.json files found under {prompts_root}")
        return

    processed = 0
    for json_path in json_files:
        rel = json_path.relative_to(prompts_root)
        video_dir = rel.parent  # e.g., training/13_007
        if args.videos and video_dir.name not in args.videos:
            continue
        masks_dir = masks_root / video_dir
        if not masks_dir.exists():
            print(f"[warn] Missing masks directory: {masks_dir}")
            continue
        with open(json_path, "r") as f:
            payload = json.load(f)
        frames_dir = Path(payload["frames_dir"])
        prompts = payload.get("prompts", [])
        if not prompts:
            continue
        frame_lookup = _build_frame_lookup(prompts)
        out_dir = out_root / video_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        mask_files = sorted(masks_dir.glob("*_mask.png"))
        if not mask_files:
            print(f"[warn] No mask files found in {masks_dir}")
            continue

        for mask_path in mask_files:
            stem = mask_path.stem.replace("_mask", "")
            frame_path = _resolve_frame_path(frames_dir, stem, frame_lookup)
            if frame_path is None:
                print(f"[warn] Could not locate frame for mask {mask_path}")
                continue
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"[warn] Failed to read frame {frame_path}")
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[warn] Failed to read mask {mask_path}")
                continue
            colored = _colorize_mask(mask, args.colormap)
            overlay = _blend(frame, colored, args.alpha)
            out_file = out_dir / f"{frame_path.stem}_overlay.png"
            cv2.imwrite(str(out_file), overlay)
        processed += 1

    print(f"[done] Overlays generated for {processed} videos. Output root: {out_root}")


def parse_args():
    ap = argparse.ArgumentParser("Render colored masks over the original frames for all prompts")
    ap.add_argument("--prompts_root", type=str, default="artifacts/anomaly_prompts")
    ap.add_argument("--masks_root", type=str, default="artifacts/sam2_masks")
    ap.add_argument("--out_root", type=str, default="artifacts/overlays")
    ap.add_argument("--split", type=str, default=None, choices=["training", "testing", None],
                    help="Limit to a specific split (default: both)")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Optional subset of video folder names to process")
    ap.add_argument("--alpha", type=float, default=0.35,
                    help="Overlay strength (0=no mask, 1=mask only)")
    ap.add_argument("--colormap", type=int, default=cv2.COLORMAP_JET,
                    help="OpenCV colormap code (default: JET)")
    return ap.parse_args()


def main():
    args = parse_args()
    render_overlays(args)


if __name__ == "__main__":
    main()
