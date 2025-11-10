#!/usr/bin/env python3
"""Extract per-detection deep features (ResNet50) from cached YOLO boxes."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


def _index_frame_files(frames_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not frames_dir.exists():
        return mapping
    for frame_path in frames_dir.iterdir():
        if frame_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            mapping[frame_path.stem] = frame_path.name
    return mapping


def _resolve_frame_name(mapping: Dict[str, str], key: str) -> Optional[str]:
    if key in mapping:
        return mapping[key]
    no_zero = key.lstrip("0")
    if no_zero and no_zero in mapping:
        return mapping[no_zero]
    pad6 = key.zfill(6)
    return mapping.get(pad6)


def _sorted_items(det_map: Dict[object, object]):
    def keyfun(raw_key):
        try:
            return (0, int(raw_key))
        except Exception:
            return (1, str(raw_key))

    return sorted(det_map.items(), key=lambda kv: keyfun(kv[0]))


def _crop(image: np.ndarray, box: np.ndarray, pad: float = 0.10) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1), (y2 - y1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    pw, ph = bw * (1.0 + pad), bh * (1.0 + pad)
    nx1 = max(0, int(cx - pw * 0.5))
    ny1 = max(0, int(cy - ph * 0.5))
    nx2 = min(w, int(cx + pw * 0.5))
    ny2 = min(h, int(cy + ph * 0.5))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return image[ny1:ny2, nx1:nx2]


def _build_model(device: torch.device) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1  # type: ignore[attr-defined]
        model = models.resnet50(weights=weights)
    except AttributeError:
        model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


def parse_args():
    ap = argparse.ArgumentParser("Extract ResNet50 features for every YOLO detection")
    ap.add_argument("--dataset_name", type=str, default="shanghaitech")
    ap.add_argument("--split", choices=["training", "testing"], required=True)
    ap.add_argument("--data_root", type=str, default="./data/cache/shanghaitech")
    ap.add_argument("--detections_root", type=str, default="./artifacts/detections")
    ap.add_argument("--frames_root", type=str, default=None,
                    help="Override frames directory (default=data_root/<split>/frames)")
    ap.add_argument("--out_root", type=str, default="./artifacts/features")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    return ap.parse_args()


def main():
    args = parse_args()
    split_tag = "train" if args.split == "training" else "test"

    data_root = Path(args.data_root)
    frames_root = Path(args.frames_root) if args.frames_root else data_root / args.split / "frames"
    detections_dir = Path(args.detections_root) / args.split
    out_dir = Path(args.out_root) / args.dataset_name / split_tag

    if not detections_dir.exists():
        raise FileNotFoundError(f"Detections directory missing: {detections_dir}")
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames directory missing: {frames_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = _build_model(device)

    videos = sorted([d for d in detections_dir.iterdir() if d.is_dir()])
    if not videos:
        raise FileNotFoundError(f"No detection folders found under {detections_dir}")

    deep_features: List[np.ndarray] = []

    with torch.no_grad():
        for video_dir in videos:
            det_path = video_dir / "detections.npy"
            if not det_path.exists():
                print(f"[warn] Missing detections file: {det_path}")
                continue
            det_map = np.load(det_path, allow_pickle=True).item()
            frame_items = _sorted_items(det_map)

            frame_dir = frames_root / video_dir.name
            frame_map = _index_frame_files(frame_dir)

            for frame_key, payload in frame_items:
                boxes = np.asarray(payload.get("boxes", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
                feats_frame = np.zeros((len(boxes), 2048), dtype=np.float32)
                if len(boxes) == 0:
                    deep_features.append(feats_frame)
                    continue

                frame_name = _resolve_frame_name(frame_map, str(frame_key))
                if frame_name is None:
                    deep_features.append(feats_frame)
                    continue
                image = cv2.imread(str(frame_dir / frame_name))
                if image is None:
                    deep_features.append(feats_frame)
                    continue

                batch_tensors: List[torch.Tensor] = []
                valid_indices: List[int] = []
                for idx, box in enumerate(boxes):
                    crop = _crop(image, box, pad=0.10)
                    if crop is None:
                        continue
                    batch_tensors.append(transform(crop))
                    valid_indices.append(idx)

                if not batch_tensors:
                    deep_features.append(feats_frame)
                    continue

                xb = torch.stack(batch_tensors, dim=0).to(device)
                for start in range(0, xb.size(0), args.batch_size):
                    end = min(start + args.batch_size, xb.size(0))
                    emb = model(xb[start:end])
                    emb = torch.flatten(emb, 1).cpu().numpy().astype(np.float32)
                    for offset, feat_vec in enumerate(emb):
                        frame_idx = valid_indices[start + offset]
                        feats_frame[frame_idx] = feat_vec

                deep_features.append(feats_frame)

    out_path = out_dir / "deep_features.npy"
    np.save(out_path, np.array(deep_features, dtype=object), allow_pickle=True)
    print(f"[save] {out_path}  (frames={len(deep_features)})")


if __name__ == "__main__":
    main()
