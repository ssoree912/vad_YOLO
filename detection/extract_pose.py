#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract COCO-17 keypoint pose features aligned to YOLO detections (ShanghaiTech VAD).
- Input:  artifacts/detections/<split>/<video>/detections.npy  (from your YOLO script)
- Frames: data/shanghaitech/<split>/frames/<video>/*.jpg|png
- Output: artifacts/features/shanghaitech/<split>/pose.npy
Each frame i -> array of shape (N_i, 34): [x1,y1, x2,y2, ..., x17,y17] normalized to the YOLO bbox.
Non-person or unmatched -> zeros(34).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F

def _load_detections_map(det_file: Path) -> Dict:
    d = np.load(det_file, allow_pickle=True).item()
    # ensure arrays
    for k, v in d.items():
        v["boxes"]   = np.asarray(v["boxes"],   dtype=np.float32).reshape(-1, 4)
        v["classes"] = np.asarray(v["classes"], dtype=np.int32).reshape(-1,)
        v["scores"]  = np.asarray(v["scores"],  dtype=np.float32).reshape(-1,)
    return d

def _index_frames(frames_dir: Path) -> Dict[str, str]:
    m = {}
    if not frames_dir.exists():
        return m
    for p in frames_dir.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            m[p.stem] = p.name
    return m

def _resolve_frame_name(mapping: Dict[str, str], key: str) -> Optional[str]:
    if key in mapping:
        return mapping[key]
    k = key.lstrip("0")
    if k and k in mapping:
        return mapping[k]
    z = key.zfill(6)
    if z in mapping:
        return mapping[z]
    return None

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
    return float(inter / (area_a + area_b - inter))

def _match_person_boxes(yolo_boxes: np.ndarray,
                        yolo_classes: np.ndarray,
                        kp_boxes: np.ndarray,
                        iou_thr: float,
                        person_class_idx: int) -> List[int]:
    """
    For each YOLO box idx -> index of matched keypoint box; -1 if none.
    Greedy per-YOLO-box best IoU match (no exclusivity required, but typical).
    """
    matches = [-1] * len(yolo_boxes)
    if len(kp_boxes) == 0:
        return matches
    for i, (box, cls) in enumerate(zip(yolo_boxes, yolo_classes)):
        if cls != person_class_idx:
            continue
        best_j, best_v = -1, 0.0
        for j, kpb in enumerate(kp_boxes):
            v = _iou_xyxy(box, kpb)
            if v > best_v:
                best_v, best_j = v, j
        if best_v >= iou_thr:
            matches[i] = best_j
    return matches

def _pose_to_34vec(keypoints_xyv: np.ndarray, yolo_box: np.ndarray,
                   vis_min: float = 1.0) -> np.ndarray:
    """
    keypoints_xyv: (17, 3) with (x,y,vis), vis in {0,1,2}
    yolo_box: [x1,y1,x2,y2]
    Returns (34,) normalized by bbox (x': (x-x1)/w, y': (y-y1)/h).
    keypoints with vis < vis_min are set to 0.
    """
    x1, y1, x2, y2 = yolo_box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    out = np.zeros((34,), dtype=np.float32)
    if keypoints_xyv.shape != (17, 3):
        return out
    for k in range(17):
        x, y, v = keypoints_xyv[k]
        if v < vis_min:
            xi = yi = 0.0
        else:
            xi = (float(x) - x1) / w
            yi = (float(y) - y1) / h
            # clamp
            xi = float(np.clip(xi, 0.0, 1.0))
            yi = float(np.clip(yi, 0.0, 1.0))
        out[2 * k + 0] = xi
        out[2 * k + 1] = yi
    return out

@torch.no_grad()
def _run_keypointrcnn(model, image_bgr: np.ndarray, conf_thr: float = 0.5):
    """
    Returns:
      kp_boxes: (M, 4) xyxy
      kp_keypoints: (M, 17, 3)  (x,y,vis)
      kp_scores: (M,)
    """
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    t = F.to_tensor(img).to(next(model.parameters()).device)
    pred = model([t])[0]
    boxes = pred["boxes"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    keypoints = pred["keypoints"].detach().cpu().numpy()  # (M, 17, 3)

    m = scores >= conf_thr
    return boxes[m], keypoints[m], scores[m]

def build_pose_for_split(data_root: Path,
                         detections_root: Path,
                         frames_root: Path,
                         out_path: Path,
                         person_class_name: str = "person",
                         iou_match_thr: float = 0.5,
                         kp_conf_thr: float = 0.5,
                         device: str = "cuda"):
    """
    Walk through all videos (using detections ordering), build per-frame (N_i, 34) arrays,
    and save a big list (one entry per frame across all videos) to out_path (pose.npy).
    """
    # Try to read class mapping from any metadata.json under detections_root
    # (assumes uniform class mapping across videos)
    class_names_map = None
    for meta in detections_root.rglob("metadata.json"):
        with open(meta, "r") as f:
            md = json.load(f)
        class_names_map = md.get("class_names", None)
        if class_names_map:
            break
    if class_names_map is None:
        # fallback to COCO default index for 'person' (0 in YOLOv5 models)
        name2idx = {"person": 0}
    else:
        # metadata.json stored { "0": "person", ... }
        idx2name = {int(k): v for k, v in class_names_map.items()}
        name2idx = {v: k for k, v in idx2name.items()}

    person_idx = int(name2idx.get(person_class_name, 0))
    print(f"[info] person_class index = {person_idx}")

    # keypoint model
    print("[load] torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)")
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()
    if device and torch.cuda.is_available() and device != "cpu":
        model = model.to("cuda")
        print("[info] device: cuda")
    else:
        print("[info] device: cpu")

    all_frames_pose: List[np.ndarray] = []
    video_dirs = sorted([d for d in detections_root.iterdir() if d.is_dir()])
    print(f"[info] videos: {len(video_dirs)}")

    total_frames = 0
    for vdir in video_dirs:
        det_map = _load_detections_map(vdir / "detections.npy")
        # Build frame filename lookup
        frm_dir = frames_root / vdir.name
        f_lookup = _index_frames(frm_dir)

        # Iterate frames in the same order as detections
        items = sorted(det_map.items(),
                       key=lambda kv: (0, f"{int(kv[0]):010d}") if str(kv[0]).isdigit() else (1, str(kv[0])))
        for frame_key, entry in items:
            fname = _resolve_frame_name(f_lookup, str(frame_key))
            if fname is None:
                # If missing, push zeros for all boxes (keeps alignment)
                nb = entry["boxes"].shape[0]
                all_frames_pose.append(np.zeros((nb, 34), dtype=np.float32))
                continue

            img_path = frm_dir / fname
            img = cv2.imread(str(img_path))
            if img is None:
                nb = entry["boxes"].shape[0]
                all_frames_pose.append(np.zeros((nb, 34), dtype=np.float32))
                continue

            # run keypoint detector on full frame
            kp_boxes, kp_kps, _ = _run_keypointrcnn(model, img, conf_thr=kp_conf_thr)

            # match YOLO -> kp
            y_boxes, y_cls = entry["boxes"], entry["classes"]
            matches = _match_person_boxes(y_boxes, y_cls, kp_boxes, iou_thr=iou_match_thr,
                                          person_class_idx=person_idx)

            # build (N, 34)
            feats = np.zeros((len(y_boxes), 34), dtype=np.float32)
            for i, j in enumerate(matches):
                if y_cls[i] != person_idx or j < 0:
                    continue
                feats[i] = _pose_to_34vec(kp_kps[j], y_boxes[i], vis_min=1.0)
            all_frames_pose.append(feats)
        total_frames += len(items)
        print(f"[done] {vdir.name}: {len(items)} frames")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.array(all_frames_pose, dtype=object))
    print(f"[save] pose list → {out_path}  (frames={total_frames})")
    print("[note] Shape per frame = (N_boxes, 34). Non-person/unmatched boxes are zeros.")
    return out_path

def parse_args():
    ap = argparse.ArgumentParser("Extract YOLO-aligned pose.npy (torchvision keypoint R-CNN)")
    ap.add_argument("--data_root", type=str, default="data/shanghaitech")
    ap.add_argument("--split", type=str, default="training", choices=["training", "testing"])
    ap.add_argument("--detections_root", type=str, required=True,
                    help="artifacts/detections/<split>")
    ap.add_argument("--frames_root", type=str, default=None,
                    help="defaults to data_root/<split>/frames")
    ap.add_argument("--out_root", type=str, default="artifacts/features/shanghaitech",
                    help="will save to <out_root>/<split>/pose.npy")
    ap.add_argument("--person_name", type=str, default="person")
    ap.add_argument("--iou_match_thr", type=float, default=0.5)
    ap.add_argument("--kp_conf_thr", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    frames_root = Path(args.frames_root) if args.frames_root else (data_root / args.split / "frames")
    detections_root = Path(args.detections_root)
    out_path = Path(args.out_root) / args.split / "pose.npy"

    print("=== Pose extraction (torchvision Keypoint R-CNN) ===")
    print({
        "data_root": str(data_root),
        "split": args.split,
        "frames_root": str(frames_root),
        "detections_root": str(detections_root),
        "out_path": str(out_path),
        "iou_match_thr": args.iou_match_thr,
        "kp_conf_thr": args.kp_conf_thr,
        "device": args.device,
    })

    build_pose_for_split(
        data_root=data_root,
        detections_root=detections_root,
        frames_root=frames_root,
        out_path=out_path,
        person_class_name=args.person_name,
        iou_match_thr=args.iou_match_thr,
        kp_conf_thr=args.kp_conf_thr,
        device=args.device,
    )

if __name__ == "__main__":
    main()