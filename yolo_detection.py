#!/usr/bin/env python3
"""
Ultralytics YOLOv5/YOLOv5u Detection Script for VAD Pipeline - ShanghaiTech
Python 3.7 Compatible (Ultralytics==8.0.151)
Extracts object detection boxes from video frames
Saves per-video dict: {frame_idx: {"boxes": (N,4), "classes": (N,), "scores": (N,)}}
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import argparse
from tqdm import tqdm
import json

def _safe_class_maps(model):
    """
    Ultralytics model.names can be list or dict depending on weights.
    Return: dict idx->name, dict name->idx (lowercased names for robust filtering).
    """
    names = model.names
    if isinstance(names, dict):
        idx2name = {int(k): str(v) for k, v in names.items()}
    else:  # list/tuple
        idx2name = {int(i): str(n) for i, n in enumerate(names)}
    name2idx = {v.lower(): k for k, v in idx2name.items()}
    return idx2name, name2idx

class UltralyticsYOLODetector:
    def __init__(self, model_path='yolov5su.pt', conf_threshold=0.30, iou_threshold=0.50,
                 device='auto', filter_classes=None, imgsz=640):
        """
        Args:
            model_path: 'yolov5su.pt' 권장(Py3.7 + Ultralytics 8.0.151 호환)
            conf_threshold: 0.25~0.35 권장
            iou_threshold: 보통 0.5
            device: 'auto' | 'cpu' | 'cuda:0' | 0 (GPU index)
            filter_classes: ['person'] 등 (None이면 전체 클래스)
            imgsz: 입력 리사이즈 (기본 640)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)

        if device in (None, 'auto'):
            self.device = (0 if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.idx2name, self.name2idx = _safe_class_maps(self.model)

        # 필터 클래스 인덱스 준비(소문자 매칭)
        self.filter_indices = None
        if filter_classes:
            want = [c.lower() for c in filter_classes]
            self.filter_indices = set([self.name2idx[c] for c in want if c in self.name2idx])

        print(f"[YOLO] weights={model_path}, device={self.device}, imgsz={self.imgsz}")
        print(f"[YOLO] conf={self.conf_threshold}, iou={self.iou_threshold}")
        if self.filter_indices is not None:
            keep_names = [self.idx2name[i] for i in sorted(self.filter_indices)]
            print(f"[YOLO] class filter: {keep_names}")

    def detect_frame(self, frame_path: Path):
        """
        Returns:
            boxes (N,4) float32 xyxy
            classes (N,) int32
            scores (N,) float32
        """
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.int32),
                    np.zeros((0,), np.float32))

        # BGR -> RGB
        rgb = frame[..., ::-1]

        # Ultralytics predict (8.0.151)
        res = self.model.predict(
            source=rgb,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.int32),
                    np.zeros((0,), np.float32))

        boxes = res[0].boxes.xyxy.cpu().numpy().astype(np.float32)
        classes = res[0].boxes.cls.cpu().numpy().astype(np.int32)
        scores = res[0].boxes.conf.cpu().numpy().astype(np.float32)

        # 클래스 필터링
        if self.filter_indices is not None:
            m = np.isin(classes, np.fromiter(self.filter_indices, dtype=np.int32))
            boxes, classes, scores = boxes[m], classes[m], scores[m]

        return boxes, classes, scores

    def process_video_frames(self, frames_dir, output_dir, video_name):
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            print(f"[warn] Frames directory not found: {frames_dir}")
            return None

        frame_files = sorted([f for f in frames_path.iterdir()
                              if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if len(frame_files) == 0:
            print(f"[warn] No frame files in {frames_dir}")
            return None

        video_output_dir = Path(output_dir) / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        all_det = {}
        total = 0
        for idx, frame_file in enumerate(tqdm(frame_files, desc=f"Detect {video_name}")):
            # 프레임 인덱스: 파일명이 숫자면 그대로, 아니면 enumerate 인덱스
            try:
                frame_idx = int(frame_file.stem)
            except ValueError:
                frame_idx = idx

            boxes, classes, scores = self.detect_frame(frame_file)
            all_det[frame_idx] = {
                "boxes": boxes,
                "classes": classes,
                "scores": scores
            }
            total += len(boxes)

        # 저장(.npy: 파이썬 dict pickled) — 로더에서 np.load(..., allow_pickle=True) 필요
        out_file = video_output_dir / "detections.npy"
        np.save(out_file, all_det, allow_pickle=True)

        # 메타데이터(JSON 직렬화 가능한 형태로)
        meta = {
            "video_name": video_name,
            "total_frames": len(frame_files),
            "total_detections": int(total),
            "avg_dets_per_frame": (float(total) / max(1, len(frame_files))),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "imgsz": self.imgsz,
            "filtered_class_indices": sorted(list(self.filter_indices)) if self.filter_indices is not None else None,
            "class_names": {str(k): v for k, v in self.idx2name.items()}
        }
        with open(video_output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[save] {out_file}  (avg {meta['avg_dets_per_frame']:.2f}/frame)")
        return all_det

def process_shanghaitech(data_root, output_root, model_path='yolov5su.pt',
                         conf=0.30, iou=0.50, imgsz=640,
                         filter_classes=['person'], splits=('training','testing')):
    det = UltralyticsYOLODetector(model_path=model_path,
                                  conf_threshold=conf,
                                  iou_threshold=iou,
                                  device='auto',
                                  filter_classes=filter_classes,
                                  imgsz=imgsz)

    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n=== Processing split: {split} ===")
        frames_root = data_root / split / "frames"
        if not frames_root.exists():
            print(f"[warn] Not found: {frames_root}")
            continue

        split_out = output_root / split
        split_out.mkdir(parents=True, exist_ok=True)

        video_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
        print(f"[info] {len(video_dirs)} videos")

        for vd in video_dirs:
            det.process_video_frames(frames_dir=vd, output_dir=split_out, video_name=vd.name)

    print(f"\n=== All done. Detections at: {output_root} ===")

def main():
    ap = argparse.ArgumentParser(description="Ultralytics YOLO detection (ShanghaiTech, Py3.7)")
    ap.add_argument("--data_root", type=str, default="./data/shanghaitech")
    ap.add_argument("--output_root", type=str, default="./artifacts/detections")
    ap.add_argument("--model", type=str, default="yolov5su.pt",
                    help="yolov5nu/su/mu/lu/xu.pt or yolov8n/s/m/l/x.pt (8.0.151 기준)")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--filter_classes", nargs="+", default=["person"])
    ap.add_argument("--splits", nargs="+", default=["training", "testing"])
    ap.add_argument("--no_filter", action="store_true")
    args = ap.parse_args()

    if args.no_filter:
        args.filter_classes = None

    print("=== YOLO Detection for VAD (ShanghaiTech) ===")
    print(vars(args))
    process_shanghaitech(data_root=args.data_root,
                         output_root=args.output_root,
                         model_path=args.model,
                         conf=args.conf,
                         iou=args.iou,
                         imgsz=args.imgsz,
                         filter_classes=args.filter_classes,
                         splits=tuple(args.splits))
    print("\nNext: feature_extraction.py / score_calibration.py / evaluate.py")
if __name__ == "__main__":
    main()