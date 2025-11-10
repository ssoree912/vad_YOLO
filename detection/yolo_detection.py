#!/usr/bin/env python3
"""
YOLOv5 (Torch Hub) Detection for VAD Pipeline - ShanghaiTech
Python 3.7 compatible (no ultralytics pip dep)
Saves per-video dict: {frame_idx: {"boxes": (N,4), "classes": (N,), "scores": (N,)}}
"""

import cv2
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse

def _names_from_model(model):
    # model.names: list or dict
    names = model.names
    if isinstance(names, dict):
        idx2name = {int(k): str(v) for k, v in names.items()}
    else:
        idx2name = {i: str(n) for i, n in enumerate(names)}
    name2idx = {v.lower(): k for k, v in idx2name.items()}
    return idx2name, name2idx

class YOLOv5HubDetector:
    def __init__(self, model_name='yolov5s', weights=None,
                 conf_threshold=0.30, iou_threshold=0.50,
                 device='auto', filter_classes=None, imgsz=640):
        """
        Args:
          model_name: 'yolov5n/s/m/l/x'
          weights:   None면 COCO pretrained, 또는 커스텀 pt 경로
        """
        if device in (None, 'auto'):
            self.device = 0 if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Torch Hub에서 YOLOv5 로드
        # 인터넷 불가 환경이면: 미리 `git clone https://github.com/ultralytics/yolov5 -b v6.2` 후
        # torch.hub.load('local/yolov5', 'yolov5s', source='local', pretrained=True) 형태로 사용 가능
        if weights is None:
            self.model = torch.hub.load('ultralytics/yolov5:v6.2', model_name, pretrained=True, trust_repo=True)
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

        self.model.to(self.device)
        self.model.conf = float(conf_threshold)
        self.model.iou = float(iou_threshold)

        self.imgsz = int(imgsz)
        self.idx2name, self.name2idx = _names_from_model(self.model)

        # 클래스 필터: e.g., ['person']
        self.filter_indices = None
        if filter_classes:
            want = [c.lower() for c in filter_classes]
            self.filter_indices = set([self.name2idx[c] for c in want if c in self.name2idx])

        print(f"[YOLOv5Hub] model={model_name or weights}, device={self.device}, imgsz={self.imgsz}")
        print(f"[YOLOv5Hub] conf={self.model.conf}, iou={self.model.iou}")
        if self.filter_indices is not None:
            keep_names = [self.idx2name[i] for i in sorted(self.filter_indices)]
            print(f"[YOLOv5Hub] class filter: {keep_names}")

    def detect_frame(self, frame_path: Path):
        img = cv2.imread(str(frame_path))
        if img is None:
            return (np.zeros((0,4), np.float32),
                    np.zeros((0,), np.int32),
                    np.zeros((0,), np.float32))

        # YOLOv5는 numpy BGR도 처리합니다.
        with torch.no_grad():
            results = self.model(img, size=self.imgsz)

        if results is None or len(results.xyxy) == 0:
            return (np.zeros((0,4), np.float32),
                    np.zeros((0,), np.int32),
                    np.zeros((0,), np.float32))

        pred = results.xyxy[0].cpu().numpy()  # [N,6]: x1,y1,x2,y2,conf,cls
        if pred.size == 0:
            return (np.zeros((0,4), np.float32),
                    np.zeros((0,), np.int32),
                    np.zeros((0,), np.float32))

        boxes  = pred[:, :4].astype(np.float32)
        scores = pred[:, 4].astype(np.float32)
        classes= pred[:, 5].astype(np.int32)

        if self.filter_indices is not None:
            m = np.isin(classes, np.fromiter(self.filter_indices, dtype=np.int32))
            boxes, classes, scores = boxes[m], classes[m], scores[m]

        return boxes, classes, scores

    def process_video_frames(self, frames_dir, output_dir, video_name):
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            print(f"[warn] Frames not found: {frames_dir}")
            return None

        frame_files = sorted([f for f in frames_path.iterdir()
                              if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if not frame_files:
            print(f"[warn] No frames in: {frames_dir}")
            return None

        out_dir = Path(output_dir) / video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        all_det = {}
        total = 0
        for idx, f in enumerate(tqdm(frame_files, desc=f"Detect {video_name}")):
            try:
                frame_idx = int(f.stem)
            except ValueError:
                frame_idx = idx

            boxes, classes, scores = self.detect_frame(f)
            all_det[frame_idx] = {"boxes": boxes, "classes": classes, "scores": scores}
            total += len(boxes)

        np.save(out_dir / "detections.npy", all_det, allow_pickle=True)

        meta = {
            "video_name": video_name,
            "total_frames": len(frame_files),
            "total_detections": int(total),
            "avg_dets_per_frame": (float(total) / max(1, len(frame_files))),
            "imgsz": self.imgsz,
            "conf_threshold": float(self.model.conf),
            "iou_threshold": float(self.model.iou),
            "filtered_class_indices": sorted(list(self.filter_indices)) if self.filter_indices else None,
            "class_names": {str(k): v for k, v in self.idx2name.items()},
        }
        with open(out_dir / "metadata.json", "w") as fp:
            json.dump(meta, fp, indent=2)

        print(f"[save] {out_dir/'detections.npy'}  (avg {meta['avg_dets_per_frame']:.2f}/frame)")
        return all_det

def process_shanghaitech(data_root, output_root, model='yolov5s', weights=None,
                         conf=0.30, iou=0.50, imgsz=640,
                         filter_classes=['person'],
                         splits=('training','testing')):
    det = YOLOv5HubDetector(model_name=model, weights=weights,
                            conf_threshold=conf, iou_threshold=iou,
                            device='auto', filter_classes=filter_classes, imgsz=imgsz)

    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n=== Process split: {split} ===")
        frames_root = data_root / split / "frames"
        if not frames_root.exists():
            print(f"[warn] not found: {frames_root}")
            continue

        split_out = output_root / split
        split_out.mkdir(parents=True, exist_ok=True)

        videos = sorted([d for d in frames_root.iterdir() if d.is_dir()])
        print(f"[info] videos: {len(videos)}")
        for vd in videos:
            det.process_video_frames(vd, split_out, vd.name)

def main():
    ap = argparse.ArgumentParser("YOLOv5 (Torch Hub) detection for VAD (ShanghaiTech)")
    ap.add_argument("--data_root", type=str, default="./data/cache/shanghaitech")
    ap.add_argument("--output_root", type=str, default="./artifacts/detections")
    ap.add_argument("--model", type=str, default="yolov5s",
                    help="yolov5n/s/m/l/x (COCO pretrained). Use --weights for custom .pt")
    ap.add_argument("--weights", type=str, default=None,
                    help="custom weights .pt path (overrides --model)")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--filter_classes", nargs="+", default=["person"])
    ap.add_argument("--splits", nargs="+", default=["training", "testing"])
    ap.add_argument("--no_filter", action="store_true")
    args = ap.parse_args()

    if args.no_filter:
        args.filter_classes = None

    print("=== YOLOv5 Torch Hub Detection ===")
    print(vars(args))
    process_shanghaitech(
        data_root=args.data_root,
        output_root=args.output_root,
        model=args.model,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        filter_classes=args.filter_classes,
        splits=tuple(args.splits)
    )
    print("\nNext: feature_extraction.py / score_calibration.py / evaluate.py")

if __name__ == "__main__":
    main()
