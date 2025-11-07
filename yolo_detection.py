#!/usr/bin/env python3
"""
Ultralytics YOLOv5 Detection Script for VAD Pipeline - ShanghaiTech Dataset
Python 3.7+ Compatible
Extracts object detection boxes from video frames
Saves in [video][frame] -> (N_i, 4) structure for VAD processing
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

class UltralyticsYOLOv5Detector:
    def __init__(self, model_path='yolov5s.pt', conf_threshold=0.25, iou_threshold=0.5, device='auto', filter_classes=None):
        """
        Initialize Ultralytics YOLOv5 detector for VAD pipeline
        
        Args:
            model_path: YOLOv5 model path (yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt)
            conf_threshold: Confidence threshold for detection (recommended: 0.25-0.35)
            iou_threshold: IoU threshold for NMS (recommended: 0.5)
            device: Device to run inference ('auto', 'cpu', 'cuda:0', etc.)
            filter_classes: List of class names to keep (e.g., ['person'] for VAD)
        """
        # Load Ultralytics YOLOv5 model
        self.model = YOLO(model_path)
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.filter_classes = filter_classes
        
        # Get class names mapping (COCO classes)
        self.class_names = self.model.names
        self.class_to_idx = {v: k for k, v in self.class_names.items()}
        
        print(f"Ultralytics YOLOv5 Model loaded: {model_path}")
        print(f"Device: {self.device}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
        if filter_classes:
            print(f"Filtering classes: {filter_classes}")
    
    def detect_frame(self, frame_path):
        """
        Detect objects in a single frame using Ultralytics YOLOv5
        
        Returns:
            boxes: (N, 4) array of bounding boxes in xyxy format
            classes: (N,) array of class indices
            scores: (N,) array of confidence scores
        """
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return np.array([]).reshape(0, 4), np.array([]), np.array([])
        
        # Run Ultralytics YOLOv5 detection
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, 
                           device=self.device, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return np.array([]).reshape(0, 4), np.array([]), np.array([])
        
        # Extract detection results
        boxes = results[0].boxes.xyxy.cpu().numpy()  # xyxy format
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        scores = results[0].boxes.conf.cpu().numpy()
        
        # Filter by class if specified (e.g., person only for VAD)
        if self.filter_classes:
            filter_indices = []
            for class_name in self.filter_classes:
                if class_name in self.class_to_idx:
                    class_idx = self.class_to_idx[class_name]
                    indices = np.where(classes == class_idx)[0]
                    filter_indices.extend(indices)
            
            if len(filter_indices) > 0:
                filter_indices = np.array(filter_indices)
                boxes = boxes[filter_indices]
                classes = classes[filter_indices]
                scores = scores[filter_indices]
            else:
                boxes = np.array([]).reshape(0, 4)
                classes = np.array([])
                scores = np.array([])
        
        return boxes, classes, scores
    
    def process_video_frames(self, frames_dir, output_dir, video_name):
        """
        Process all frames in a video directory
        Output format: [video][frame] -> (N_i, 4) structure for VAD
        
        Args:
            frames_dir: Directory containing frame images
            output_dir: Directory to save detection results  
            video_name: Name of the video (for organizing output)
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            print(f"Frames directory not found: {frames_dir}")
            return
        
        # Get sorted list of frame files
        frame_files = sorted([f for f in frames_path.iterdir() 
                            if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        
        if len(frame_files) == 0:
            print(f"No frame files found in {frames_dir}")
            return
        
        print(f"Processing {len(frame_files)} frames for video: {video_name}")
        
        # Create output directory
        video_output_dir = Path(output_dir) / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each frame and store in VAD format
        all_detections = {}
        total_detections = 0
        
        for frame_file in tqdm(frame_files, desc=f"Detecting {video_name}"):
            # Extract frame index from filename
            try:
                frame_idx = int(frame_file.stem)
            except ValueError:
                # If filename is not a number, use index in sorted list
                frame_idx = frame_files.index(frame_file)
            
            boxes, classes, scores = self.detect_frame(frame_file)
            
            # Store in VAD format: [video][frame] -> (N_i, 4)
            all_detections[frame_idx] = {
                'boxes': boxes,      # (N_i, 4) xyxy format for VAD
                'classes': classes,  # (N_i,) class indices
                'scores': scores     # (N_i,) confidence scores
            }
            total_detections += len(boxes)
        
        # Save detections as .npy file (VAD pipeline format)
        detection_file = video_output_dir / 'detections.npy'
        np.save(detection_file, all_detections)
        
        # Save metadata for debugging and analysis
        metadata = {
            'video_name': video_name,
            'total_frames': len(frame_files),
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / len(frame_files) if len(frame_files) > 0 else 0,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'filter_classes': self.filter_classes,
            'class_names': self.class_names
        }
        
        metadata_file = video_output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved detections: {detection_file}")
        print(f"Total detections: {total_detections} ({total_detections/len(frame_files):.1f} per frame)")
        
        return all_detections

def process_shanghaitech(data_root, output_root, model_path='yolov5s.pt', 
                        conf=0.3, iou=0.5, filter_classes=['person'], 
                        splits=['training', 'testing']):
    """
    Process ShanghaiTech dataset for VAD pipeline using Ultralytics YOLOv5
    Expected structure: data_root/{split}/frames/{video_name}/*.jpg
    Output: output_root/{split}/{video_name}/detections.npy
    """
    detector = UltralyticsYOLOv5Detector(model_path=model_path, conf_threshold=conf, 
                                        iou_threshold=iou, filter_classes=filter_classes)
    
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    for split in splits:
        print(f"\n=== Processing {split} split ===")
        
        # Check frames directory
        frames_root = data_root / split / 'frames'
        if not frames_root.exists():
            print(f"Frames directory not found: {frames_root}")
            print("Please extract frames first using extract_shanghaitech_frames.py")
            continue
        
        split_output_dir = output_root / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all video directories
        video_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
        
        print(f"Found {len(video_dirs)} videos in {split} split")
        
        for video_dir in video_dirs:
            video_name = video_dir.name
            print(f"\nProcessing video: {video_name}")
            
            detector.process_video_frames(
                frames_dir=video_dir,
                output_dir=split_output_dir,
                video_name=video_name
            )
        
        print(f"Completed {split} split")
    
    print(f"\n=== All detections saved to {output_root} ===")

def main():
    parser = argparse.ArgumentParser(description='Ultralytics YOLOv5 Detection for VAD Pipeline (ShanghaiTech)')
    parser.add_argument('--data_root', type=str, default='./shanghaitech',
                       help='Root directory of ShanghaiTech dataset')
    parser.add_argument('--output_root', type=str, default='./artifacts/detections',
                       help='Output directory for detections')
    parser.add_argument('--model', type=str, default='yolov5s.pt',
                       help='Ultralytics YOLOv5 model (yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt)')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold (0.25-0.35 recommended for VAD)')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS')
    parser.add_argument('--filter_classes', nargs='+', default=['person'],
                       help='Classes to filter (person recommended for VAD)')
    parser.add_argument('--splits', nargs='+', default=['training', 'testing'],
                       help='Dataset splits to process')
    parser.add_argument('--no_filter', action='store_true',
                       help='Disable class filtering (detect all classes)')
    
    args = parser.parse_args()
    
    if args.no_filter:
        args.filter_classes = None
    
    print("=== Ultralytics YOLOv5 Detection for VAD Pipeline (ShanghaiTech) ===")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Model: {args.model}")
    print(f"Confidence: {args.conf}")
    print(f"IoU: {args.iou}")
    print(f"Filter classes: {args.filter_classes}")
    print(f"Splits: {args.splits}")
    
    process_shanghaitech(
        data_root=args.data_root,
        output_root=args.output_root,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        filter_classes=args.filter_classes,
        splits=args.splits
    )
    
    print("\n=== Detection completed ===")
    print("Next step: Run feature extraction (RAFT, Pose) for VAD training")

if __name__ == "__main__":
    main()