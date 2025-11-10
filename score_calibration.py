import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import faiss

def _maybe_gpu_index(index_cpu, use_gpu: bool, device: int = 0):
    if not use_gpu:
        return index_cpu
    try:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, device, index_cpu)
    except Exception as exc:
        print(f"[warn] GPU FAISS unavailable ({exc}); falling back to CPU")
        return index_cpu


def compute_calibration_parameters(args, root):
    train_clip_lengths = np.load(os.path.join(root, args.dataset_name, 'train_clip_lengths.npy'))

    feats_root = Path(args.features_root) / args.dataset_name

    train_poses = np.load(feats_root / 'train' / 'pose.npy', allow_pickle=True)
    train_deep_features = np.load(feats_root / 'train' / 'deep_features.npy', allow_pickle=True)


    all_ranges = np.arange(0, len(train_deep_features))
    features_scores = []
    pose_scores = []

    prev = 0
    for i in tqdm(range(len(train_clip_lengths))):
        cur = train_clip_lengths[i]
        cur_video_range = np.arange(prev, cur)
        complement_indices = np.setdiff1d(all_ranges, cur_video_range)

        rest_deep_features = train_deep_features[complement_indices]
        rest_deep_features = np.concatenate(rest_deep_features, 0)

        cur_deep_features = train_deep_features[cur_video_range]
        cur_deep_features = np.concatenate(cur_deep_features, 0)

        index = faiss.IndexFlatL2(rest_deep_features.shape[1])
        index_deep_features = _maybe_gpu_index(index, args.faiss_use_gpu, args.faiss_device)
        index_deep_features.add(rest_deep_features.astype(np.float32))

        D, I = index_deep_features.search(cur_deep_features.astype(np.float32), 1)
        score_deep_features = np.mean(D, axis=1)
        features_scores.append(score_deep_features)

        rest_poses = train_poses[complement_indices]
        without_empty_frames = []
        for i in tqdm(range(len(rest_poses))):
            if len(rest_poses[i]):
                without_empty_frames.append(rest_poses[i])
        rest_poses = np.concatenate(without_empty_frames, 0)
        # rest_poses = np.concatenate(rest_poses, 0)

        cur_poses = train_poses[cur_video_range]
        without_empty_frames = []
        for i in tqdm(range(len(cur_poses))):
            if len(cur_poses[i]):
                without_empty_frames.append(cur_poses[i])
        cur_poses = np.concatenate(without_empty_frames, 0)
        # cur_poses = np.concatenate(cur_poses, 0)

        index = faiss.IndexFlatL2(rest_poses.shape[1])
        index_poses = _maybe_gpu_index(index, args.faiss_use_gpu, args.faiss_device)
        index_poses.add(rest_poses.astype(np.float32))

        D, I = index_poses.search(cur_poses.astype(np.float32), 1)
        score_poses = np.mean(D, axis=1)
        pose_scores.append(score_poses)

        prev = cur

    features_scores = np.concatenate(features_scores, 0)
    pose_scores = np.concatenate(pose_scores, 0)

    np.save(feats_root / 'train_pose_scores.npy', pose_scores)
    np.save(feats_root / 'train_deep_features_scores.npy', features_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    parser.add_argument("--data_root", type=str, default="data/cache", help="dataset root directory")
    parser.add_argument("--features_root", type=str, default="artifacts/features",
                        help="Root directory storing extracted features (e.g., data/cache/extracted)")
    parser.add_argument("--faiss_use_gpu", action="store_true", help="Enable FAISS GPU indices")
    parser.add_argument("--faiss_device", type=int, default=0, help="GPU device id for FAISS indices")
    args = parser.parse_args()
    root = args.data_root
    compute_calibration_parameters(args, root)
