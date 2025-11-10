#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make anomaly prompts (robust SAM2 prompts) by combining YOLO detections with Attribute-VAD scores.
Memory-safe version:
 - Skips FAISS build/search for disabled features (pose/deep)
 - Optional GPU FAISS with FP16 + temp memory cap
 - Optional IVF/IVFPQ indices
"""

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # CPU-only or no FAISS path (will error if pose/deep enabled without FAISS)


# ----------------------------- Small utils -----------------------------

def _log(msg: str) -> None:
    print(f"[step] {msg}", flush=True)

def _warn(msg: str) -> None:
    print(f"[warn] {msg}", flush=True)

def _load_object_list(path: Path) -> List[np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return [np.asarray(x) for x in arr]

def _concat_non_empty(arrays: Sequence[np.ndarray]) -> np.ndarray:
    pieces = [np.asarray(a) for a in arrays if len(a)]
    if not pieces:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(pieces, axis=0)

def _make_placeholder_list(n_items: int, dim: int) -> List[np.ndarray]:
    # returns n_items empty arrays so indexing lengths line up
    return [np.zeros((0, dim), dtype=np.float32) for _ in range(n_items)]

def _normalize(scores: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float32)
    denom = max(max_v - min_v, 1e-6)
    return np.clip((scores - min_v) / denom, 0.0, 1.0).astype(np.float32)

def _ensure_2d(array: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(array)
    if arr.size == 0:
        return np.zeros((0, dim), dtype=np.float32)
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def _align_pose_scores(
    pose_scores: np.ndarray, classes: np.ndarray, num_boxes: int, person_class: int
) -> np.ndarray:
    aligned = np.zeros((num_boxes,), dtype=np.float32)
    if pose_scores.size == 0 or num_boxes == 0:
        return aligned
    if pose_scores.shape[0] == num_boxes:
        aligned[:] = pose_scores
        return aligned
    person_mask = classes == person_class
    person_count = int(np.count_nonzero(person_mask))
    if pose_scores.shape[0] == person_count:
        aligned[person_mask] = pose_scores
        return aligned
    # fallback: left-align
    limit = min(pose_scores.shape[0], num_boxes)
    aligned[:limit] = pose_scores[:limit]
    return aligned

def _build_clip_ranges(lengths: np.ndarray) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    prev = 0
    for L in lengths.astype(int).tolist():
        ranges.append((prev, prev + L))
        prev += L
    return ranges

def _frame_sort_key(key) -> Tuple[int, str]:
    try:
        return (0, f"{int(key):010d}")
    except (TypeError, ValueError):
        return (1, str(key))

def _center_xyxy(box: np.ndarray) -> List[float]:
    return [float(0.5 * (box[0] + box[2])), float(0.5 * (box[1] + box[3]))]

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

def _greedy_match(prev_boxes: np.ndarray, curr_boxes: np.ndarray, iou_thr: float) -> List[int]:
    if prev_boxes.size == 0 or curr_boxes.size == 0:
        return [-1] * len(curr_boxes)
    assigned: List[int] = [-1] * len(curr_boxes)
    used_prev = set()
    for ci, cbox in enumerate(curr_boxes):
        best_i, best_v = -1, 0.0
        for pi, pbox in enumerate(prev_boxes):
            if pi in used_prev:
                continue
            iou = _iou_xyxy(pbox, cbox)
            if iou > best_v:
                best_v, best_i = iou, pi
        if best_v >= iou_thr:
            assigned[ci] = best_i
            used_prev.add(best_i)
    return assigned

def _build_tracks_per_video(frames_boxes: List[np.ndarray], iou_thr: float) -> List[List[int]]:
    per_frame_tids: List[List[int]] = []
    prev_map: Dict[int, int] = {}
    next_tid = 0
    for t, boxes in enumerate(frames_boxes):
        tids = [-1] * len(boxes)
        if t == 0:
            for i in range(len(boxes)):
                tids[i] = next_tid
                next_tid += 1
            prev_map = {i: tids[i] for i in range(len(boxes))}
            per_frame_tids.append(tids)
            continue
        prev_boxes = frames_boxes[t - 1]
        match = _greedy_match(prev_boxes, boxes, iou_thr)
        new_prev_map = {}
        for i, prev_idx in enumerate(match):
            if prev_idx >= 0 and prev_idx in prev_map:
                tids[i] = prev_map[prev_idx]
            else:
                tids[i] = next_tid
                next_tid += 1
            new_prev_map[i] = tids[i]
        prev_map = new_prev_map
        per_frame_tids.append(tids)
    return per_frame_tids

def _count_support(
    frames_boxes: List[np.ndarray],
    per_frame_tids: List[List[int]],
    target_tid: int,
    ref_frame: int,
    ref_det: int,
    window: int,
    iou_thr: float,
    direction: int,
) -> int:
    count = 0
    ref_box = frames_boxes[ref_frame][ref_det]
    for offset in range(1, window + 1):
        idx = ref_frame + direction * offset
        if idx < 0 or idx >= len(frames_boxes):
            break
        tids = per_frame_tids[idx]
        boxes = frames_boxes[idx]
        for det_idx, tid in enumerate(tids):
            if tid == target_tid:
                if _iou_xyxy(boxes[det_idx], ref_box) >= iou_thr:
                    count += 1
                break
    return count

def _robust_prompts(
    frames_boxes: List[np.ndarray],
    per_frame_tids: List[List[int]],
    anomaly_mask: List[np.ndarray],
    frame_components: List[Dict[str, np.ndarray]],
    frame_keys: List[str],
    frame_order_to_global: List[int],
    save_interval: int,
    window: int,
    iou_thr: float,
    min_hits: int,
    frame_file_lookup: Dict[str, Optional[str]],
    include_components: bool,
) -> List[Dict[str, object]]:
    prompts: List[Dict[str, object]] = []
    for t, boxes in enumerate(frames_boxes):
        if save_interval > 0 and (t % save_interval) != 0:
            continue
        mask = anomaly_mask[t]
        if not np.any(mask):
            continue
        tids = per_frame_tids[t]
        comps = frame_components[t]
        key = frame_keys[t]
        frame_file = frame_file_lookup.get(key)
        for det_idx, flag in enumerate(mask):
            if not flag:
                continue
            tid = tids[det_idx]
            if tid < 0:
                continue
            support_prev = _count_support(
                frames_boxes, per_frame_tids, tid, t, det_idx, window, iou_thr, direction=-1
            )
            support_next = _count_support(
                frames_boxes, per_frame_tids, tid, t, det_idx, window, iou_thr, direction=1
            )
            if support_prev < min_hits and support_next < min_hits:
                continue
            box = boxes[det_idx]
            prompt = {
                "frame_order": t,
                "frame_index": frame_order_to_global[t],
                "frame_key": key,
                "frame_file": frame_file,
                "bbox": box.astype(float).tolist(),
                "center": _center_xyxy(box),
                "track_id": int(tid),
                "score": float(comps["total"][det_idx]),
            }
            if include_components and "velocity" in comps:
                pose_arr = comps.get("pose")
                deep_arr = comps.get("deep")
                comp_payload = {
                    "velocity": float(comps["velocity"][det_idx]) if comps["velocity"].size else 0.0,
                    "pose": float(pose_arr[det_idx]) if pose_arr is not None and pose_arr.size else 0.0,
                    "deep": float(deep_arr[det_idx]) if deep_arr is not None and deep_arr.size else 0.0,
                }
                prompt["components"] = comp_payload
            prompts.append(prompt)
    return prompts


# ----------------------------- FAISS helpers -----------------------------

def _have_gpu_faiss() -> bool:
    return (faiss is not None) and hasattr(faiss, "StandardGpuResources")

def _batched_search(index, queries: np.ndarray, k: int, batch_size: int):
    if queries.shape[0] == 0:
        return np.zeros((0, k), dtype=np.float32), np.zeros((0, k), dtype=np.int64)
    queries = queries.astype(np.float32)
    D = np.empty((queries.shape[0], k), dtype=np.float32)
    I = np.empty((queries.shape[0], k), dtype=np.int64)
    for start in range(0, queries.shape[0], batch_size):
        end = min(start + batch_size, queries.shape[0])
        d, i = index.search(queries[start:end], k)
        D[start:end] = d
        I[start:end] = i
    return D, I

def _cpu_train_sample(vectors: np.ndarray, max_n: int) -> np.ndarray:
    if vectors.shape[0] <= max_n:
        return vectors.astype(np.float32)
    sel = np.random.choice(vectors.shape[0], max_n, replace=False)
    return vectors[sel].astype(np.float32)


def _db_sample(vectors: np.ndarray, max_n: int) -> np.ndarray:
    if max_n <= 0 or vectors.shape[0] <= max_n:
        return vectors
    sel = np.random.choice(vectors.shape[0], max_n, replace=False)
    return vectors[sel]


def _maybe_limit_temp(res, mb: int) -> None:
    try:
        if mb and hasattr(res, "setTempMemory"):
            res.setTempMemory(mb * 1024 * 1024)
    except Exception:
        pass


def _gpu_clone(index_cpu, device: int, temp_mb: int, use_fp16: bool = True):
    res = faiss.StandardGpuResources()
    _maybe_limit_temp(res, temp_mb)
    opts = faiss.GpuClonerOptions()
    opts.useFloat16 = bool(use_fp16)
    index_gpu = faiss.index_cpu_to_gpu(res, device, index_cpu, opts)
    return index_gpu, res


def _batch_add(index, vectors: np.ndarray, divisor: int) -> None:
    arr = vectors.astype(np.float32)
    if arr.shape[0] == 0:
        return
    chunk = arr.shape[0] // divisor if divisor > 0 else arr.shape[0]
    chunk = max(8192, chunk)
    chunk = min(65536, chunk)
    chunk = min(chunk, arr.shape[0])
    for start in range(0, arr.shape[0], chunk):
        index.add(arr[start:start + chunk])


def _build_flat_index(vectors: np.ndarray, add_vectors: np.ndarray,
                      use_gpu: bool, device: int, temp_mb: int, use_fp16: bool):
    dim = vectors.shape[1] if vectors.size else add_vectors.shape[1]
    target = add_vectors if add_vectors.size else vectors
    cpu = faiss.IndexFlatL2(dim)
    if use_gpu and _have_gpu_faiss():
        try:
            idx, res = _gpu_clone(cpu, device, temp_mb, use_fp16)
            _batch_add(idx, target, divisor=32)
            return idx, res
        except Exception as exc:
            _warn(f"GPU flat index build failed ({exc}); using CPU")
    _batch_add(cpu, target, divisor=32)
    return cpu, None


def _build_ivf_flat_index(vectors: np.ndarray, add_vectors: np.ndarray, nlist: int, nprobe: int,
                          use_gpu: bool, device: int, temp_mb: int, train_max: int):
    dim = vectors.shape[1] if vectors.size else add_vectors.shape[1]
    target = add_vectors if add_vectors.size else vectors
    nlist = max(1, min(nlist, vectors.shape[0]))
    quant = faiss.IndexFlatL2(dim)
    ivf_cpu = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_L2)
    train = _cpu_train_sample(vectors, train_max)
    ivf_cpu.train(train)
    ivf_cpu.nprobe = nprobe
    if use_gpu and _have_gpu_faiss():
        try:
            idx, res = _gpu_clone(ivf_cpu, device, temp_mb)
            idx.nprobe = nprobe
            _batch_add(idx, target, divisor=32)
            return idx, res
        except Exception as exc:
            _warn(f"GPU IVF-Flat build failed ({exc}); using CPU")
    _batch_add(ivf_cpu, target, divisor=32)
    return ivf_cpu, None


def _build_ivfpq_index(vectors: np.ndarray, add_vectors: np.ndarray, nlist: int, m: int, pq_bits: int, nprobe: int,
                       use_gpu: bool, device: int, temp_mb: int, train_max: int):
    dim = vectors.shape[1] if vectors.size else add_vectors.shape[1]
    target = add_vectors if add_vectors.size else vectors
    if dim % m != 0:
        raise ValueError(f"PQ subquantizers m must divide dim: dim={dim}, m={m}")
    nlist = max(1, min(nlist, vectors.shape[0]))
    quant = faiss.IndexFlatL2(dim)
    ivfpq_cpu = faiss.IndexIVFPQ(quant, dim, nlist, m, pq_bits)
    train = _cpu_train_sample(vectors, train_max)
    ivfpq_cpu.train(train)
    ivfpq_cpu.nprobe = nprobe
    if use_gpu and _have_gpu_faiss():
        try:
            idx, res = _gpu_clone(ivfpq_cpu, device, temp_mb, use_fp16=True)
            idx.nprobe = nprobe
            _batch_add(idx, target, divisor=32)
            return idx, res
        except Exception as exc:
            _warn(f"GPU IVFPQ build failed ({exc}); using CPU")
    _batch_add(ivfpq_cpu, target, divisor=32)
    return ivfpq_cpu, None


def _build_hnsw_index(vectors: np.ndarray, add_vectors: np.ndarray, m: int, ef_search: int):
    dim = vectors.shape[1] if vectors.size else add_vectors.shape[1]
    target = add_vectors if add_vectors.size else vectors
    idx = faiss.IndexHNSWFlat(dim, m)
    idx.hnsw.efSearch = ef_search
    _batch_add(idx, target, divisor=16)
    return idx, None


def _build_index(vectors: np.ndarray, args, tag: str):
    if vectors.size == 0:
        return None, None
    kind = args.faiss_index.lower()
    use_gpu = bool(args.faiss_use_gpu)
    device = int(args.faiss_device)
    temp_mb = int(args.faiss_temp_mem_mb)
    add_vectors = _db_sample(vectors, getattr(args, "faiss_db_sample", 0))
    _log(f"Building {tag} index (type={kind}, gpu={use_gpu})")
    if kind == "flat":
        return _build_flat_index(vectors, add_vectors, use_gpu, device, temp_mb, use_fp16=False)
    if kind == "flat_fp16":
        return _build_flat_index(vectors, add_vectors, use_gpu or args.faiss_use_gpu, device, temp_mb, use_fp16=True)
    if kind == "ivf_flat":
        return _build_ivf_flat_index(
            vectors,
            add_vectors,
            args.faiss_nlist,
            args.faiss_nprobe,
            use_gpu,
            device,
            temp_mb,
            args.faiss_train_sample,
        )
    if kind == "ivfpq":
        return _build_ivfpq_index(
            vectors,
            add_vectors,
            args.faiss_nlist,
            args.faiss_pq_m,
            args.faiss_pq_bits,
            args.faiss_nprobe,
            use_gpu,
            device,
            temp_mb,
            args.faiss_train_sample,
        )
    if kind == "hnsw":
        return _build_hnsw_index(vectors, add_vectors, args.hnsw_m, args.hnsw_efsearch)
    raise ValueError(f"Unknown faiss_index {kind}")


# ----------------------------- Scoring -----------------------------

def _gaussian_mixture(train_velocity: np.ndarray, components: int) -> GaussianMixture:
    gm = GaussianMixture(n_components=components, random_state=0)
    gm.fit(train_velocity.astype(np.float32))
    return gm

def _score_frame_components(
    velocity: np.ndarray,
    pose: np.ndarray,
    deep: np.ndarray,
    classes: np.ndarray,
    models: Dict[str, object],
    use_pose: bool,
    use_deep: bool,
    person_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_boxes = velocity.shape[0]
    vel_scores = (
        np.zeros((0,), dtype=np.float32)
        if num_boxes == 0
        else -models["velocity_gmm"].score_samples(velocity.astype(np.float32))
    )
    vel_scores = _normalize(vel_scores, models["min_velocity"], models["max_velocity"])

    pose_aligned = np.zeros((num_boxes,), dtype=np.float32)
    if use_pose and (pose.size > 0) and (models.get("pose_index") is not None):
        D_pose, _ = _batched_search(models["pose_index"], pose.astype(np.float32), 1, models["faiss_search_batch"])
        pose_norm = _normalize(D_pose[:, 0], models["min_pose"], models["max_pose"])
        pose_aligned = _align_pose_scores(pose_norm, classes, num_boxes, person_class)

    deep_scores = np.zeros((num_boxes,), dtype=np.float32)
    if use_deep and (deep.size > 0) and (models.get("deep_index") is not None):
        D_deep, _ = _batched_search(models["deep_index"], deep.astype(np.float32), 1, models["faiss_search_batch"])
        deep_scores = _normalize(D_deep[:, 0], models["min_deep"], models["max_deep"])

    total = vel_scores
    if use_pose:
        total = total + pose_aligned
    if use_deep:
        total = total + deep_scores
    return total.astype(np.float32), vel_scores, pose_aligned, deep_scores


# ----------------------------- Main -----------------------------

def parse_args():
    ap = argparse.ArgumentParser("Attribute-VAD based anomaly prompt builder (memory-safe)")
    ap.add_argument("--dataset_name", type=str, default="shanghaitech")
    ap.add_argument("--split", type=str, default="testing", choices=["training", "testing"])
    ap.add_argument("--data_root", type=str, default="./data/cache/shanghaitech")
    ap.add_argument("--extracted_root", type=str, default="./artifacts/features")
    ap.add_argument("--detections_dir", type=str, required=True,
                    help="Directory with per-video detections.npy files for the split")
    ap.add_argument("--frames_root", type=str, default=None,
                    help="Override frames dir (defaults to data_root/<split>/frames)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--videos", nargs="+", default=None, help="Optional subset of video folder names")

    ap.add_argument("--tau", type=float, default=None, help="Manual anomaly threshold")
    ap.add_argument("--tau_percentile", type=float, default=97.5)

    ap.add_argument("--k", type=int, default=10, help="Temporal window for robustness filtering")
    ap.add_argument("--h", type=float, default=0.5, help="IoU threshold for filtering / tracking")
    ap.add_argument("--m", type=int, default=3, help="Required hits within window")
    ap.add_argument("--l", type=int, default=5, help="Prompt save interval")
    ap.add_argument("--track_iou", type=float, default=0.5, help="IoU threshold for track linking")
    ap.add_argument("--person_class", type=int, default=0)

    ap.add_argument("--disable_pose", action="store_true")
    ap.add_argument("--enable_deep_features", action="store_true")
    ap.add_argument("--disable_deep_features", action="store_true")

    # FAISS options
    ap.add_argument("--faiss_use_gpu", action="store_true")
    ap.add_argument("--faiss_device", type=int, default=0,
                    help="GPU device id (0 = first visible device)")
    ap.add_argument("--faiss_index", type=str, default="flat",
                    choices=["flat", "flat_fp16", "ivf_flat", "ivfpq", "hnsw"])
    ap.add_argument("--faiss_gpu_batch", type=int, default=65536,
                    help="Batch size for adding vectors to GPU index")
    ap.add_argument("--faiss_search_batch", type=int, default=32768,
                    help="Batch size for FAISS search calls")
    ap.add_argument("--faiss_temp_mem_mb", type=int, default=256,
                    help="Temporary memory cap (MB) for GPU FAISS")
    ap.add_argument("--faiss_nlist", type=int, default=4096)
    ap.add_argument("--faiss_nprobe", type=int, default=16)
    ap.add_argument("--faiss_pq_m", type=int, default=8)
    ap.add_argument("--faiss_pq_bits", type=int, default=8)
    ap.add_argument("--faiss_train_sample", type=int, default=200000,
                    help="Maximum vectors to use when training IVF/PQ (CPU)")
    ap.add_argument("--faiss_db_sample", type=int, default=0,
                    help="Maximum database vectors to add to FAISS index (0 = all)")
    ap.add_argument("--hnsw_m", type=int, default=32)
    ap.add_argument("--hnsw_efsearch", type=int, default=64)
    ap.add_argument("--components_fp16", action="store_true",
                    help="Store frame components as float16 to save memory")
    ap.add_argument("--no_components_in_prompts", action="store_true",
                    help="Skip embedding velocity/pose/deep components in prompts JSON")

    return ap.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.data_root)
    extracted_root = Path(args.extracted_root) / args.dataset_name
    detections_dir = Path(args.detections_dir)
    frames_root = (Path(args.frames_root)
                   if args.frames_root else dataset_dir / args.split / "frames")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    split_tag = "train" if args.split == "training" else "test"
    train_tag = "train"

    # Decide feature usage BEFORE loading/building indices
    if args.enable_deep_features:
        use_deep = True
    elif args.disable_deep_features:
        use_deep = False
    else:
        use_deep = (args.dataset_name != "shanghaitech")  # default off on Shanghaitech
    use_pose = not args.disable_pose

    if (use_pose or use_deep) and faiss is None:
        raise ImportError("FAISS is required when pose or deep features are enabled.")

    # Load training velocity (always needed)
    train_vel_path = extracted_root / train_tag / "velocity.npy"
    _log("Loading training velocity")
    train_velocity = _load_object_list(train_vel_path)
    train_velocity_concat = _concat_non_empty(train_velocity)
    n_train_frames = len(train_velocity)
    vel_dim = train_velocity_concat.shape[1] if train_velocity_concat.size else 2  # safe default

    # Conditionally load training pose/deep (or build placeholders)
    if use_pose:
        train_pose_path = extracted_root / train_tag / "pose.npy"
        _log("Loading training pose")
        train_pose = _load_object_list(train_pose_path)
        train_pose_concat = _concat_non_empty(train_pose)
        pose_dim = train_pose_concat.shape[1] if train_pose_concat.size else 34
        if len(train_pose) != n_train_frames:
            _warn("train pose length != train velocity length; filling with empty arrays")
            if len(train_pose) < n_train_frames:
                train_pose += _make_placeholder_list(n_train_frames - len(train_pose), pose_dim)
            else:
                train_pose = train_pose[:n_train_frames]
    else:
        pose_dim = 34
        train_pose = _make_placeholder_list(n_train_frames, pose_dim)
        train_pose_concat = np.zeros((0, pose_dim), dtype=np.float32)

    if use_deep:
        train_deep_path = extracted_root / train_tag / "deep_features.npy"
        _log("Loading training deep features")
        train_deep = _load_object_list(train_deep_path)
        train_deep_concat = _concat_non_empty(train_deep)
        deep_dim = train_deep_concat.shape[1] if train_deep_concat.size else 512
        if len(train_deep) != n_train_frames:
            _warn("train deep length != train velocity length; filling with empty arrays")
            if len(train_deep) < n_train_frames:
                train_deep += _make_placeholder_list(n_train_frames - len(train_deep), deep_dim)
            else:
                train_deep = train_deep[:n_train_frames]
    else:
        deep_dim = 512
        train_deep = _make_placeholder_list(n_train_frames, deep_dim)
        train_deep_concat = np.zeros((0, deep_dim), dtype=np.float32)

    # Velocity GMM
    comps = 2 if args.dataset_name == "ped2" else 5
    _log(f"Fitting velocity GMM with {comps} components on {train_velocity_concat.shape[0]} boxes")
    velocity_gmm = _gaussian_mixture(train_velocity_concat, comps)
    train_velocity_scores = -velocity_gmm.score_samples(train_velocity_concat)
    min_velocity = float(np.min(train_velocity_scores)) if train_velocity_scores.size else 0.0
    max_velocity = float(np.percentile(train_velocity_scores, 99.9)) if train_velocity_scores.size else 1.0

    # Build indices (pose/deep) only if used
    pose_index = pose_res = None
    deep_index = deep_res = None

    if use_pose and train_pose_concat.size:
        _log(f"Building FAISS index for pose (index={args.faiss_index})")
        pose_index, pose_res = _build_index(train_pose_concat, args, "pose")

    if use_deep and train_deep_concat.size:
        _log(f"Building FAISS index for deep (index={args.faiss_index})")
        deep_index, deep_res = _build_index(train_deep_concat, args, "deep")

    # Estimate train distribution for normalization of pose/deep
    if use_pose and train_pose_concat.size and (pose_index is not None):
        D_pose_train, _ = _batched_search(pose_index, train_pose_concat.astype(np.float32), 2, args.faiss_search_batch)
        train_pose_scores = D_pose_train[:, 1]
        min_pose = float(np.min(train_pose_scores))
        max_pose = float(np.percentile(train_pose_scores, 99.9))
    else:
        min_pose, max_pose = 0.0, 1.0

    if use_deep and train_deep_concat.size and (deep_index is not None):
        D_deep_train, _ = _batched_search(deep_index, train_deep_concat.astype(np.float32), 2, args.faiss_search_batch)
        train_deep_scores = D_deep_train[:, 1]
        min_deep = float(np.min(train_deep_scores))
        max_deep = float(np.percentile(train_deep_scores, 99.9))
    else:
        min_deep, max_deep = 0.0, 1.0

    # Load classes lists (train/test)
    train_classes_path = dataset_dir / f"{args.dataset_name}_bboxes_train_classes.npy"
    test_classes_path = dataset_dir / f"{args.dataset_name}_bboxes_{('test' if split_tag=='test' else 'train')}_classes.npy"
    _log("Loading classes lists")
    train_classes = _load_object_list(train_classes_path) if train_classes_path.exists() else [np.zeros((len(x),), np.int32) for x in train_velocity]
    # Load test features before checking lengths
    test_velocity = _load_object_list(extracted_root / split_tag / "velocity.npy")
    n_test_frames = len(test_velocity)

    if use_pose:
        test_pose = _load_object_list(extracted_root / split_tag / "pose.npy")
        if len(test_pose) != n_test_frames:
            _warn("test pose length != test velocity length; aligning with placeholders")
            if len(test_pose) < n_test_frames:
                test_pose += _make_placeholder_list(n_test_frames - len(test_pose), pose_dim)
            else:
                test_pose = test_pose[:n_test_frames]
    else:
        test_pose = _make_placeholder_list(n_test_frames, pose_dim)

    if use_deep:
        test_deep = _load_object_list(extracted_root / split_tag / "deep_features.npy")
        if len(test_deep) != n_test_frames:
            _warn("test deep length != test velocity length; aligning with placeholders")
            if len(test_deep) < n_test_frames:
                test_deep += _make_placeholder_list(n_test_frames - len(test_deep), deep_dim)
            else:
                test_deep = test_deep[:n_test_frames]
    else:
        test_deep = _make_placeholder_list(n_test_frames, deep_dim)

    test_classes = _load_object_list(test_classes_path) if test_classes_path.exists() else [np.zeros((len(x),), np.int32) for x in test_velocity]

    # Align train lengths
    if len(train_classes) != n_train_frames:
        _warn("train classes length != train velocity length; aligning with placeholders")
        if len(train_classes) < n_train_frames:
            train_classes += [np.zeros((len(train_velocity[i]),), np.int32) for i in range(len(train_classes), n_train_frames)]
        else:
            train_classes = train_classes[:n_train_frames]

    # Models pack
    models = {
        "velocity_gmm": velocity_gmm,
        "min_velocity": min_velocity,
        "max_velocity": max_velocity,
        "pose_index": pose_index,
        "deep_index": deep_index,
        "min_pose": min_pose,
        "max_pose": max_pose,
        "min_deep": min_deep,
        "max_deep": max_deep,
        "faiss_search_batch": int(args.faiss_search_batch),
    }

    # Scoring helpers
    comp_dtype = np.float16 if args.components_fp16 else np.float32

    def _score_split(features: Dict[str, List[np.ndarray]], classes_list: List[np.ndarray], store_components: bool):
        outputs = []
        for idx, vel in enumerate(tqdm(features["velocity"], desc=f"Scoring {len(features['velocity'])} frames ({'store' if store_components else 'train'})")):
            vel_arr = _ensure_2d(vel, vel_dim)
            pose_arr = _ensure_2d(features["pose"][idx], pose_dim) if use_pose else np.zeros((0, pose_dim), np.float32)
            deep_arr = _ensure_2d(features["deep"][idx], deep_dim) if use_deep else np.zeros((0, deep_dim), np.float32)
            cls_arr = np.asarray(classes_list[idx], dtype=np.int32) if idx < len(classes_list) else np.zeros((vel_arr.shape[0],), np.int32)
            if cls_arr.shape[0] != vel_arr.shape[0]:
                if cls_arr.size == 0:
                    cls_arr = np.zeros((vel_arr.shape[0],), dtype=np.int32)
                else:
                    cls_arr = np.resize(cls_arr, vel_arr.shape[0])
            total, vel_scores, pose_scores, deep_scores = _score_frame_components(
                vel_arr, pose_arr, deep_arr, cls_arr, models, use_pose, use_deep, args.person_class
            )
            if store_components:
                entry = {
                    "total": total.astype(comp_dtype, copy=False),
                    "classes": cls_arr,
                }
                if not args.no_components_in_prompts:
                    entry["velocity"] = vel_scores.astype(comp_dtype, copy=False)
                    entry["pose"] = (pose_scores if use_pose else np.zeros_like(total)).astype(comp_dtype, copy=False)
                    entry["deep"] = (deep_scores if use_deep else np.zeros_like(total)).astype(comp_dtype, copy=False)
                outputs.append(entry)
            else:
                outputs.append(total.astype(comp_dtype, copy=False))
        return outputs

    # Train distribution for tau
    train_features = {"velocity": train_velocity, "pose": train_pose, "deep": train_deep}
    _log(f"Scoring training frames ({n_train_frames}) to derive tau distribution")
    train_scores = _score_split(train_features, train_classes, store_components=False)
    gc.collect()

    # Test scores + components
    test_features = {"velocity": test_velocity, "pose": test_pose, "deep": test_deep}
    _log(f"Scoring {split_tag} frames ({n_test_frames}) and storing components")
    test_scores = _score_split(test_features, test_classes, store_components=True)
    gc.collect()

    # Tau
    if args.tau is not None:
        tau = float(args.tau)
        tau_source = "manual"
    else:
        total_counts = sum(len(s) for s in train_scores if len(s))
        if total_counts == 0:
            sample_vals = np.array([0.0], dtype=np.float32)
        else:
            target = min(200000, total_counts)
            if target >= total_counts:
                sample_vals = np.concatenate([np.asarray(s, dtype=np.float32) for s in train_scores if len(s)], axis=0)
            else:
                rng = np.random.default_rng(0)
                chosen = np.sort(rng.choice(total_counts, target, replace=False))
                sample_vals = np.empty(target, dtype=np.float32)
                ptr = 0
                global_idx = 0
                for s in train_scores:
                    arr = np.asarray(s, dtype=np.float32).ravel()
                    count = arr.size
                    if count == 0:
                        continue
                    while ptr < target and chosen[ptr] < global_idx + count:
                        local = chosen[ptr] - global_idx
                        sample_vals[ptr] = arr[int(local)]
                        ptr += 1
                    global_idx += count
                    if ptr >= target:
                        break
        tau = float(np.percentile(sample_vals, args.tau_percentile))
        tau_source = f"train_percentile_{args.tau_percentile}" if total_counts else "default"
    print(f"[info] Using tau={tau:.4f} ({tau_source}) with pose={use_pose}, deep={use_deep}")

    # Metadata
    _log("Loading clip length metadata")
    clip_lengths = np.load(dataset_dir / f"{split_tag}_clip_lengths.npy")
    clip_ranges = _build_clip_ranges(clip_lengths)

    video_dirs = sorted([d for d in detections_dir.iterdir() if d.is_dir()])
    if len(video_dirs) != len(clip_ranges):
        raise ValueError(f"Video count mismatch: detections={len(video_dirs)} vs clip_lengths={len(clip_ranges)}")

    requested = set(args.videos) if args.videos else None
    summary = []
    _log(f"Processing {len(video_dirs)} videos for split '{args.split}'")
    for vid_idx, (video_dir, (start, end)) in enumerate(zip(video_dirs, clip_ranges)):
        _log(f"Video {vid_idx + 1}/{len(video_dirs)}: {video_dir.name}")
        if requested and video_dir.name not in requested:
            continue
        det_file = video_dir / "detections.npy"
        if not det_file.exists():
            _warn(f"Missing detections for {video_dir.name}, skipping.")
            continue
        det_map = np.load(det_file, allow_pickle=True).item()
        frame_items = sorted(det_map.items(), key=lambda kv: _frame_sort_key(kv[0]))
        expected_frames = end - start
        if expected_frames != len(frame_items):
            _warn(f"Frame count mismatch for {video_dir.name}: expected {expected_frames}, got {len(frame_items)}. Skipping.")
            continue

        frames_boxes: List[np.ndarray] = []
        frame_components: List[Dict[str, np.ndarray]] = []
        frame_keys: List[str] = []
        frame_ord_to_global: List[int] = []
        anomaly_mask: List[np.ndarray] = []

        frames_dir = frames_root / video_dir.name
        # Build stem->filename map for this video
        frame_lookup = {}
        if frames_dir.exists():
            for p in frames_dir.iterdir():
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    frame_lookup[p.stem] = p.name

        processed = True
        for local_idx, (frame_key, entry) in enumerate(frame_items):
            global_idx = start + local_idx
            boxes = np.asarray(entry["boxes"], dtype=np.float32)
            # number of boxes must match per-box features (velocity)
            test_box_arr = np.asarray(test_velocity[global_idx])
            if boxes.shape[0] != test_box_arr.shape[0]:
                _warn(f"Detection/VAD bbox count mismatch at {video_dir.name} frame {frame_key}; skipping video.")
                processed = False
                break
            comp = test_scores[global_idx]
            mask = comp["total"] > tau
            frames_boxes.append(boxes)
            frame_components.append(comp)
            frame_keys.append(str(frame_key))
            frame_ord_to_global.append(global_idx)
            anomaly_mask.append(mask)

        if not processed:
            continue

        per_frame_tids = _build_tracks_per_video(frames_boxes, args.track_iou)
        prompts = _robust_prompts(
            frames_boxes=frames_boxes,
            per_frame_tids=per_frame_tids,
            anomaly_mask=anomaly_mask,
            frame_components=frame_components,
            frame_keys=frame_keys,
            frame_order_to_global=frame_ord_to_global,
            save_interval=args.l,
            window=args.k,
            iou_thr=args.h,
            min_hits=args.m,
            frame_file_lookup={k: frame_lookup.get(k) for k in frame_keys},
            include_components=not args.no_components_in_prompts,
        )

        video_out = out_root / args.split / video_dir.name
        video_out.mkdir(parents=True, exist_ok=True)
        out_json = video_out / "robust_prompts.json"
        payload = {
            "dataset": args.dataset_name,
            "split": args.split,
            "video": video_dir.name,
            "frames_dir": str(frames_dir),
            "frame_keys": frame_keys,
            "params": {
                "tau": tau,
                "tau_source": tau_source,
                "k": args.k,
                "h": args.h,
                "m": args.m,
                "l": args.l,
                "track_iou": args.track_iou,
                "use_pose": use_pose,
                "use_deep": use_deep,
            },
            "prompts": prompts,
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        tuples = [
            (
                p["frame_key"],
                np.array(p["bbox"], dtype=np.float32),
                np.array(p["center"], dtype=np.float32),
                p["track_id"],
                p["score"],
            )
            for p in prompts
        ]
        np.save(video_out / "filtered_boxes.npy", np.array(tuples, dtype=object))
        summary.append((video_dir.name, len(prompts)))
        print(f"[save] {out_json}  (#prompts={len(prompts)})")
        gc.collect()

    if not summary:
        _warn("No videos were processed; check --videos or input paths.")
    else:
        print("=== Prompt summary ===")
        for name, count in summary:
            print(f"{name}: {count}")


if __name__ == "__main__":
    main()
