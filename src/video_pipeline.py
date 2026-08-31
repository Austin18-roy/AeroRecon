"""
AeroRecon Video Pipeline
========================
Full end-to-end pipeline: UAV video → keyframes → YOLO11s → Depth Anything V2 → Dense 3D Map

Stages implemented:
  Stage 1a (MVD)    : Keyframe extraction + Laplacian sharpness scoring
  Stage 1b          : YOLO11s + ByteTrack persistent object tracking
  Stage 2           : Depth Anything V2 relative depth → .npy + PNG + metadata
  Stage 3 (Future)  : Incremental mapping placeholder
  Stage 4 (Future)  : Rescue AI navigation placeholder
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────
# Sharpness metric (Laplacian variance)
# ─────────────────────────────────────────────

def get_blur_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─────────────────────────────────────────────
# Stage 1a: Intelligent Keyframe Extraction
# ─────────────────────────────────────────────

def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    max_frames: int = 10,
    min_sharpness_threshold: float = 50.0,
    progress_callback: Optional[callable] = None,
) -> List[Dict]:
    """
    Extracts intelligently selected keyframes from a UAV video.
    Uses Laplacian sharpness scoring and temporal segment partitioning
    to ensure both clarity and spatial flight coverage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for f in output_dir.glob("*.png"):
        try:
            f.unlink()
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no frames or cannot be read.")

    target_count = min(max_frames, max(1, total_frames // 5))
    segment_size = total_frames / target_count

    extracted = []

    for seg_idx in range(target_count):
        start = int(seg_idx * segment_size)
        end   = int((seg_idx + 1) * segment_size)
        step  = max(1, (end - start) // 10)

        best_frame, best_score, best_idx = None, -1.0, start

        for fi in range(start, end, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            score = get_blur_score(frame)
            if score > best_score:
                best_score, best_frame, best_idx = score, frame, fi

        if best_frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ret, best_frame = cap.read()
            best_idx, best_score = start, 0.0

        if best_frame is not None:
            stem_num  = seg_idx * 100
            stem_str  = f"{stem_num:06d}"
            out_path  = output_dir / f"{stem_str}.png"
            cv2.imwrite(str(out_path), best_frame)

            extracted.append({
                "stem": stem_str,
                "filename": out_path.name,
                "path": out_path,
                "frame_idx": best_idx,
                "timestamp": best_idx / fps,
                "sharpness": best_score,
                "resolution": (width, height),
            })

        if progress_callback:
            progress_callback(
                (seg_idx + 1) / target_count,
                f"Extracted keyframe {seg_idx + 1}/{target_count}",
            )

    cap.release()
    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b: YOLO11s Object Detection + ByteTrack Persistent Tracking
# ─────────────────────────────────────────────────────────────────────────────

# COCO class IDs for the target categories we care about:
#   Persons: 0
#   Vehicles: 1 (bicycle), 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
#   Animals: 14 (bird), 15 (cat), 16 (dog), 17 (horse), 18 (sheep),
#            19 (cow), 20 (elephant), 21 (bear), 22 (zebra), 23 (giraffe)
_TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

_COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}


def run_yolo_on_keyframes(
    image_paths: List[Path],
    output_dir: Path,
    model_path: str = "yolo11s.pt",
    conf: float = 0.30,
    imgsz: int = 1280,
    frame_timestamps: Optional[Dict[str, float]] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, int]:
    """
    Runs YOLO11s detection with ByteTrack persistent object tracking across keyframes.

    Outputs:
      - Annotated JPEG images (for the existing Streamlit UI display).
      - ``detections.jsonl`` - one JSON record per detection per frame, containing:
          frame_id, timestamp_sec, track_id, class_id, class_name, confidence,
          bbox [x1, y1, x2, y2]
      - ``detections_summary.json`` - per-frame counts + run-level statistics.

    Args:
      image_paths:      Ordered list of keyframe paths (must be consistent between runs).
      output_dir:       Destination for annotated images and metadata files.
      model_path:       YOLO model filename relative to project root.
      conf:             Confidence threshold (0-1).
      imgsz:            Inference image size.
      frame_timestamps: Optional mapping {frame_stem: timestamp_sec}. If None the
                        frame index is used as a surrogate.
      progress_callback: Optional callback(fraction, message).

    Returns:
      Dict mapping frame stem to detection count (same contract as before).
    """
    import json
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale outputs from any previous run
    for stale in output_dir.glob("*.jpg"):
        try:
            stale.unlink()
        except Exception:
            pass
    for stale in output_dir.glob("*.jsonl"):
        try:
            stale.unlink()
        except Exception:
            pass
    for stale in output_dir.glob("*.json"):
        try:
            stale.unlink()
        except Exception:
            pass

    model_file = ROOT / model_path
    if not model_file.exists():
        fallback = ROOT / "yolo11n.pt"
        model_file = fallback if fallback.exists() else Path(model_path)

    model = YOLO(str(model_file))

    detection_counts: Dict[str, int] = {}
    jsonl_path = output_dir / "detections.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_fh:
        for i, img_path in enumerate(image_paths):
            frame_stem = img_path.stem
            ts_sec = (
                frame_timestamps.get(frame_stem, float(i))
                if frame_timestamps
                else float(i)
            )

            try:
                results = model.track(
                    source=str(img_path),
                    tracker="bytetrack.yaml",
                    persist=True,
                    device="cpu",
                    conf=conf,
                    imgsz=imgsz,
                    classes=_TARGET_CLASSES,
                    save=False,
                    verbose=False,
                )
            except Exception:
                results = model.predict(
                    source=str(img_path),
                    device="cpu",
                    conf=conf,
                    imgsz=imgsz,
                    classes=_TARGET_CLASSES,
                    save=False,
                    verbose=False,
                )

            result = results[0]
            boxes = result.boxes
            frame_count = 0

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                track_ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if boxes.id is not None
                    else [None] * len(xyxy)
                )

                for bbox, cls_id, conf_val, tid in zip(xyxy, cls_ids, confs, track_ids):
                    x1, y1, x2, y2 = bbox.tolist()
                    record = {
                        "frame_id": frame_stem,
                        "timestamp_sec": round(ts_sec, 4),
                        "track_id": int(tid) if tid is not None else None,
                        "class_id": int(cls_id),
                        "class_name": _COCO_CLASS_NAMES.get(int(cls_id), f"class_{cls_id}"),
                        "confidence": round(float(conf_val), 4),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    }
                    jsonl_fh.write(json.dumps(record) + "\n")
                    frame_count += 1

            detection_counts[frame_stem] = frame_count
            annotated = result.plot()
            cv2.imwrite(str(output_dir / f"{frame_stem}.jpg"), annotated)

            if progress_callback:
                progress_callback(
                    (i + 1) / len(image_paths),
                    f"Detected frame {i + 1}/{len(image_paths)}: {frame_count} objects",
                )

    total_detections = sum(detection_counts.values())
    unique_classes: Dict[str, int] = {}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cn = rec.get("class_name", "unknown")
                unique_classes[cn] = unique_classes.get(cn, 0) + 1
    except Exception:
        pass

    summary = {
        "frames_processed": len(image_paths),
        "total_detections": total_detections,
        "per_frame_counts": detection_counts,
        "class_totals": unique_classes,
        "tracker": "ByteTrack (ultralytics built-in)",
        "model": str(model_file.name),
        "conf_threshold": conf,
        "target_classes": _TARGET_CLASSES,
    }
    summary_path = output_dir / "detections_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    return detection_counts


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Depth Anything V2 – Relative Depth Estimation
# ─────────────────────────────────────────────────────────────────────────────

_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def run_depth_on_keyframes(
    image_paths: List[Path],
    output_dir: Path,
    frame_timestamps: Optional[Dict[str, float]] = None,
    progress_callback: Optional[callable] = None,
) -> List[Path]:
    """
    Runs Depth Anything V2 relative depth estimation on all keyframes.

    For every processed frame this function saves:
      depth_{stem}.npy   -- lossless float32 normalized relative depth [0, 1].
                            Normalised so 0 = furthest, 1 = nearest.
                            Machine-usable representation for downstream 3D
                            reconstruction.
      depth_{stem}.png   -- uint8 greyscale visualization (backward-compatible
                            with the existing 3D reconstruction pipeline which
                            reads these PNGs).

    A ``depth_metadata.json`` file is written to output_dir summarising every
    processed frame with:
      frame_id, source_filename, timestamp_sec, width, height,
      depth_npy, depth_png, model, depth_type.

    IMPORTANT: This function preserves the original relative depth output of
    Depth Anything V2. It does NOT convert relative depth to metric distance.

    Args:
      image_paths:       Ordered list of keyframe image Paths to process.
      output_dir:        Directory for all depth outputs (npy + png + json).
      frame_timestamps:  Optional {frame_stem: float} mapping for metadata.
                         If None, the frame index is used as a surrogate.
      progress_callback: Optional callback(fraction, message) for UI progress.

    Returns:
      List of Paths to the PNG visualization files (one per successfully
      processed frame, same order as image_paths). Unchanged contract for
      the downstream reconstruction stage.
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale depth outputs so previous runs cannot contaminate
    for stale in output_dir.glob("depth_*.png"):
        try:
            stale.unlink()
        except Exception:
            pass
    for stale in output_dir.glob("depth_*.npy"):
        try:
            stale.unlink()
        except Exception:
            pass
    stale_meta = output_dir / "depth_metadata.json"
    if stale_meta.exists():
        try:
            stale_meta.unlink()
        except Exception:
            pass

    # Load model once (lazy import to keep startup fast when not running depth)
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline(
        task="depth-estimation",
        model=_DEPTH_MODEL_ID,
        device=-1,  # CPU
    )

    depth_png_paths: List[Path] = []
    metadata_records: List[Dict] = []

    for i, img_path in enumerate(image_paths):
        frame_stem = img_path.stem
        ts_sec = (
            frame_timestamps.get(frame_stem, float(i))
            if frame_timestamps
            else float(i)
        )

        # Load image safely -- skip corrupt/unreadable frames
        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception as load_err:
            if progress_callback:
                progress_callback(
                    (i + 1) / len(image_paths),
                    f"Depth frame {i + 1}/{len(image_paths)}: SKIPPED ({load_err})",
                )
            continue

        img_w, img_h = pil_image.size  # PIL gives (width, height)

        # Run inference
        try:
            result = pipe(pil_image)
        except Exception as inf_err:
            if progress_callback:
                progress_callback(
                    (i + 1) / len(image_paths),
                    f"Depth frame {i + 1}/{len(image_paths)}: INFERENCE ERROR ({inf_err})",
                )
            continue

        # Extract float32 predicted depth tensor
        # result["predicted_depth"] is torch.Tensor (H, W) float32.
        # Normalise to [0, 1]: 0 = furthest, 1 = nearest.
        predicted_depth = result["predicted_depth"]
        depth_float = predicted_depth.detach().float().numpy()  # (H, W) float32

        d_min = float(depth_float.min())
        d_max = float(depth_float.max())
        if d_max > d_min:
            depth_norm = (depth_float - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_float)

        # Save lossless float32 .npy
        npy_path = output_dir / f"depth_{frame_stem}.npy"
        np.save(str(npy_path), depth_norm.astype(np.float32))

        # Save uint8 PNG visualization (backward-compatible)
        depth_vis_pil = result["depth"]  # PIL Image, mode L, uint8
        png_path = output_dir / f"depth_{frame_stem}.png"
        depth_vis_pil.save(str(png_path))
        depth_png_paths.append(png_path)

        # Accumulate metadata record
        metadata_records.append({
            "frame_id": frame_stem,
            "source_filename": img_path.name,
            "timestamp_sec": round(ts_sec, 4),
            "width": img_w,
            "height": img_h,
            "depth_npy": npy_path.name,
            "depth_png": png_path.name,
            "depth_min_raw": round(d_min, 6),
            "depth_max_raw": round(d_max, 6),
            "model": _DEPTH_MODEL_ID,
            "depth_type": "relative",
        })

        if progress_callback:
            progress_callback(
                (i + 1) / len(image_paths),
                f"Depth map {i + 1}/{len(image_paths)}: {frame_stem}",
            )

    # Write depth_metadata.json
    metadata_payload = {
        "model": _DEPTH_MODEL_ID,
        "depth_type": "relative",
        "depth_npy_format": "float32 normalized [0=furthest, 1=nearest]",
        "frames": metadata_records,
    }
    meta_path = output_dir / "depth_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata_payload, mf, indent=2)

    return depth_png_paths


# ─────────────────────────────────────────────
# Camera pose estimation helpers
# ─────────────────────────────────────────────

def _estimate_intrinsics(w: int, h: int) -> Tuple[float, float, float, float]:
    """Estimate focal length from image size assuming 70 degree horizontal FOV."""
    fov_h_rad = math.radians(70)
    fx = (w / 2.0) / math.tan(fov_h_rad / 2.0)
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    return fx, fy, cx, cy


def _optical_flow_delta(
    prev_gray: np.ndarray, curr_gray: np.ndarray
) -> Tuple[float, float, float]:
    """
    Estimates translation (dx, dy) and rotation (dtheta) between two grayscale frames
    using sparse Lucas-Kanade optical flow on detected good features.
    Returns (dx, dy, dtheta) in pixel units.
    """
    pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=10
    )
    if pts is None or len(pts) < 10:
        return 0.0, 0.0, 0.0

    pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None,
        winSize=(21, 21), maxLevel=3,
    )
    status = status.flatten()
    good_prev = pts[status == 1].reshape(-1, 2)
    good_next = pts_next[status == 1].reshape(-1, 2)

    if len(good_prev) < 4:
        return 0.0, 0.0, 0.0

    disp = good_next - good_prev
    dx = float(np.median(disp[:, 0]))
    dy = float(np.median(disp[:, 1]))

    angles_prev = np.arctan2(good_prev[:, 1] - prev_gray.shape[0] / 2,
                             good_prev[:, 0] - prev_gray.shape[1] / 2)
    angles_next = np.arctan2(good_next[:, 1] - curr_gray.shape[0] / 2,
                             good_next[:, 0] - curr_gray.shape[1] / 2)
    dtheta = float(np.median(angles_next - angles_prev))

    return dx, dy, dtheta


# ─────────────────────────────────────────────
# Stage 1d + 2: Dense 3D Point Cloud Builder
# ─────────────────────────────────────────────

def estimate_point_cloud_and_trajectory(
    image_paths: List[Path],
    depth_paths: List[Path],
    output_ply_path: Path,
    detection_data: Optional[Dict[str, int]] = None,
    max_points_per_frame: int = 800,
    progress_callback: Optional[callable] = None,
) -> Tuple[List[Dict], int]:
    """
    Builds a dense, RGB-colored 3D point cloud from multi-view keyframes and depth maps.
    Uses optical-flow-based camera pose estimation between consecutive frames.

    Each frame contributes dense colored 3D points by:
      1. Sampling pixels uniformly (or using depth saliency for richer coverage).
      2. Unprojecting (u, v, d) to (X, Y, Z) using estimated camera intrinsics.
      3. Transforming into a shared world frame using accumulated camera poses.

    Returns the list of recovered camera poses and the total point count.
    """
    output_ply_path.parent.mkdir(parents=True, exist_ok=True)

    all_pts   : List[List[float]] = []
    all_colors: List[List[int]]   = []
    camera_poses: List[Dict]       = []

    # Accumulated camera pose (simple integrated visual odometry)
    cam_pos   = np.zeros(3, dtype=np.float64)
    cam_yaw   = 0.0
    cam_pitch = 0.0

    prev_gray: Optional[np.ndarray] = None

    for idx, (img_p, dep_p) in enumerate(zip(image_paths, depth_paths)):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        # Load depth -- prefer .npy (float32 normalized) if available
        npy_path = dep_p.parent / (dep_p.stem + ".npy")
        if npy_path.exists():
            depth_arr = np.load(str(npy_path)).astype(np.float64)
            # Resize if needed
            if depth_arr.shape != (h, w):
                from PIL import Image as _PIL
                depth_pil = _PIL.fromarray(depth_arr.astype(np.float32))
                depth_arr = np.array(
                    depth_pil.resize((w, h), Image.BILINEAR)
                ).astype(np.float64)
        else:
            depth_pil = Image.open(dep_p)
            depth_arr = np.array(
                depth_pil.resize((w, h), Image.BILINEAR)
            ).astype(np.float64) / 255.0

        curr_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            dx_px, dy_px, dtheta = _optical_flow_delta(prev_gray, curr_gray)
            world_scale = 0.012
            step_x =  dx_px * world_scale
            step_y = -dy_px * world_scale * 0.5
            step_z =  0.6 + abs(dy_px) * 0.003

            cam_yaw   += dtheta * 0.15
            cam_pitch += dy_px * 0.0001

            cam_pos[0] += step_x * math.cos(cam_yaw) - step_z * math.sin(cam_yaw)
            cam_pos[1] += step_y
            cam_pos[2] += step_x * math.sin(cam_yaw) + step_z * math.cos(cam_yaw)
        else:
            cam_pos = np.zeros(3, dtype=np.float64)

        camera_poses.append({
            "id":     idx + 1,
            "name":   img_p.name,
            "center": cam_pos.copy(),
            "yaw":    cam_yaw,
        })
        prev_gray = curr_gray

        fx, fy, cx, cy = _estimate_intrinsics(w, h)

        stride = max(4, int(math.sqrt(h * w / max_points_per_frame)))
        ys, xs = np.meshgrid(
            np.arange(0, h, stride),
            np.arange(0, w, stride),
            indexing="ij",
        )
        ys = ys.flatten()
        xs = xs.flatten()

        d_raw = depth_arr[ys, xs]
        metric_z = (1.5 + (1.0 - d_raw) * 3.5)

        valid = (metric_z > 0.5) & (metric_z < 6.0)
        ys, xs, metric_z = ys[valid], xs[valid], metric_z[valid]

        if len(ys) == 0:
            continue

        x_cam = (xs - cx) * metric_z / fx
        y_cam = (ys - cy) * metric_z / fy
        z_cam = metric_z

        cos_y, sin_y = math.cos(cam_yaw), math.sin(cam_yaw)
        x_world = x_cam * cos_y - z_cam * sin_y + cam_pos[0]
        y_world = y_cam                            + cam_pos[1]
        z_world = x_cam * sin_y + z_cam * cos_y  + cam_pos[2]

        r = img_rgb[ys, xs, 0].astype(np.uint8)
        g = img_rgb[ys, xs, 1].astype(np.uint8)
        b = img_rgb[ys, xs, 2].astype(np.uint8)

        for px, py, pz, pr, pg, pb in zip(
            x_world.tolist(), y_world.tolist(), z_world.tolist(),
            r.tolist(), g.tolist(), b.tolist()
        ):
            all_pts.append([px, py, pz])
            all_colors.append([pr, pg, pb])

        if progress_callback:
            progress_callback(
                (idx + 1) / len(image_paths),
                f"Reconstructed 3D geometry: view {idx + 1}/{len(image_paths)}",
            )

    if not all_pts:
        all_pts    = [[0.0, 0.0, 0.0]]
        all_colors = [[200, 200, 200]]

    pts_arr  = np.array(all_pts,    dtype=np.float32)
    cols_arr = np.array(all_colors, dtype=np.uint8)

    max_total = 60_000
    if len(pts_arr) > max_total:
        idx_s = np.random.choice(len(pts_arr), max_total, replace=False)
        pts_arr  = pts_arr[idx_s]
        cols_arr = cols_arr[idx_s]

    for ax in range(3):
        col = pts_arr[:, ax]
        mu, sigma = col.mean(), col.std()
        if sigma > 0:
            keep = np.abs(col - mu) < 3.5 * sigma
            pts_arr  = pts_arr[keep]
            cols_arr = cols_arr[keep]

    vertices = np.zeros(
        len(pts_arr),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertices["x"],    vertices["y"],     vertices["z"]     = pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2]
    vertices["red"],  vertices["green"], vertices["blue"]  = cols_arr[:, 0], cols_arr[:, 1], cols_arr[:, 2]

    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(output_ply_path))

    return camera_poses, len(pts_arr)
