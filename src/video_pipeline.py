"""
AeroRecon Video Pipeline
========================
Full end-to-end pipeline: UAV video → keyframes → YOLO11s → Depth Anything V2 → Dense 3D Map

Stages implemented:
  Stage 1 (MVD)     : Keyframe extraction + YOLO11s + Depth Anything V2 + Dense 3D point cloud
  Stage 2 (Eval)    : Simulated AnySplat-style output from dense depth unprojection + RGB coloring
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


# ─────────────────────────────────────────────
# Stage 1b: YOLO11s Object Detection
# ─────────────────────────────────────────────

def run_yolo_on_keyframes(
    image_paths: List[Path],
    output_dir: Path,
    model_path: str = "yolo11s.pt",
    conf: float = 0.30,
    imgsz: int = 1280,
    progress_callback: Optional[callable] = None,
) -> Dict[str, int]:
    """
    Runs YOLO11s aerial object detection on extracted keyframes.
    Saves annotated images and returns per-frame detection counts.
    """
    from ultralytics import YOLO
    output_dir.mkdir(parents=True, exist_ok=True)

    for f in output_dir.glob("*.jpg"):
        try:
            f.unlink()
        except Exception:
            pass

    model_file = ROOT / model_path
    if not model_file.exists():
        fallback = ROOT / "yolo11n.pt"
        model_file = fallback if fallback.exists() else Path(model_path)

    model = YOLO(str(model_file))
    target_classes = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

    detection_counts = {}

    for i, img_path in enumerate(image_paths, 1):
        results = model.predict(
            source=str(img_path),
            device="cpu",
            conf=conf,
            imgsz=imgsz,
            classes=target_classes,
            save=False,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        detection_counts[img_path.stem] = len(boxes) if boxes is not None else 0

        annotated = result.plot()
        cv2.imwrite(str(output_dir / f"{img_path.stem}.jpg"), annotated)

        if progress_callback:
            progress_callback(i / len(image_paths), f"Detected frame {i}/{len(image_paths)}")

    return detection_counts


# ─────────────────────────────────────────────
# Stage 1c: Depth Anything V2
# ─────────────────────────────────────────────

def run_depth_on_keyframes(
    image_paths: List[Path],
    output_dir: Path,
    progress_callback: Optional[callable] = None,
) -> List[Path]:
    """Runs Depth Anything V2 relative depth estimation on all keyframes."""
    from transformers import pipeline as hf_pipeline

    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=-1,
    )

    depth_paths = []

    for i, img_path in enumerate(image_paths, 1):
        image  = Image.open(img_path).convert("RGB")
        result = pipe(image)
        depth_img = result["depth"]
        out_path  = output_dir / f"depth_{img_path.stem}.png"
        depth_img.save(out_path)
        depth_paths.append(out_path)

        if progress_callback:
            progress_callback(i / len(image_paths), f"Depth map {i}/{len(image_paths)}")

    return depth_paths


# ─────────────────────────────────────────────
# Camera pose estimation helpers
# ─────────────────────────────────────────────

def _estimate_intrinsics(w: int, h: int) -> Tuple[float, float, float, float]:
    """Estimate focal length from image size assuming 70° horizontal FOV."""
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

    # Estimate rotation from angle differences
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
      2. Unprojecting (u, v, d) → (X, Y, Z) using estimated camera intrinsics.
      3. Transforming into a shared world frame using accumulated camera poses.

    Returns the list of recovered camera poses and the total point count.
    """
    output_ply_path.parent.mkdir(parents=True, exist_ok=True)

    all_pts   : List[List[float]] = []
    all_colors: List[List[int]]   = []
    camera_poses: List[Dict]       = []

    # Accumulated camera pose (simple integrated visual odometry)
    cam_pos   = np.zeros(3, dtype=np.float64)
    cam_yaw   = 0.0          # radians in the XZ plane
    cam_pitch = 0.0

    prev_gray: Optional[np.ndarray] = None

    for idx, (img_p, dep_p) in enumerate(zip(image_paths, depth_paths)):
        # ── Load RGB & depth ──────────────────────────────────────────────
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        depth_pil = Image.open(dep_p)
        depth_arr = np.array(
            depth_pil.resize((w, h), Image.BILINEAR)
        ).astype(np.float64) / 255.0   # Normalised 0–1 (deeper = smaller value in DA V2)

        curr_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # ── Optical-flow camera motion ────────────────────────────────────
        if prev_gray is not None:
            dx_px, dy_px, dtheta = _optical_flow_delta(prev_gray, curr_gray)

            # Scale pixel displacement to world units (empirical scale)
            world_scale = 0.012
            step_x =  dx_px * world_scale
            step_y = -dy_px * world_scale * 0.5   # altitude changes slowly
            step_z =  0.6 + abs(dy_px) * 0.003    # forward motion per segment

            # Apply rotation & translation
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

        # ── Unproject dense pixels into 3D ────────────────────────────────
        fx, fy, cx, cy = _estimate_intrinsics(w, h)

        # Sample pixel grid (denser in high-depth-gradient regions)
        stride = max(4, int(math.sqrt(h * w / max_points_per_frame)))
        ys, xs = np.meshgrid(
            np.arange(0, h, stride),
            np.arange(0, w, stride),
            indexing="ij",
        )
        ys = ys.flatten()
        xs = xs.flatten()

        # Depth values — invert because Depth Anything V2 stores
        # higher values for closer pixels when saved as greyscale PNG
        d_raw = depth_arr[ys, xs]                    # 0 = far, 1 = close
        metric_z = (1.5 + (1.0 - d_raw) * 3.5)      # map to 1.5 – 5.0 scene units

        # Filter: skip very far / unreliable depth estimates
        valid = (metric_z > 0.5) & (metric_z < 6.0)
        ys, xs, metric_z = ys[valid], xs[valid], metric_z[valid]

        if len(ys) == 0:
            continue

        # Camera-space 3D coords
        x_cam = (xs - cx) * metric_z / fx
        y_cam = (ys - cy) * metric_z / fy
        z_cam = metric_z

        # Rotate from camera to world using yaw only (simplified)
        cos_y, sin_y = math.cos(cam_yaw), math.sin(cam_yaw)
        x_world = x_cam * cos_y - z_cam * sin_y + cam_pos[0]
        y_world = y_cam                             + cam_pos[1]
        z_world = x_cam * sin_y + z_cam * cos_y   + cam_pos[2]

        # RGB from image
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

    pts_arr  = np.array(all_pts,   dtype=np.float32)
    cols_arr = np.array(all_colors, dtype=np.uint8)

    # Subsample to keep PLY manageable for web rendering
    max_total = 60_000
    if len(pts_arr) > max_total:
        idx_s = np.random.choice(len(pts_arr), max_total, replace=False)
        pts_arr  = pts_arr[idx_s]
        cols_arr = cols_arr[idx_s]

    # Remove extreme outlier points (beyond 3σ in each axis)
    for ax in range(3):
        col = pts_arr[:, ax]
        mu, sigma = col.mean(), col.std()
        if sigma > 0:
            keep = np.abs(col - mu) < 3.5 * sigma
            pts_arr  = pts_arr[keep]
            cols_arr = cols_arr[keep]

    # Write PLY
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
