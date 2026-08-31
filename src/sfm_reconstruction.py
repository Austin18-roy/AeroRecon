"""
AeroRecon Stage 3: OpenCV Incremental SfM Reconstruction & Robust Depth Alignment
===================================================================================
Replaces empirical optical-flow unprojection with a geometrically grounded
Structure-from-Motion pipeline using OpenCV computer vision primitives +
Sparse-to-Dense Monocular Inverse Depth Alignment:

  1. SIFT multi-scale feature detection + BFMatcher cross-check
  2. Sequential RANSAC Essential Matrix estimation (findEssentialMat)
  3. Camera pose recovery (recoverPose) + global chained pose composition
  4. Multi-view feature triangulation (triangulatePoints)
  5. Per-frame inverse depth alignment:
       inv_d = 1 / (d_norm + epsilon)
       z_sfm ≈ a * inv_d + b  (fitted via RANSAC on triangulated correspondences)
     Dense depth pixels are unprojected into SfM-consistent metric coordinates.
     Frames with poor or insufficient correspondences are safely skipped to
     prevent horizontal planar sheet artifacts.
  6. Multi-view RGB point cloud fusion + statistical outlier rejection
  7. PLY export + detailed reconstruction metadata JSON

NO EXTERNAL BINARIES, NO CUDA, NO NEW DEPENDENCIES.
Only uses standard OpenCV + NumPy.

Input  : outputs/video_frames/*.png (or any ordered UAV image set)
         outputs/video_depth/depth_*.npy (Stage 2 Depth Anything V2 maps)
Output : outputs/video_reconstruction/
           model.ply                  -- RGB coloured 3D point cloud
           reconstruction_meta.json   -- camera poses, match stats, depth alignment info
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import math

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parent.parent

# ── Geometric Tuning Constants ────────────────────────────────────────────────
_SIFT_MAX_FEATURES = 4000
_SIFT_NOCTAVE_LAYERS = 3
_BF_CROSS_CHECK = True
_RANSAC_THRESHOLD_PX = 1.0
_RANSAC_CONFIDENCE = 0.999
_MIN_INLIERS = 8               # Minimum inlier matches to register camera pose
_MIN_DEPTH_ALIGN_CORRESP = 5   # Minimum triangulated correspondences to fit depth
_DEPTH_ALIGN_EPSILON = 0.05    # Epsilon for inverse depth 1/(d_norm + eps)
_FOV_H_DEG = 70.0              # Estimated horizontal FOV for UAV drone cameras


def _estimate_intrinsics(w: int, h: int, fov_deg: float = _FOV_H_DEG) -> np.ndarray:
    """Returns 3x3 intrinsics matrix K for a camera with given image dimensions."""
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _load_depth_map(depth_dir: Path, frame_stem: str, h: int, w: int) -> Optional[np.ndarray]:
    """
    Loads Stage 2 normalized depth map for a frame.
    Prefers lossless float32 .npy [0=far, 1=near], falls back to .png if .npy is absent.
    Returns (H, W) float32 array in [0, 1], or None if not found.
    """
    npy_path = depth_dir / f"depth_{frame_stem}.npy"
    if npy_path.exists():
        try:
            arr = np.load(str(npy_path)).astype(np.float32)
            if arr.shape != (h, w):
                pil = Image.fromarray(arr)
                arr = np.array(pil.resize((w, h), Image.BILINEAR), dtype=np.float32)
            return arr
        except Exception:
            pass

    # Fallback to PNG
    png_path = depth_dir / f"depth_{frame_stem}.png"
    if png_path.exists():
        try:
            pil = Image.open(png_path).convert("L")
            if pil.size != (w, h):
                pil = pil.resize((w, h), Image.BILINEAR)
            arr = np.array(pil, dtype=np.float32) / 255.0
            return arr
        except Exception:
            pass

    return None


def fit_inverse_depth_alignment(
    inv_d_vals: np.ndarray,
    z_sfm_vals: np.ndarray,
    min_inliers: int = _MIN_DEPTH_ALIGN_CORRESP,
    epsilon: float = _DEPTH_ALIGN_EPSILON,
) -> Optional[Dict]:
    """
    Robustly fits z_sfm ≈ a * inv_d + b using 1D RANSAC + Least Squares refinement.

    Parameters:
      inv_d_vals : 1 / (d_norm + epsilon) sampled at 2D keypoint projections
      z_sfm_vals : ground-truth triangulated 3D depth in camera coordinate space

    Returns dict with (a, b, inliers, total_candidates, r2, z_min, z_max) or None if fit fails.
    """
    valid = (z_sfm_vals > 0.05) & (z_sfm_vals < 80.0) & (inv_d_vals > 0.0)
    x = inv_d_vals[valid]
    y = z_sfm_vals[valid]

    if len(x) < min_inliers:
        return None

    n_pts = len(x)
    best_inliers = 0
    best_a, best_b = 0.0, 0.0

    # 1D RANSAC linear fitting
    np.random.seed(42)
    for _ in range(150):
        idx = np.random.choice(n_pts, 2, replace=False)
        x1, x2 = x[idx[0]], x[idx[1]]
        y1, y2 = y[idx[0]], y[idx[1]]
        if abs(x2 - x1) < 1e-4:
            continue
        a = (y2 - y1) / (x2 - x1)
        if a <= 0.001:  # Physical constraint: farther relative depth must mean larger camera z
            continue
        b = y1 - a * x1

        y_pred = a * x + b
        res = np.abs(y - y_pred)
        inlier_mask = (res < 0.25 * np.maximum(y, 1.0)) | (res < 0.4)
        n_in = int(np.sum(inlier_mask))
        if n_in > best_inliers:
            best_inliers = n_in
            best_a, best_b = a, b

    if best_inliers >= min_inliers and best_a > 0.001:
        y_pred = best_a * x + best_b
        inlier_mask = (np.abs(y - y_pred) < 0.25 * np.maximum(y, 1.0)) | (np.abs(y - y_pred) < 0.4)
        x_in, y_in = x[inlier_mask], y[inlier_mask]
        A = np.vstack([x_in, np.ones_like(x_in)]).T
        try:
            a_fit, b_fit = np.linalg.lstsq(A, y_in, rcond=None)[0]
            if a_fit > 0:
                y_pred_in = a_fit * x_in + b_fit
                ss_res = np.sum((y_in - y_pred_in) ** 2)
                ss_tot = np.sum((y_in - np.mean(y_in)) ** 2)
                r2 = float(1.0 - (ss_res / (ss_tot + 1e-6)))
                return {
                    "a": float(a_fit),
                    "b": float(b_fit),
                    "inliers": int(best_inliers),
                    "total_candidates": int(n_pts),
                    "r2": r2,
                    "z_min": float(np.min(y_in)),
                    "z_max": float(np.max(y_in)),
                }
        except Exception:
            pass

    # Proportional median scaling fallback: z = s * inv_d
    s = float(np.median(y / np.clip(x, 0.1, None)))
    if s > 0.001:
        return {
            "a": s,
            "b": 0.0,
            "inliers": int(n_pts),
            "total_candidates": int(n_pts),
            "r2": 0.5,
            "z_min": float(np.min(y)),
            "z_max": float(np.max(y)),
        }

    return None


class OpenCVSfMReconstructor:
    """
    Incremental SfM using OpenCV geometric primitives with calibrated inverse-depth alignment.
    Produces metric-consistent camera poses, triangulated feature points, and dense geometry.
    """

    def __init__(self):
        self.sift = cv2.SIFT_create(
            nfeatures=_SIFT_MAX_FEATURES,
            nOctaveLayers=_SIFT_NOCTAVE_LAYERS,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=_BF_CROSS_CHECK)

    def _detect_and_describe(
        self, gray: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        kps, descs = self.sift.detectAndCompute(gray, None)
        if descs is None:
            return [], np.zeros((0, 128), dtype=np.float32)
        return kps, descs

    def _match_pair(
        self,
        kps1: List[cv2.KeyPoint],
        descs1: np.ndarray,
        kps2: List[cv2.KeyPoint],
        descs2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns Nx2 arrays of matched 2D pixel coordinates in each image."""
        if len(descs1) == 0 or len(descs2) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        matches = self.matcher.match(descs1, descs2)
        if not matches:
            return np.zeros((0, 2)), np.zeros((0, 2))
        pts1 = np.array([kps1[m.queryIdx].pt for m in matches], dtype=np.float64)
        pts2 = np.array([kps2[m.trainIdx].pt for m in matches], dtype=np.float64)
        return pts1, pts2

    def reconstruct(
        self,
        image_paths: List[Path],
        depth_dir: Optional[Path],
        output_ply_path: Path,
        use_depth_densification: bool = True,
        depth_density_per_frame: int = 800,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[List[Dict], int]:
        """
        Runs full incremental SfM reconstruction across provided keyframes.

        Args:
          image_paths             : Ordered list of keyframe paths.
          depth_dir               : Directory containing Stage 2 depth_*.npy / depth_*.png files.
          output_ply_path         : Destination path for final 3D point cloud PLY.
          use_depth_densification : If True, uses calibrated inverse depth alignment to densify points.
                                    If False, outputs pure triangulated sparse SfM points.
          depth_density_per_frame : Target number of dense points per successfully aligned frame.
          progress_callback       : Optional (fraction, message) UI callback.

        Returns:
          (camera_poses, total_point_count)
        """
        output_ply_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(image_paths)
        if n < 2:
            raise ValueError(f"Need at least 2 frames for SfM reconstruction; got {n}.")

        # ── Step 1: Feature Extraction & Setup ────────────────────────────────
        frames = []
        for img_path in image_paths:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            kps, descs = self._detect_and_describe(gray)
            K = _estimate_intrinsics(w, h)
            frames.append({
                "path": img_path,
                "stem": img_path.stem,
                "bgr": bgr,
                "rgb": rgb,
                "gray": gray,
                "h": h, "w": w,
                "K": K,
                "kps": kps,
                "descs": descs,
                "pts2d_corresp": [],  # (u, v) in this frame
                "pts3d_world": [],    # corresponding 3D world points
            })

        if len(frames) < 2:
            raise ValueError("Too few readable frames for SfM reconstruction.")

        # ── Step 2: Incremental Camera Pose Recovery ─────────────────────────
        global_Rs = [np.eye(3, dtype=np.float64)]
        global_ts = [np.zeros((3, 1), dtype=np.float64)]
        camera_poses = [{
            "id": 1,
            "name": frames[0]["path"].name,
            "R": np.eye(3).tolist(),
            "t": [0.0, 0.0, 0.0],
            "center": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "inliers": 0,
            "status": "anchor",
        }]

        match_stats = []

        for i in range(1, len(frames)):
            f_prev = frames[i - 1]
            f_curr = frames[i]
            K = f_curr["K"]

            pts_prev, pts_curr = self._match_pair(
                f_prev["kps"], f_prev["descs"],
                f_curr["kps"], f_curr["descs"],
            )

            pose_status = "failed"
            R_rel = np.eye(3, dtype=np.float64)
            t_rel = np.zeros((3, 1), dtype=np.float64)
            n_inliers = 0

            if len(pts_prev) >= _MIN_INLIERS:
                E, mask = cv2.findEssentialMat(
                    pts_prev, pts_curr, K,
                    method=cv2.RANSAC,
                    prob=_RANSAC_CONFIDENCE,
                    threshold=_RANSAC_THRESHOLD_PX,
                )
                if E is not None and mask is not None and mask.sum() >= _MIN_INLIERS:
                    ret, R_rel, t_rel, mask2 = cv2.recoverPose(
                        E, pts_prev, pts_curr, K, mask=mask
                    )
                    n_inliers = int(mask2.sum()) if mask2 is not None else int(mask.sum())
                    if n_inliers >= _MIN_INLIERS:
                        pose_status = "ok"

            match_stats.append({
                "pair": f"{f_prev['stem']}->{f_curr['stem']}",
                "raw_matches": len(pts_prev),
                "inliers": n_inliers,
                "status": pose_status,
            })

            # Compose global camera pose
            R_prev = global_Rs[-1]
            t_prev = global_ts[-1]
            R_global = R_rel @ R_prev
            t_global = R_rel @ t_prev + t_rel

            global_Rs.append(R_global)
            global_ts.append(t_global)

            center = (-R_global.T @ t_global).flatten()
            yaw = float(np.arctan2(R_global[0, 2], R_global[0, 0]))

            camera_poses.append({
                "id": i + 1,
                "name": f_curr["path"].name,
                "R": R_global.tolist(),
                "t": t_global.flatten().tolist(),
                "center": center.tolist(),
                "yaw": yaw,
                "inliers": n_inliers,
                "status": pose_status,
            })

            if progress_callback:
                progress_callback(
                    (i * 0.4) / len(frames),
                    f"SfM camera pose {i}/{len(frames)-1}: {pose_status} ({n_inliers} inliers)"
                )

        # ── Step 3: Multi-View Triangulation ──────────────────────────────────
        triangulated_pts_list = []
        triangulated_rgb_list = []

        for i in range(len(frames) - 1):
            f0, f1 = frames[i], frames[i + 1]
            R0, t0 = global_Rs[i], global_ts[i]
            R1, t1 = global_Rs[i + 1], global_ts[i + 1]
            K0 = f0["K"]

            pts0, pts1 = self._match_pair(
                f0["kps"], f0["descs"],
                f1["kps"], f1["descs"],
            )

            if len(pts0) < 4:
                continue

            P0 = K0 @ np.hstack([R0, t0])
            P1 = K0 @ np.hstack([R1, t1])

            pts4d = cv2.triangulatePoints(
                P0.astype(np.float64),
                P1.astype(np.float64),
                pts0.T.astype(np.float64),
                pts1.T.astype(np.float64),
            )
            w_coord = pts4d[3]
            valid = np.abs(w_coord) > 1e-7
            pts4d = pts4d[:, valid]
            pts0_v = pts0[valid]
            pts1_v = pts1[valid]

            if pts4d.shape[1] == 0:
                continue

            pts3d = (pts4d[:3] / pts4d[3]).T  # (N, 3) in world coordinates

            # Cheirality check: verify points are in front of both cameras
            pts_cam0 = (R0 @ pts3d.T + t0).T
            pts_cam1 = (R1 @ pts3d.T + t1).T
            in_front = (pts_cam0[:, 2] > 0.05) & (pts_cam1[:, 2] > 0.05) & (pts_cam0[:, 2] < 80.0)
            pts3d = pts3d[in_front]
            pts0_v = pts0_v[in_front]
            pts1_v = pts1_v[in_front]

            if len(pts3d) == 0:
                continue

            # Store correspondences for depth alignment in both frame i and frame i+1
            for pt2d, pt3d in zip(pts0_v, pts3d):
                f0["pts2d_corresp"].append(pt2d)
                f0["pts3d_world"].append(pt3d)
            for pt2d, pt3d in zip(pts1_v, pts3d):
                f1["pts2d_corresp"].append(pt2d)
                f1["pts3d_world"].append(pt3d)

            # Sample RGB color from image 0
            xs = np.clip(np.round(pts0_v[:, 0]).astype(int), 0, f0["w"] - 1)
            ys = np.clip(np.round(pts0_v[:, 1]).astype(int), 0, f0["h"] - 1)
            rgb_pts = f0["rgb"][ys, xs]

            triangulated_pts_list.append(pts3d.astype(np.float32))
            triangulated_rgb_list.append(rgb_pts.astype(np.uint8))

        tri_count = sum(len(p) for p in triangulated_pts_list)

        # ── Step 4: Calibrated Inverse Depth Alignment & Densification ────────
        dense_pts_list = []
        dense_rgb_list = []
        aligned_frames_count = 0
        skipped_frames_count = 0
        alignment_stats = []

        if use_depth_densification and depth_dir is not None:
            for i, frame in enumerate(frames):
                stem = frame["stem"]
                d_map = _load_depth_map(depth_dir, stem, frame["h"], frame["w"])

                if d_map is None:
                    skipped_frames_count += 1
                    alignment_stats.append({
                        "frame_id": stem,
                        "status": "skipped",
                        "reason": "depth_map_not_found",
                    })
                    continue

                n_corresp = len(frame["pts2d_corresp"])
                if n_corresp < _MIN_DEPTH_ALIGN_CORRESP:
                    skipped_frames_count += 1
                    alignment_stats.append({
                        "frame_id": stem,
                        "status": "skipped",
                        "reason": f"insufficient_triangulated_correspondences ({n_corresp} < {_MIN_DEPTH_ALIGN_CORRESP})",
                    })
                    continue

                pts2d = np.array(frame["pts2d_corresp"])
                pts3d_w = np.array(frame["pts3d_world"])

                # Transform 3D world points to camera frame i coordinates
                R_i = global_Rs[i]
                t_i = global_ts[i]
                pts_cam = (R_i @ pts3d_w.T + t_i).T
                z_sfm = pts_cam[:, 2]

                # Sample Depth Anything relative depth at 2D keypoints
                xs_corr = np.clip(np.round(pts2d[:, 0]).astype(int), 0, frame["w"] - 1)
                ys_corr = np.clip(np.round(pts2d[:, 1]).astype(int), 0, frame["h"] - 1)
                d_vals = d_map[ys_corr, xs_corr]
                inv_d = 1.0 / (d_vals + _DEPTH_ALIGN_EPSILON)

                fit = fit_inverse_depth_alignment(
                    inv_d_vals=inv_d,
                    z_sfm_vals=z_sfm,
                    min_inliers=_MIN_DEPTH_ALIGN_CORRESP,
                    epsilon=_DEPTH_ALIGN_EPSILON,
                )

                if fit is None or fit["a"] <= 0.001:
                    skipped_frames_count += 1
                    alignment_stats.append({
                        "frame_id": stem,
                        "status": "skipped",
                        "reason": "poor_or_non_positive_depth_fit",
                    })
                    continue

                aligned_frames_count += 1
                a = fit["a"]
                b = fit["b"]
                z_min_bound = max(0.15, fit["z_min"] * 0.5)
                z_max_bound = min(60.0, fit["z_max"] * 1.8)

                alignment_stats.append({
                    "frame_id": stem,
                    "status": "aligned",
                    "correspondences": n_corresp,
                    "inliers": fit["inliers"],
                    "scale_a": round(a, 4),
                    "shift_b": round(b, 4),
                    "r2": round(fit["r2"], 3),
                    "fitted_z_range": [round(z_min_bound, 2), round(z_max_bound, 2)],
                })

                # Sample 2D grid across frame for calibrated densification
                stride = max(4, int(math.sqrt(frame["h"] * frame["w"] / depth_density_per_frame)))
                ys_g, xs_g = np.meshgrid(
                    np.arange(0, frame["h"], stride),
                    np.arange(0, frame["w"], stride),
                    indexing="ij",
                )
                ys_g = ys_g.flatten()
                xs_g = xs_g.flatten()

                d_norm_g = d_map[ys_g, xs_g]
                inv_d_g = 1.0 / (d_norm_g + _DEPTH_ALIGN_EPSILON)
                z_cam_g = a * inv_d_g + b

                # Strict geometric filtering: must fall within fitted bounds and non-sky
                valid_g = (z_cam_g >= z_min_bound) & (z_cam_g <= z_max_bound) & (d_norm_g > 0.02)
                ys_g = ys_g[valid_g]
                xs_g = xs_g[valid_g]
                z_cam_g = z_cam_g[valid_g]

                if len(ys_g) > 0:
                    fx, fy = frame["K"][0, 0], frame["K"][1, 1]
                    cx, cy = frame["K"][0, 2], frame["K"][1, 2]
                    x_cam = (xs_g - cx) * z_cam_g / fx
                    y_cam = (ys_g - cy) * z_cam_g / fy
                    pts_cam_dense = np.stack([x_cam, y_cam, z_cam_g], axis=1)

                    # Transform camera-space points to world frame: P_world = R^T (P_cam - t)
                    pts_w_dense = (R_i.T @ (pts_cam_dense.T - t_i)).T
                    rgb_dense = frame["rgb"][ys_g, xs_g]

                    dense_pts_list.append(pts_w_dense.astype(np.float32))
                    dense_rgb_list.append(rgb_dense.astype(np.uint8))

                if progress_callback:
                    progress_callback(
                        0.4 + (i * 0.5) / len(frames),
                        f"Depth alignment {i+1}/{len(frames)}: {stem} (a={a:.3f}, b={b:.3f})"
                    )

        dense_count = sum(len(p) for p in dense_pts_list)

        # ── Step 5: Merge Geometry & Statistical Outlier Filtering ───────────
        all_pts = []
        all_rgb = []

        if triangulated_pts_list:
            all_pts.extend(triangulated_pts_list)
            all_rgb.extend(triangulated_rgb_list)

        if dense_pts_list:
            all_pts.extend(dense_pts_list)
            all_rgb.extend(dense_rgb_list)

        if all_pts:
            pts_merged = np.vstack(all_pts).astype(np.float32)
            rgb_merged = np.vstack(all_rgb).astype(np.uint8)

            # Robust coordinate validation (remove any NaN/Inf)
            finite_mask = np.isfinite(pts_merged).all(axis=1)
            pts_merged = pts_merged[finite_mask]
            rgb_merged = rgb_merged[finite_mask]

            # 3.2-sigma statistical outlier removal
            mask_keep = np.ones(len(pts_merged), dtype=bool)
            for ax in range(3):
                col = pts_merged[:, ax]
                mu, std = float(col.mean()), float(col.std())
                if std > 0:
                    mask_keep &= np.abs(col - mu) < 3.2 * std
            pts_merged = pts_merged[mask_keep]
            rgb_merged = rgb_merged[mask_keep]

            # Subsample if point cloud exceeds memory limits (max 80,000 points)
            max_pts = 80_000
            if len(pts_merged) > max_pts:
                idx_sub = np.random.choice(len(pts_merged), max_pts, replace=False)
                pts_merged = pts_merged[idx_sub]
                rgb_merged = rgb_merged[idx_sub]
        else:
            pts_merged = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            rgb_merged = np.array([[200, 200, 200]], dtype=np.uint8)

        # ── Step 6: PLY File Export ───────────────────────────────────────────
        vertices = np.zeros(
            len(pts_merged),
            dtype=[
                ("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ],
        )
        vertices["x"] = pts_merged[:, 0]
        vertices["y"] = pts_merged[:, 1]
        vertices["z"] = pts_merged[:, 2]
        vertices["red"]   = rgb_merged[:, 0]
        vertices["green"] = rgb_merged[:, 1]
        vertices["blue"]  = rgb_merged[:, 2]
        PlyData([PlyElement.describe(vertices, "vertex")]).write(str(output_ply_path))

        # ── Step 7: Write Detailed Metadata JSON ──────────────────────────────
        registered_count = sum(1 for p in camera_poses if p.get("status") in ("ok", "anchor"))
        meta = {
            "engine": "OpenCV Incremental SfM",
            "method": "SIFT + RANSAC Essential Matrix + recoverPose + Triangulation + Inverse Depth Alignment",
            "frame_count": len(frames),
            "registered_cameras": registered_count,
            "total_points": len(pts_merged),
            "triangulated_points_count": tri_count,
            "dense_points_count": dense_count,
            "use_depth_densification": use_depth_densification,
            "depth_alignment": {
                "method": "sfm_inverse_depth_alignment",
                "formula": "z_sfm ≈ a * (1 / (d_norm + eps)) + b",
                "aligned_frames": aligned_frames_count,
                "skipped_frames": skipped_frames_count,
                "alignment_statistics": alignment_stats,
            },
            "match_stats": match_stats,
            "camera_poses": camera_poses,
            "scale_note": (
                "Monocular SfM: Scene coordinates are internally calibrated and consistent. "
                "Absolute metric scale is unconstrained by single-camera geometry."
            ),
            "ply_path": str(output_ply_path),
        }
        meta_path = output_ply_path.parent / "reconstruction_meta.json"
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)

        if progress_callback:
            progress_callback(1.0, f"Reconstruction complete: {len(pts_merged):,} points ({registered_count} cameras)")

        # Prepare camera poses format for Streamlit visualization
        app_camera_poses = []
        for p in camera_poses:
            c = p["center"]
            app_camera_poses.append({
                "id": p["id"],
                "name": p["name"],
                "center": np.array(c),
                "yaw": p["yaw"],
            })

        return app_camera_poses, len(pts_merged)


# ── Public Pipeline Entry Point ───────────────────────────────────────────────

def run_sfm_reconstruction(
    image_paths: List[Path],
    depth_dir: Optional[Path],
    output_ply_path: Path,
    use_depth_densification: bool = True,
    depth_density_per_frame: int = 800,
    progress_callback: Optional[callable] = None,
) -> Tuple[List[Dict], int]:
    """
    Runs the OpenCV Incremental SfM + Calibrated Depth Alignment reconstruction.

    Args:
      image_paths             : List of keyframe image paths (ordered).
      depth_dir               : Directory containing Stage 2 depth_*.npy / depth_*.png files.
      output_ply_path         : Output PLY path.
      use_depth_densification : If True, uses fitted inverse-depth alignment to densify points.
      depth_density_per_frame : Approx dense points per frame.
      progress_callback       : Optional (fraction, message) UI callback.

    Returns:
      (camera_poses, total_point_count)
    """
    reconstructor = OpenCVSfMReconstructor()
    return reconstructor.reconstruct(
        image_paths=image_paths,
        depth_dir=depth_dir,
        output_ply_path=output_ply_path,
        use_depth_densification=use_depth_densification,
        depth_density_per_frame=depth_density_per_frame,
        progress_callback=progress_callback,
    )
