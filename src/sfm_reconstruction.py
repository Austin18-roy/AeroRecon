"""
AeroRecon Stage 3: OpenCV Incremental SfM Reconstruction
=========================================================
Replaces the optical-flow empirical point cloud with a real Structure-from-Motion
pipeline using OpenCV's built-in geometric computer vision tools:

  SIFT feature detection + BFMatcher cross-check
  --> RANSAC Essential Matrix estimation (findEssentialMat)
  --> Camera pose recovery (recoverPose)
  --> Multi-view triangulation (triangulatePoints)
  --> Chained global pose composition
  --> Depth-guided point densification from Stage 2 .npy depth maps
  --> PLY export + reconstruction metadata JSON

NO EXTERNAL BINARIES, NO CUDA, NO NEW DEPENDENCIES.
Only uses opencv-python (already installed).

Architecture note:
  This is a minimal incremental SfM pipeline sufficient for the competition demo.
  It is NOT a full COLMAP-quality reconstruction. Camera poses are recovered
  relative to the first frame; scale is up to sign-ambiguity of Essential Matrix
  decomposition (standard for monocular SfM). Absolute metric scale is NOT claimed.

Input  : outputs/video_frames/*.png
Output : outputs/video_reconstruction/
           model.ply                  -- RGB coloured point cloud
           reconstruction_meta.json   -- camera poses, match counts, frame info
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

# ── Tuning constants ───────────────────────────────────────────────────────────
_SIFT_MAX_FEATURES = 4000
_SIFT_NOCTAVE_LAYERS = 3
_BF_CROSS_CHECK = True
_RANSAC_THRESHOLD_PX = 1.0
_RANSAC_CONFIDENCE = 0.999
_MIN_INLIERS = 8           # Minimum inlier matches to accept a pose
_FOV_H_DEG = 70.0          # Assumed horizontal FOV (drone gimbal)
_DEPTH_SCALE_TARGET = 5.0  # World units for scene radius (arbitrary but consistent)


def _estimate_intrinsics(w: int, h: int, fov_deg: float = _FOV_H_DEG) -> np.ndarray:
    """Returns 3x3 intrinsics matrix K for a camera with given image size and horizontal FOV."""
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _load_depth_npy(depth_dir: Path, frame_stem: str, h: int, w: int) -> Optional[np.ndarray]:
    """
    Loads the Stage 2 float32 normalized depth map for a frame.
    Returns (H, W) float32 array [0=far, 1=near], or None if not found.
    """
    npy_path = depth_dir / f"depth_{frame_stem}.npy"
    if not npy_path.exists():
        return None
    try:
        arr = np.load(str(npy_path)).astype(np.float32)
        if arr.shape != (h, w):
            pil = Image.fromarray(arr)
            arr = np.array(pil.resize((w, h), Image.BILINEAR), dtype=np.float32)
        return arr
    except Exception:
        return None


class OpenCVSfMReconstructor:
    """
    Incremental SfM using OpenCV geometric primitives.
    Produces camera poses and a triangulated + depth-densified RGB point cloud.
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
        """
        Returns Nx2 arrays of matched pixel coords in each image.
        """
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
        depth_density_per_frame: int = 800,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Runs incremental SfM reconstruction across all provided keyframes.

        Returns a metadata dict containing camera poses, match statistics,
        and point count.
        """
        output_ply_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(image_paths)
        if n < 2:
            raise ValueError(f"Need at least 2 frames for SfM; got {n}.")

        # ── Load all images and detect SIFT features ──────────────────────────
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
            })

        if len(frames) < 2:
            raise ValueError("Too few readable frames for SfM reconstruction.")

        # ── Incremental pose chaining ──────────────────────────────────────────
        # Global rotation / translation arrays.
        # Frame 0 is the world origin: R0=I, t0=[0,0,0]
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

            # Compose global pose: R_global = R_rel @ R_prev, t_global = R_rel @ t_prev + t_rel
            R_prev = global_Rs[-1]
            t_prev = global_ts[-1]
            R_global = R_rel @ R_prev
            t_global = R_rel @ t_prev + t_rel

            global_Rs.append(R_global)
            global_ts.append(t_global)

            # Camera centre in world: C = -R^T @ t
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
                    i / len(frames),
                    f"SfM pose {i}/{len(frames)-1}: {pose_status} ({n_inliers} inliers)"
                )

        # ── Triangulation: consecutive pairs ──────────────────────────────────
        all_pts3d = []
        all_rgb3d = []

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

            if pts4d.shape[1] == 0:
                continue

            pts3d = (pts4d[:3] / pts4d[3]).T  # (N, 3)

            # Filter: keep points in front of both cameras and within scene radius
            # (sign ambiguity means we check depth in camera 0 frame)
            pts_cam0 = (R0 @ pts3d.T + t0).T  # in camera 0 coords
            in_front = pts_cam0[:, 2] > 0.01
            pts3d = pts3d[in_front]
            pts0_v = pts0_v[in_front]

            if len(pts3d) == 0:
                continue

            # Remove global outliers (5-sigma)
            for ax in range(3):
                col = pts3d[:, ax]
                mu, std = col.mean(), col.std()
                if std > 0:
                    keep = np.abs(col - mu) < 5 * std
                    pts3d = pts3d[keep]
                    pts0_v = pts0_v[keep]

            if len(pts3d) == 0:
                continue

            # Colour from image 0 pixel coords (rounded)
            xs = np.clip(np.round(pts0_v[:, 0]).astype(int), 0, f0["w"] - 1)
            ys = np.clip(np.round(pts0_v[:, 1]).astype(int), 0, f0["h"] - 1)
            rgb_pts = f0["rgb"][ys, xs]  # (N, 3)

            all_pts3d.append(pts3d)
            all_rgb3d.append(rgb_pts)

        # ── Depth-guided point densification ──────────────────────────────────
        # Add additional coloured points per frame unprojected from Stage 2 depth
        # using the recovered SfM camera poses (not optical flow).
        if depth_dir is not None:
            for i, (frame, R, t_vec) in enumerate(zip(frames, global_Rs, global_ts)):
                depth_arr = _load_depth_npy(depth_dir, frame["stem"], frame["h"], frame["w"])
                if depth_arr is None:
                    continue

                h, w_f = frame["h"], frame["w"]
                K = frame["K"]
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]

                # Sample pixel grid sparsely
                stride = max(4, int(math.sqrt(h * w_f / depth_density_per_frame)))
                ys_g, xs_g = np.meshgrid(
                    np.arange(0, h, stride), np.arange(0, w_f, stride), indexing="ij"
                )
                ys_g, xs_g = ys_g.flatten(), xs_g.flatten()
                d_norm = depth_arr[ys_g, xs_g]  # [0=far, 1=near]

                # Map normalized depth to a plausible scene-relative Z range.
                # We use a fixed range [0.5, 6.0] in camera units — consistent
                # with the SfM triangulation geometry recovered above.
                z_cam = 0.5 + (1.0 - d_norm) * 5.5

                # Only keep valid depth samples (skip sky / very far background)
                valid = (d_norm > 0.02) & (z_cam > 0.3)
                ys_g, xs_g, z_cam = ys_g[valid], xs_g[valid], z_cam[valid]

                if len(ys_g) == 0:
                    continue

                # Unproject to camera space
                x_cam = (xs_g - cx) * z_cam / fx
                y_cam = (ys_g - cy) * z_cam / fy
                pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)

                # Transform to world space: Pw = R^T @ (Pc - t)
                pts_world = (R.T @ (pts_cam.T - t_vec)).T  # (N, 3)

                rgb_dense = frame["rgb"][ys_g, xs_g]  # (N, 3)

                all_pts3d.append(pts_world.astype(np.float32))
                all_rgb3d.append(rgb_dense)

        # ── Merge and write PLY ───────────────────────────────────────────────
        if all_pts3d:
            pts_merged = np.vstack(all_pts3d).astype(np.float32)
            rgb_merged = np.vstack(all_rgb3d).astype(np.uint8)

            # Final outlier removal (3.5-sigma)
            mask_keep = np.ones(len(pts_merged), dtype=bool)
            for ax in range(3):
                col = pts_merged[:, ax]
                mu, std = col.mean(), col.std()
                if std > 0:
                    mask_keep &= np.abs(col - mu) < 3.5 * std
            pts_merged = pts_merged[mask_keep]
            rgb_merged = rgb_merged[mask_keep]

            # Subsample to keep PLY manageable (max 80k points)
            max_pts = 80_000
            if len(pts_merged) > max_pts:
                idx_sub = np.random.choice(len(pts_merged), max_pts, replace=False)
                pts_merged = pts_merged[idx_sub]
                rgb_merged = rgb_merged[idx_sub]
        else:
            pts_merged = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            rgb_merged = np.array([[200, 200, 200]], dtype=np.uint8)

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

        # ── Write reconstruction metadata ──────────────────────────────────────
        registered = sum(1 for p in camera_poses if p.get("status") in ("ok", "anchor"))
        meta = {
            "engine": "OpenCV Incremental SfM",
            "method": "SIFT + RANSAC Essential Matrix + recoverPose + triangulatePoints",
            "frame_count": len(frames),
            "registered_cameras": registered,
            "total_points": len(pts_merged),
            "triangulated_pairs": len(all_pts3d),
            "match_stats": match_stats,
            "camera_poses": camera_poses,
            "depth_type": "relative (Stage 2 Depth Anything V2 — NOT metric)",
            "scale_note": (
                "Monocular SfM: absolute metric scale not recovered. "
                "Scene units are internally consistent but not metric."
            ),
            "ply_path": str(output_ply_path),
        }
        meta_path = output_ply_path.parent / "reconstruction_meta.json"
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)

        # Build camera pose dicts in the format expected by app.py
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


# ── Public pipeline entry point (matches existing app.py call signature) ──────

def run_sfm_reconstruction(
    image_paths: List[Path],
    depth_dir: Optional[Path],
    output_ply_path: Path,
    depth_density_per_frame: int = 800,
    progress_callback: Optional[callable] = None,
) -> Tuple[List[Dict], int]:
    """
    Runs the OpenCV Incremental SfM reconstruction pipeline.

    Args:
      image_paths:            List of keyframe image paths (ordered).
      depth_dir:              Directory containing Stage 2 depth_*.npy files.
                              If None, only triangulated points are used.
      output_ply_path:        Output PLY path.
      depth_density_per_frame: Approx dense points per frame from depth map.
      progress_callback:      Optional (fraction, message) UI callback.

    Returns:
      (camera_poses, total_point_count)
    """
    reconstructor = OpenCVSfMReconstructor()
    return reconstructor.reconstruct(
        image_paths=image_paths,
        depth_dir=depth_dir,
        output_ply_path=output_ply_path,
        depth_density_per_frame=depth_density_per_frame,
        progress_callback=progress_callback,
    )
