"""
AeroRecon Semantic 3D Spatial Association & Object Localization Module
=======================================================================
Projects 2D YOLO detections and ByteTrack tracks into 3D world-space coordinates
using calibrated Stage 2 Depth Anything V2 maps and Stage 3 OpenCV SfM camera poses.

Pipeline:
  YOLO bbox + Stage 2 Relative Depth + Camera Intrinsics K + SfM Camera Pose (R, t)
  --> Robust Interior Depth Sampling
  --> SfM Inverse-Depth Calibrated Unprojection
  --> World Coordinate Transformation (monocular_sfm)
  --> Track-Level Multi-View Fusion (median XYZ across observations)
  --> Semantic 3D Markers & Summary JSON Generation

Outputs:
  outputs/video_semantic/
    semantic_objects.json    -- fused 3D objects and individual localized observations
    semantic_markers.json    -- 3D marker pins for Plotly/WebGL scene rendering
    semantic_summary.json    -- detection & localization category statistics
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import math
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Category & Visual Configuration ───────────────────────────────────────────
CATEGORY_MAP: Dict[str, str] = {
    # Persons
    "person": "person",
    # Vehicles
    "bicycle": "vehicle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    # Animals
    "bird": "animal",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "elephant": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal",
}

CATEGORY_COLORS: Dict[str, str] = {
    "person": "#22c55e",   # Emerald Green
    "vehicle": "#38bdf8",  # Sky Blue
    "animal": "#f59e0b",   # Amber
    "other": "#a855f7",    # Purple
}

CATEGORY_SYMBOLS: Dict[str, str] = {
    "person": "cross",
    "vehicle": "diamond",
    "animal": "circle",
    "other": "square",
}


def _estimate_intrinsics(w: int, h: int, fov_deg: float = 70.0) -> Tuple[float, float, float, float]:
    """Computes (fx, fy, cx, cy) from image dimensions and estimated horizontal FOV."""
    fov_rad = math.radians(fov_deg)
    fx = (w / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return fx, fy, cx, cy


class Semantic3DManager:
    """
    Manages 3D semantic grounding, object localization, and multi-view track fusion.
    """

    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else ROOT / "outputs"
        self.epsilon = 0.05
        self.coordinate_system = "monocular_sfm"

    def load_detections(self, detections_jsonl_path: Path) -> List[Dict]:
        """Loads 2D YOLO detections from detections.jsonl."""
        detections = []
        p = Path(detections_jsonl_path)
        if not p.exists():
            return detections

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    detections.append(record)
                except Exception:
                    continue
        return detections

    def load_depth_map(self, depth_dir: Path, frame_id: str, height: int, width: int) -> Optional[np.ndarray]:
        """
        Loads Stage 2 normalized depth map.
        Prefers float32 .npy, falls back to .png if .npy is absent.
        """
        depth_dir = Path(depth_dir)
        npy_path = depth_dir / f"depth_{frame_id}.npy"
        if npy_path.exists():
            try:
                arr = np.load(str(npy_path)).astype(np.float32)
                if arr.shape != (height, width):
                    pil = Image.fromarray(arr)
                    arr = np.array(pil.resize((width, height), Image.BILINEAR), dtype=np.float32)
                return arr
            except Exception:
                pass

        png_path = depth_dir / f"depth_{frame_id}.png"
        if png_path.exists():
            try:
                pil = Image.open(png_path).convert("L")
                if pil.size != (width, height):
                    pil = pil.resize((width, height), Image.BILINEAR)
                arr = np.array(pil, dtype=np.float32) / 255.0
                return arr
            except Exception:
                pass

        return None

    def load_reconstruction_metadata(self, recon_meta_path: Path) -> Dict[str, Any]:
        """Loads Stage 3 reconstruction metadata (camera poses and alignment stats)."""
        p = Path(recon_meta_path)
        if not p.exists():
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def sample_interior_depth(
        self,
        depth_map: np.ndarray,
        bbox: List[float],
        num_samples_per_axis: int = 5,
    ) -> Optional[float]:
        """
        Robustly samples relative depth from the central 50% interior of a 2D bounding box.
        Avoids edge boundaries to minimize background depth contamination.
        """
        x1, y1, x2, y2 = bbox
        h_img, w_img = depth_map.shape[:2]

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return None

        # Sample from the central 50% region
        inner_x1 = max(0.0, min(float(w_img - 1), x1 + 0.25 * bw))
        inner_x2 = max(0.0, min(float(w_img - 1), x2 - 0.25 * bw))
        inner_y1 = max(0.0, min(float(h_img - 1), y1 + 0.25 * bh))
        inner_y2 = max(0.0, min(float(h_img - 1), y2 - 0.25 * bh))

        if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
            cx = int(np.clip((x1 + x2) / 2.0, 0, w_img - 1))
            cy = int(np.clip((y1 + y2) / 2.0, 0, h_img - 1))
            return float(depth_map[cy, cx])

        xs = np.linspace(inner_x1, inner_x2, num_samples_per_axis)
        ys = np.linspace(inner_y1, inner_y2, num_samples_per_axis)
        grid_x, grid_y = np.meshgrid(xs, ys)

        sample_x = np.clip(np.round(grid_x.flatten()).astype(int), 0, w_img - 1)
        sample_y = np.clip(np.round(grid_y.flatten()).astype(int), 0, h_img - 1)

        d_samples = depth_map[sample_y, sample_x]
        valid_samples = d_samples[d_samples > 0.01]

        if len(valid_samples) == 0:
            cx = int(np.clip((x1 + x2) / 2.0, 0, w_img - 1))
            cy = int(np.clip((y1 + y2) / 2.0, 0, h_img - 1))
            return float(depth_map[cy, cx])

        return float(np.median(valid_samples))

    def project_detection_to_3d(
        self,
        bbox: List[float],
        depth_map: np.ndarray,
        R_mat: np.ndarray,
        t_vec: np.ndarray,
        intrinsics: Tuple[float, float, float, float],
        alignment_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Projects a single 2D bounding box to 3D world coordinates.

        Args:
          bbox             : [x1, y1, x2, y2] pixel coordinates.
          depth_map        : Normalized float32 relative depth map [0, 1].
          R_mat            : 3x3 global camera rotation matrix.
          t_vec            : 3x1 or 3-element global camera translation vector.
          intrinsics       : (fx, fy, cx, cy)
          alignment_params : {'scale_a': a, 'shift_b': b, 'fitted_z_range': [z_min, z_max]}

        Returns:
          Dict with world_position, camera_depth, and confidence if successful, else None.
        """
        x1, y1, x2, y2 = bbox
        cx_box = (x1 + x2) / 2.0
        cy_box = (y1 + y2) / 2.0

        d_med = self.sample_interior_depth(depth_map, bbox)
        if d_med is None:
            return None

        # Compute metric-aligned camera depth z_cam
        if alignment_params and alignment_params.get("scale_a", 0) > 0:
            a = float(alignment_params["scale_a"])
            b = float(alignment_params.get("shift_b", 0.0))
            z_bounds = alignment_params.get("fitted_z_range", [0.2, 50.0])
            inv_d = 1.0 / (d_med + self.epsilon)
            z_cam = a * inv_d + b
            z_cam = max(float(z_bounds[0]), min(float(z_bounds[1]), z_cam))
        else:
            # Fallback when alignment parameters are absent: conservative relative scaling
            z_cam = 1.0 + (1.0 - d_med) * 4.0

        if z_cam <= 0.05 or not np.isfinite(z_cam):
            return None

        fx, fy, px, py = intrinsics
        x_cam = (cx_box - px) * z_cam / fx
        y_cam = (cy_box - py) * z_cam / fy
        P_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)

        t_arr = np.array(t_vec, dtype=np.float64).flatten()
        R_arr = np.array(R_mat, dtype=np.float64)

        # Coordinate transformation to world frame: P_world = R^T (P_cam - t)
        P_world = R_arr.T @ (P_cam - t_arr)

        if not np.all(np.isfinite(P_world)):
            return None

        return {
            "world_position": [round(float(P_world[0]), 3), round(float(P_world[1]), 3), round(float(P_world[2]), 3)],
            "camera_depth": round(float(z_cam), 3),
            "depth_sample": round(float(d_med), 4),
        }

    def process_semantic_mapping(
        self,
        detections_path: Path,
        depth_dir: Path,
        recon_meta_path: Path,
        output_dir: Optional[Path] = None,
        image_dims: Tuple[int, int] = (4096, 2160),
    ) -> Dict[str, Any]:
        """
        Executes end-to-end 3D semantic mapping across all frames and performs track-level fusion.
        """
        output_dir = Path(output_dir) if output_dir else self.workspace_dir / "video_semantic"
        output_dir.mkdir(parents=True, exist_ok=True)

        detections = self.load_detections(detections_path)
        recon_meta = self.load_reconstruction_metadata(recon_meta_path)

        camera_poses = recon_meta.get("camera_poses", [])
        alignment_stats = recon_meta.get("depth_alignment", {}).get("alignment_statistics", [])

        # Index camera poses by frame stem
        poses_by_frame: Dict[str, Dict] = {}
        for cam in camera_poses:
            c_name = cam.get("name", "")
            stem = Path(c_name).stem
            poses_by_frame[stem] = cam
            poses_by_frame[c_name] = cam

        # Index depth alignment parameters by frame stem
        align_by_frame: Dict[str, Dict] = {}
        for stat in alignment_stats:
            fid = stat.get("frame_id", "")
            align_by_frame[fid] = stat

        width, height = image_dims
        intrinsics = _estimate_intrinsics(width, height)

        localized_observations = []
        unlocalized_count = 0

        # Step 1: Localize individual 2D detections in 3D
        for det in detections:
            frame_id = det.get("frame_id", "")
            bbox = det.get("bbox", [])
            class_name = det.get("class_name", "object")
            category = CATEGORY_MAP.get(class_name, "other")
            track_id = det.get("track_id")
            confidence = det.get("confidence", 0.0)
            timestamp_sec = det.get("timestamp_sec", 0.0)

            if len(bbox) != 4:
                unlocalized_count += 1
                continue

            cam_info = poses_by_frame.get(frame_id)
            if not cam_info or "R" not in cam_info or "t" not in cam_info:
                unlocalized_count += 1
                continue

            depth_map = self.load_depth_map(depth_dir, frame_id, height, width)
            if depth_map is None:
                unlocalized_count += 1
                continue

            align_info = align_by_frame.get(frame_id)
            proj_res = self.project_detection_to_3d(
                bbox=bbox,
                depth_map=depth_map,
                R_mat=np.array(cam_info["R"]),
                t_vec=np.array(cam_info["t"]),
                intrinsics=intrinsics,
                alignment_params=align_info,
            )

            if proj_res is None:
                unlocalized_count += 1
                continue

            obs_record = {
                "track_id": track_id,
                "class_id": det.get("class_id"),
                "class_name": class_name,
                "category": category,
                "confidence": round(float(confidence), 4),
                "source_frame_id": frame_id,
                "timestamp_sec": round(float(timestamp_sec), 4),
                "bbox": [round(float(b), 1) for b in bbox],
                "camera_depth": proj_res["camera_depth"],
                "world_position": {
                    "x": proj_res["world_position"][0],
                    "y": proj_res["world_position"][1],
                    "z": proj_res["world_position"][2],
                },
                "coordinate_system": self.coordinate_system,
                "localization_status": "localized",
            }
            localized_observations.append(obs_record)

        # Step 2: Track-Level Multi-View Fusion
        # Group observations by track_id + class_name (or single instance if track_id is None)
        track_groups: Dict[str, List[Dict]] = {}
        for idx, obs in enumerate(localized_observations):
            tid = obs.get("track_id")
            cname = obs.get("class_name", "object")
            if tid is not None:
                key = f"track_{tid}_{cname}"
            else:
                key = f"det_{obs['source_frame_id']}_{idx}_{cname}"

            if key not in track_groups:
                track_groups[key] = []
            track_groups[key].append(obs)

        fused_objects = []
        markers = []

        category_counts = {"persons": 0, "vehicles": 0, "animals": 0}
        class_counts: Dict[str, int] = {}

        for group_key, obs_list in track_groups.items():
            first_obs = obs_list[0]
            cname = first_obs["class_name"]
            cat = first_obs["category"]
            tid = first_obs["track_id"]

            # Compute robust median XYZ across all observations for this track
            all_x = [o["world_position"]["x"] for o in obs_list]
            all_y = [o["world_position"]["y"] for o in obs_list]
            all_z = [o["world_position"]["z"] for o in obs_list]

            fused_xyz = [
                round(float(np.median(all_x)), 3),
                round(float(np.median(all_y)), 3),
                round(float(np.median(all_z)), 3),
            ]

            mean_conf = round(float(np.mean([o["confidence"] for o in obs_list])), 4)
            source_frames = sorted(list(set(o["source_frame_id"] for o in obs_list)))

            fused_obj = {
                "track_id": tid,
                "class_name": cname,
                "category": cat,
                "confidence": mean_conf,
                "observation_count": len(obs_list),
                "source_frames": source_frames,
                "world_position": fused_xyz,
                "coordinate_system": self.coordinate_system,
                "localization_status": "localized",
            }
            fused_objects.append(fused_obj)

            # Update category and class counts
            if cat == "person":
                category_counts["persons"] += 1
            elif cat == "vehicle":
                category_counts["vehicles"] += 1
            elif cat == "animal":
                category_counts["animals"] += 1
            class_counts[cname] = class_counts.get(cname, 0) + 1

            # Format label for 3D marker
            track_str = f"#{tid}" if tid is not None else ""
            label_text = f"{cname.capitalize()} {track_str}".strip()

            marker = {
                "track_id": tid,
                "class_name": cname,
                "category": cat,
                "label": label_text,
                "confidence": mean_conf,
                "observation_count": len(obs_list),
                "world_position": fused_xyz,
                "color": CATEGORY_COLORS.get(cat, "#38bdf8"),
                "symbol": CATEGORY_SYMBOLS.get(cat, "diamond"),
            }
            markers.append(marker)

        # Step 3: Write Output JSON Files
        # 1. semantic_objects.json
        objects_payload = {
            "coordinate_system": self.coordinate_system,
            "total_objects": len(fused_objects),
            "objects": fused_objects,
            "observations": localized_observations,
        }
        with open(output_dir / "semantic_objects.json", "w", encoding="utf-8") as f:
            json.dump(objects_payload, f, indent=2)

        # 2. semantic_markers.json
        with open(output_dir / "semantic_markers.json", "w", encoding="utf-8") as f:
            json.dump(markers, f, indent=2)

        # 3. semantic_summary.json
        summary_payload = {
            "total_detections": len(detections),
            "localized_detections": len(localized_observations),
            "unlocalized_detections": unlocalized_count,
            "total_fused_objects": len(fused_objects),
            "category_counts": category_counts,
            "class_counts": class_counts,
            "coordinate_system": self.coordinate_system,
            "source_reconstruction": recon_meta.get("engine", "OpenCV Incremental SfM"),
            "source_depth_model": "depth-anything/Depth-Anything-V2-Small-hf",
        }
        with open(output_dir / "semantic_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        return summary_payload


# ── Standalone Entry Point ───────────────────────────────────────────────────

def run_semantic_3d_pipeline(
    detections_path: Path,
    depth_dir: Path,
    recon_meta_path: Path,
    output_dir: Path,
    image_dims: Tuple[int, int] = (4096, 2160),
) -> Dict[str, Any]:
    """
    Public entry point for Stage 4 Semantic 3D mapping pipeline.
    """
    manager = Semantic3DManager(workspace_dir=output_dir.parent)
    return manager.process_semantic_mapping(
        detections_path=detections_path,
        depth_dir=depth_dir,
        recon_meta_path=recon_meta_path,
        output_dir=output_dir,
        image_dims=image_dims,
    )
