"""
AeroRecon Semantic 3D Spatial Association Architecture (Future / Experimental)
=============================================================================
Defines clean structural interfaces for associating 2D YOLO detections with reconstructed
3D spatial environments, camera poses, and depth maps.

Architecture Concept:
  2D Detection Bounding Boxes (YOLO11s)
  + Camera Pose Matrix (COLMAP / AI Agent)
  + Dense Depth Map (Depth Anything V2)
  -----------------------------------------
  = Georeferenced 3D Semantic Bounding Volumes & Hazard Points
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class Semantic3DManager:
    """
    Interface manager for 3D semantic grounding and projection.
    Provides decoupled accessors and geometric projection stubs.
    """

    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path("outputs")
        self.is_experimental = True

    def get_detections(self, detections_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Retrieves 2D bounding boxes and class labels from detection results.
        Interface function: decoupled from specific detector backend.
        """
        target_dir = Path(detections_dir) if detections_dir else self.workspace_dir / "detections"
        records = {}
        if target_dir.exists():
            for p in sorted(target_dir.glob("*.jpg")):
                records[p.stem] = {"annotated_image": str(p), "status": "Available"}
        return records

    def get_depth(self, depth_dir: Optional[Path] = None) -> Dict[str, str]:
        """
        Retrieves depth map paths for available frames.
        Interface function: decoupled from depth estimation model.
        """
        target_dir = Path(depth_dir) if depth_dir else self.workspace_dir / "depth"
        records = {}
        if target_dir.exists():
            for p in sorted(target_dir.glob("depth_*.png")):
                stem = p.stem.replace("depth_", "")
                records[stem] = str(p)
        return records

    def get_camera_poses(self, cameras_list: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Retrieves estimated camera extrinsics and focal parameters.
        """
        return cameras_list if cameras_list is not None else []

    def get_point_cloud(self, ply_path: Optional[Path] = None) -> Optional[str]:
        """
        Retrieves path to reconstructed 3D point cloud model.
        """
        p = Path(ply_path) if ply_path else self.workspace_dir / "colmap" / "model.ply"
        return str(p) if p.exists() else None

    def project_detection_to_3d(
        self,
        bbox_2d: Tuple[float, float, float, float],
        depth_map: np.ndarray,
        camera_pose: Dict,
        camera_intrinsics: Tuple[float, float, float, float],
    ) -> Dict:
        """
        Conceptual geometric projection: Projects 2D bounding box center + median depth
        into 3D world coordinate frame (X, Y, Z).

        Note: Marked experimental. Stored as clean interface for future semantic mapping.
        """
        xmin, ymin, xmax, ymax = bbox_2d
        cx_2d = int((xmin + xmax) / 2.0)
        cy_2d = int((ymin + ymax) / 2.0)

        h, w = depth_map.shape[:2]
        cx_2d = max(0, min(cx_2d, w - 1))
        cy_2d = max(0, min(cy_2d, h - 1))

        # Sample depth
        d_val = float(depth_map[cy_2d, cx_2d])
        fx, fy, px, py = camera_intrinsics
        cam_center = camera_pose.get("center", np.zeros(3))

        # Unproject stub
        z_metric = 1.0 + (1.0 - d_val) * 4.0
        x_cam = (cx_2d - px) * z_metric / fx
        y_cam = (cy_2d - py) * z_metric / fy

        world_3d = (float(cam_center[0] + x_cam), float(cam_center[1] + y_cam), float(cam_center[2] + z_metric))

        return {
            "center_3d": world_3d,
            "depth_sample": d_val,
            "status": "Experimental Interface Projection",
        }
