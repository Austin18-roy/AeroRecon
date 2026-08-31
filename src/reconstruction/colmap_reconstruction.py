"""
COLMAP Sparse Reconstruction Backend
====================================
Wraps and parses COLMAP Structure-from-Motion (SfM) sparse reconstruction artifacts
(images.bin, cameras.bin, points3D.bin, model.ply).
"""

from pathlib import Path
from typing import List, Dict, Optional
import struct
import numpy as np
from plyfile import PlyData


def qvec2rotmat(qvec: tuple = (1.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    """Converts a quaternion into a 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def load_colmap_cameras(images_bin_path: Path) -> List[Dict]:
    """Extracts camera poses and calculates world coordinates C = -R^T * t."""
    images_bin_path = Path(images_bin_path)
    if not images_bin_path.exists():
        return []
    try:
        cameras = []
        with open(images_bin_path, "rb") as f:
            num_reg = struct.unpack("Q", f.read(8))[0]
            for _ in range(num_reg):
                img_id = struct.unpack("I", f.read(4))[0]
                qvec = struct.unpack("4d", f.read(32))
                tvec = struct.unpack("3d", f.read(24))
                cam_id = struct.unpack("I", f.read(4))[0]
                name_chars = []
                while True:
                    c = f.read(1)
                    if c == b"\x00":
                        break
                    name_chars.append(c)
                name = b"".join(name_chars).decode("utf-8", errors="ignore")
                num_2d = struct.unpack("Q", f.read(8))[0]
                f.seek(num_2d * 24, 1)

                R = qvec2rotmat(qvec)
                center = -R.T @ np.array(tvec)
                cameras.append({
                    "id": img_id,
                    "name": name,
                    "center": center,
                })
        cameras.sort(key=lambda c: c["name"])
        return cameras
    except Exception:
        return []


def load_ply_points(ply_path: Path) -> Optional[Dict]:
    """Loads PLY points and colors."""
    p = Path(ply_path)
    if not p.exists():
        return None
    try:
        ply = PlyData.read(p)
        v = ply["vertex"].data
        x = np.array(v["x"], dtype=np.float32)
        y = np.array(v["y"], dtype=np.float32)
        z = np.array(v["z"], dtype=np.float32)
        has_rgb = all(c in v.dtype.names for c in ("red", "green", "blue"))
        if has_rgb:
            r = np.array(v["red"], dtype=np.uint8)
            g = np.array(v["green"], dtype=np.uint8)
            b = np.array(v["blue"], dtype=np.uint8)
            colors = [f"rgb({ri},{gi},{bi})" for ri, gi, bi in zip(r, g, b)]
        else:
            colors = "#38bdf8"
        return {"x": x, "y": y, "z": z, "colors": colors, "count": len(x)}
    except Exception:
        return None
