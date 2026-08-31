import json
from pathlib import Path
import numpy as np
from plyfile import PlyData

ply_path = Path("outputs/video_reconstruction/model.ply")
markers_path = Path("outputs/video_semantic/semantic_markers.json")
recon_meta_path = Path("outputs/video_reconstruction/reconstruction_meta.json")

print("=== SPATIAL COORDINATE DIAGNOSTIC ===")

cloud_pts = None
if ply_path.exists():
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    px = np.array(v["x"], dtype=np.float64)
    py = np.array(v["y"], dtype=np.float64)
    pz = np.array(v["z"], dtype=np.float64)
    print(f"PLY Points: {len(px):,}")
    print(f"  X bounds: [{px.min():.3f}, {px.max():.3f}] (span={np.ptp(px):.3f}, mean={px.mean():.3f})")
    print(f"  Y bounds: [{py.min():.3f}, {py.max():.3f}] (span={np.ptp(py):.3f}, mean={py.mean():.3f})")
    print(f"  Z bounds: [{pz.min():.3f}, {pz.max():.3f}] (span={np.ptp(pz):.3f}, mean={pz.mean():.3f})")
    cloud_pts = np.column_stack([px, py, pz])

if markers_path.exists():
    with open(markers_path, "r", encoding="utf-8") as f:
        markers = json.load(f)
    print(f"\nSemantic Markers Count: {len(markers)}")
    if markers:
        mx = [m["world_position"][0] for m in markers]
        my = [m["world_position"][1] for m in markers]
        mz = [m["world_position"][2] for m in markers]
        print(f"  Marker X bounds: [{min(mx):.3f}, {max(mx):.3f}] (span={np.ptp(mx):.3f}, mean={np.mean(mx):.3f})")
        print(f"  Marker Y bounds: [{min(my):.3f}, {max(my):.3f}] (span={np.ptp(my):.3f}, mean={np.mean(my):.3f})")
        print(f"  Marker Z bounds: [{min(mz):.3f}, {max(mz):.3f}] (span={np.ptp(mz):.3f}, mean={np.mean(mz):.3f})")

        if cloud_pts is not None:
            dists = []
            inside_box = 0
            x_min, x_max = px.min(), px.max()
            y_min, y_max = py.min(), py.max()
            z_min, z_max = pz.min(), pz.max()

            for m in markers:
                m_pos = np.array(m["world_position"], dtype=np.float64)
                # Compute distance to nearest cloud point (subsample cloud for speed if large)
                sample_step = max(1, len(cloud_pts) // 5000)
                sub_cloud = cloud_pts[::sample_step]
                d = np.min(np.linalg.norm(sub_cloud - m_pos, axis=1))
                dists.append(d)
                if (x_min <= m_pos[0] <= x_max) and (y_min <= m_pos[1] <= y_max) and (z_min <= m_pos[2] <= z_max):
                    inside_box += 1

            print(f"\nMarker to Scene Proximity Analysis:")
            print(f"  Markers inside PLY bounding box: {inside_box} / {len(markers)} ({inside_box/len(markers)*100:.1f}%)")
            print(f"  Nearest PLY point dist (min):    {min(dists):.3f} m (SfM units)")
            print(f"  Nearest PLY point dist (median): {np.median(dists):.3f} m (SfM units)")
            print(f"  Nearest PLY point dist (mean):   {np.mean(dists):.3f} m (SfM units)")
            print(f"  Nearest PLY point dist (max):    {max(dists):.3f} m (SfM units)")

            print("\nSample Markers (first 5):")
            for i, m in enumerate(markers[:5]):
                lbl = m.get("label", "Object")
                pos = m["world_position"]
                print(f"  - Marker {i+1} [{lbl}]: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) -> nearest_cloud_pt={dists[i]:.2f}")

if recon_meta_path.exists():
    with open(recon_meta_path, "r", encoding="utf-8") as f:
        rmeta = json.load(f)
    cams = rmeta.get("camera_poses", [])
    print(f"\nCamera Poses Count: {len(cams)}")
    if cams:
        cx = [c["center"][0] for c in cams]
        cy = [c["center"][1] for c in cams]
        cz = [c["center"][2] for c in cams]
        print(f"  Camera X bounds: [{min(cx):.3f}, {max(cx):.3f}]")
        print(f"  Camera Y bounds: [{min(cy):.3f}, {max(cy):.3f}]")
        print(f"  Camera Z bounds: [{min(cz):.3f}, {max(cz):.3f}]")
