import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.sfm_reconstruction import run_sfm_reconstruction

images = sorted((ROOT / "data" / "input" / "seq38" / "Images").glob("*.png"))
depth_dir = ROOT / "outputs" / "test_depth_align"
out_ply = ROOT / "outputs" / "sfm_quality_benchmark" / "model.ply"
out_ply.parent.mkdir(parents=True, exist_ok=True)

print(f"Running 10-frame SfM with calibrated inverse depth alignment...")
t0 = time.time()
cams, pt_count = run_sfm_reconstruction(
    image_paths=images,
    depth_dir=depth_dir,
    output_ply_path=out_ply,
    use_depth_densification=True,
    depth_density_per_frame=900,
    progress_callback=lambda p, m: print(f"  {int(p*100):3d}% {m}"),
)
elapsed = time.time() - t0

meta = json.loads((out_ply.parent / "reconstruction_meta.json").read_text(encoding="utf-8"))
print()
print("=" * 60)
print(f"Engine                 : {meta['engine']}")
print(f"Total frames           : {meta['frame_count']}")
print(f"Registered cameras     : {meta['registered_cameras']}")
print(f"Triangulated points    : {meta['triangulated_points_count']:,}")
print(f"Dense aligned points   : {meta['dense_points_count']:,}")
print(f"Total merged points    : {meta['total_points']:,}")
print(f"Aligned frames count   : {meta['depth_alignment']['aligned_frames']}")
print(f"Skipped frames count   : {meta['depth_alignment']['skipped_frames']}")
print(f"Runtime                : {elapsed:.2f}s")
for st in meta['depth_alignment']['alignment_statistics']:
    fid = st['frame_id']
    status = st['status']
    if status == 'aligned':
        print(f"  Frame {fid}: aligned | a={st.get('scale_a')}, b={st.get('shift_b')}, r2={st.get('r2')}, z_range={st.get('fitted_z_range')}")
    else:
        print(f"  Frame {fid}: skipped | reason={st.get('reason')}")
print("=" * 60)
