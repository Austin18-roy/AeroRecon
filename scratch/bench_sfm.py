import time, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sfm_reconstruction import run_sfm_reconstruction

images = sorted(Path("data/input/seq38/Images").glob("*.png"))
depth_dir = Path("outputs/video_depth")
out_ply = Path("outputs/sfm_bench_test/model.ply")
out_ply.parent.mkdir(parents=True, exist_ok=True)

print(f"Running SfM on {len(images)} images...")
t0 = time.time()

cams, pts = run_sfm_reconstruction(
    image_paths=images,
    depth_dir=depth_dir,
    output_ply_path=out_ply,
    depth_density_per_frame=900,
    progress_callback=lambda p, m: print(f"  {int(p*100):3d}% {m}"),
)
elapsed = time.time() - t0

meta = json.loads((out_ply.parent / "reconstruction_meta.json").read_text())
print()
print("=" * 55)
print(f"Engine        : {meta['engine']}")
print(f"Frames        : {meta['frame_count']}")
print(f"Registered    : {meta['registered_cameras']}")
print(f"Total points  : {meta['total_points']:,}")
print(f"Runtime (s)   : {elapsed:.1f}")
for ms in meta["match_stats"]:
    print(f"  Pair {ms['pair']}: {ms['inliers']} inliers [{ms['status']}]")
print("=" * 55)
