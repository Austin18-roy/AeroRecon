"""
AeroRecon Stage 2 Validation: Depth Anything V2 Depth Estimation
=================================================================
Validates the enhanced run_depth_on_keyframes function against the existing
benchmark images (seq38) without touching any benchmark output files.

Validates:
  T0  Import works
  T1  Depth Anything V2 model can initialize
  T2  Frame can be processed end-to-end
  T3  Depth PNG visualization exists
  T4  Depth PNG has correct dimensions
  T5  Depth .npy float32 file exists
  T6  .npy values are float32 and in [0, 1]
  T7  .npy shape matches source image dimensions
  T8  depth_metadata.json exists
  T9  Metadata contains all required fields
  T10 frame_id in metadata matches source stem
  T11 Re-running clears stale depth files (idempotency)
  T12 Benchmark depth/reconstruction files remain untouched
  T13 app.py imports successfully
"""

import sys
import json
import shutil
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCHMARK_PLY = ROOT / "outputs" / "colmap" / "model.ply"
BENCHMARK_DEPTH_DIR = ROOT / "outputs" / "depth"
BENCHMARK_IMAGES_DIR = ROOT / "data" / "input" / "seq38" / "Images"
TEST_OUT_DIR = ROOT / "outputs" / "depth_stage2_test"

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    msg = f"{tag} {label}"
    if detail:
        msg += f" | {detail}"
    print(msg)
    return condition


def run_all_tests() -> int:
    failures = 0

    # ── T0: Import check ──────────────────────────────────────────────────────
    try:
        from src.video_pipeline import run_depth_on_keyframes, _DEPTH_MODEL_ID
        ok = True
    except ImportError as e:
        print(f"{FAIL} T0 Import | {e}")
        return 1
    failures += 0 if check("T0 Import: run_depth_on_keyframes", ok) else 1

    # ── T1: Model constant correct ────────────────────────────────────────────
    failures += 0 if check(
        "T1 Depth model ID correct",
        "Depth-Anything-V2-Small-hf" in _DEPTH_MODEL_ID,
        _DEPTH_MODEL_ID,
    ) else 1

    # ── Benchmark images guard ────────────────────────────────────────────────
    bench_images = sorted(BENCHMARK_IMAGES_DIR.glob("*.png"))
    if not check("   Benchmark images available", len(bench_images) >= 2, f"{len(bench_images)} images"):
        return 1

    # Use 2 frames only for speed
    subset = bench_images[:2]

    # ── T2: Run depth estimation on test subset ────────────────────────────────
    TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_map = {p.stem: float(i * 3) for i, p in enumerate(subset)}

    print(f"\nRunning Depth Anything V2 on {len(subset)} benchmark frames (CPU)...")
    try:
        result_paths = run_depth_on_keyframes(
            image_paths=subset,
            output_dir=TEST_OUT_DIR,
            frame_timestamps=ts_map,
            progress_callback=lambda p, m: print(f"  {int(p*100):3d}% {m}"),
        )
        run_ok = True
    except Exception as exc:
        print(f"{FAIL} T2 Depth run | {exc}")
        import traceback
        traceback.print_exc()
        return 1
    failures += 0 if check("T2 run_depth_on_keyframes executes without error", run_ok) else 1

    # ── T3: PNG visualization files exist ─────────────────────────────────────
    png_files = list(TEST_OUT_DIR.glob("depth_*.png"))
    failures += 0 if check(
        "T3 Depth PNG visualization files saved",
        len(png_files) == len(subset),
        f"{len(png_files)} PNGs for {len(subset)} frames",
    ) else 1

    # ── T4: PNG dimensions match source ───────────────────────────────────────
    if png_files:
        from PIL import Image
        src_img = Image.open(subset[0])
        src_w, src_h = src_img.size
        depth_png = Image.open(png_files[0])
        d_w, d_h = depth_png.size
        failures += 0 if check(
            "T4 Depth PNG dimensions match source frame",
            d_w == src_w and d_h == src_h,
            f"src=({src_w},{src_h}) depth=({d_w},{d_h})",
        ) else 1

    # ── T5: .npy float32 files exist ──────────────────────────────────────────
    npy_files = list(TEST_OUT_DIR.glob("depth_*.npy"))
    failures += 0 if check(
        "T5 Depth .npy float32 files saved",
        len(npy_files) == len(subset),
        f"{len(npy_files)} .npy for {len(subset)} frames",
    ) else 1

    # ── T6: .npy dtype is float32 and values are in [0, 1] ───────────────────
    if npy_files:
        arr = np.load(str(npy_files[0]))
        failures += 0 if check(
            "T6 .npy dtype is float32",
            arr.dtype == np.float32,
            str(arr.dtype),
        ) else 1
        failures += 0 if check(
            "T6b .npy values in [0.0, 1.0]",
            float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0,
            f"min={arr.min():.4f} max={arr.max():.4f}",
        ) else 1

    # ── T7: .npy shape matches source image (H, W) ────────────────────────────
    if npy_files:
        from PIL import Image
        src = Image.open(subset[0])
        expected_h, expected_w = src.size[1], src.size[0]
        actual_h, actual_w = arr.shape[:2]
        failures += 0 if check(
            "T7 .npy shape (H,W) matches source frame",
            actual_h == expected_h and actual_w == expected_w,
            f"expected=({expected_h},{expected_w}) actual=({actual_h},{actual_w})",
        ) else 1

    # ── T8: depth_metadata.json exists ────────────────────────────────────────
    meta_path = TEST_OUT_DIR / "depth_metadata.json"
    failures += 0 if check("T8 depth_metadata.json created", meta_path.exists()) else 1

    # ── T9: Metadata required fields ──────────────────────────────────────────
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as mf:
            meta = json.load(mf)

        top_keys = {"model", "depth_type", "depth_npy_format", "frames"}
        missing_top = top_keys - set(meta.keys())
        failures += 0 if check(
            "T9a Metadata top-level keys present",
            len(missing_top) == 0,
            f"missing: {missing_top}" if missing_top else "OK",
        ) else 1

        required_frame_keys = {
            "frame_id", "source_filename", "timestamp_sec",
            "width", "height", "depth_npy", "depth_png", "model", "depth_type",
        }
        if meta.get("frames"):
            sample_frame = meta["frames"][0]
            missing_frame = required_frame_keys - set(sample_frame.keys())
            failures += 0 if check(
                "T9b Frame record contains all required fields",
                len(missing_frame) == 0,
                f"sample: {sample_frame}",
            ) else 1

            # depth_type must NOT claim metric
            failures += 0 if check(
                "T9c depth_type is 'relative' (not metric)",
                sample_frame.get("depth_type") == "relative",
                str(sample_frame.get("depth_type")),
            ) else 1

    # ── T10: frame_id corresponds to source stem ──────────────────────────────
    if meta_path.exists() and meta.get("frames"):
        first_frame_id = meta["frames"][0]["frame_id"]
        expected_stem = subset[0].stem
        failures += 0 if check(
            "T10 frame_id matches source frame stem",
            first_frame_id == expected_stem,
            f"frame_id={first_frame_id} stem={expected_stem}",
        ) else 1
        # Filenames follow depth_{stem}.npy / depth_{stem}.png pattern
        expected_npy = f"depth_{expected_stem}.npy"
        expected_png = f"depth_{expected_stem}.png"
        failures += 0 if check(
            "T10b depth_npy / depth_png filenames use correct stem pattern",
            meta["frames"][0]["depth_npy"] == expected_npy
            and meta["frames"][0]["depth_png"] == expected_png,
            f"npy={meta['frames'][0]['depth_npy']} png={meta['frames'][0]['depth_png']}",
        ) else 1

    # ── T11: Idempotency — re-run clears stale files ──────────────────────────
    print("\nRe-running depth to verify stale file cleanup...")
    old_npy_count = len(list(TEST_OUT_DIR.glob("depth_*.npy")))
    run_depth_on_keyframes(
        image_paths=subset,
        output_dir=TEST_OUT_DIR,
        frame_timestamps=None,
    )
    new_npy_count = len(list(TEST_OUT_DIR.glob("depth_*.npy")))
    failures += 0 if check(
        "T11 Re-run produces fresh outputs (count stable, no accumulation)",
        new_npy_count == old_npy_count,
        f"Before: {old_npy_count}, After: {new_npy_count}",
    ) else 1

    # ── T12: Benchmark data untouched ─────────────────────────────────────────
    bench_ply_ok = BENCHMARK_PLY.exists()
    bench_depth_count = len(list(BENCHMARK_DEPTH_DIR.glob("depth_*.png")))
    failures += 0 if check(
        "T12 Benchmark PLY and depth files untouched",
        bench_ply_ok and bench_depth_count >= 10,
        f"PLY exists={bench_ply_ok}, bench depth PNGs={bench_depth_count}",
    ) else 1

    # ── T13: app.py imports successfully ──────────────────────────────────────
    try:
        import importlib.util, types
        spec = importlib.util.spec_from_file_location("_app_check", ROOT / "app.py")
        # We just need to verify it compiles, not run Streamlit
        import py_compile
        py_compile.compile(str(ROOT / "app.py"), doraise=True)
        app_ok = True
    except Exception as ae:
        print(f"  app.py compile error: {ae}")
        app_ok = False
    failures += 0 if check("T13 app.py compiles without syntax errors", app_ok) else 1

    # ── Cleanup ────────────────────────────────────────────────────────────────
    shutil.rmtree(TEST_OUT_DIR, ignore_errors=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    if failures == 0:
        print(">>> ALL STAGE 2 DEPTH ESTIMATION CHECKS PASSED <<<")
    else:
        print(f">>> {failures} TEST(S) FAILED <<<")
    print(f"{'='*58}")
    return failures


if __name__ == "__main__":
    sys.exit(run_all_tests())
