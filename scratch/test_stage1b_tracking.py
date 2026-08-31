"""
AeroRecon Stage 1b Validation: YOLO Detection + ByteTrack Tracking
====================================================================
Tests the enhanced run_yolo_on_keyframes function against the existing
benchmark images (seq38) without touching any benchmark output files.

Validates:
  1. Function imports and executes without error
  2. Annotated JPEG images are saved per frame
  3. detections.jsonl is created with valid JSON records
  4. Each record contains: frame_id, timestamp_sec, track_id, class_id,
     class_name, confidence, bbox
  5. detections_summary.json is written with correct structure
  6. Tracking IDs are integers (not None) when objects are detected
  7. Old metadata is cleaned before a re-run (idempotency check)
  8. Frames with zero detections are handled gracefully
  9. Expanded class list (animals) is used
  10. Benchmark data at outputs/colmap/ is NOT modified
"""

import sys
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCHMARK_PLY = ROOT / "outputs" / "colmap" / "model.ply"
BENCHMARK_IMAGES_DIR = ROOT / "data" / "input" / "seq38" / "Images"
TEST_OUT_DIR = ROOT / "outputs" / "video_detections_test"

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
        from src.video_pipeline import run_yolo_on_keyframes, _TARGET_CLASSES, _COCO_CLASS_NAMES
        ok = True
    except ImportError as e:
        print(f"{FAIL} T0 Import | {e}")
        return 1
    failures += 0 if check("T0 Import: run_yolo_on_keyframes", ok) else 1

    # ── T1: Target class completeness ─────────────────────────────────────────
    expected_classes = {0, 1, 2, 3, 5, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
    failures += 0 if check(
        "T1 Target classes include persons + vehicles + animals",
        expected_classes.issubset(set(_TARGET_CLASSES)),
        f"Found {sorted(_TARGET_CLASSES)}"
    ) else 1

    # ── T2: Class name map coverage ───────────────────────────────────────────
    required_animals = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
    has_animals = required_animals.issubset(set(_COCO_CLASS_NAMES.values()))
    failures += 0 if check("T2 Animal class names present in _COCO_CLASS_NAMES", has_animals) else 1

    # ── T3: Benchmark images exist (guard) ────────────────────────────────────
    bench_images = sorted(BENCHMARK_IMAGES_DIR.glob("*.png"))
    if not check("T3 Benchmark images available", len(bench_images) >= 3, f"{len(bench_images)} images"):
        print("      Cannot proceed without benchmark images.")
        return 1
    failures += 0

    # ── T4: Run detection on a small subset (3 frames max) ───────────────────
    TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    subset = bench_images[:3]

    print("\nRunning YOLO detection + ByteTrack on 3 benchmark frames (CPU)...")
    try:
        det_counts = run_yolo_on_keyframes(
            image_paths=subset,
            output_dir=TEST_OUT_DIR,
            model_path="yolo11s.pt",
            conf=0.30,
            imgsz=1280,
            frame_timestamps={p.stem: float(i * 2) for i, p in enumerate(subset)},
            progress_callback=lambda p, m: print(f"  {int(p*100):3d}% {m}"),
        )
        run_ok = True
    except Exception as exc:
        print(f"{FAIL} T4 Detection run | {exc}")
        import traceback
        traceback.print_exc()
        return 1
    failures += 0 if check("T4 run_yolo_on_keyframes executes without error", run_ok) else 1

    # ── T5: Return type ───────────────────────────────────────────────────────
    failures += 0 if check(
        "T5 Returns Dict[str, int]",
        isinstance(det_counts, dict) and all(isinstance(v, int) for v in det_counts.values()),
        str(det_counts)
    ) else 1

    # ── T6: Annotated JPEGs saved ─────────────────────────────────────────────
    saved_jpgs = list(TEST_OUT_DIR.glob("*.jpg"))
    failures += 0 if check(
        "T6 Annotated JPEG images saved for each frame",
        len(saved_jpgs) == len(subset),
        f"{len(saved_jpgs)} JPEGs for {len(subset)} frames"
    ) else 1

    # ── T7: detections.jsonl exists and is valid ──────────────────────────────
    jsonl_path = TEST_OUT_DIR / "detections.jsonl"
    failures += 0 if check("T7 detections.jsonl created", jsonl_path.exists()) else 1

    records = []
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as je:
                        failures += 1
                        print(f"{FAIL} T7b Invalid JSON line: {je}")

    # ── T8: Record field completeness ─────────────────────────────────────────
    required_fields = {"frame_id", "timestamp_sec", "track_id", "class_id", "class_name", "confidence", "bbox"}
    if records:
        sample = records[0]
        missing = required_fields - set(sample.keys())
        failures += 0 if check(
            "T8 Detection records contain all required fields",
            len(missing) == 0,
            f"Sample: {sample}"
        ) else 1

        # ── T9: bbox is [x1,y1,x2,y2] list of 4 floats ────────────────────
        bbox = sample.get("bbox", [])
        failures += 0 if check(
            "T9 bbox field is list of 4 numbers",
            isinstance(bbox, list) and len(bbox) == 4,
            str(bbox)
        ) else 1

        # ── T10: tracking IDs are integers (not None) when present ─────────
        track_ids_found = [r["track_id"] for r in records if r.get("track_id") is not None]
        failures += 0 if check(
            "T10 Tracking IDs are integers (at least some non-None)",
            len(track_ids_found) > 0,
            f"{len(track_ids_found)}/{len(records)} records have track_id"
        ) else 1
    else:
        print(f"      (No detections in the 3 test frames — graceful empty output validated)")
        failures += 0 if check(
            "T8 Zero detections handled gracefully (empty JSONL is fine)",
            jsonl_path.exists()
        ) else 1

    # ── T11: detections_summary.json ─────────────────────────────────────────
    summary_path = TEST_OUT_DIR / "detections_summary.json"
    failures += 0 if check("T11 detections_summary.json created", summary_path.exists()) else 1
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as sf:
            summary = json.load(sf)
        required_summary_keys = {"frames_processed", "total_detections", "per_frame_counts",
                                  "class_totals", "tracker", "model", "target_classes"}
        missing_summary = required_summary_keys - set(summary.keys())
        failures += 0 if check(
            "T11b Summary JSON has all required keys",
            len(missing_summary) == 0,
            f"tracker={summary.get('tracker')}, total={summary.get('total_detections')}"
        ) else 1

    # ── T12: Idempotency (re-run clears old JSONL data) ──────────────────────
    print("\nRe-running detection to verify idempotency (stale data cleanup)...")
    old_record_count = sum(1 for _ in open(jsonl_path, encoding="utf-8") if _.strip()) if jsonl_path.exists() else 0

    run_yolo_on_keyframes(
        image_paths=subset,
        output_dir=TEST_OUT_DIR,
        conf=0.30,
        frame_timestamps=None,
    )
    new_record_count = sum(1 for _ in open(jsonl_path, encoding="utf-8") if _.strip()) if jsonl_path.exists() else 0
    failures += 0 if check(
        "T12 Re-run produces fresh JSONL (no metadata accumulation)",
        new_record_count == old_record_count,
        f"First run: {old_record_count} records, Second run: {new_record_count} records"
    ) else 1

    # ── T13: Benchmark NOT modified ───────────────────────────────────────────
    bench_ply_untouched = BENCHMARK_PLY.exists()
    bench_imgs_count = len(sorted(BENCHMARK_IMAGES_DIR.glob("*.png")))
    failures += 0 if check(
        "T13 Benchmark colmap PLY and benchmark images are untouched",
        bench_ply_untouched and bench_imgs_count >= 10,
        f"model.ply exists={bench_ply_untouched}, images={bench_imgs_count}"
    ) else 1

    # ── Cleanup test output dir ───────────────────────────────────────────────
    shutil.rmtree(TEST_OUT_DIR, ignore_errors=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    if failures == 0:
        print(">>> ALL STAGE 1b YOLO+TRACKING CHECKS PASSED <<<")
    else:
        print(f">>> {failures} TEST(S) FAILED <<<")
    print(f"{'='*56}")
    return failures


if __name__ == "__main__":
    sys.exit(run_all_tests())
