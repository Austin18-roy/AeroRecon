"""
AeroRecon Video Frame Ingestion & Keyframe Selection Module
===========================================================
Extracts, inspects, and filters frames from UAV flight video recordings (.mp4, .mov, .avi, .mkv).

Features:
  - Video Inspection (Duration, FPS, Frame Count, Resolution W x H)
  - Configurable Sampling (Every N frames or Target Frame Count)
  - Keyframe Selection — Prototype (Laplacian Blur & Quality Scoring)
  - Isolated Output Path Handling (outputs/video_frames/)
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_VIDEO_FRAMES_DIR = ROOT / "outputs" / "video_frames"


def inspect_video(video_path: Path) -> Dict:
    """
    Inspects video file parameters using OpenCV.
    Returns: filename, duration_seconds, fps, total_frames, width, height, resolution_str
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    duration_sec = total_frames / fps if fps > 0 else 0.0

    cap.release()

    return {
        "filename": video_path.name,
        "duration_sec": duration_sec,
        "duration_str": f"{int(duration_sec // 60)}m {int(duration_sec % 60):02d}s" if duration_sec >= 60 else f"{duration_sec:.1f}s",
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "resolution_str": f"{width} × {height}",
        "path": str(video_path),
    }


def compute_blur_score(image: np.ndarray) -> float:
    """
    Computes Laplacian variance sharpness score.
    Higher values indicate sharp, in-focus imagery.
    """
    if image is None or image.size == 0:
        return 0.0
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_video_frames(
    video_path: Path,
    output_dir: Optional[Path] = None,
    target_frames: int = 20,
    sampling_step: Optional[int] = None,
    use_keyframe_selection: bool = True,
    progress_callback: Optional[callable] = None,
) -> List[Dict]:
    """
    Extracts sampled or keyframe-filtered frames from video.

    Args:
      video_path: Path to video file.
      output_dir: Destination folder (defaults to outputs/video_frames/).
      target_frames: Number of frames to extract.
      sampling_step: If specified, extracts every N frames instead of target count.
      use_keyframe_selection: If True, uses Laplacian blur filtering to select sharpest frames.
      progress_callback: Optional callback(fraction, message) for UI progress bars.

    Returns:
      List of dicts: [{"id": int, "path": Path, "frame_idx": int, "timestamp": float, "blur_score": float}]
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir) if output_dir else DEFAULT_VIDEO_FRAMES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    info = inspect_video(video_path)
    total_frames = info["total_frames"]
    fps = info["fps"]

    if total_frames <= 0:
        raise ValueError(f"Video has 0 frames or cannot be read: {video_path}")

    # Determine frame indices to inspect
    if sampling_step and sampling_step > 0:
        candidate_indices = list(range(0, total_frames, sampling_step))
        target_frames = len(candidate_indices)
    else:
        target_frames = max(1, min(target_frames, total_frames))
        candidate_indices = list(range(0, total_frames))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    extracted_records = []

    if use_keyframe_selection and len(candidate_indices) > target_frames:
        # Keyframe Selection — Prototype: Temporal Segment Partitioning + Sharpness Ranking
        num_segments = target_frames
        seg_size = len(candidate_indices) / num_segments

        for seg_i in range(num_segments):
            seg_start = int(seg_i * seg_size)
            seg_end = int((seg_i + 1) * seg_size)
            seg_candidates = candidate_indices[seg_start:seg_end]

            best_frame = None
            best_score = -1.0
            best_idx = seg_candidates[0] if seg_candidates else 0

            # Subsample candidates within segment for speed
            step_inside = max(1, len(seg_candidates) // 6)
            for f_idx in seg_candidates[::step_inside]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                score = compute_blur_score(frame)
                if score > best_score:
                    best_score = score
                    best_frame = frame
                    best_idx = f_idx

            if best_frame is not None:
                out_name = f"frame_{seg_i:04d}.png"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), best_frame)
                extracted_records.append({
                    "id": seg_i + 1,
                    "name": out_name,
                    "path": out_path,
                    "frame_idx": best_idx,
                    "timestamp": round(best_idx / fps, 2) if fps > 0 else 0.0,
                    "blur_score": round(best_score, 1),
                })

            if progress_callback:
                progress_callback(
                    (seg_i + 1) / num_segments,
                    f"Keyframe Selection — Prototype: Extracting segment {seg_i + 1}/{num_segments}"
                )

    else:
        # Uniform sampling interval
        step = max(1, total_frames // target_frames)
        selected_indices = list(range(0, total_frames, step))[:target_frames]

        for idx, f_idx in enumerate(selected_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            score = compute_blur_score(frame)
            out_name = f"frame_{idx:04d}.png"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame)

            extracted_records.append({
                "id": idx + 1,
                "name": out_name,
                "path": out_path,
                "frame_idx": f_idx,
                "timestamp": round(f_idx / fps, 2) if fps > 0 else 0.0,
                "blur_score": round(score, 1),
            })

            if progress_callback:
                progress_callback(
                    (idx + 1) / len(selected_indices),
                    f"Uniform Sampling: Extracting frame {idx + 1}/{len(selected_indices)}"
                )

    cap.release()
    return extracted_records
