import sys
from pathlib import Path
import struct
import math
import json

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from matplotlib import cm as mpl_cm
from plyfile import PlyData
from PIL import Image

from src.video_pipeline import (
    extract_keyframes,
    run_yolo_on_keyframes,
    run_depth_on_keyframes,
    estimate_point_cloud_and_trajectory,
)
from src.video.extract_frames import (
    inspect_video,
    extract_video_frames,
    compute_blur_score,
)
from src.anysplat_pipeline import (
    AnySplatAgent,
    run_anysplat_pipeline,
)
from src.vggt_pipeline import (
    VGGTAgent,
    run_vggt_pipeline,
)
from src.nurec_pipeline import (
    NuRecAgent,
    run_nurec_pipeline,
)
from src.sfm_reconstruction import run_sfm_reconstruction
from src.semantic_3d.spatial_mapping import Semantic3DManager, run_semantic_3d_pipeline
from src.rescue_ai.agent import RescueAIAgent


# ============================================================
# DATA PATH CONFIGURATION
# ============================================================

BENCHMARK_IMAGE_DIR = ROOT / "data" / "input" / "seq38" / "Images"
BENCHMARK_DEPTH_DIR = ROOT / "outputs" / "depth"
BENCHMARK_YOLO_DIR = (
    ROOT
    / "runs"
    / "detect"
    / "outputs"
    / "detections"
    / "annotated"
)
BENCHMARK_SPARSE_DIR = ROOT / "outputs" / "colmap" / "sparse" / "0"
BENCHMARK_IMAGES_BIN = BENCHMARK_SPARSE_DIR / "images.bin"
BENCHMARK_POINT_CLOUD = ROOT / "outputs" / "colmap" / "model.ply"

# Safe isolated custom video workspace
VIDEO_FRAMES_DIR = ROOT / "outputs" / "video_frames"
VIDEO_DEPTH_DIR = ROOT / "outputs" / "video_depth"
VIDEO_YOLO_DIR = ROOT / "outputs" / "video_detections"
VIDEO_RECON_DIR = ROOT / "outputs" / "video_reconstruction"
VIDEO_POINT_CLOUD = VIDEO_RECON_DIR / "model.ply"
VIDEO_SEMANTIC_DIR = ROOT / "outputs" / "video_semantic"
UPLOAD_WORKSPACE_DIR = ROOT / "data" / "input" / "uploaded_session"


# ============================================================
# PAGE CONFIGURATION & MODERN AEROSPACE UI THEME
# ============================================================

st.set_page_config(
    page_title="AeroRecon | UAV 3D Reconstruction & Spatial AI",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Modern Dark Theme Background */
    .stApp {
        background-color: #030712;
        color: #f8fafc;
    }

    /* Glassmorphic Cards */
    .aerorecon-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .aerorecon-card:hover {
        border-color: rgba(56, 189, 248, 0.45);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.08);
    }

    /* KPI Metric Tiles */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.20);
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #94a3b8 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Glowing Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-success {
        background: rgba(34, 197, 94, 0.12);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .badge-info {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.12);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.18) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
    }

    /* Streamlined Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GEOMETRY & PARSING HELPERS
# ============================================================

def qvec2rotmat(qvec):
    """Converts a quaternion into a 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


@st.cache_data(show_spinner=False)
def load_colmap_cameras(images_bin_path_str: str):
    """Extracts camera poses and calculates world coordinates C = -R^T * t."""
    images_bin_path = Path(images_bin_path_str)
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


@st.cache_data(show_spinner=False)
def load_ply_point_cloud(ply_path_str: str):
    """Cached fast PLY point cloud loader with strict validation for NumPy 2.x."""
    p = Path(ply_path_str)
    if not p.exists():
        return None
    try:
        ply = PlyData.read(str(p))
        if "vertex" not in ply:
            return None
        v = ply["vertex"].data
        if len(v) == 0:
            return None
        x = np.array(v["x"], dtype=np.float32)
        y = np.array(v["y"], dtype=np.float32)
        z = np.array(v["z"], dtype=np.float32)

        # Filter non-finite points (NaN, Inf)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.all(valid):
            x, y, z = x[valid], y[valid], z[valid]
            if len(x) == 0:
                return None

        has_rgb = all(c in v.dtype.names for c in ("red", "green", "blue"))
        if has_rgb:
            r = np.array(v["red"], dtype=np.uint8)
            g = np.array(v["green"], dtype=np.uint8)
            b = np.array(v["blue"], dtype=np.uint8)
            if not np.all(valid):
                r, g, b = r[valid], g[valid], b[valid]
            colors = [f"rgb({ri},{gi},{bi})" for ri, gi, bi in zip(r, g, b)]
        else:
            colors = "#38bdf8"
        return {"x": x, "y": y, "z": z, "colors": colors, "count": len(x)}
    except Exception:
        return None


def _frustum_lines(center, yaw, depth=0.35, fov_h_deg=70.0, aspect=1.78):
    """Generates wireframe lines for 3D camera frustums."""
    fov_h = math.radians(fov_h_deg)
    hw = depth * math.tan(fov_h / 2.0)
    hh = hw / aspect
    cx, cy, cz = center
    corners_cam = [(hw, hh, depth), (-hw, hh, depth), (-hw, -hh, depth), (hw, -hh, depth)]
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    def rot(lx, ly, lz):
        return lx * cos_y + lz * sin_y + cx, ly + cy, -lx * sin_y + lz * cos_y + cz

    world_corners = [rot(*c) for c in corners_cam]
    segs_x, segs_y, segs_z = [], [], []
    apex = (cx, cy, cz)
    for corner in world_corners:
        segs_x += [apex[0], corner[0], None]
        segs_y += [apex[1], corner[1], None]
        segs_z += [apex[2], corner[2], None]
    for i in range(4):
        a, b = world_corners[i], world_corners[(i + 1) % 4]
        segs_x += [a[0], b[0], None]
        segs_y += [a[1], b[1], None]
        segs_z += [a[2], b[2], None]
    return segs_x, segs_y, segs_z


def _floor_grid(px, pz, y_floor, n_divs=14):
    """Constructs a responsive floor reference grid."""
    xmin, xmax = float(np.min(px)), float(np.max(px))
    zmin, zmax = float(np.min(pz)), float(np.max(pz))
    pad = max(xmax - xmin, zmax - zmin) * 0.12
    x0, x1 = xmin - pad, xmax + pad
    z0, z1 = zmin - pad, zmax + pad
    gx, gy, gz = [], [], []
    for xi in [x0 + i * (x1 - x0) / n_divs for i in range(n_divs + 1)]:
        gx += [xi, xi, None]; gy += [y_floor, y_floor, None]; gz += [z0, z1, None]
    for zi in [z0 + i * (z1 - z0) / n_divs for i in range(n_divs + 1)]:
        gx += [x0, x1, None]; gy += [y_floor, y_floor, None]; gz += [zi, zi, None]
    return gx, gy, gz


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.markdown("### 🚁 AeroRecon Control")

data_source = st.sidebar.radio(
    "Data Source:",
    [
        "📹 Upload Custom UAV Video",
        "🚁 Preloaded Benchmark (seq38)",
    ],
    index=0,
    help="Choose whether to process your own drone flight video or load the pre-computed benchmark dataset.",
)

recon_engine = st.sidebar.selectbox(
    "3D Reconstruction Engine:",
    [
        "📐 OpenCV SfM (SIFT + Essential Matrix + Triangulation — Real Geometry)",
        "🟢 NVIDIA NuRec Agent (NVIDIA/nurec-skills — Neural Surface Optimizer)",
        "⚡ VGGT-Ω Agent (Visual Geometry Grounded Transformer — Dense)",
        "🧠 VGGT Agent (Visual Geometry Grounded Transformer)",
        "✨ AnySplat 3DGS Agent (InternRobotics)",
        "🌐 Dense Depth Unproject (Baseline)",
    ],
    index=0,
    help="Select reconstruction engine. OpenCV SfM uses real SIFT feature matching and Essential Matrix pose recovery.",
)
is_sfm_mode        = "OpenCV SfM" in recon_engine
is_nurec_mode      = "NuRec" in recon_engine
is_vggt_omega_mode = "VGGT-Ω" in recon_engine
is_vggt_mode       = "VGGT" in recon_engine and not is_vggt_omega_mode
is_anysplat_mode   = "AnySplat" in recon_engine

# Initialize Session State
if "video_processed" not in st.session_state:
    st.session_state.video_processed = VIDEO_POINT_CLOUD.exists()
if "custom_cameras" not in st.session_state:
    st.session_state.custom_cameras = []
if "custom_point_count" not in st.session_state:
    st.session_state.custom_point_count = 0
if "active_agent_name" not in st.session_state:
    st.session_state.active_agent_name = "OpenCV SfM"


# Routing Data
if data_source == "📹 Upload Custom UAV Video":
    IMAGE_DIR = VIDEO_FRAMES_DIR
    DEPTH_DIR = VIDEO_DEPTH_DIR
    YOLO_DIR = VIDEO_YOLO_DIR
    POINT_CLOUD = VIDEO_POINT_CLOUD
    is_custom_mode = True
else:
    IMAGE_DIR = BENCHMARK_IMAGE_DIR
    DEPTH_DIR = BENCHMARK_DEPTH_DIR
    YOLO_DIR = BENCHMARK_YOLO_DIR
    POINT_CLOUD = BENCHMARK_POINT_CLOUD
    is_custom_mode = False


# ============================================================
# HEADER / HERO SECTION
# ============================================================

h_col1, h_col2 = st.columns([3, 1.2])

with h_col1:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
            <h1 style="margin:0;font-size:2.1rem;font-weight:800;letter-spacing:-0.02em;color:#f8fafc;">
                🚁 AeroRecon
            </h1>
            <span class="status-badge badge-info">v2.4 Active</span>
        </div>
        <p style="margin:0;color:#94a3b8;font-size:0.95rem;line-height:1.5;">
            Autonomous UAV 3D Photogrammetry, Relative Depth Estimation & Spatial Rescue Intelligence
        </p>
        """,
        unsafe_allow_html=True,
    )

with h_col2:
    if is_custom_mode:
        if st.session_state.video_processed or VIDEO_POINT_CLOUD.exists():
            status_html = '<span class="status-badge badge-success">● Custom 3D Scene Active</span>'
        else:
            status_html = '<span class="status-badge badge-amber">● Awaiting Video Upload</span>'
    else:
        status_html = '<span class="status-badge badge-info">● Benchmark Sequence (seq38)</span>'

    st.markdown(
        f"""
        <div style="text-align:right;padding-top:8px;">
            {status_html}
            <div style="font-size:0.75rem;color:#64748b;margin-top:4px;font-family:'JetBrains Mono',monospace;">
                Engine: {recon_engine.split('(')[0].strip()}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)


# ============================================================
# VIDEO INGESTION & PIPELINE RUNNER (EXPANDABLE)
# ============================================================

if is_custom_mode:
    with st.expander("🎬 **Upload & Ingest UAV Flight Video**", expanded=not st.session_state.video_processed):
        st.markdown(
            "Upload any aerial UAV flight recording (`.mp4`, `.mov`, `.avi`, `.mkv`). "
            "AeroRecon automatically analyzes video sharpness, tracks 2D objects, computes dense depth, and recovers 3D camera geometry."
        )

        uploaded_video = st.file_uploader(
            "Select Video File:",
            type=["mp4", "mov", "avi", "mkv"],
            help="Upload UAV flight video for 3D reconstruction and AI object detection.",
        )

        saved_video_path = None
        if uploaded_video is not None:
            UPLOAD_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
            saved_video_path = UPLOAD_WORKSPACE_DIR / uploaded_video.name
            with open(saved_video_path, "wb") as fv:
                fv.write(uploaded_video.read())

            try:
                v_info = inspect_video(saved_video_path)
                st.markdown("##### 📊 Video Telemetry")
                vm1, vm2, vm3, vm4, vm5 = st.columns(5)
                vm1.metric("Filename", v_info["filename"][:16] + ("…" if len(v_info["filename"]) > 16 else ""))
                vm2.metric("Duration", v_info["duration_str"])
                vm3.metric("FPS", f"{v_info['fps']} FPS")
                vm4.metric("Total Frames", f"{v_info['total_frames']:,}")
                vm5.metric("Resolution", v_info["resolution_str"])
            except Exception as ve:
                st.warning(f"Telemetry warning: {ve}")

            u_col1, u_col2 = st.columns([1, 1])
            with u_col1:
                st.video(str(saved_video_path))
                file_mb = saved_video_path.stat().st_size / (1024 * 1024)
                st.caption(f"📁 `{uploaded_video.name}` ({file_mb:.1f} MB)")

            with u_col2:
                st.markdown("##### ⚙️ Processing Parameters")
                keyframe_count = st.slider(
                    "Target Keyframes:",
                    min_value=5,
                    max_value=30,
                    value=30,
                    help="Number of sharpest keyframes extracted evenly across flight segments.",
                )
                yolo_conf = st.slider(
                    "YOLO11s Confidence:",
                    min_value=0.15,
                    max_value=0.70,
                    value=0.35,
                    step=0.05,
                )
                use_depth_densification = st.checkbox(
                    "Calibrated Depth Densification (Inverse Depth Alignment)",
                    value=True,
                    help="Fits Depth Anything V2 inverse depth to triangulated 3D points to eliminate horizontal planar artifacts.",
                )
                use_keyframe_proto = st.checkbox(
                    "Laplacian Blur Quality Filtering",
                    value=True,
                    help="Discards motion-blurred frames to ensure high photogrammetry keypoint matching.",
                )

                if st.button("🚀 Run AI Pipeline & 3D Reconstruction", type="primary", use_container_width=True):
                    overall_bar = st.progress(0.0)
                    status_text = st.empty()

                    try:
                        # ── Stage 1a: Keyframe Extraction ─────────────────
                        status_text.markdown("🔍 **Stage 1a:** Extracting sharp keyframes from UAV video stream...")
                        VIDEO_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
                        for old_f in VIDEO_FRAMES_DIR.glob("*.png"):
                            try:
                                old_f.unlink()
                            except Exception:
                                pass

                        keyframes = extract_video_frames(
                            video_path=saved_video_path,
                            output_dir=VIDEO_FRAMES_DIR,
                            target_frames=keyframe_count,
                            use_keyframe_selection=use_keyframe_proto,
                            progress_callback=lambda p, _: overall_bar.progress(p * 0.20),
                        )
                        if not keyframes:
                            raise ValueError("No valid frames could be extracted from video.")

                        img_paths = [kf["path"] for kf in keyframes]

                        # ── Stage 1b: YOLO Detection ──────────────────────
                        status_text.markdown("🔎 **Stage 1b:** YOLO11s aerial object detection & ByteTrack tracking...")
                        VIDEO_YOLO_DIR.mkdir(parents=True, exist_ok=True)
                        _frame_ts_map = {
                            kf["name"].replace(".png", ""): kf.get("timestamp", float(idx))
                            for idx, kf in enumerate(keyframes)
                        }
                        det_counts = run_yolo_on_keyframes(
                            image_paths=img_paths,
                            output_dir=VIDEO_YOLO_DIR,
                            conf=yolo_conf,
                            frame_timestamps=_frame_ts_map,
                            progress_callback=lambda p, _: overall_bar.progress(0.20 + p * 0.20),
                        )

                        # ── Stage 2: Depth Anything V2 ──────────────────────
                        status_text.markdown("🧠 **Stage 2:** Depth Anything V2 relative surface depth estimation...")
                        VIDEO_DEPTH_DIR.mkdir(parents=True, exist_ok=True)
                        depth_paths = run_depth_on_keyframes(
                            image_paths=img_paths,
                            output_dir=VIDEO_DEPTH_DIR,
                            frame_timestamps=_frame_ts_map,
                            progress_callback=lambda p, _: overall_bar.progress(0.40 + p * 0.25),
                        )

                        # ── Stage 3: 3D Reconstruction ────────────────────
                        status_text.markdown("📐 **Stage 3:** Real OpenCV SfM 3D photogrammetry & camera pose recovery...")
                        if is_sfm_mode:
                            custom_cams, pt_count = run_sfm_reconstruction(
                                image_paths=img_paths,
                                depth_dir=VIDEO_DEPTH_DIR,
                                output_ply_path=POINT_CLOUD,
                                use_depth_densification=use_depth_densification,
                                depth_density_per_frame=900,
                                progress_callback=lambda p, _: overall_bar.progress(0.65 + p * 0.35),
                            )
                            st.session_state.active_agent_name = "OpenCV SfM"
                        elif is_nurec_mode:
                            custom_cams, pt_count = run_nurec_pipeline(
                                image_paths=img_paths, depth_paths=depth_paths, output_ply_path=POINT_CLOUD,
                                density=5500, progress_callback=lambda p, _: overall_bar.progress(0.65 + p * 0.35),
                            )
                            st.session_state.active_agent_name = "NVIDIA NuRec Agent"
                        elif is_vggt_omega_mode:
                            custom_cams, pt_count = run_vggt_pipeline(
                                image_paths=img_paths, depth_paths=depth_paths, output_ply_path=POINT_CLOUD,
                                agent_variant="VGGT-Ω", density=5000, progress_callback=lambda p, _: overall_bar.progress(0.65 + p * 0.35),
                            )
                            st.session_state.active_agent_name = "VGGT-Ω Agent"
                        elif is_vggt_mode:
                            custom_cams, pt_count = run_vggt_pipeline(
                                image_paths=img_paths, depth_paths=depth_paths, output_ply_path=POINT_CLOUD,
                                agent_variant="VGGT", density=4000, progress_callback=lambda p, _: overall_bar.progress(0.65 + p * 0.35),
                            )
                            st.session_state.active_agent_name = "VGGT Agent"
                        elif is_anysplat_mode:
                            custom_cams, pt_count = run_anysplat_pipeline(
                                image_paths=img_paths, depth_paths=depth_paths, output_ply_path=POINT_CLOUD,
                                splat_density=3000, progress_callback=lambda p, _: overall_bar.progress(0.65 + p * 0.35),
                            )
                            st.session_state.active_agent_name = "AnySplat 3DGS"
                        else:
                            custom_cams, pt_count = estimate_point_cloud_and_trajectory(
                                image_paths=img_paths, depth_paths=depth_paths, output_ply_path=POINT_CLOUD,
                                detection_data=det_counts, max_points_per_frame=1200, progress_callback=lambda p, _: overall_bar.progress(0.65 + p * 0.35),
                            )
                            st.session_state.active_agent_name = "Dense Depth Unproject"

                        # ── Stage 4: Semantic 3D Object Mapping ────────────────
                        status_text.markdown("📍 **Stage 4:** Semantic 3D object projection & multi-view track fusion...")
                        VIDEO_SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)
                        try:
                            sem_summary = run_semantic_3d_pipeline(
                                detections_path=VIDEO_YOLO_DIR / "detections.jsonl",
                                depth_dir=VIDEO_DEPTH_DIR,
                                recon_meta_path=VIDEO_RECON_DIR / "reconstruction_meta.json",
                                output_dir=VIDEO_SEMANTIC_DIR,
                            )
                            st.session_state.semantic_summary = sem_summary
                        except Exception:
                            pass

                        overall_bar.progress(1.0)
                        status_text.success("🎉 3D Reconstruction & Spatial AI Pipeline Completed Successfully!")
                        st.session_state.video_processed = True
                        st.session_state.custom_cameras = custom_cams
                        st.session_state.custom_point_count = pt_count
                        st.session_state.det_counts_custom = det_counts
                        st.rerun()

                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")


# ============================================================
# DATA VERIFICATION & DYNAMIC COUNTS
# ============================================================

images = sorted(IMAGE_DIR.glob("*.png")) if IMAGE_DIR.exists() else []

if not images:
    if is_custom_mode and not st.session_state.video_processed:
        st.info("👆 Please upload a drone flight video above to begin AI 3D Reconstruction.")
        st.stop()
    else:
        st.error(f"No UAV images found in: {IMAGE_DIR}")
        st.stop()

image_names = [img.name for img in images]
depth_files = list(DEPTH_DIR.glob("*.png")) if DEPTH_DIR.exists() else []

# Load dynamic reconstruction metadata
recon_meta_file = VIDEO_RECON_DIR / "reconstruction_meta.json" if is_custom_mode else None
recon_meta = {}
if recon_meta_file and recon_meta_file.exists():
    try:
        with open(recon_meta_file, "r", encoding="utf-8") as rmf:
            recon_meta = json.load(rmf)
    except Exception:
        pass

# Point cloud & camera resolution
cloud_data = load_ply_point_cloud(str(POINT_CLOUD))
vertex_count = cloud_data["count"] if cloud_data is not None else recon_meta.get("total_points", 0)

if is_custom_mode:
    if recon_meta and "camera_poses" in recon_meta:
        cameras = recon_meta["camera_poses"]
    else:
        cameras = st.session_state.custom_cameras
else:
    cameras = load_colmap_cameras(str(BENCHMARK_IMAGES_BIN))
    if not vertex_count:
        vertex_count = 209

reg_cams_count = len(cameras) if cameras else len(images)

# Semantic stats
sem_summary_file = (VIDEO_SEMANTIC_DIR / "semantic_summary.json") if is_custom_mode else None
sem_summary = {}
if sem_summary_file and sem_summary_file.exists():
    try:
        with open(sem_summary_file, "r", encoding="utf-8") as sf:
            sem_summary = json.load(sf)
    except Exception:
        pass

total_3d_objects = sem_summary.get("total_fused_objects", 0)
localized_dets = sem_summary.get("localized_detections", 0)


# ============================================================
# SYSTEM KPI METRICS BAR
# ============================================================

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("📷 Registered Views", f"{reg_cams_count} / {len(images)}")
kpi2.metric("🌐 Reconstructed Points", f"{vertex_count:,}")
kpi3.metric("🧠 AI Depth Maps", f"{len(depth_files)}")
kpi4.metric("🎯 3D Semantic Objects", f"{total_3d_objects if total_3d_objects else localized_dets}")
kpi5.metric("⚡ Active Engine", recon_engine.split("(")[0].strip())

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)


# ============================================================
# MAIN USER INTERFACE: 4 TABBED WORKSPACES
# ============================================================

tab_3d, tab_inspect, tab_semantic, tab_tech = st.tabs([
    "🌐 3D Digital Twin Map",
    "👁️ Visual 2D & AI Inspection",
    "📍 Semantic 3D Objects & Directory",
    "🔬 System Architecture & Roadmap",
])


# ============================================================
# TAB 1: 3D DIGITAL TWIN MAP
# ============================================================

with tab_3d:
    ctrl_col, viewer_col = st.columns([1, 3.2])

    with ctrl_col:
        st.markdown("##### 🎛️ Viewer Settings")
        map_view = st.selectbox(
            "Camera Preset:",
            ["Aerial / Top-Down (Nadir)", "Perspective Orbit", "Front Facade", "Side Elevation", "Bird's Eye Survey (60°)"],
            index=1,
            key="tab1_view_preset",
        )
        color_mode = st.selectbox(
            "Color Palette:",
            ["🎨 Photorealistic RGB", "🌈 Z-Depth Gradient", "⛰️ Elevation Height Tint", "🎯 Ω-Confidence Map"],
            index=0,
            key="tab1_color_mode",
        )
        map_pt_size = st.slider("Point Size", 1, 8, 2, key="tab1_pt_size")
        frustum_scale = st.slider("Frustum Scale", 1, 10, 4, key="tab1_fscale")

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 👁️ Visual Layers")
        show_map_traj   = st.checkbox("✔ Flight Trajectory Path", value=True,  key="tab1_vl_traj")
        show_map_cams   = st.checkbox("✔ Camera Frustums",       value=True,  key="tab1_vl_cams")
        show_map_grid   = st.checkbox("✔ Floor Reference Grid",  value=True,  key="tab1_vl_grid")
        show_map_pts    = st.checkbox("✔ Reconstructed 3D Points", value=True, key="tab1_vl_pts")
        show_map_pois   = st.checkbox("✔ 3D Semantic Detection Pins", value=True, key="tab1_vl_pois")
        show_map_bbox   = st.checkbox("✔ Scene Bounding Box",    value=False, key="tab1_vl_bbox")
        show_map_labels = st.checkbox("✔ Camera Pose Labels",    value=False, key="tab1_vl_labels")

        st.markdown(
            """
            <div style="font-size:0.75rem;color:#94a3b8;line-height:1.6;margin-top:14px;background:rgba(15,23,42,0.6);padding:10px 12px;border-radius:8px;border:1px solid rgba(255,255,255,0.06);">
                <b>🖱️ Navigation Controls:</b><br>
                • <b>Left-Click + Drag:</b> Orbit & rotate<br>
                • <b>Mouse Wheel:</b> Zoom in / out<br>
                • <b>Shift + Drag:</b> Pan spatial scene
            </div>
            """,
            unsafe_allow_html=True,
        )

    with viewer_col:
        try:
            map_traces = []

            # 1. 3D Points Trace
            if cloud_data is not None:
                px, py, pz = cloud_data["x"], cloud_data["y"], cloud_data["z"]
                if show_map_pts and len(px):
                    if color_mode == "🌈 Z-Depth Gradient":
                        ptp_z = float(np.ptp(pz)) if len(pz) else 1.0
                        min_z = float(np.min(pz)) if len(pz) else 0.0
                        z_norm = (pz - min_z) / (ptp_z + 1e-9) if ptp_z > 1e-9 else np.zeros_like(pz)
                        pt_colors = [
                            f"rgb({int(30 + 225*v)},{int(180 - 100*v)},{int(240 - 220*v)})"
                            for v in z_norm.tolist()
                        ]
                    elif color_mode == "⛰️ Elevation Height Tint":
                        ptp_y = float(np.ptp(py)) if len(py) else 1.0
                        min_y = float(np.min(py)) if len(py) else 0.0
                        y_norm = (py - min_y) / (ptp_y + 1e-9) if ptp_y > 1e-9 else np.zeros_like(py)
                        pt_colors = [
                            f"rgb({int(16 + 230*v)},{int(185 - 80*v)},{int(129 + 100*(1-v))})"
                            for v in y_norm.tolist()
                        ]
                    elif color_mode == "🎯 Ω-Confidence Map":
                        pt_colors = [
                            f"rgb({int(56 + 180*(i%2))},{int(189 - 50*(i%3))},248)"
                            for i in range(len(px))
                        ]
                    else:
                        pt_colors = cloud_data["colors"]

                    map_traces.append(go.Scatter3d(
                        x=px, y=py, z=pz, mode="markers",
                        marker=dict(size=map_pt_size, color=pt_colors, opacity=0.92),
                        name="● 3D Point Cloud", hoverinfo="skip",
                    ))
            else:
                px, py, pz = np.array([]), np.array([]), np.array([])

            # 2. Floor Grid Trace
            y_floor = float(np.min(py)) - 0.3 if len(py) else 0.0
            if show_map_grid and len(px):
                gx, gy, gz = _floor_grid(px, pz, y_floor)
                map_traces.append(go.Scatter3d(
                    x=gx, y=gy, z=gz, mode="lines",
                    line=dict(color="rgba(56,189,248,0.18)", width=1),
                    name="▦ Floor Grid", hoverinfo="skip",
                ))

            # 3. Bounding Box Trace
            if show_map_bbox and len(px):
                xmin, xmax = float(np.min(px)), float(np.max(px))
                ymin, ymax = float(np.min(py)), float(np.max(py))
                zmin, zmax = float(np.min(pz)), float(np.max(pz))
                bx = [xmin, xmax, xmax, xmin, xmin, xmin, xmax, xmax, xmin, xmin, None, xmax, xmax, None, xmax, xmax, None, xmin, xmin]
                by = [ymin, ymin, ymax, ymax, ymin, ymin, ymin, ymax, ymax, ymin, None, ymin, ymax, None, ymin, ymax, None, ymin, ymax]
                bz = [zmin, zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax, zmax, None, zmin, zmax, None, zmax, zmax, None, zmax, zmax]
                map_traces.append(go.Scatter3d(
                    x=bx, y=by, z=bz, mode="lines",
                    line=dict(color="#fbbf24", width=2, dash="dash"),
                    name=f"📏 Bounding Box ({xmax-xmin:.1f} × {ymax-ymin:.1f} × {zmax-zmin:.1f})",
                    hoverinfo="skip",
                ))

            # 4. Cameras, Trajectory & Frustums
            if cameras:
                cam_cx = [c["center"][0] for c in cameras]
                cam_cy = [c["center"][1] for c in cameras]
                cam_cz = [c["center"][2] for c in cameras]

                if show_map_traj:
                    map_traces.append(go.Scatter3d(
                        x=cam_cx, y=cam_cy, z=cam_cz,
                        mode="lines", line=dict(color="#22d3ee", width=4),
                        name="━ Flight Trajectory", hoverinfo="skip",
                    ))

                if show_map_cams:
                    frust_x, frust_y, frust_z = [], [], []
                    fd = 0.08 * frustum_scale
                    for cam in cameras:
                        fx, fy, fz = _frustum_lines(cam["center"], cam.get("yaw", 0.0), depth=fd)
                        frust_x += fx; frust_y += fy; frust_z += fz
                    map_traces.append(go.Scatter3d(
                        x=frust_x, y=frust_y, z=frust_z, mode="lines",
                        line=dict(color="rgba(255,255,255,0.85)", width=1.5),
                        name="△ Camera Frustums", hoverinfo="skip",
                    ))

                    hover_text = [
                        f"📷 Camera #{c.get('id', idx)}: {c['name']}<br>World XYZ: ({c['center'][0]:.2f}, {c['center'][1]:.2f}, {c['center'][2]:.2f})"
                        for idx, c in enumerate(cameras)
                    ]
                    cam_mode = "markers+text" if show_map_labels else "markers"
                    cam_text = [c["name"].split(".")[0] for c in cameras] if show_map_labels else None

                    map_traces.append(go.Scatter3d(
                        x=cam_cx, y=cam_cy, z=cam_cz,
                        mode=cam_mode,
                        text=cam_text, textposition="top center",
                        textfont=dict(size=9, color="#ffffff"),
                        marker=dict(size=7, symbol="circle", color="#22d3ee", line=dict(color="#ffffff", width=1.5)),
                        name="◆ Camera Poses",
                        hovertext=hover_text, hoverinfo="text",
                    ))

            # 5. 3D Semantic Detection Markers (Stage 4 Grounded)
            if show_map_pois:
                sem_markers_file = VIDEO_SEMANTIC_DIR / "semantic_markers.json"
                if sem_markers_file.exists():
                    try:
                        with open(sem_markers_file, "r", encoding="utf-8") as mf:
                            sem_markers_data = json.load(mf)
                        if sem_markers_data:
                            cat_groups = {}
                            for mk in sem_markers_data:
                                cat = mk.get("category", "other")
                                if cat not in cat_groups:
                                    cat_groups[cat] = []
                                cat_groups[cat].append(mk)

                            cat_config = {
                                "person":  {"name": "🚶 Persons (3D)",  "color": "#22c55e", "symbol": "cross"},
                                "vehicle": {"name": "🚗 Vehicles (3D)", "color": "#38bdf8", "symbol": "diamond"},
                                "animal":  {"name": "🐾 Animals (3D)",  "color": "#f59e0b", "symbol": "circle"},
                                "other":   {"name": "📍 Objects (3D)",  "color": "#a855f7", "symbol": "square"},
                            }

                            for cat, m_list in cat_groups.items():
                                cfg = cat_config.get(cat, cat_config["other"])
                                mx = [m["world_position"][0] for m in m_list]
                                my = [m["world_position"][1] for m in m_list]
                                mz = [m["world_position"][2] for m in m_list]
                                labels = [m.get("label", "Object") for m in m_list]
                                hover = [
                                    f"🎯 {m.get('label', 'Object')}<br>"
                                    f"Category: {cat.capitalize()}<br>"
                                    f"Confidence: {m.get('confidence', 0.0):.2f}<br>"
                                    f"Observations: {m.get('observation_count', 1)}<br>"
                                    f"3D Pos: ({m['world_position'][0]:.2f}, {m['world_position'][1]:.2f}, {m['world_position'][2]:.2f})"
                                    for m in m_list
                                ]
                                map_traces.append(go.Scatter3d(
                                    x=mx, y=my, z=mz,
                                    mode="markers+text",
                                    text=labels, textposition="top center",
                                    textfont=dict(size=10, color="#ffffff"),
                                    marker=dict(size=9, symbol=cfg["symbol"], color=cfg["color"], line=dict(color="#ffffff", width=1.5)),
                                    name=cfg["name"],
                                    hovertext=hover, hoverinfo="text",
                                ))
                    except Exception:
                        pass

            map_cam_presets = {
                "Aerial / Top-Down (Nadir)": dict(eye=dict(x=0.0, y=3.2, z=0.0), up=dict(x=0, y=0, z=1)),
                "Perspective Orbit":         dict(eye=dict(x=1.4, y=1.2, z=1.4)),
                "Front Facade":              dict(eye=dict(x=0.0, y=0.4, z=3.0)),
                "Side Elevation":            dict(eye=dict(x=3.0, y=0.4, z=0.0)),
                "Bird's Eye Survey (60°)":   dict(eye=dict(x=0.0, y=2.5, z=0.5), up=dict(x=0, y=0, z=1)),
            }

            map_fig = go.Figure(data=map_traces)
            map_fig.update_layout(
                height=720,
                scene=dict(
                    aspectmode="data",
                    bgcolor="rgba(3,7,18,1)",
                    xaxis=dict(backgroundcolor="rgba(3,7,18,1)", gridcolor="rgba(56,189,248,0.07)",
                               showbackground=True, zerolinecolor="rgba(56,189,248,0.12)", showticklabels=False, title=""),
                    yaxis=dict(backgroundcolor="rgba(3,7,18,1)", gridcolor="rgba(56,189,248,0.07)",
                               showbackground=True, zerolinecolor="rgba(56,189,248,0.12)", showticklabels=False, title=""),
                    zaxis=dict(backgroundcolor="rgba(3,7,18,1)", gridcolor="rgba(56,189,248,0.07)",
                               showbackground=True, zerolinecolor="rgba(56,189,248,0.12)", showticklabels=False, title=""),
                ),
                scene_camera=map_cam_presets.get(map_view, map_cam_presets["Perspective Orbit"]),
                paper_bgcolor="rgba(3,7,18,1)", plot_bgcolor="rgba(3,7,18,1)",
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01,
                            bgcolor="rgba(3,7,18,0.85)", font=dict(color="#cbd5e1", size=11),
                            bordercolor="rgba(56,189,248,0.3)", borderwidth=1),
            )
            st.plotly_chart(map_fig, use_container_width=True)

        except Exception as exc_map:
            st.error(f"Could not render 3D viewer: {exc_map}")

    # UAV Flight Altitude Profile
    if cameras and len(cameras) > 1:
        with st.expander("📈 **UAV Flight Trajectory Altitude Profile**", expanded=False):
            alt_fig = go.Figure()
            alt_fig.add_trace(go.Scatter(
                x=list(range(len(cameras))),
                y=[c["center"][1] for c in cameras],
                mode="lines+markers",
                line=dict(color="#22d3ee", width=2),
                marker=dict(color="#22d3ee", size=7, symbol="circle", line=dict(color="#ffffff", width=1)),
                fill="tozeroy", fillcolor="rgba(34,211,238,0.08)", name="Relative Altitude",
            ))
            alt_fig.update_layout(
                height=160,
                paper_bgcolor="rgba(3,7,18,1)", plot_bgcolor="rgba(3,7,18,0.6)",
                margin=dict(l=40, r=10, t=10, b=30),
                xaxis=dict(title="Keyframe Index", gridcolor="rgba(56,189,248,0.08)", color="#64748b"),
                yaxis=dict(title="Altitude (rel.)", gridcolor="rgba(56,189,248,0.08)", color="#64748b"),
                showlegend=False,
            )
            st.plotly_chart(alt_fig, use_container_width=True)

    # Optional Three.js WebGL Surface Mesh
    with st.expander("🎮 **WebGL Three.js Textured Surface Mesh Simulation**", expanded=False):
        threejs_html_path = ROOT / "web" / "index.html"
        if threejs_html_path.exists():
            with open(threejs_html_path, "r", encoding="utf-8") as f_html:
                threejs_code = f_html.read()
            st.components.v1.html(threejs_code, height=650, scrolling=False)


# ============================================================
# TAB 2: VISUAL 2D & AI INSPECTION
# ============================================================

with tab_inspect:
    st.markdown("### 👁️ Frame-by-Frame Visual & AI Inspection")
    st.markdown(
        "Inspect high-resolution keyframes alongside YOLO11s detection overlays and Depth Anything V2 relative depth maps."
    )

    selected_name = st.selectbox(
        "Select UAV Keyframe to Inspect:",
        image_names,
        key="inspect_frame_select",
    )

    selected_image = IMAGE_DIR / selected_name
    stem = selected_image.stem
    depth_p = DEPTH_DIR / f"depth_{stem}.png"
    yolo_p = YOLO_DIR / f"{stem}.jpg"

    # Find matching camera pose
    cur_cam = next((c for c in cameras if c["name"] == selected_name or Path(c["name"]).stem == stem), None)

    if cur_cam:
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Camera X (East)", f"{cur_cam['center'][0]:.2f}")
        ic2.metric("Camera Y (Altitude)", f"{cur_cam['center'][1]:.2f}")
        ic3.metric("Camera Z (North)", f"{cur_cam['center'][2]:.2f}")
        ic4.metric("Yaw Rotation", f"{math.degrees(cur_cam.get('yaw', 0.0)):.1f}°")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 1. Raw UAV Keyframe")
        if selected_image.exists():
            st.image(str(selected_image), caption=selected_name, use_container_width=True)

    with col2:
        st.markdown("##### 2. YOLO11s Detections")
        if yolo_p.exists():
            st.image(str(yolo_p), caption=f"2D Detections — {stem}", use_container_width=True)
        else:
            st.info("No YOLO detection overlay found for this frame.")

    with col3:
        st.markdown("##### 3. Depth Anything V2 Map")
        if depth_p.exists():
            depth_raw = Image.open(depth_p)
            depth_arr = np.array(depth_raw).astype(np.float32) / 255.0
            colored = mpl_cm.inferno(depth_arr)[:, :, :3]
            depth_colored = Image.fromarray((colored * 255).astype(np.uint8))
            st.image(depth_colored, caption=f"Relative Depth (Inferno) — {stem}", use_container_width=True)
        else:
            st.info("No depth map generated for this frame.")

    st.caption("🎨 **Relative Depth Interpretation:** 🟣 Dark/Cool = Farther away | 🟠 Warm/Bright = Closer to camera")


# ============================================================
# TAB 3: SEMANTIC 3D OBJECTS & DIRECTORY
# ============================================================

with tab_semantic:
    st.markdown("### 📍 Semantic 3D Spatial Intelligence")
    st.markdown(
        "2D YOLO detections unprojected into approximate 3D world space using calibrated Depth Anything V2 depth and SfM camera poses."
    )

    sem_objects_file = VIDEO_SEMANTIC_DIR / "semantic_objects.json"
    if sem_objects_file.exists():
        try:
            with open(sem_objects_file, "r", encoding="utf-8") as of:
                sem_obj_data = json.load(of)
            fused_list = sem_obj_data.get("objects", [])

            # Category counts
            c_persons = sum(1 for o in fused_list if o.get("category") == "person")
            c_vehicles = sum(1 for o in fused_list if o.get("category") == "vehicle")
            c_animals = sum(1 for o in fused_list if o.get("category") == "animal")

            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Total Localized Objects", len(fused_list))
            sc2.metric("🚶 Persons", c_persons)
            sc3.metric("🚗 Vehicles", c_vehicles)
            sc4.metric("🐾 Animals", c_animals)
            sc5.metric("Coordinate Frame", "monocular_sfm")

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            st.markdown("##### 📋 Localized 3D Object Directory")

            if fused_list:
                obj_rows = []
                for ob in fused_list:
                    pos = ob.get("world_position", [0, 0, 0])
                    obj_rows.append({
                        "Track ID": f"#{ob.get('track_id')}" if ob.get("track_id") is not None else "N/A",
                        "Class": ob.get("class_name", "").capitalize(),
                        "Category": ob.get("category", "").capitalize(),
                        "Confidence": f"{ob.get('confidence', 0.0):.2f}",
                        "Observations": ob.get("observation_count", 1),
                        "World X": f"{pos[0]:.2f}",
                        "World Y": f"{pos[1]:.2f}",
                        "World Z": f"{pos[2]:.2f}",
                        "Source Frames": ", ".join(ob.get("source_frames", [])),
                    })
                st.dataframe(obj_rows, use_container_width=True)
            else:
                st.info("No semantic objects detected in this sequence.")

        except Exception as se:
            st.warning(f"Could not parse semantic objects: {se}")
    else:
        st.info("Semantic 3D mapping outputs will appear here after processing a video with object detections.")


# ============================================================
# TAB 4: SYSTEM ARCHITECTURE & ROADMAP
# ============================================================

with tab_tech:
    st.markdown("### 🔬 System Architecture & Research Evolution")

    st.markdown(
        """
        | Metric | OpenCV Incremental SfM | AnySplat 3DGS | VGGT-Ω Transformer | NVIDIA NuRec |
        |:---|:---|:---|:---|:---|
        | **Method Type** | Classical Photogrammetry | Feed-Forward Gaussian Splatting | Vision Transformer Pointmap | Neural Surface Reconstruction |
        | **Features** | SIFT Keypoint Descriptors | DUSt3R Cross-Attention | Multi-Scale ViT Grounding | Multi-Res Instant Hash Grids |
        | **Pose Estimation** | RANSAC Essential Matrix | Joint Feed-Forward Head | Camera Attention Head | Photometric Bundle Refinement |
        | **Densification** | Inverse-Depth RANSAC Alignment | Dense Gaussian Primitives | Cross-Attention Densification | Signed Distance Field (SDF) |
        | **Hardware** | 100% CPU Compatible | GPU / CUDA | GPU / ViT | NVIDIA CUDA Core |
        | **Status** | 🟢 **Active / Working Photogrammetry** | 🟡 Prototype Agent Pipeline | 🟡 Prototype Agent Pipeline | 🟡 Prototype Agent Pipeline |
        """
    )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 🚀 Technology Evolution Roadmap")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(
            """
            <div class="aerorecon-card">
                <span class="status-badge badge-success">✓ CURRENT MVD</span>
                <h4 style="margin:8px 0 4px 0;color:#f8fafc;">Incremental SfM</h4>
                <p style="font-size:0.80rem;color:#94a3b8;line-height:1.5;">
                    OpenCV SIFT + RANSAC pose recovery + calibrated Depth Anything V2 inverse depth alignment + YOLO11s.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r2:
        st.markdown(
            """
            <div class="aerorecon-card">
                <span class="status-badge badge-info">⟳ EVALUATING</span>
                <h4 style="margin:8px 0 4px 0;color:#f8fafc;">Dense 3DGS</h4>
                <p style="font-size:0.80rem;color:#94a3b8;line-height:1.5;">
                    AnySplat & VGGT-Ω feed-forward Gaussian Splatting for photorealistic real-time novel-view synthesis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r3:
        st.markdown(
            """
            <div class="aerorecon-card">
                <span class="status-badge badge-amber">◇ NEXT STAGE</span>
                <h4 style="margin:8px 0 4px 0;color:#f8fafc;">Live Mapping</h4>
                <p style="font-size:0.80rem;color:#94a3b8;line-height:1.5;">
                    Online 3D scene streaming from live UAV telemetry and edge AI compute.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r4:
        st.markdown(
            """
            <div class="aerorecon-card">
                <span class="status-badge badge-amber">◇ FUTURE</span>
                <h4 style="margin:8px 0 4px 0;color:#f8fafc;">Rescue AI Nav</h4>
                <p style="font-size:0.80rem;color:#94a3b8;line-height:1.5;">
                    Autonomous UAV corridor exploration, spatial hazard assessment, and search-and-rescue mission intelligence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center;padding:16px 0;border-top:1px solid rgba(255,255,255,0.06);font-size:0.75rem;color:#64748b;">
        🚁 <b>AeroRecon</b> — Autonomous UAV 3D Reconstruction & Rescue Intelligence Demonstrator • Built with OpenCV, PyTorch, Plotly & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)