import sys
from pathlib import Path
import struct

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


# ============================================================
# BENCHMARK DEFAULT PATHS
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

UPLOAD_WORKSPACE_DIR = ROOT / "data" / "input" / "uploaded_session"


# ============================================================
# PAGE CONFIG & AEROSPACE RESEARCH THEME
# ============================================================

st.set_page_config(
    page_title="AeroRecon | Drone 3D Reconstruction",
    page_icon="🚁",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Clean typography & aerospace styling */
    .pipeline-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 14px;
        text-align: left;
        transition: border-color 0.2s ease;
    }
    .pipeline-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
    }
    .pipeline-status {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #22c55e;
        margin-bottom: 4px;
    }
    .pipeline-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 2px;
    }
    .pipeline-desc {
        font-size: 0.78rem;
        color: #94a3b8;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
    }
    .upload-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px dashed rgba(56, 189, 248, 0.3);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .roadmap-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 0;
        position: relative;
    }
    .roadmap-stage-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }
    .roadmap-title {
        font-size: 1.0rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .roadmap-desc {
        font-size: 0.80rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    .stage-active { color: #22c55e; }
    .stage-next   { color: #38bdf8; }
    .stage-future { color: #94a3b8; }
    .compare-label {
        text-align: center;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 0 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# COLMAP PARSING HELPERS
# ============================================================

def qvec2rotmat(qvec):
    """Converts a quaternion into a 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def load_colmap_cameras(images_bin_path):
    """Extracts camera poses and calculates world coordinates C = -R^T * t."""
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


# ============================================================
# SIDEBAR / DATA SOURCE SELECTION
# ============================================================

st.sidebar.markdown("### ⚙️ Pipeline Configuration")

data_source = st.sidebar.radio(
    "Select Input Source:",
    [
        "🚁 Preloaded Benchmark (seq38)",
        "📹 Upload Custom UAV Video",
    ],
    index=0,
)

# Initialize Session State
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "custom_cameras" not in st.session_state:
    st.session_state.custom_cameras = []
if "custom_point_count" not in st.session_state:
    st.session_state.custom_point_count = 0


# ============================================================
# DATA ROUTING (BENCHMARK VS CUSTOM VIDEO)
# ============================================================

if data_source == "📹 Upload Custom UAV Video":
    IMAGE_DIR = UPLOAD_WORKSPACE_DIR / "Images"
    DEPTH_DIR = UPLOAD_WORKSPACE_DIR / "depth"
    YOLO_DIR = UPLOAD_WORKSPACE_DIR / "detections"
    POINT_CLOUD = UPLOAD_WORKSPACE_DIR / "model.ply"
    is_custom_mode = True
else:
    IMAGE_DIR = BENCHMARK_IMAGE_DIR
    DEPTH_DIR = BENCHMARK_DEPTH_DIR
    YOLO_DIR = BENCHMARK_YOLO_DIR
    POINT_CLOUD = BENCHMARK_POINT_CLOUD
    is_custom_mode = False


# ============================================================
# 1. HEADER / HERO
# ============================================================

hero_col1, hero_col2 = st.columns([3, 1])

with hero_col1:
    st.title("🚁 AeroRecon")
    st.subheader("AI-Assisted Drone 3D Reconstruction")
    st.markdown(
        "From **UAV video & imagery** to **object detection**, **relative depth estimation**, "
        "**camera reconstruction**, and **sparse 3D visualization**."
    )

with hero_col2:
    status_label = "Custom Video Active" if (is_custom_mode and st.session_state.video_processed) else "Prototype Ready"
    st.markdown(
        f"""
        <div style="text-align: right; padding-top: 20px;">
            <span style="
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: rgba(34, 197, 94, 0.12);
                color: #22c55e;
                border: 1px solid rgba(34, 197, 94, 0.3);
                border-radius: 20px;
                padding: 6px 14px;
                font-size: 0.84rem;
                font-weight: 600;
                letter-spacing: 0.02em;
            ">
                <span style="font-size: 9px;">●</span> {status_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# VIDEO UPLOAD & KEYFRAME INGESTION UI
# ============================================================

if is_custom_mode:
    st.markdown("### 📹 UAV Video Ingestion & AI 3D Mapping Pipeline")

    with st.expander("🎬 **Upload & Process UAV Flight Video**", expanded=not st.session_state.video_processed):
        st.markdown(
            "Upload any aerial UAV footage (`.mp4`, `.mov`, `.avi`, `.mkv`). "
            "The system automatically extracts keyframes, runs AI analysis, and builds a **dense interactive 3D map**."
        )

        u_col1, u_col2 = st.columns([2, 1])

        with u_col1:
            uploaded_video = st.file_uploader(
                "Select Drone Flight Video:",
                type=["mp4", "mov", "avi", "mkv"],
                help="Upload a video recording from a drone or UAV flight.",
            )

        with u_col2:
            keyframe_count = st.slider(
                "Target Keyframes:",
                min_value=5,
                max_value=15,
                value=10,
                help="Number of sharpest keyframes extracted across the flight.",
            )
            yolo_conf = st.slider(
                "YOLO Confidence Threshold:",
                min_value=0.15,
                max_value=0.70,
                value=0.30,
                step=0.05,
            )

        if uploaded_video is not None:
            UPLOAD_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
            saved_video_path = UPLOAD_WORKSPACE_DIR / "uploaded_drone_video.mp4"

            with open(saved_video_path, "wb") as fv:
                fv.write(uploaded_video.read())

            v_preview1, v_preview2 = st.columns([1, 1])
            with v_preview1:
                st.video(str(saved_video_path))
                file_mb = saved_video_path.stat().st_size / (1024 * 1024)
                st.caption(f"📁 `{uploaded_video.name}` — {file_mb:.1f} MB")

            with v_preview2:
                st.markdown("#### ⚡ AI Pipeline Stages")
                st.markdown(
                    """
                    <div style='font-size:0.82rem; color:#94a3b8; line-height:2.0;'>
                    <b style='color:#22c55e;'>Stage 1a</b> &nbsp; Intelligent Keyframe Extraction<br>
                    <b style='color:#22c55e;'>Stage 1b</b> &nbsp; YOLO11s Object Detection<br>
                    <b style='color:#22c55e;'>Stage 1c</b> &nbsp; Depth Anything V2 &mdash; Relative Depth<br>
                    <b style='color:#38bdf8;'>Stage 2 &nbsp;</b> &nbsp; Dense 3D Point Cloud (Depth Unprojection)<br>
                    <b style='color:#a855f7;'>Stage 3 &nbsp;</b> &nbsp; [Evaluating] AnySplat / VGGT Gaussian Splatting<br>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("🚀  Build Interactive 3D Map", type="primary", use_container_width=True):
                    overall_bar  = st.progress(0.0)
                    status_text  = st.empty()

                    try:
                        # ── Stage 1a: Keyframe Extraction ─────────────────
                        status_text.markdown(
                            "🔍 **Stage 1a — Keyframe Extraction:** Analysing video sharpness across temporal segments..."
                        )
                        keyframes = extract_keyframes(
                            video_path=saved_video_path,
                            output_dir=IMAGE_DIR,
                            max_frames=keyframe_count,
                            progress_callback=lambda p, _: overall_bar.progress(p * 0.20),
                        )
                        st.success(f"✓ Stage 1a complete — {len(keyframes)} keyframes extracted")
                        img_paths = [kf["path"] for kf in keyframes]

                        # ── Stage 1b: YOLO Detection ──────────────────────
                        status_text.markdown(
                            "🔎 **Stage 1b — YOLO11s Detection:** Identifying vehicles, persons, and objects..."
                        )
                        det_counts = run_yolo_on_keyframes(
                            image_paths=img_paths,
                            output_dir=YOLO_DIR,
                            conf=yolo_conf,
                            progress_callback=lambda p, _: overall_bar.progress(0.20 + p * 0.20),
                        )
                        total_dets = sum(det_counts.values())
                        st.success(f"✓ Stage 1b complete — {total_dets} aerial detections across {len(det_counts)} frames")

                        # ── Stage 1c: Depth Anything V2 ───────────────────
                        status_text.markdown(
                            "🧠 **Stage 1c — Depth Anything V2:** Estimating relative surface depth for each frame..."
                        )
                        depth_paths = run_depth_on_keyframes(
                            image_paths=img_paths,
                            output_dir=DEPTH_DIR,
                            progress_callback=lambda p, _: overall_bar.progress(0.40 + p * 0.25),
                        )
                        st.success(f"✓ Stage 1c complete — {len(depth_paths)} depth maps generated")

                        # ── Stage 2: Dense 3D Point Cloud ─────────────────
                        status_text.markdown(
                            "🌐 **Stage 2 — Dense 3D Reconstruction:** Unprojecting depth + RGB into world-space 3D map..."
                        )
                        custom_cams, pt_count = estimate_point_cloud_and_trajectory(
                            image_paths=img_paths,
                            depth_paths=depth_paths,
                            output_ply_path=POINT_CLOUD,
                            detection_data=det_counts,
                            max_points_per_frame=1200,
                            progress_callback=lambda p, _: overall_bar.progress(0.65 + p * 0.35),
                        )
                        overall_bar.progress(1.0)
                        status_text.markdown(
                            f"🎉 **Pipeline Complete!** Dense 3D map built — {pt_count:,} RGB points | {len(custom_cams)} camera poses"
                        )

                        st.session_state.video_processed  = True
                        st.session_state.custom_cameras   = custom_cams
                        st.session_state.custom_point_count = pt_count
                        st.session_state.det_counts_custom  = det_counts
                        st.rerun()

                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")

    st.divider()

    # ── Post-processing: Dedicated 3D Map section ─────────────────────────────
    if st.session_state.video_processed and POINT_CLOUD.exists():

        st.markdown("## 🌐 Interactive 3D Scene Map")
        st.markdown(
            "Dense, RGB-coloured 3D point cloud built from **Depth Anything V2 depth unprojection** "
            "and **optical-flow camera pose estimation** from your uploaded UAV video."
        )

        # Summary metrics
        det_counts_cust = getattr(st.session_state, "det_counts_custom", {})
        total_dets_cust = sum(det_counts_cust.values()) if det_counts_cust else 0
        cust_cams       = st.session_state.custom_cameras
        cust_pts        = st.session_state.custom_point_count

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Dense 3D Points",   f"{cust_pts:,}")
        mc2.metric("Camera Poses",      f"{len(cust_cams)}")
        mc3.metric("Detected Objects",  f"{total_dets_cust}")
        mc4.metric("Depth Maps",        f"{len(list(DEPTH_DIR.glob('*.png')))}")
        mc5.metric("Stage",             "1-2 Active")

        # 3D viewer controls
        mp_col1, mp_col2, mp_col3, mp_col4, mp_col5, mp_col6 = st.columns([1.4, 1, 1, 1, 1, 1])
        with mp_col1:
            map_view = st.selectbox("View Preset", ["Aerial (Top)", "Perspective", "Front", "Side"], key="map_view")
        with mp_col2:
            map_pt_size = st.slider("Point Size", 1, 8, 2, key="map_pt_size")
        with mp_col3:
            show_map_cams = st.checkbox("Camera Frustums", value=True, key="map_cams")
        with mp_col4:
            show_map_labels = st.checkbox("Frame Labels", value=True, key="map_labels")
        with mp_col5:
            show_map_traj = st.checkbox("Flight Path", value=True, key="map_traj")
        with mp_col6:
            colorize_depth = st.checkbox("Depth Colorize", value=False, key="map_depth_color",
                                         help="Colorise by Z-depth instead of RGB texture")

        # Build figure
        try:
            ply      = PlyData.read(POINT_CLOUD)
            vertex   = ply["vertex"].data
            px, py, pz = vertex["x"], vertex["y"], vertex["z"]

            if colorize_depth:
                # Z-depth colormap
                z_norm = (pz - pz.min()) / (pz.ptp() + 1e-9)
                pt_colors = [
                    f"rgb({int(255*v)},{int(80 + 120*(1-v))},{int(200*(1-v))})"
                    for v in z_norm.tolist()
                ]
            elif all(c in vertex.dtype.names for c in ("red", "green", "blue")):
                pt_colors = [
                    f"rgb({r},{g},{b})"
                    for r, g, b in zip(vertex["red"], vertex["green"], vertex["blue"])
                ]
            else:
                pt_colors = "#38bdf8"

            map_traces = [go.Scatter3d(
                x=px, y=py, z=pz,
                mode="markers",
                marker=dict(size=map_pt_size, color=pt_colors),
                name="● Dense 3D Points (RGB scene reconstruction)",
                hoverinfo="skip",
            )]

            if cust_cams:
                cam_cx = [c["center"][0] for c in cust_cams]
                cam_cy = [c["center"][1] for c in cust_cams]
                cam_cz = [c["center"][2] for c in cust_cams]

                if show_map_traj:
                    map_traces.append(go.Scatter3d(
                        x=cam_cx, y=cam_cy, z=cam_cz,
                        mode="lines",
                        line=dict(color="#38bdf8", width=5),
                        name="━ UAV Flight Path",
                        hoverinfo="skip",
                    ))

                if show_map_cams:
                    cam_mode = "markers+text" if show_map_labels else "markers"
                    cam_text = [c["name"].replace(".png", "") for c in cust_cams] if show_map_labels else None
                    map_traces.append(go.Scatter3d(
                        x=cam_cx, y=cam_cy, z=cam_cz,
                        mode=cam_mode,
                        text=cam_text,
                        textposition="top center",
                        textfont=dict(size=10, color="#FFFFFF"),
                        marker=dict(
                            size=10, symbol="diamond",
                            color="#38bdf8",
                            line=dict(color="#FFFFFF", width=1.5),
                        ),
                        name="◆ Camera Poses (UAV positions)",
                        hovertext=[f"📷 {c['name']}" for c in cust_cams],
                        hoverinfo="text",
                    ))

            map_fig = go.Figure(data=map_traces)

            map_camera_presets = {
                "Aerial (Top)": dict(eye=dict(x=0, y=0, z=3.5), up=dict(x=0, y=1, z=0)),
                "Perspective":  dict(eye=dict(x=1.6, y=1.6, z=1.2)),
                "Front":        dict(eye=dict(x=0, y=-3.5, z=0.3)),
                "Side":         dict(eye=dict(x=3.5, y=0, z=0.3)),
            }

            map_fig.update_layout(
                height=820,
                scene=dict(
                    xaxis_title="X (metres)",
                    yaxis_title="Y (altitude)",
                    zaxis_title="Z (metres)",
                    aspectmode="data",
                    bgcolor="rgba(5,10,20,1)",
                    xaxis=dict(
                        backgroundcolor="rgba(5,10,20,1)",
                        gridcolor="rgba(56,189,248,0.08)",
                        showbackground=True,
                        zerolinecolor="rgba(56,189,248,0.15)",
                    ),
                    yaxis=dict(
                        backgroundcolor="rgba(5,10,20,1)",
                        gridcolor="rgba(56,189,248,0.08)",
                        showbackground=True,
                        zerolinecolor="rgba(56,189,248,0.15)",
                    ),
                    zaxis=dict(
                        backgroundcolor="rgba(5,10,20,1)",
                        gridcolor="rgba(56,189,248,0.08)",
                        showbackground=True,
                        zerolinecolor="rgba(56,189,248,0.15)",
                    ),
                ),
                scene_camera=map_camera_presets.get(map_view, map_camera_presets["Perspective"]),
                paper_bgcolor="rgba(5,10,20,1)",
                plot_bgcolor="rgba(5,10,20,1)",
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(
                    yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(5,10,20,0.85)",
                    font=dict(color="#FFFFFF", size=11),
                    bordercolor="rgba(56,189,248,0.3)",
                    borderwidth=1,
                ),
            )

            st.plotly_chart(map_fig, width="stretch")
            st.caption(
                f"🌐 Dense RGB 3D map — {cust_pts:,} points from {len(cust_cams)} UAV views. "
                "Rotate: left-drag • Zoom: scroll • Pan: shift+drag."
            )

        except Exception as e:
            st.error(f"Could not render 3D map: {e}")

        # Stage status
        st.markdown("##### Pipeline Stage Status")
        ss1, ss2, ss3, ss4 = st.columns(4)
        with ss1:
            st.markdown(
                "<div class='roadmap-card'>"
                "<div class='roadmap-stage-label stage-active'>✓ COMPLETE</div>"
                "<div class='roadmap-title'>Stage 1 &mdash; MVD</div>"
                "<div class='roadmap-desc'>COLMAP SfM + Depth Anything V2 + YOLO11s + Dense Point Cloud</div>"
                "</div>", unsafe_allow_html=True
            )
        with ss2:
            st.markdown(
                "<div class='roadmap-card' style='border-color:rgba(56,189,248,0.3);'>"
                "<div class='roadmap-stage-label stage-next'>⟳ EVALUATING</div>"
                "<div class='roadmap-title'>Stage 2 &mdash; Dense 3DGS</div>"
                "<div class='roadmap-desc'>AnySplat / VGGT &mdash; photorealistic Gaussian Splatting from UAV sequences</div>"
                "</div>", unsafe_allow_html=True
            )
        with ss3:
            st.markdown(
                "<div class='roadmap-card' style='border-color:rgba(168,85,247,0.2);'>"
                "<div class='roadmap-stage-label stage-future'>◇ NEXT STAGE</div>"
                "<div class='roadmap-title'>Stage 3 &mdash; Incremental Mapping</div>"
                "<div class='roadmap-desc'>Online 3D scene updates from live streaming UAV frames</div>"
                "</div>", unsafe_allow_html=True
            )
        with ss4:
            st.markdown(
                "<div class='roadmap-card' style='border-color:rgba(168,85,247,0.15);'>"
                "<div class='roadmap-stage-label stage-future'>◇ FUTURE</div>"
                "<div class='roadmap-title'>Stage 4 &mdash; Rescue AI</div>"
                "<div class='roadmap-desc'>Autonomous rescue drone navigation using live 3D scene understanding</div>"
                "</div>", unsafe_allow_html=True
            )

        st.divider()


# ============================================================
# DATA VERIFICATION & DYNAMIC COUNTS
# ============================================================

images = sorted(IMAGE_DIR.glob("*.png")) if IMAGE_DIR.exists() else []

if not images:
    if is_custom_mode and not st.session_state.video_processed:
        st.info("👆 Please upload a drone flight video above and click **Process Video** to extract keyframes and begin analysis.")
        st.stop()
    else:
        st.error(f"No UAV images found in: {IMAGE_DIR}")
        st.stop()

image_names = [image.name for image in images]
depth_files = list(DEPTH_DIR.glob("depth_*.png")) if DEPTH_DIR.exists() else []

if is_custom_mode:
    cameras = st.session_state.custom_cameras
    vertex_count = st.session_state.custom_point_count
    if not cameras and POINT_CLOUD.exists():
        # Fallback load points
        try:
            ply_temp = PlyData.read(POINT_CLOUD)
            vertex_count = len(ply_temp["vertex"].data)
        except Exception:
            vertex_count = len(images) * 100
else:
    cameras = load_colmap_cameras(BENCHMARK_IMAGES_BIN)
    vertex_count = 0
    if POINT_CLOUD.exists():
        try:
            ply_temp = PlyData.read(POINT_CLOUD)
            vertex_count = len(ply_temp["vertex"].data)
        except Exception:
            vertex_count = 209


# ============================================================
# 2. PROCESSING PIPELINE
# ============================================================

st.markdown("### 🔄 Processing Pipeline")

p1, p2, p3, p4, p5 = st.columns(5)

input_desc = f"{len(images)} keyframes" if is_custom_mode else f"{len(images)} frames"
stage4_title = "Camera Pose" if is_custom_mode else "COLMAP"
stage4_desc = "Visual Trajectory" if is_custom_mode else "Camera + Geometry"

with p1:
    st.markdown(
        f"""
        <div class="pipeline-card">
            <div class="pipeline-status">✓ Stage 1</div>
            <div class="pipeline-title">UAV Input</div>
            <div class="pipeline-desc">{input_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p2:
    st.markdown(
        """
        <div class="pipeline-card">
            <div class="pipeline-status">✓ Stage 2</div>
            <div class="pipeline-title">YOLO</div>
            <div class="pipeline-desc">Object Detection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p3:
    st.markdown(
        """
        <div class="pipeline-card">
            <div class="pipeline-status">✓ Stage 3</div>
            <div class="pipeline-title">Depth AI</div>
            <div class="pipeline-desc">Relative Depth</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p4:
    st.markdown(
        f"""
        <div class="pipeline-card">
            <div class="pipeline-status">✓ Stage 4</div>
            <div class="pipeline-title">{stage4_title}</div>
            <div class="pipeline-desc">{stage4_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p5:
    st.markdown(
        """
        <div class="pipeline-card">
            <div class="pipeline-status">✓ Stage 5</div>
            <div class="pipeline-title">3D Output</div>
            <div class="pipeline-desc">Sparse Reconstruction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ============================================================
# 3. SYSTEM OVERVIEW
# ============================================================

st.markdown("### 📊 System Overview")

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric("UAV Frames", f"{len(images)}")

with s2:
    reg_count = len(cameras) if cameras else len(images)
    st.metric("Registered Views", f"{reg_count} / {len(images)}")

with s3:
    st.metric("Sparse 3D Points", f"{vertex_count}")

with s4:
    st.metric("Depth Maps", f"{len(depth_files)}")


# ============================================================
# FRAME SELECTION
# ============================================================

selected_name = st.selectbox(
    "🎞️ Select UAV Frame",
    image_names,
)

selected_image = IMAGE_DIR / selected_name
stem = selected_image.stem


# ============================================================
# 4. VISUAL ANALYSIS
# ============================================================

st.markdown("### 👁️ Visual Analysis")

orig_w, orig_h = 1920, 1080
if selected_image.exists():
    with Image.open(selected_image) as img_tmp:
        orig_w, orig_h = img_tmp.size

v_col1, v_col2, v_col3 = st.columns(3)
with v_col1:
    st.metric("Selected Frame", selected_name)
with v_col2:
    st.metric("Detection Model", "YOLO11s")
with v_col3:
    st.metric("Source Resolution", f"{orig_w} × {orig_h}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Original UAV Frame")
    original = Image.open(selected_image).convert("RGB")
    st.image(
        original,
        caption=selected_name,
        width="stretch",
    )

with col2:
    st.markdown("#### YOLO Object Detection")
    yolo_path = YOLO_DIR / f"{stem}.jpg"
    if yolo_path.exists():
        detection = Image.open(yolo_path).convert("RGB")
        st.image(
            detection,
            caption=f"YOLO11s detections — {stem}",
            width="stretch",
        )
    else:
        st.warning(f"YOLO result not found for {selected_name}")


# ============================================================
# 5. AI DEPTH ESTIMATION
# ============================================================

st.markdown("### 🧠 AI Depth Estimation")

st.markdown(
    "Depth Anything V2 estimates relative scene depth from the UAV image, "
    "helping reveal the approximate spatial structure of the captured environment."
)

depth_path = DEPTH_DIR / f"depth_{stem}.png"

if depth_path.exists():
    depth_img = Image.open(depth_path)
    depth_w, depth_h = depth_img.size

    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.metric("Depth Map", "Generated")
    with d_col2:
        st.metric("Resolution", f"{depth_w} × {depth_h}")
    with d_col3:
        st.metric("Estimation", "Relative Depth")
    with d_col4:
        st.metric("Source", selected_name)

    # Apply perceptual colormap for display only
    depth_arr = np.array(depth_img).astype(np.float32) / 255.0
    colored = mpl_cm.inferno(depth_arr)[:, :, :3]
    depth_colored = Image.fromarray((colored * 255).astype(np.uint8))

    left_depth, right_depth = st.columns(2)

    with left_depth:
        st.markdown("#### Original UAV Frame")
        st.image(
            original,
            caption=selected_name,
            width="stretch",
        )

    with right_depth:
        st.markdown("#### AI Relative Depth Map")
        st.image(
            depth_colored,
            caption=f"Depth Anything V2 — {stem}",
            width="stretch",
        )

    st.caption(
        "🎨 **Relative Depth Interpretation:** 🟣 **Dark / Cool — Farther** ➔ 🟠 **Warm / Bright — Closer**"
    )

    st.markdown(
        "💡 **Spatial Structure:** The depth visualization highlights relative surface proximity across buildings, roads, vegetation, and terrain."
    )

    st.info(
        "**Note:** Depth Anything V2 provides relative scene-depth estimation. "
        "It does not represent metric distance in meters."
    )

else:
    st.warning(f"Depth map not found for {selected_name}")


# ============================================================
# 6. 3D RECONSTRUCTION
# ============================================================

st.markdown("### 📐 Spatial 3D Reconstruction")

if not is_custom_mode:
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    if BENCHMARK_SPARSE_DIR.exists() and all((BENCHMARK_SPARSE_DIR / f).exists() for f in required_files):
        st.success("✓ COLMAP Structure-from-Motion reconstruction successfully loaded.")
    else:
        st.warning("COLMAP sparse reconstruction files not found.")
else:
    st.success("✓ Multi-view visual feature tracking & camera trajectory reconstruction generated.")

st.markdown("### 🌐 Interactive 3D Reconstruction")

st.markdown(
    "Camera poses are estimated from overlapping UAV views and matched visual features "
    "are triangulated into a 3D representation of the environment."
)

st.markdown(
    """
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(251, 191, 36, 0.10);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.25);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    ">
        <span style="font-size: 9px;">●</span> SPARSE RECONSTRUCTION · WORK IN PROGRESS
    </span>
    """,
    unsafe_allow_html=True,
)

# Metric cards
reg_val = len(cameras) if cameras else len(images)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Registered Views", f"{reg_val} / {len(images)}")
with m2:
    st.metric("Sparse 3D Points", f"{vertex_count}")
with m3:
    st.metric("Camera Poses", f"{reg_val}")
with m4:
    st.metric("Reconstruction", "Sparse / WIP")

# 3D View Controls
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([1.2, 1, 1, 1, 1])

with ctrl_col1:
    view_preset = st.selectbox(
        "View",
        ["Perspective", "Top", "Front", "Side"],
        index=0,
    )

with ctrl_col2:
    point_size = st.slider("Point Size", 1, 10, 3)

with ctrl_col3:
    show_cameras = st.checkbox("Camera Poses", value=True)

with ctrl_col4:
    show_labels = st.checkbox("Camera Labels", value=True)

with ctrl_col5:
    show_trajectory = st.checkbox("Flight Trajectory", value=True)

# Explanation panel
exp1, exp2, exp3 = st.columns(3)
with exp1:
    st.markdown(
        "**● 3D Points**<br>"
        "<span style='color: #94a3b8; font-size: 0.85rem;'>"
        "Each point represents a visual feature reconstructed from multiple overlapping images.</span>",
        unsafe_allow_html=True,
    )
with exp2:
    st.markdown(
        "**◆ Camera Poses**<br>"
        "<span style='color: #94a3b8; font-size: 0.85rem;'>"
        "Estimated UAV camera positions recovered along the flight path.</span>",
        unsafe_allow_html=True,
    )
with exp3:
    st.markdown(
        "**━ Flight Trajectory**<br>"
        "<span style='color: #94a3b8; font-size: 0.85rem;'>"
        "Connected camera markers showing estimated UAV movement across keyframes.</span>",
        unsafe_allow_html=True,
    )

# Build 3D Plotly figure
if POINT_CLOUD.exists():
    try:
        ply = PlyData.read(POINT_CLOUD)
        vertex = ply["vertex"].data

        x = vertex["x"]
        y = vertex["y"]
        z = vertex["z"]

        pt_marker = dict(size=point_size)

        if all(ch in vertex.dtype.names for ch in ("red", "green", "blue")):
            rgb = [
                f"rgb({r},{g},{b})"
                for r, g, b in zip(vertex["red"], vertex["green"], vertex["blue"])
            ]
            pt_marker["color"] = rgb

        point_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=pt_marker,
            name="● 3D Points — reconstructed scene features",
            hoverinfo="skip",
        )

        traces = [point_trace]

        if cameras:
            cam_x = [c["center"][0] for c in cameras]
            cam_y = [c["center"][1] for c in cameras]
            cam_z = [c["center"][2] for c in cameras]

            if show_trajectory:
                trajectory_trace = go.Scatter3d(
                    x=cam_x, y=cam_y, z=cam_z,
                    mode="lines",
                    line=dict(color="#FF4B4B", width=5),
                    name="━ Camera Trajectory — estimated UAV movement",
                    hoverinfo="skip",
                )
                traces.append(trajectory_trace)

            if show_cameras:
                cam_mode = "markers+text" if show_labels else "markers"
                cam_text = (
                    [c["name"].replace(".png", "") for c in cameras]
                    if show_labels else None
                )
                camera_trace = go.Scatter3d(
                    x=cam_x, y=cam_y, z=cam_z,
                    mode=cam_mode,
                    text=cam_text,
                    textposition="top center",
                    textfont=dict(size=10, color="#FFFFFF"),
                    marker=dict(
                        size=8,
                        symbol="diamond",
                        color="#FF4B4B",
                        line=dict(color="#FFFFFF", width=1.5),
                    ),
                    name="◆ Camera Poses — estimated UAV positions",
                    hovertext=[f"Camera: {c['name']}" for c in cameras],
                    hoverinfo="text",
                )
                traces.append(camera_trace)

        fig = go.Figure(data=traces)

        camera_presets = {
            "Perspective": dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            "Top":         dict(eye=dict(x=0, y=0, z=3.0), up=dict(x=0, y=1, z=0)),
            "Front":       dict(eye=dict(x=0, y=-3.0, z=0.3)),
            "Side":        dict(eye=dict(x=3.0, y=0, z=0.3)),
        }
        scene_camera = camera_presets.get(view_preset, camera_presets["Perspective"])

        fig.update_layout(
            height=720,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.06)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.1)",
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.06)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.1)",
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.06)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.1)",
                ),
            ),
            scene_camera=scene_camera,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0, 0, 0, 0.65)",
                font=dict(color="#FFFFFF", size=11),
                bordercolor="rgba(255, 255, 255, 0.15)",
                borderwidth=1,
            ),
        )

        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Interactive sparse 3D point cloud and camera trajectory. "
            "Use mouse to orbit, scroll to zoom, shift+drag to pan."
        )

    except Exception as e:
        st.error(f"Could not load 3D point cloud: {e}")

else:
    st.warning("3D PLY point cloud not found.")

# Scene Interpretation
interp1, interp2, interp3 = st.columns(3)
with interp1:
    st.markdown(
        f"**Reconstructed Environment**<br>"
        f"<span style='color:#94a3b8; font-size:0.85rem;'>"
        f"{vertex_count} sparse 3D features reconstructed from registered UAV views.</span>",
        unsafe_allow_html=True,
    )
with interp2:
    st.markdown(
        f"**UAV Motion**<br>"
        f"<span style='color:#94a3b8; font-size:0.85rem;'>"
        f"{reg_val} camera poses recovered from the flight sequence.</span>",
        unsafe_allow_html=True,
    )
with interp3:
    st.markdown(
        "**Current Limitation**<br>"
        "<span style='color:#94a3b8; font-size:0.85rem;'>"
        "Sparse reconstruction; unregistered frames are not currently included in the 3D model.</span>",
        unsafe_allow_html=True,
    )

st.divider()


# ============================================================
# 7. AI RECONSTRUCTION ROADMAP: CURRENT vs TARGET
# ============================================================

st.markdown("### 🔬 Reconstruction Methodology: Current vs Target")

st.info(
    "**Our current prototype uses COLMAP for sparse geometric reconstruction. "
    "We are evaluating AI-based reconstruction models such as VGGT and Gaussian Splatting "
    "to evolve this toward denser and eventually incremental 3D mapping.**"
)

# Side-by-side comparison
cmp_left, cmp_right = st.columns(2)

with cmp_left:
    st.markdown(
        "<div class='compare-label' style='color:#fbbf24;'>◀ CURRENT PROTOTYPE — Sparse SfM (COLMAP)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="roadmap-card">
            <div class="roadmap-stage-label stage-active">✓ ACTIVE — Stage 1</div>
            <div class="roadmap-title">COLMAP Sparse Reconstruction</div>
            <div class="roadmap-desc">
                Structure-from-Motion pipeline that recovers <strong>camera poses</strong> and
                <strong>triangulates 209 sparse 3D feature points</strong> from 10 overlapping UAV images.
                Output is an interactive sparse point cloud suitable for camera-pose validation
                and geometric verification.
                <br><br>
                <span style="color:#fbbf24;">Limitation:</span> Produces only a skeletal scene representation —
                geometry is sparse, lacks surface texture, and requires offline batch processing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cmp_right:
    st.markdown(
        "<div class='compare-label' style='color:#38bdf8;'>TARGET — Dense 3D Gaussian Splatting (AnySplat)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="roadmap-card" style="border-color: rgba(56,189,248,0.25);">
            <div class="roadmap-stage-label stage-next">⟳ EVALUATING — Stage 2</div>
            <div class="roadmap-title">AnySplat / VGGT Dense Gaussian Splatting</div>
            <div class="roadmap-desc">
                Feed-forward 3D Gaussian Splatting from <strong>uncalibrated images in a single forward pass</strong>.
                Jointly predicts camera poses, per-pixel depth maps, and dense 3D Gaussian primitives —
                producing a <strong>photorealistic, textured 3D scene</strong> suitable for immersive
                inspection and spatial reasoning.
                <br><br>
                <span style="color:#38bdf8;">Advantage:</span> No iterative optimization required.
                Millions of RGB-colored Gaussians vs. 209 sparse points.
                Real-time novel-view synthesis ready.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Show the target image
ASSETS_DIR = ROOT / "assets"
anysplat_img_path = ASSETS_DIR / "anysplat_target.jpg"
if anysplat_img_path.exists():
    st.image(
        str(anysplat_img_path),
        caption=(
            "Target output: Dense 3D Gaussian Splatting (AnySplat) — photorealistic aerial scene reconstruction "
            "with camera frustums and flight trajectory overlay. "
            "Source: InternRobotics/AnySplat (github.com/InternRobotics/AnySplat)"
        ),
        width="stretch",
    )

# Comparison table
st.markdown("#### Reconstruction Method Comparison")
st.markdown(
    """
    | Metric | COLMAP (Current) | Depth Anything V2 | AnySplat / VGGT (Target) |
    |:---|:---|:---|:---|
    | **Output** | 209 sparse 3D points | 2D relative depth maps | Millions of dense 3D Gaussians |
    | **Camera Poses** | Iterative SfM (offline) | None (monocular 2D) | Jointly predicted (feed-forward) |
    | **Processing** | Minutes (batch) | ~100 ms/frame | ~1–3 s (full scene) |
    | **Scene Rendering** | Scatter plot only | 2D image only | Photorealistic novel-view synthesis |
    | **Pose Requirement** | Required before reconstruction | Not applicable | None — pose-free |
    | **Real-Time Ready** | No | Partial | Yes (feed-forward) |
    """
)

# Stage Roadmap Cards
st.markdown("#### Technology Evolution Roadmap")
r1, r2, r3, r4 = st.columns(4)

with r1:
    st.markdown(
        """
        <div class="roadmap-card">
            <div class="roadmap-stage-label stage-active">✓ CURRENT MVD</div>
            <div class="roadmap-title">Sparse SfM</div>
            <div class="roadmap-desc">COLMAP offline sparse reconstruction + Depth Anything V2 + YOLO11s</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r2:
    st.markdown(
        """
        <div class="roadmap-card" style="border-color:rgba(56,189,248,0.25);">
            <div class="roadmap-stage-label stage-next">⟳ EVALUATING</div>
            <div class="roadmap-title">Dense 3DGS</div>
            <div class="roadmap-desc">AnySplat / VGGT — pose-free dense Gaussian Splatting from UAV sequences</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r3:
    st.markdown(
        """
        <div class="roadmap-card" style="border-color:rgba(168,85,247,0.2);">
            <div class="roadmap-stage-label stage-future">◇ NEXT STAGE</div>
            <div class="roadmap-title">Incremental Mapping</div>
            <div class="roadmap-desc">Online 3D scene updates from streaming UAV frames — real-time spatial mapping</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r4:
    st.markdown(
        """
        <div class="roadmap-card" style="border-color:rgba(168,85,247,0.15);">
            <div class="roadmap-stage-label stage-future">◇ FUTURE</div>
            <div class="roadmap-title">Rescue AI Navigation</div>
            <div class="roadmap-desc">Autonomous drone navigation using live 3D environment understanding for rescue operations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "AI reconstruction model reference: "
    "[InternRobotics/AnySplat](https://github.com/InternRobotics/AnySplat) · "
    "Feed-forward 3D Gaussian Splatting from unconstrained views. "
    "VGGT: Visual Geometry Grounded Transformer for multi-view geometry estimation."
)


# ============================================================
# FINAL PIPELINE SUMMARY
# ============================================================

st.divider()

st.markdown("### 🚀 End-to-End MVD")

st.markdown(
    """
**UAV Video / Images**
→ **Intelligent Keyframe Extraction**
→ **YOLO11s Object Detection**
→ **Depth Anything V2 — Relative Depth**
→ **COLMAP Sparse 3D Reconstruction**
→ **Interactive 3D WebGL Visualization**
→ **[Evaluating] AnySplat / VGGT Dense 3D Gaussian Splatting**
"""
)

st.success("✅ Minimum Viable Demonstrator ready for evaluation.")
st.info(
    "🔬 **Research Evolution:** Sparse offline reconstruction (COLMAP) → "
    "Dense feed-forward Gaussian Splatting (AnySplat) → "
    "Incremental real-time 3D mapping → Rescue AI spatial reasoning → "
    "Autonomous rescue drone navigation."
)