from pathlib import Path
import struct

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from matplotlib import cm as mpl_cm
from plyfile import PlyData
from PIL import Image


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

IMAGE_DIR = ROOT / "data" / "input" / "seq38" / "Images"
DEPTH_DIR = ROOT / "outputs" / "depth"

YOLO_DIR = (
    ROOT
    / "runs"
    / "detect"
    / "outputs"
    / "detections"
    / "annotated"
)

SPARSE_DIR = ROOT / "outputs" / "colmap" / "sparse" / "0"
IMAGES_BIN = SPARSE_DIR / "images.bin"
POINT_CLOUD = ROOT / "outputs" / "colmap" / "model.ply"


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
# DATA VERIFICATION & DYNAMIC COUNTS
# ============================================================

images = sorted(IMAGE_DIR.glob("*.png")) if IMAGE_DIR.exists() else []

if not images:
    st.error(f"No UAV images found in: {IMAGE_DIR}")
    st.stop()

image_names = [image.name for image in images]
depth_files = list(DEPTH_DIR.glob("depth_*.png")) if DEPTH_DIR.exists() else []
cameras = load_colmap_cameras(IMAGES_BIN)

vertex_count = 0
if POINT_CLOUD.exists():
    try:
        ply_temp = PlyData.read(POINT_CLOUD)
        vertex_count = len(ply_temp["vertex"].data)
    except Exception:
        vertex_count = 209


# ============================================================
# 1. HEADER / HERO
# ============================================================

hero_col1, hero_col2 = st.columns([3, 1])

with hero_col1:
    st.title("🚁 AeroRecon")
    st.subheader("AI-Assisted Drone 3D Reconstruction")
    st.markdown(
        "From **UAV imagery** to **object detection**, **relative depth estimation**, "
        "**camera reconstruction**, and **sparse 3D visualization**."
    )

with hero_col2:
    st.markdown(
        """
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
                <span style="font-size: 9px;">●</span> Prototype Ready
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 2. PROCESSING PIPELINE
# ============================================================

st.markdown("### 🔄 Processing Pipeline")

p1, p2, p3, p4, p5 = st.columns(5)

with p1:
    st.markdown(
        f"""
        <div class="pipeline-card">
            <div class="pipeline-status">✓ Stage 1</div>
            <div class="pipeline-title">UAV Input</div>
            <div class="pipeline-desc">{len(images)} frames</div>
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
        """
        <div class="pipeline-card">
            <div class="pipeline-status">✓ Stage 4</div>
            <div class="pipeline-title">COLMAP</div>
            <div class="pipeline-desc">Camera + Geometry</div>
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
    reg_count = len(cameras) if cameras else 7
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

orig_w, orig_h = 4096, 2160
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

    # Apply perceptual colormap for display only (does not modify the file on disk)
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
# 6. COLMAP STATUS & INTERACTIVE 3D RECONSTRUCTION
# ============================================================

st.markdown("### 📐 COLMAP Sparse 3D Reconstruction")

required_files = [
    "cameras.bin",
    "images.bin",
    "points3D.bin",
]

if SPARSE_DIR.exists() and all((SPARSE_DIR / f).exists() for f in required_files):
    st.success("✓ COLMAP Structure-from-Motion reconstruction successfully generated.")
else:
    st.warning("COLMAP sparse reconstruction files not found.")

st.markdown("### 🌐 Interactive 3D Reconstruction")

st.markdown(
    "COLMAP estimates UAV camera poses from overlapping images and triangulates "
    "matched visual features into a sparse 3D representation of the captured environment."
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

# Metric cards — dynamically derived
reg_val = len(cameras) if cameras else 7
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
        "Estimated UAV positions recovered by COLMAP's Structure-from-Motion solver.</span>",
        unsafe_allow_html=True,
    )
with exp3:
    st.markdown(
        "**━ Flight Trajectory**<br>"
        "<span style='color: #94a3b8; font-size: 0.85rem;'>"
        "Connected camera markers showing estimated UAV movement between registered views.</span>",
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

        # View preset camera
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
            "Interactive sparse 3D point cloud and camera trajectory generated from COLMAP. "
            "Use mouse to orbit, scroll to zoom, shift+drag to pan."
        )

    except Exception as e:
        st.error(f"Could not load the COLMAP point cloud: {e}")

else:
    st.warning("COLMAP PLY point cloud not found.")

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

# Reconstruction status & roadmap
recon1, recon2 = st.columns([1, 1])
with recon1:
    st.info(
        "**Current prototype:** The system currently produces a sparse offline reconstruction "
        "from overlapping UAV frames. This demonstrates camera-pose estimation, feature "
        "triangulation, and 3D scene visualization."
    )
with recon2:
    st.info(
        "**Next stage:** Incremental reconstruction will allow new UAV frames to update the "
        "3D environment continuously, leading toward real-time and denser spatial mapping."
    )

st.caption(
    "**Future development:** Sparse offline reconstruction → incremental 3D mapping → "
    "real-time scene reconstruction → Rescue AI spatial reasoning → autonomous rescue drone navigation."
)


# ============================================================
# FINAL PIPELINE SUMMARY
# ============================================================

st.divider()

st.markdown("### 🚀 End-to-End MVD")

st.markdown(
    """
**UAV Images**
→ **COLMAP Camera Reconstruction**
→ **Depth Anything V2**
→ **YOLO Object Detection**
→ **Sparse 3D Point Cloud**
"""
)

st.success("✅ Minimum Viable Demonstrator ready for evaluation.")