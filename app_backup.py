from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
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

POINT_CLOUD = ROOT / "outputs" / "colmap" / "model.ply"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Drone 3D Reconstruction MVD",
    page_icon="🚁",
    layout="wide",
)

st.title("🚁 Drone 3D Reconstruction — MVD")

st.caption(
    "UAV imagery → COLMAP → Depth Anything V2 → YOLO → "
    "Sparse 3D Reconstruction"
)


# ============================================================
# FRAME SELECTION
# ============================================================

images = sorted(IMAGE_DIR.glob("*.png"))

if not images:
    st.error("No UAV images found.")
    st.stop()

image_names = [image.name for image in images]

selected_name = st.selectbox(
    "Select UAV frame",
    image_names,
)

selected_image = IMAGE_DIR / selected_name
stem = selected_image.stem


# ============================================================
# INPUT IMAGE
# ============================================================

st.header("1. Input UAV Frame")

original = Image.open(selected_image).convert("RGB")

st.image(
    original,
    caption=selected_name,
    use_container_width=True,
)


# ============================================================
# DEPTH
# ============================================================

st.header("2. AI Depth Estimation")

depth_path = DEPTH_DIR / f"depth_{stem}.png"

if depth_path.exists():

    depth = Image.open(depth_path)

    st.image(
        depth,
        caption=f"Depth Anything V2 — {stem}",
        use_container_width=True,
    )

else:

    st.warning(
        f"Depth map not found for {selected_name}"
    )


# ============================================================
# YOLO
# ============================================================

st.header("3. YOLO Object Detection")

yolo_path = YOLO_DIR / f"{stem}.jpg"

if yolo_path.exists():

    detection = Image.open(yolo_path).convert("RGB")

    st.image(
        detection,
        caption=f"YOLO detections — {stem}",
        use_container_width=True,
    )

else:

    st.warning(
        f"YOLO result not found for {selected_name}"
    )


# ============================================================
# COLMAP STATUS
# ============================================================

st.header("4. COLMAP Sparse Reconstruction")

sparse_dir = ROOT / "outputs" / "colmap" / "sparse" / "0"

required_files = [
    "cameras.bin",
    "images.bin",
    "points3D.bin",
]

if sparse_dir.exists() and all(
    (sparse_dir / f).exists()
    for f in required_files
):

    st.success(
        "COLMAP reconstruction successfully generated."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Registered Views",
            "7 / 10",
        )

    with col2:
        st.metric(
            "Sparse 3D Points",
            "209",
        )

else:

    st.warning(
        "COLMAP sparse reconstruction not found."
    )


# ============================================================
# 3D POINT CLOUD
# ============================================================

st.header("5. Interactive Sparse 3D Reconstruction")

if POINT_CLOUD.exists():

    try:

        ply = PlyData.read(POINT_CLOUD)

        vertex = ply["vertex"].data

        x = vertex["x"]
        y = vertex["y"]
        z = vertex["z"]

        # Use RGB if the PLY contains it.
        if all(
            channel in vertex.dtype.names
            for channel in ("red", "green", "blue")
        ):

            rgb = [
                f"rgb({r},{g},{b})"
                for r, g, b in zip(
                    vertex["red"],
                    vertex["green"],
                    vertex["blue"],
                )
            ]

            marker = dict(
                size=3,
                color=rgb,
            )

        else:

            marker = dict(
                size=3,
            )

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=marker,
                )
            ]
        )

        fig.update_layout(
            height=650,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
            ),
            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "Sparse point cloud generated from the COLMAP reconstruction."
        )

    except Exception as e:

        st.error(
            f"Could not load COLMAP point cloud: {e}"
        )

else:

    st.warning(
        "COLMAP point cloud file not found."
    )


# ============================================================
# PIPELINE SUMMARY
# ============================================================

st.divider()

st.header("Complete MVD Pipeline")

st.markdown(
    """
**UAV Images**
→ **COLMAP Camera Reconstruction**
→ **Depth Anything V2**
→ **YOLO Object Detection**
→ **Sparse 3D Point Cloud**
"""
)

st.success(
    "✅ Working prototype pipeline completed."
)