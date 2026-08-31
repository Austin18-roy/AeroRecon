---
name: anysplat
description: Comprehensive guide and operational skill for AnySplat (Feed-Forward 3D Gaussian Splatting from Unconstrained Views). Use when implementing, configuring, integrating, or running AnySplat for pose-free 3DGS reconstruction, novel view synthesis, depth map prediction, camera pose estimation from uncalibrated multi-view UAV/robotics images, or upgrading sparse SfM pipelines to dense Gaussian Splatting.
---

# AnySplat: Feed-Forward 3D Gaussian Splatting Skill Guide

[AnySplat](https://github.com/InternRobotics/AnySplat) is an open-source framework developed by InternRobotics for **pose-free, feed-forward 3D Gaussian Splatting (3DGS)** from unconstrained, uncalibrated image collections.

Unlike traditional 3DGS or NeRF pipelines that require precomputed Structure-from-Motion (SfM/COLMAP) camera poses and lengthy per-scene iterative optimization, AnySplat predicts **camera poses, dense depth maps, and 3D Gaussian parameters in a single forward pass**.

---

## 1. Technical Architecture

```
                    ┌────────────────────────┐
                    │ Uncalibrated Images   │ (e.g. UAV flight frames)
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ Transformer Geometry   │
                    │ Encoder (DUSt3R-based) │
                    └─────┬──────┬──────┬────┘
                          │      │      │
           ┌──────────────┘      │      └──────────────┐
           ▼                     ▼                     ▼
┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
│ Gaussian Head (FG) │ │ Depth Head (FD)    │ │ Camera Head (FC)   │
│ μ, σ, r, s, c      │ │ Depth Map D        │ │ Camera Poses p     │
└──────────┬─────────┘ └─────────┬──────────┘ └─────────┬──────────┘
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 │
                    ┌────────────▼───────────┐
                    │ Differentiable         │
                    │ Voxelization Module    │
                    └────────────┬───────────┘
                                 │
                    ┌────────────▼───────────┐
                    │ Pre-Voxel 3D Gaussians │ (Exportable as .ply / .splat)
                    └────────────┬───────────┘
                                 │
                    ┌────────────▼───────────┐
                    │ Real-Time Rendering &  │
                    │ Novel View Synthesis   │
                    └────────────────────────┘
```

### Key Components

1. **Geometry Encoder:** Multi-view vision transformer processing uncalibrated input image sets.
2. **Three Decoupled Heads:**
   - **$F_G$ (Gaussian Head):** Predicts 3D Gaussian primitive parameters: center position $\mu$, opacity $\sigma$, rotation quaternion $r$, scaling vector $s$, and harmonic color features $c$.
   - **$F_D$ (Depth Head):** Decodes per-frame dense depth maps.
   - **$F_C$ (Camera Head):** Predicts camera extrinsics and focal length intrinsics $p$.
3. **Differentiable Voxelization:** Aggregates and prunes dense pixel-wise Gaussians into structured 3D space, eliminating redundant floaters.
4. **Supervision:** Dual-loss scheme combining RGB multi-view rendering loss against ground-truth and geometry loss against VGGT/DUSt3R geometric priors.

---

## 2. Installation & Prerequisites

### Hardware & Environment Requirements
- **OS:** Linux or Windows (with CUDA support)
- **Python:** 3.10+
- **PyTorch:** 2.2.0+ with CUDA 12.1+
- **GPU:** NVIDIA GPU with ≥ 12GB VRAM (24GB recommended for multi-view inference)

### Setup Steps
```bash
# 1. Clone repository
git clone https://github.com/InternRobotics/AnySplat.git
cd AnySplat

# 2. Environment creation
conda create -y -n anysplat python=3.10
conda activate anysplat

# 3. Install PyTorch with CUDA
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Install requirements & submodules (diff-gaussian-rasterization)
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
```

### Pretrained Model Weights
- **Hugging Face Model Hub:** [`lhjiang/anysplat`](https://huggingface.co/lhjiang/anysplat)
- Download weights and place them in the `checkpoints/` directory:
```bash
huggingface-cli download lhjiang/anysplat --local-dir ./checkpoints
```

---

## 3. Core Inference & Usage Workflow

### Python API Inference Script Example
```python
import os
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T

# Load AnySplat pipeline
from anysplat.models import AnySplatModel
from anysplat.utils.render import render_gaussians
from anysplat.utils.ply_export import save_ply_gaussians

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = AnySplatModel.from_pretrained("./checkpoints").to(device)
model.eval()

# Load uncalibrated UAV / multi-view images
image_paths = sorted(Path("data/input/seq38/Images").glob("*.png"))
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

images_tensor = torch.stack([transform(Image.open(p).convert("RGB")) for p in image_paths]).unsqueeze(0).to(device)

# Single forward pass: extracts 3D Gaussians, depth, and camera poses
with torch.no_grad():
    outputs = model(images_tensor)
    gaussians = outputs["gaussians"]       # 3D Gaussian primitives
    pred_depth = outputs["depth"]          # Predicted dense depth maps
    pred_poses = outputs["camera_poses"]   # Predicted camera poses

# Export 3D Gaussian Splat PLY
save_ply_gaussians(gaussians, "outputs/anysplat/scene_splat.ply")
print("Exported AnySplat 3DGS model to outputs/anysplat/scene_splat.ply")
```

---

## 4. Integration with AeroRecon Drone Reconstruction

AnySplat offers significant advantages for autonomous drone and robotics 3D reconstruction:

| Capability | COLMAP (Current SfM) | Depth Anything V2 | AnySplat (3DGS) |
| :--- | :--- | :--- | :--- |
| **Output Type** | Sparse Point Cloud (209 pts) | 2D Relative Depth Maps | Dense 3D Gaussians (Millions) |
| **Camera Poses** | Required / Computed via SfM | None (Monocular) | Jointly predicted in 1 pass |
| **Processing Time** | Minutes (Iterative Bundle Adj.) | Fast (~100ms/frame) | Fast (~1s for multi-view batch) |
| **Rendering** | Scatter points / Mesh | 2D Image only | Photorealistic Novel View Synthesis |
| **Real-Time Readiness** | Offline only | Near real-time | Feed-forward real-time pipeline |

### Recommended AeroRecon Integration Roadmap:
1. **Stage 1 (Current):** Sparse SfM (COLMAP) + Monocular Relative Depth (Depth Anything V2) + YOLO11s.
2. **Stage 2 (Hybrid):** Use AnySplat to produce dense 3D Gaussian Splatting `.ply` directly from raw UAV sequences without relying on slow bundle adjustment.
3. **Stage 3 (Real-Time Deployment):** Stream UAV frames directly into AnySplat for instant 3D environment generation and obstacle hazard volume estimation for autonomous flight planning.

---

## 5. Troubleshooting & Best Practices

1. **VRAM Optimization:**
   - AnySplat processes multi-view cross-attention tokens. If running out of memory (OOM), downsample input images to $512 \times 512$ or $384 \times 384$ before feeding them into the transformer encoder.
   - Use mixed precision (`torch.cuda.amp.autocast()`).
2. **Coordinate System Conventions:**
   - AnySplat predicts camera poses in the OpenCV/COLMAP camera coordinate frame ($+X$ right, $+Y$ down, $+Z$ forward).
3. **Web Viewer Compatibility:**
   - Exported `.ply` files contain standard Gaussian properties (`x, y, z`, `f_dc_*`, `opacity`, `scale_*`, `rot_*`). They can be rendered in real-time in browsers via WebGL / Three.js 3DGS viewers (e.g. AntSplat, SuperSplat, or PlayCanvas engine).
