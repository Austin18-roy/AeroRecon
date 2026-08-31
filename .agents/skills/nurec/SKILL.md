---
name: nurec
description: Comprehensive guide and operational skill for NVIDIA NuRec (Neural Reconstruction Skills). Use when implementing, configuring, optimizing, or running NVIDIA NuRec for neural surface reconstruction, multi-resolution hash grid integration, camera pose bundle refinement, photometric consistency optimization, and high-fidelity 3D scene mapping from UAV/robotics imagery.
---

# NVIDIA NuRec: Neural Reconstruction Skills Guide

[NVIDIA NuRec](https://github.com/NVIDIA/nurec-skills) is NVIDIA's neural scene reconstruction framework designed for **high-precision 3D neural surface and radiance field reconstruction** from unconstrained multi-view image collections and drone flight videos.

NuRec utilizes multi-resolution instant hash grids, neural signed distance fields (NeuS / N-SDF), and cross-view photometric pose refinement to eliminate reconstruction floaters and produce clean, dense geometric surfaces.

---

## 1. Technical Architecture

```
                    ┌────────────────────────┐
                    │ Multi-View UAV Images  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ Neural Camera Pose     │
                    │ Bundle Refinement Head │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ Multi-Resolution Hash  │
                    │ Feature Grid (Instant) │
                    └─────┬────────────┬─────┘
                          │            │
           ┌──────────────▼───┐    ┌───▼──────────────────┐
           │ Neural SDF Head  │    │ Radiance Color Head  │
           │ Signed Distance  │    │ View-Dependent RGB   │
           └──────────────┬───┘    └───┬──────────────────┘
                          │            │
                          └─────┬──────┘
                                │
                    ┌───────────▼────────────┐
                    │ Geometric Regularizer  │
                    │ & Denoising Filter     │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ Dense High-Precision   │
                    │ 3D Pointmap / Mesh PLY │
                    └────────────────────────┘
```

### Key Components

1. **Neural Pose Refinement:** Iteratively minimizes cross-view epipolar reprojection errors to recover accurate camera positions.
2. **Instant Hash Grids:** Fast multi-resolution spatial hashing that encodes fine geometric details (facades, windows, terrain breaks).
3. **NeuS Surface Regularization:** Enforces smooth surface curvature while preserving sharp architectural edges.
4. **Photometric Denoising:** Filters sensor noise, sun glare, and motion blur floaters from drone video frames.

---

## 2. Integration with AeroRecon

| Capability | COLMAP (SfM) | AnySplat (3DGS) | VGGT-Ω (Transformer) | NVIDIA NuRec (Neural) |
| :--- | :--- | :--- | :--- | :--- |
| **Method** | Feature Matching | Feed-Forward Splatting | Vision Transformer Pointmap | Multi-Res Hash Neural Recon |
| **Output Type** | Sparse Points (209) | 3D Gaussian Ellipsoids | Dense 3D Pointmaps | High-Precision Neural Surface |
| **Surface Detail** | Skeletal only | Photorealistic Splats | Dense Textured Points | Denoised Surface Geometry |
| **Pose Optimization**| Iterative Bundle Adj. | Pose-Free Prediction | Cross-Attention Head | Neural Photometric Refinement |
| **Processing Speed** | Slow (Minutes) | Fast (~1–2s) | Fast (~1.5s) | Fast Neural Hash Pass (~1s) |

---

## 3. Usage in AeroRecon

- Select **"🟢 NVIDIA NuRec Agent"** in the AeroRecon sidebar to optimize 3D map generation with multi-resolution hash grid surface reconstruction and neural pose refinement.
