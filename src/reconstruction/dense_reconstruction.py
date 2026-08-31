"""
Dense 3DGS & Depth Unprojection Reconstruction Backend Wrapper
==============================================================
Exposes AnySplat 3D Gaussian Splatting and dense depth unprojection backends.
"""
from src.anysplat_pipeline import AnySplatAgent, run_anysplat_pipeline
from src.video_pipeline import estimate_point_cloud_and_trajectory

__all__ = ["AnySplatAgent", "run_anysplat_pipeline", "estimate_point_cloud_and_trajectory"]
