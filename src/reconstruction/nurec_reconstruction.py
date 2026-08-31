"""
NVIDIA NuRec Neural Surface Reconstruction Backend Wrapper
==========================================================
Exposes NVIDIA NuRec multi-resolution hash grid neural surface reconstruction.
"""
from src.nurec_pipeline import NuRecAgent, run_nurec_pipeline

__all__ = ["NuRecAgent", "run_nurec_pipeline"]
