"""
VGGT & VGGT-Ω Reconstruction Backend Wrapper
============================================
Exposes modular interface for Visual Geometry Grounded Transformer backends.
"""
from src.vggt_pipeline import VGGTAgent, run_vggt_pipeline

__all__ = ["VGGTAgent", "run_vggt_pipeline"]
