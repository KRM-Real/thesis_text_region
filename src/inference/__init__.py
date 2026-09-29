"""Shared inference entry point for CLI and Streamlit."""

from .service import InferenceResult, predict_image_bytes, predict_image_path

__all__ = ["InferenceResult", "predict_image_bytes", "predict_image_path"]
