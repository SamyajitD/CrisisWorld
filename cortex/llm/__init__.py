"""LLM-backed Cortex roles via HuggingFace Inference API."""

from .provider import HuggingFaceProvider
from .roles import LLMRole

__all__ = ["HuggingFaceProvider", "LLMRole"]
