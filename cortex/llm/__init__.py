"""LLM-backed Cortex roles via HuggingFace Inference API."""

from cortex.llm.provider import HuggingFaceProvider
from cortex.llm.roles import LLMRole

__all__ = ["HuggingFaceProvider", "LLMRole"]
