"""HuggingFace Inference API provider for LLM-backed roles."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from huggingface_hub import InferenceClient

_log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAYS = (1.0, 2.0, 4.0)


class HuggingFaceProvider:
    """Calls HuggingFace Inference API and extracts JSON responses."""

    def __init__(
        self,
        api_key: str,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        timeout_s: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("HuggingFace API key is required")
        self._client = InferenceClient(token=api_key, model=model, timeout=timeout_s)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.total_calls = 0
        self.total_fallbacks = 0
        self.total_response_time = 0.0

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system: str, user: str) -> dict[str, Any]:
        """Call HF API, extract JSON from response text.

        Returns parsed dict or raises ValueError if JSON extraction fails.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Phase 1: API call with retry (only retries network/API errors)
        text = ""
        for attempt in range(_MAX_RETRIES):
            try:
                start = time.monotonic()
                response = self._client.chat_completion(
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                elapsed = time.monotonic() - start
                self.total_calls += 1
                self.total_response_time += elapsed
                text = response.choices[0].message.content or ""
                break  # API succeeded
            except Exception as exc:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_DELAYS[attempt]
                    _log.warning(
                        "HF API call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    _log.error("HF API call failed after %d retries: %s", _MAX_RETRIES, exc)
                    raise

        # Phase 2: JSON extraction (no retry -- parse failure means bad output)
        return self._extract_json(text)

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract first JSON object from LLM response text.

        Uses bracket-counting to handle arbitrarily nested JSON.
        """
        # Strip code fences if present
        stripped = re.sub(r"```(?:json)?\s*", "", text)
        stripped = stripped.replace("```", "")

        # Find first '{' and count brackets to find matching '}'
        start = stripped.find("{")
        if start == -1:
            raise ValueError(f"No JSON found in response: {text[:200]}")

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(stripped)):
            c = stripped[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : i + 1]
                    return json.loads(candidate)

        raise ValueError(f"Unbalanced braces in response: {text[:200]}")

    def reset_stats(self) -> None:
        """Reset call statistics (per episode)."""
        self.total_calls = 0
        self.total_fallbacks = 0
        self.total_response_time = 0.0
