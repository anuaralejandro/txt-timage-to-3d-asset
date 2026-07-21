"""
ComfyUI-LocalAssetFactory · Ollama Client
HTTP client for the local Ollama server.

Features:
- Chat/generate completions via the Ollama REST API.
- JSON repair for malformed LLM outputs.
- Retry on transient failures.
- Never logs or exposes secrets.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

import requests

from .. import config
from ..utilities.logging_utils import get_logger
from ..utilities.retry import retry

log = get_logger(__name__)


class OllamaError(RuntimeError):
    """Raised when an Ollama API call fails fatally."""


class OllamaClient:
    """Thin wrapper around the local Ollama HTTP API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: int = 2,
    ):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.DEFAULT_TIMEOUT_SECONDS
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Quick connectivity check."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return names of locally available models."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            data = r.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception as exc:
            log.warning("Could not list Ollama models: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @retry(max_retries=2, delay=2.0, exceptions=(requests.RequestException,))
    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        format_json: bool = False,
    ) -> str:
        """Send a generation request to Ollama and return the full response text."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"

        log.info("Ollama generate request → model=%s (json=%s)", self.model, format_json)

        r = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        response_text = data.get("response", "")
        log.info("Ollama response length: %d chars", len(response_text))
        return response_text

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Generate a JSON response, with repair and retry on parse failure.

        Strategy:
        1. Ask Ollama with ``format=json``.
        2. If JSON is invalid, attempt local repair.
        3. If still invalid, make a second call asking only for corrected JSON.
        4. If still invalid, raise ``OllamaError``.
        """
        raw = self.generate(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            format_json=True,
        )

        parsed = self._try_parse_json(raw)
        if parsed is not None:
            return parsed

        # Attempt local repair
        repaired = self._repair_json(raw)
        if repaired is not None:
            log.warning("JSON repaired locally from LLM output.")
            return repaired

        # Second call — correction prompt
        log.warning("LLM JSON invalid, requesting correction …")
        correction_prompt = (
            "The following JSON is malformed. Please return ONLY the corrected JSON, "
            "with no markdown formatting, no explanation, no extra text:\n\n"
            f"{raw}"
        )
        raw2 = self.generate(
            correction_prompt,
            system="You are a JSON repair assistant. Output only valid JSON.",
            temperature=0.1,
            max_tokens=max_tokens,
            format_json=True,
        )
        parsed2 = self._try_parse_json(raw2)
        if parsed2 is not None:
            return parsed2

        repaired2 = self._repair_json(raw2)
        if repaired2 is not None:
            return repaired2

        raise OllamaError(
            "Ollama returned invalid JSON after two attempts. "
            f"Last raw output (truncated): {raw2[:500]}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Try to parse *text* as JSON.  Returns None on failure."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _repair_json(text: str) -> Optional[Dict[str, Any]]:
        """Attempt common repairs on malformed JSON strings."""
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = cleaned.strip().rstrip("`")

        # Try extracting the first JSON object
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Try fixing trailing commas
        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        return None
