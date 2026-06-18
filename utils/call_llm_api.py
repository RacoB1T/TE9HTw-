"""
Lightweight OpenAI-compatible LLM API client.

Used by ``post_process_critical_paths.py`` for LLM-based judgment of
prediction quality (judge function). Replace ``MODEL_ENDPOINT`` values
with your own API endpoints before use.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — edit these for your own API endpoints
# ---------------------------------------------------------------------------

# OpenAI-compatible API base URLs keyed by model name.
# The default values point to common self-hosted (vLLM/SGLang) endpoints.
MODEL_ENDPOINT: Dict[str, str] = {
    "doubao-pro-128k": "http://localhost:8000/v1",
}

# Shared httpx / requests client.  Create your own or use the default.
# For OpenAI-compatible APIs you can use the openai Python package:
#
#   from openai import OpenAI
#   api_client = OpenAI(base_url=MODEL_ENDPOINT["doubao-pro-128k"], api_key="not-needed")
#
# The default below is a placeholder — replace with a real client.
api_client: Any = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_with_messages(
    client: Any,
    endpoint_url: str,
    prompt: str,
    *,
    max_attempts: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    model: str = "default",
    **kwargs: Any,
) -> Optional[str]:
    """Call an OpenAI-compatible chat completions endpoint.

    Parameters
    ----------
    client :
        An OpenAI-compatible client (``openai.OpenAI`` instance) or ``None``
        to use raw ``requests``.
    endpoint_url :
        Base URL of the API (e.g. ``http://localhost:8000/v1``).
    prompt :
        The user message to send.
    max_attempts :
        Number of retries on failure.
    max_new_tokens :
        Maximum tokens to generate.
    temperature :
        Sampling temperature.

    Returns
    -------
    str or None
        The model response text, or ``None`` if all attempts fail.
    """
    if client is not None:
        return _call_with_openai_client(
            client, model, prompt, max_attempts, max_new_tokens, temperature, **kwargs
        )
    return _call_with_requests(
        endpoint_url, prompt, max_attempts, max_new_tokens, temperature, **kwargs
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _call_with_openai_client(
    client: Any,
    model: str,
    prompt: str,
    max_attempts: int,
    max_new_tokens: int,
    temperature: float,
    **kwargs: Any,
) -> Optional[str]:
    import openai

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            return response.choices[0].message.content
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as exc:
            logger.warning("API call attempt %d/%d failed: %s", attempt, max_attempts, exc)
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
        except Exception as exc:
            logger.error("Unexpected error on attempt %d: %s", attempt, exc)
            if attempt < max_attempts:
                time.sleep(2 ** attempt)

    logger.error("All %d API attempts failed.", max_attempts)
    return None


def _call_with_requests(
    endpoint_url: str,
    prompt: str,
    max_attempts: int,
    max_new_tokens: int,
    temperature: float,
    **kwargs: Any,
) -> Optional[str]:
    import requests

    url = endpoint_url.rstrip("/") + "/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        **kwargs,
    }

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            logger.warning("Request attempt %d/%d failed: %s", attempt, max_attempts, exc)
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Malformed response on attempt %d: %s", attempt, exc)
            if attempt < max_attempts:
                time.sleep(2 ** attempt)

    logger.error("All %d request attempts failed.", max_attempts)
    return None
