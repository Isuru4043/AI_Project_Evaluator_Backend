"""
Unified LLM service — single entry point for all model calls in the viva pipeline.

DESIGN GOAL:
    Every agent (Analyzer, Strategist, Questioner, Critic) calls `llm_call` here.
    When we swap providers (Gemini → OpenAI/Claude/local), only this file changes.
    Agent code stays untouched.

CURRENT BACKEND: Vertex AI Gemini via the google-genai SDK
"""

import json
import logging
import os
import re
import time
from typing import Any, Optional

from django.conf import settings

from AI_Evaluator_Backend.llm import get_llm

logger = logging.getLogger(__name__)


class LLMQuotaError(Exception):
    """Raised when the provider returns a quota / rate-limit (429) error.

    Distinct from generic failures so views can show a clear 'service busy,
    try again shortly' message instead of shipping an empty/neutral result.
    """
    def __init__(self, message: str, retry_after_seconds: Optional[int] = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        '429' in text
        or 'resource_exhausted' in text
        or 'quota' in text
        or 'rate limit' in text
        or 'rate-limit' in text
    )


def _extract_retry_after(exc: Exception) -> Optional[int]:
    import re as _re
    m = _re.search(r'retry in ([\d.]+)\s*s', str(exc), _re.IGNORECASE)
    if m:
        try:
            return int(float(m.group(1)))
        except ValueError:
            return None
    return None


# =============================================================================
# Model registry — semantic names mapped to provider-specific identifiers.
# Agents request models by purpose, not by provider-specific name.
#
# Env-overridable for speed/quality A/B testing:
#   flash-lite ~ 1s/call   |   flash ~ 6s/call
#   reasoning = Analyzer + Questioner (latency hotspot)
#   fast      = Critic, code summaries, image captions
# =============================================================================

MODEL_REGISTRY = {
    'default':   os.getenv('LLM_DEFAULT_MODEL',   settings.GEMINI_MODEL),
    'fast':      os.getenv('LLM_FAST_MODEL',      settings.GEMINI_MODEL),
    'reasoning': os.getenv('LLM_REASONING_MODEL', settings.GEMINI_MODEL),
}


# =============================================================================
# Shared client — initialized lazily and cached by AI_Evaluator_Backend.llm.
# =============================================================================

def _get_client():
    return get_llm()


# =============================================================================
# Public API — what every agent calls.
# =============================================================================

def llm_call(
    prompt: str,
    model: str = 'default',
    expect_json: bool = False,
    max_retries: int = 2,
    fallback: Optional[Any] = None,
) -> Any:
    """
    Single entry point for LLM calls.

    Args:
        prompt:       The full prompt to send.
        model:        Semantic model name from MODEL_REGISTRY.
        expect_json:  If True, attempt JSON parsing and return dict/list.
        max_retries:  How many times to retry on transient failure.
        fallback:     Value to return if all retries fail. If None, raises.

    Returns:
        - If expect_json=True: parsed dict/list (or fallback on JSON failure)
        - Else: raw text string

    Raises:
        RuntimeError on persistent provider failures (only if fallback is None).
    """
    return _llm_call_internal(
        prompt=prompt,
        model=model,
        expect_json=expect_json,
        max_retries=max_retries,
        fallback=fallback,
        image_bytes=None,
    )


def llm_call_with_image(
    prompt: str,
    image_bytes: bytes,
    image_mime: str = 'image/png',
    model: str = 'default',
    expect_json: bool = False,
    max_retries: int = 2,
    fallback: Optional[Any] = None,
) -> Any:
    """
    Multimodal call — same contract as llm_call, but with one image attached.

    Used by the image captioner during indexing. Gemini handles the vision
    part natively; if you swap providers later, only this function changes.
    """
    return _llm_call_internal(
        prompt=prompt,
        model=model,
        expect_json=expect_json,
        max_retries=max_retries,
        fallback=fallback,
        image_bytes=image_bytes,
        image_mime=image_mime,
    )


def _llm_call_internal(
    prompt: str,
    model: str,
    expect_json: bool,
    max_retries: int,
    fallback: Optional[Any],
    image_bytes: Optional[bytes] = None,
    image_mime: str = 'image/png',
) -> Any:
    model_id = MODEL_REGISTRY.get(model, MODEL_REGISTRY['default'])
    client = _get_client()

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()

            # Build contents: text-only or [image, text] for multimodal
            if image_bytes is not None:
                from google.genai import types
                contents = [
                    types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
                    prompt,
                ]
            else:
                contents = prompt

            response = client.models.generate_content(
                model=model_id,
                contents=contents,
            )
            latency_ms = int((time.time() - t0) * 1000)
            raw_text = (response.text or '').strip()

            logger.info(
                'llm_call ok model=%s latency=%dms chars_in=%d chars_out=%d image=%s',
                model_id, latency_ms, len(prompt), len(raw_text),
                'yes' if image_bytes else 'no',
            )

            if not expect_json:
                return raw_text

            parsed = _parse_json(raw_text)
            if parsed is not None:
                return parsed

            # JSON parse failed — count as a retryable error
            last_error = ValueError('LLM returned malformed JSON')
            logger.warning(
                'llm_call json_parse_failed attempt=%d model=%s preview=%r',
                attempt, model_id, raw_text[:200],
            )

        except Exception as exc:
            last_error = exc

            # Quota / rate-limit (429): retrying within this request is
            # pointless — the daily cap won't clear in milliseconds. Raise
            # a typed error so the caller can show a clean message.
            if _is_quota_error(exc):
                logger.error(
                    'llm_call QUOTA error model=%s: %s', model_id, str(exc)[:200],
                )
                raise LLMQuotaError(
                    'AI service quota exceeded.',
                    retry_after_seconds=_extract_retry_after(exc),
                )

            logger.warning(
                'llm_call error attempt=%d model=%s err=%s',
                attempt, model_id, exc,
            )
            # Exponential backoff: 0.5s, 1s, 2s
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))

    # All retries exhausted
    if fallback is not None:
        logger.error('llm_call exhausted retries, using fallback. last_error=%s', last_error)
        return fallback

    raise RuntimeError(f'LLM call failed after {max_retries + 1} attempts: {last_error}')


# =============================================================================
# JSON parsing — robust against markdown fences, leading/trailing text.
# =============================================================================

_FENCE_RE = re.compile(r'^```(?:json)?\s*|\s*```$', re.MULTILINE)


def _parse_json(text: str) -> Optional[Any]:
    """
    Best-effort JSON extraction from LLM output. Returns None on failure.

    Handles:
      - Plain JSON
      - Markdown-fenced JSON (```json ... ```)
      - JSON embedded in surrounding prose (extracts first {...} or [...])
    """
    if not text:
        return None

    # Strip markdown fences
    cleaned = _FENCE_RE.sub('', text).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting the largest JSON object or array
    for opener, closer in [('{', '}'), ('[', ']')]:
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    return None
