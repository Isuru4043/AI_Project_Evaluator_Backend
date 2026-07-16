"""
Unified LLM service — single entry point for all model calls in the viva pipeline.

DESIGN GOAL:
    Every agent (Analyzer, Strategist, Questioner, Critic) calls `llm_call` here.
    When we swap providers (Gemini → OpenAI/Claude/local), only this file changes.
    Agent code stays untouched.

CURRENT BACKEND: Google Gemini (google-genai SDK)
"""

import json
import logging
import os
import re
import time
from typing import Any, Optional

from django.conf import settings

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


def _is_model_unavailable(exc: Exception) -> bool:
    """A 404 for the model itself — retired, or closed to new projects.

    Retrying is pointless; the next model in the chain is the only way out.
    """
    text = str(exc).lower()
    return (
        '404' in text
        or 'not_found' in text
        or 'no longer available' in text
        or 'is not found' in text
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
#
# Each purpose maps to a CHAIN, tried in order. A model is skipped when it is
# out of quota (429) or unavailable to this project (404); the next one takes
# over. The free-tier request cap is counted per project *per model*, so a
# fallback gets a genuinely separate allowance rather than the same wall.
#
# Set an env var to a comma-separated list to override a chain, e.g.
#   LLM_DEFAULT_MODEL=gemini-3.5-flash,gemini-flash-latest
# =============================================================================

# Ordered best-first, then degrading to lite. Every entry is verified reachable
# on our key: a model that always 404s would burn a round trip on every call
# before falling through, so dead rungs are worse than no rung. Notably the
# gemini-2.5-* family is closed to new projects ("no longer available to new
# users") and must not be used as a fallback.
DEFAULT_MODEL_CHAIN = [
    'gemini-3.5-flash',
    'gemini-3-flash-preview',
    'gemini-3.1-flash-lite',
    'gemini-flash-lite-latest',
]


def _chain_from_env(var: str, default: list) -> list:
    raw = os.getenv(var, '')
    models = [m.strip() for m in raw.split(',') if m.strip()]
    return models or list(default)


MODEL_REGISTRY = {
    'default':   os.getenv('LLM_DEFAULT_MODEL',   settings.GEMINI_MODEL),
    'fast':      os.getenv('LLM_FAST_MODEL',      settings.GEMINI_MODEL),
    'reasoning': os.getenv('LLM_REASONING_MODEL', settings.GEMINI_MODEL),
}


# =============================================================================
# Lazy-loaded client — avoid initialization cost at import time.
# =============================================================================

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(
            vertexai=True,
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.GOOGLE_CLOUD_LOCATION,
        )
    return _client


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
    chain = MODEL_REGISTRY.get(model) or MODEL_REGISTRY['default']
    client = _get_client()

    last_error = None
    quota_error = None

    for model_id in chain:
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

                # Quota (429) and model-unavailable (404) are properties of the
                # model, not of this attempt — no amount of retrying clears
                # them. Abandon this model and let the chain move on.
                if _is_quota_error(exc):
                    quota_error = exc
                    logger.warning(
                        'llm_call QUOTA on model=%s, falling through: %s',
                        model_id, str(exc)[:160],
                    )
                    break

                if _is_model_unavailable(exc):
                    logger.warning(
                        'llm_call model=%s unavailable, falling through: %s',
                        model_id, str(exc)[:160],
                    )
                    break

                logger.warning(
                    'llm_call error attempt=%d model=%s err=%s',
                    attempt, model_id, exc,
                )
                # Exponential backoff: 0.5s, 1s, 2s
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** attempt))

    # Every model in the chain failed.
    if fallback is not None:
        logger.error(
            'llm_call chain exhausted (%s), using fallback. last_error=%s',
            ','.join(chain), last_error,
        )
        return fallback

    # A quota wall anywhere in the chain is the more actionable diagnosis:
    # surface it as the typed error so views show "try again later" rather
    # than a generic failure.
    if quota_error is not None:
        logger.error('llm_call QUOTA exhausted across chain=%s', ','.join(chain))
        raise LLMQuotaError(
            'AI service quota exceeded.',
            retry_after_seconds=_extract_retry_after(quota_error),
        )

    raise RuntimeError(
        f'LLM call failed on every model in chain ({",".join(chain)}): {last_error}'
    )


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
