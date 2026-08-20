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
from viva_evaluator.services.llm_telemetry import (
    estimate_cost_usd,
    extract_usage,
    record_llm_call,
)

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


def _as_chain(value) -> list:
    """Coerce a model spec into a LIST of model ids.

    A chain must never be left as a bare string: `for model_id in chain`
    iterates a string CHARACTER BY CHARACTER, so 'gemini-3.1-flash-lite'
    becomes 21 requests for models named 'g', 'e', 'm', … each rejected with
    400 INVALID_ARGUMENT, and the chain then reports itself 'exhausted'.
    Normalizing here means a string can never reach the loop again.
    """
    if isinstance(value, str):
        return [m.strip() for m in value.split(',') if m.strip()]
    return [m for m in (value or []) if m]


def _chain_from_env(var: str, default: list) -> list:
    return _as_chain(os.getenv(var, '')) or list(default)


def _default_chain() -> list:
    """GEMINI_MODEL leads — it is what this deployment is configured and billed
    against — with DEFAULT_MODEL_CHAIN supplying the fallback rungs behind it.
    De-duplicated so the primary is never retried as its own fallback."""
    lead = _as_chain(settings.GEMINI_MODEL)
    return lead + [m for m in DEFAULT_MODEL_CHAIN if m not in lead]


MODEL_REGISTRY = {
    'default':   _chain_from_env('LLM_DEFAULT_MODEL',   _default_chain()),
    'fast':      _chain_from_env('LLM_FAST_MODEL',      _default_chain()),
    'reasoning': _chain_from_env('LLM_REASONING_MODEL', _default_chain()),
}

# A caller may still supply a broad semantic model, but stable operation names
# are more precise and make model experiments possible without editing agents.
OPERATION_MODEL_ROUTES = {
    'answer_analysis': 'reasoning',
    'question_generation': 'reasoning',
    'question_repair': 'fast',
    'question_critic': 'fast',
    'response_triage': 'fast',
    'fairness_review': 'fast',
    'fairness_consistency': 'fast',
    'fairness_charitable': 'fast',
    'fairness_self_correction': 'fast',
}

OPERATION_MODEL_ENV = {
    'answer_analysis': 'LLM_ANSWER_ANALYSIS_MODEL',
    'question_generation': 'LLM_QUESTION_GENERATION_MODEL',
    'question_repair': 'LLM_QUESTION_REPAIR_MODEL',
    'question_critic': 'LLM_QUESTION_CRITIC_MODEL',
    'response_triage': 'LLM_RESPONSE_TRIAGE_MODEL',
    'fairness_review': 'LLM_FAIRNESS_REVIEW_MODEL',
}

# Defaults retain useful RAG evidence while bounding pathological prompts.
# Trimming keeps both the instruction prefix and output-schema suffix.
DEFAULT_PROMPT_BUDGETS = {
    'answer_analysis': 24_000,
    'question_generation': 20_000,
    'question_repair': 12_000,
    'question_critic': 14_000,
    'response_triage': 6_000,
    'fairness_review': 10_000,
    'fairness_consistency': 8_000,
    'fairness_charitable': 8_000,
    'fairness_self_correction': 8_000,
}


def _operation_env_suffix(operation: str) -> str:
    return re.sub(r'[^A-Z0-9]+', '_', str(operation or '').upper()).strip('_')


def _resolve_model_chain(model: str, operation: str) -> tuple[list, str]:
    operation = str(operation or 'unspecified')
    env_name = OPERATION_MODEL_ENV.get(operation)
    override = _as_chain(os.getenv(env_name, '')) if env_name else []
    if override:
        return override, f'operation_override:{operation}'

    semantic_route = OPERATION_MODEL_ROUTES.get(operation, model)
    default_chain = MODEL_REGISTRY.get('default') or _default_chain()
    chain = _as_chain(MODEL_REGISTRY.get(semantic_route)) or _as_chain(default_chain)
    return chain, semantic_route


def _apply_prompt_budget(prompt: str, operation: str) -> tuple[str, int, bool]:
    suffix = _operation_env_suffix(operation)
    env_value = os.getenv(f'LLM_PROMPT_MAX_CHARS_{suffix}', '') if suffix else ''
    try:
        budget = int(env_value) if env_value else int(
            DEFAULT_PROMPT_BUDGETS.get(operation, 0)
        )
    except (TypeError, ValueError):
        budget = int(DEFAULT_PROMPT_BUDGETS.get(operation, 0))

    if budget <= 0 or len(prompt) <= budget:
        return prompt, max(0, budget), False

    marker = '\n\n[... middle context trimmed to prompt budget ...]\n\n'
    usable = max(2, budget - len(marker))
    head_chars = max(1, int(usable * 0.65))
    tail_chars = max(1, usable - head_chars)
    return prompt[:head_chars] + marker + prompt[-tail_chars:], budget, True


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
    operation: str = 'unspecified',
) -> Any:
    """
    Single entry point for LLM calls.

    Args:
        prompt:       The full prompt to send.
        model:        Semantic model name from MODEL_REGISTRY.
        expect_json:  If True, attempt JSON parsing and return dict/list.
        max_retries:  How many times to retry on transient failure.
        fallback:     Value to return if all retries fail. If None, raises.
        operation:    Stable semantic label used by per-turn telemetry.

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
        operation=operation,
    )


def llm_call_with_image(
    prompt: str,
    image_bytes: bytes,
    image_mime: str = 'image/png',
    model: str = 'default',
    expect_json: bool = False,
    max_retries: int = 2,
    fallback: Optional[Any] = None,
    operation: str = 'unspecified',
) -> Any:
    """
    Multimodal call — same contract as llm_call, but with one image attached.
    """
    return _llm_call_internal(
        prompt=prompt,
        model=model,
        expect_json=expect_json,
        max_retries=max_retries,
        fallback=fallback,
        image_bytes=image_bytes,
        image_mime=image_mime,
        operation=operation,
    )


def llm_call_with_media(
    prompt: str,
    media_bytes: bytes,
    mime_type: str = 'image/png',
    model: str = 'default',
    expect_json: bool = False,
    max_retries: int = 2,
    fallback: Optional[Any] = None,
    operation: str = 'unspecified',
) -> Any:
    """
    Multimodal call — accepts any image or audio bytes (e.g. audio/webm, audio/mp3, image/png)
    and processes them using Google GenAI SDK.
    """
    return _llm_call_internal(
        prompt=prompt,
        model=model,
        expect_json=expect_json,
        max_retries=max_retries,
        fallback=fallback,
        image_bytes=media_bytes,
        image_mime=mime_type,
        operation=operation,
    )


def _llm_call_internal(
    prompt: str,
    model: str,
    expect_json: bool,
    max_retries: int,
    fallback: Optional[Any],
    image_bytes: Optional[bytes] = None,
    image_mime: str = 'image/png',
    operation: str = 'unspecified',
) -> Any:
    operation = str(operation or 'unspecified')
    chain, model_route = _resolve_model_chain(model, operation)
    original_prompt_chars = len(prompt)
    prompt, prompt_budget_chars, prompt_truncated = _apply_prompt_budget(
        prompt,
        operation,
    )
    client = _get_client()

    last_error = None
    quota_error = None
    call_t0 = time.perf_counter()
    provider_latency_ms = 0
    provider_attempt_count = 0
    models_attempted = []
    usage_input_tokens = 0
    usage_output_tokens = 0
    usage_total_tokens = 0
    usage_response_count = 0
    response_count = 0
    response_chars_total = 0
    estimated_cost = 0.0
    priced_response_count = 0

    def finish(
        *,
        status: str,
        actual_model: Optional[str] = None,
        fallback_used: bool = False,
        error: Optional[Exception] = None,
    ) -> None:
        token_usage_complete = (
            response_count > 0 and usage_response_count == response_count
        )
        cost_complete = (
            token_usage_complete
            and priced_response_count == usage_response_count
        )
        record_llm_call({
            'operation': operation,
            'semantic_model': model,
            'model_route': model_route,
            'actual_model': actual_model,
            'status': status,
            'fallback_used': fallback_used,
            'latency_ms': max(
                0,
                int((time.perf_counter() - call_t0) * 1000),
            ),
            'provider_latency_ms': provider_latency_ms,
            'attempt_count': provider_attempt_count,
            'retry_count': max(0, provider_attempt_count - 1),
            'models_attempted': list(models_attempted),
            'input_tokens': (
                usage_input_tokens if token_usage_complete else None
            ),
            'output_tokens': (
                usage_output_tokens if token_usage_complete else None
            ),
            'total_tokens': (
                usage_total_tokens if token_usage_complete else None
            ),
            'estimated_cost_usd': (
                round(estimated_cost, 8) if cost_complete else None
            ),
            'chars_in': len(prompt) * provider_attempt_count,
            'chars_out': response_chars_total,
            'prompt_original_chars': original_prompt_chars,
            'prompt_sent_chars': len(prompt),
            'prompt_budget_chars': prompt_budget_chars,
            'prompt_truncated': prompt_truncated,
            'has_media': image_bytes is not None,
            'error_type': type(error).__name__ if error is not None else '',
        })

    for model_id in chain:
        for attempt in range(max_retries + 1):
            attempt_t0 = time.perf_counter()
            attempt_latency_recorded = False
            provider_attempt_count += 1
            models_attempted.append(model_id)
            try:
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
                latency_ms = int((time.perf_counter() - attempt_t0) * 1000)
                provider_latency_ms += latency_ms
                attempt_latency_recorded = True
                response_count += 1
                raw_text = (response.text or '').strip()
                response_chars_total += len(raw_text)
                usage = extract_usage(response)
                if usage['total_tokens'] is not None:
                    usage_response_count += 1
                    usage_input_tokens += int(usage['input_tokens'] or 0)
                    usage_output_tokens += int(usage['output_tokens'] or 0)
                    usage_total_tokens += int(usage['total_tokens'] or 0)
                    response_cost = estimate_cost_usd(
                        model_id,
                        usage['input_tokens'],
                        usage['output_tokens'],
                    )
                    if response_cost is not None:
                        estimated_cost += response_cost
                        priced_response_count += 1

                logger.info(
                    'llm_call ok model=%s latency=%dms chars_in=%d chars_out=%d image=%s',
                    model_id, latency_ms, len(prompt), len(raw_text),
                    'yes' if image_bytes else 'no',
                )

                if not expect_json:
                    finish(
                        status='success',
                        actual_model=model_id,
                    )
                    return raw_text

                parsed = _parse_json(raw_text)
                if parsed is not None:
                    finish(
                        status='success',
                        actual_model=model_id,
                    )
                    return parsed

                # JSON parse failed — count as a retryable error
                last_error = ValueError('LLM returned malformed JSON')
                logger.warning(
                    'llm_call json_parse_failed attempt=%d model=%s preview=%r',
                    attempt, model_id, raw_text[:200],
                )

            except Exception as exc:
                if not attempt_latency_recorded:
                    provider_latency_ms += int(
                        (time.perf_counter() - attempt_t0) * 1000
                    )
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
        finish(
            status='fallback',
            fallback_used=True,
            error=last_error,
        )
        return fallback

    # A quota wall anywhere in the chain is the more actionable diagnosis:
    # surface it as the typed error so views show "try again later" rather
    # than a generic failure.
    if quota_error is not None:
        logger.error('llm_call QUOTA exhausted across chain=%s', ','.join(chain))
        error = LLMQuotaError(
            'AI service quota exceeded.',
            retry_after_seconds=_extract_retry_after(quota_error),
        )
        finish(status='error', error=error)
        raise error

    error = RuntimeError(
        f'LLM call failed on every model in chain ({",".join(chain)}): {last_error}'
    )
    finish(status='error', error=error)
    raise error


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
