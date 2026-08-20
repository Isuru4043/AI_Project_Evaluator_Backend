"""Per-request LLM telemetry for the viva pipeline.

The collector is deliberately in-memory: a pipeline request opens one trace,
all LLM calls append JSON-safe metrics, and the orchestration layer persists a
single summary with the answer/question audit.  No prompts or model responses
are recorded, so student content is not duplicated into logs or audit JSON.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

from django.conf import settings


logger = logging.getLogger(__name__)

_CURRENT_COLLECTOR: ContextVar[Optional["LLMTelemetryCollector"]] = ContextVar(
    "viva_llm_telemetry_collector",
    default=None,
)


def _non_negative_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, number)


def extract_usage(response: Any) -> Dict[str, Optional[int]]:
    """Read token counts from Google GenAI objects or test dictionaries."""
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None and isinstance(response, dict):
        metadata = response.get("usage_metadata")

    def read(*names: str) -> Optional[int]:
        for name in names:
            value = (
                metadata.get(name)
                if isinstance(metadata, dict)
                else getattr(metadata, name, None)
            )
            normalized = _non_negative_int(value)
            if normalized is not None:
                return normalized
        return None

    input_tokens = read("prompt_token_count", "input_token_count")
    output_tokens = read("candidates_token_count", "output_token_count")
    total_tokens = read("total_token_count")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def estimate_cost_usd(
    model_id: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[float]:
    """Estimate cost from deployment-configured prices per million tokens.

    Prices are intentionally configuration, not source-code constants, because
    provider pricing changes.  Supported keys are ``input``/``output`` or
    ``input_per_million``/``output_per_million``.
    """
    if input_tokens is None or output_tokens is None:
        return None
    pricing = getattr(settings, "LLM_MODEL_PRICING", {}) or {}
    model_pricing = pricing.get(model_id)
    if not isinstance(model_pricing, dict):
        return None
    try:
        input_rate = float(
            model_pricing.get(
                "input_per_million",
                model_pricing.get("input"),
            )
        )
        output_rate = float(
            model_pricing.get(
                "output_per_million",
                model_pricing.get("output"),
            )
        )
    except (TypeError, ValueError):
        return None
    estimated = (
        input_tokens * input_rate + output_tokens * output_rate
    ) / 1_000_000
    return round(estimated, 8)


@dataclass
class LLMTelemetryCollector:
    trace_id: str
    trace_kind: str
    session_id: str = ""
    question_id: str = ""
    started_monotonic: float = field(default_factory=time.perf_counter)
    calls: list[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def record(self, call: Dict[str, Any]) -> None:
        with self._lock:
            self.calls.append(dict(call))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            calls = [dict(call) for call in self.calls]

        known_input = [
            call["input_tokens"]
            for call in calls
            if call.get("input_tokens") is not None
        ]
        known_output = [
            call["output_tokens"]
            for call in calls
            if call.get("output_tokens") is not None
        ]
        known_total = [
            call["total_tokens"]
            for call in calls
            if call.get("total_tokens") is not None
        ]
        known_costs = [
            call["estimated_cost_usd"]
            for call in calls
            if call.get("estimated_cost_usd") is not None
        ]
        summary = {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "trace_kind": self.trace_kind,
            "session_id": self.session_id,
            "question_id": self.question_id,
            "duration_ms": max(
                0,
                int((time.perf_counter() - self.started_monotonic) * 1000),
            ),
            "call_count": len(calls),
            "successful_call_count": sum(
                call.get("status") == "success" for call in calls
            ),
            "fallback_call_count": sum(
                call.get("fallback_used") is True for call in calls
            ),
            "failed_call_count": sum(
                call.get("status") == "error" for call in calls
            ),
            "provider_attempt_count": sum(
                int(call.get("attempt_count", 0)) for call in calls
            ),
            "retry_count": sum(
                int(call.get("retry_count", 0)) for call in calls
            ),
            "provider_latency_ms": sum(
                int(call.get("provider_latency_ms", 0)) for call in calls
            ),
            "input_characters": sum(
                int(call.get("chars_in", 0)) for call in calls
            ),
            "prompt_original_characters": sum(
                int(call.get("prompt_original_chars", 0)) for call in calls
            ),
            "prompt_sent_characters": sum(
                int(call.get("prompt_sent_chars", 0)) for call in calls
            ),
            "prompt_truncated_call_count": sum(
                call.get("prompt_truncated") is True for call in calls
            ),
            "output_characters": sum(
                int(call.get("chars_out", 0)) for call in calls
            ),
            "input_tokens": sum(known_input),
            "output_tokens": sum(known_output),
            "total_tokens": sum(known_total),
            "token_usage_call_count": len(known_total),
            "token_usage_complete": len(known_total) == len(calls),
            "estimated_cost_usd": round(sum(known_costs), 8),
            "costed_call_count": len(known_costs),
            "cost_estimate_complete": len(known_costs) == len(calls),
            "calls": calls,
        }
        logger.info(
            "llm_trace trace_id=%s kind=%s calls=%d attempts=%d retries=%d "
            "tokens=%d estimated_cost_usd=%.8f token_complete=%s cost_complete=%s",
            self.trace_id,
            self.trace_kind,
            summary["call_count"],
            summary["provider_attempt_count"],
            summary["retry_count"],
            summary["total_tokens"],
            summary["estimated_cost_usd"],
            summary["token_usage_complete"],
            summary["cost_estimate_complete"],
        )
        return summary


@contextmanager
def collect_llm_telemetry(
    *,
    trace_kind: str,
    session_id: str = "",
    question_id: str = "",
) -> Iterator[LLMTelemetryCollector]:
    collector = LLMTelemetryCollector(
        trace_id=str(uuid.uuid4()),
        trace_kind=str(trace_kind),
        session_id=str(session_id or ""),
        question_id=str(question_id or ""),
    )
    token = _CURRENT_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _CURRENT_COLLECTOR.reset(token)


def record_llm_call(call: Dict[str, Any]) -> None:
    collector = _CURRENT_COLLECTOR.get()
    if collector is not None:
        collector.record(call)


def submit_with_telemetry_context(executor, function, /, *args, **kwargs):
    """Submit work while retaining the request's telemetry ContextVar."""
    if _CURRENT_COLLECTOR.get() is None:
        return executor.submit(function, *args, **kwargs)
    context = copy_context()
    return executor.submit(context.run, function, *args, **kwargs)
