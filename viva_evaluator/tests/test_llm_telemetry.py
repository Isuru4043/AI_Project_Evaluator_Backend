import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")

from django.test import override_settings

from viva_evaluator.services.llm_service import llm_call
from viva_evaluator.services.evaluation.metrics import (
    compute_llm_telemetry_metrics,
    compute_tts_metrics,
)
from viva_evaluator.management.commands.question_validation_report import (
    _evaluate_gates,
)
from viva_evaluator.services.llm_telemetry import (
    collect_llm_telemetry,
    submit_with_telemetry_context,
)


class _FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class LLMTelemetryTests(TestCase):
    def test_operation_route_uses_fast_chain_for_question_repair(self):
        models = _FakeModels(response=SimpleNamespace(text="repaired", usage_metadata=None))
        client = SimpleNamespace(models=models)

        with (
            patch(
                "viva_evaluator.services.llm_service._get_client",
                return_value=client,
            ),
            patch.dict(
                "viva_evaluator.services.llm_service.MODEL_REGISTRY",
                {
                    "default": ["default-model"],
                    "reasoning": ["reasoning-model"],
                    "fast": ["fast-repair-model"],
                },
            ),
        ):
            result = llm_call(
                "repair this question",
                model="reasoning",
                operation="question_repair",
            )

        self.assertEqual(result, "repaired")
        self.assertEqual(models.calls[0]["model"], "fast-repair-model")

    def test_prompt_budget_preserves_instruction_head_and_schema_tail(self):
        models = _FakeModels(response=SimpleNamespace(text="ok", usage_metadata=None))
        client = SimpleNamespace(models=models)
        prompt = "HEAD-INSTRUCTIONS\n" + ("x" * 400) + "\nJSON-SCHEMA-TAIL"

        with (
            patch(
                "viva_evaluator.services.llm_service._get_client",
                return_value=client,
            ),
            patch.dict(
                "viva_evaluator.services.llm_service.MODEL_REGISTRY",
                {"default": ["test-model"], "fast": ["test-model"]},
            ),
            patch.dict(
                "viva_evaluator.services.llm_service.DEFAULT_PROMPT_BUDGETS",
                {"question_critic": 160},
            ),
            patch.dict(
                os.environ,
                {"LLM_PROMPT_MAX_CHARS_QUESTION_CRITIC": "160"},
            ),
            collect_llm_telemetry(trace_kind="question") as telemetry,
        ):
            llm_call(prompt, model="fast", operation="question_critic")
            summary = telemetry.snapshot()

        sent = models.calls[0]["contents"]
        self.assertLessEqual(len(sent), 160)
        self.assertTrue(sent.startswith("HEAD-INSTRUCTIONS"))
        self.assertTrue(sent.endswith("JSON-SCHEMA-TAIL"))
        self.assertTrue(summary["calls"][0]["prompt_truncated"])
        self.assertEqual(summary["calls"][0]["prompt_original_chars"], len(prompt))

    @override_settings(
        LLM_MODEL_PRICING={"test-model": {"input": 1.0, "output": 2.0}}
    )
    def test_llm_call_records_model_tokens_latency_and_cost(self):
        response = SimpleNamespace(
            text='{"ok": true}',
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=20,
                total_token_count=120,
            ),
        )
        client = SimpleNamespace(models=_FakeModels(response=response))

        with (
            patch(
                "viva_evaluator.services.llm_service._get_client",
                return_value=client,
            ),
            patch.dict(
                "viva_evaluator.services.llm_service.MODEL_REGISTRY",
                {"fast": ["test-model"]},
            ),
            collect_llm_telemetry(
                trace_kind="answer_turn",
                session_id="session-1",
            ) as telemetry,
        ):
            result = llm_call(
                "prompt",
                model="fast",
                expect_json=True,
                operation="response_triage",
            )
            summary = telemetry.snapshot()

        self.assertEqual(result, {"ok": True})
        self.assertEqual(summary["call_count"], 1)
        self.assertEqual(summary["provider_attempt_count"], 1)
        self.assertEqual(summary["total_tokens"], 120)
        self.assertTrue(summary["token_usage_complete"])
        self.assertTrue(summary["cost_estimate_complete"])
        self.assertEqual(summary["estimated_cost_usd"], 0.00014)
        call = summary["calls"][0]
        self.assertEqual(call["operation"], "response_triage")
        self.assertEqual(call["actual_model"], "test-model")
        self.assertEqual(call["semantic_model"], "fast")

    def test_fallback_and_failed_provider_attempt_are_visible(self):
        client = SimpleNamespace(models=_FakeModels(error=RuntimeError("down")))

        with (
            patch(
                "viva_evaluator.services.llm_service._get_client",
                return_value=client,
            ),
            patch.dict(
                "viva_evaluator.services.llm_service.MODEL_REGISTRY",
                {"fast": ["test-model"]},
            ),
            collect_llm_telemetry(trace_kind="answer_turn") as telemetry,
        ):
            result = llm_call(
                "prompt",
                model="fast",
                max_retries=0,
                fallback={"safe": True},
                operation="response_triage",
            )
            summary = telemetry.snapshot()

        self.assertEqual(result, {"safe": True})
        self.assertEqual(summary["fallback_call_count"], 1)
        self.assertEqual(summary["provider_attempt_count"], 1)
        self.assertEqual(summary["calls"][0]["status"], "fallback")
        self.assertEqual(summary["calls"][0]["error_type"], "RuntimeError")

    def test_parallel_worker_call_joins_the_parent_trace(self):
        response = SimpleNamespace(
            text="ok",
            usage_metadata=SimpleNamespace(
                prompt_token_count=3,
                candidates_token_count=1,
                total_token_count=4,
            ),
        )
        client = SimpleNamespace(models=_FakeModels(response=response))

        with (
            patch(
                "viva_evaluator.services.llm_service._get_client",
                return_value=client,
            ),
            patch.dict(
                "viva_evaluator.services.llm_service.MODEL_REGISTRY",
                {"fast": ["test-model"]},
            ),
            collect_llm_telemetry(trace_kind="answer_turn") as telemetry,
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            future = submit_with_telemetry_context(
                executor,
                llm_call,
                "prompt",
                model="fast",
                operation="parallel_check",
            )
            self.assertEqual(future.result(), "ok")
            summary = telemetry.snapshot()

        self.assertEqual(summary["call_count"], 1)
        self.assertEqual(summary["calls"][0]["operation"], "parallel_check")

    def test_persisted_trace_metrics_deduplicate_and_compute_percentiles(self):
        first = {
            "trace_id": "trace-1",
            "duration_ms": 100,
            "call_count": 2,
            "retry_count": 1,
            "fallback_call_count": 0,
            "input_characters": 50,
            "output_characters": 10,
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
            "token_usage_call_count": 2,
            "costed_call_count": 2,
            "estimated_cost_usd": 0.01,
            "calls": [
                {"operation": "answer_analysis", "total_tokens": 20},
                {"operation": "response_triage", "total_tokens": 5},
            ],
        }
        second = {
            "trace_id": "trace-2",
            "duration_ms": 900,
            "call_count": 1,
            "retry_count": 0,
            "fallback_call_count": 1,
            "input_characters": 30,
            "output_characters": 5,
            "input_tokens": 8,
            "output_tokens": 2,
            "total_tokens": 10,
            "token_usage_call_count": 1,
            "costed_call_count": 0,
            "estimated_cost_usd": 0,
            "calls": [
                {"operation": "question_generation", "total_tokens": 10},
            ],
        }

        metrics = compute_llm_telemetry_metrics([first, first, second])

        self.assertEqual(metrics["turn_count"], 2)
        self.assertEqual(metrics["llm_call_count"], 3)
        self.assertEqual(metrics["p50_turn_latency_ms"], 100.0)
        self.assertEqual(metrics["p95_turn_latency_ms"], 900.0)
        self.assertEqual(metrics["total_tokens"], 35)
        self.assertEqual(metrics["token_usage_coverage"], 1.0)
        self.assertEqual(metrics["cost_estimate_coverage"], 0.6667)
        self.assertEqual(
            metrics["operation_call_distribution"]["answer_analysis"],
            1,
        )

    def test_tts_metrics_report_readiness_cache_waste_latency_and_cost(self):
        metrics = compute_tts_metrics(
            [
                {
                    "tts": {
                        "enabled": True,
                        "status": "ready",
                        "cache_hit": False,
                        "characters": 100,
                        "generation_latency_ms": 220,
                        "speculative_wasted": False,
                    }
                },
                {
                    "tts": {
                        "enabled": True,
                        "status": "ready",
                        "cache_hit": True,
                        "characters": 100,
                        "generation_latency_ms": 20,
                        "speculative_wasted": True,
                    }
                },
                {"tts": {"enabled": True, "status": "failed"}},
            ],
            price_per_1000_characters_usd=0.2,
        )

        self.assertEqual(metrics["enabled_question_count"], 3)
        self.assertEqual(metrics["ready_rate"], 0.6667)
        self.assertEqual(metrics["cache_hit_rate"], 0.5)
        self.assertEqual(metrics["speculative_waste_count"], 1)
        self.assertEqual(metrics["generated_characters"], 100)
        self.assertEqual(metrics["estimated_cost_usd"], 0.02)
        self.assertEqual(metrics["p95_generation_latency_ms"], 220.0)

    def test_performance_gates_cover_latency_cost_calls_and_quality(self):
        options = {
            "min_turns": 20,
            "max_p95_latency_ms": 10_000,
            "max_mean_calls_per_turn": 4,
            "max_degraded_rate": 0.1,
            "max_fallback_rate": 0.05,
            "min_tier1_pass_rate": 0.95,
            "max_mean_cost_per_turn_usd": 0.01,
        }
        gates = _evaluate_gates(
            {
                "degraded_validation_rate": 0.02,
                "fallback_rate": 0.01,
                "tier1_pass_rate": 0.98,
            },
            {
                "turn_count": 30,
                "p95_turn_latency_ms": 9_000,
                "mean_calls_per_turn": 3.2,
                "mean_estimated_cost_per_turn_usd": 0.005,
            },
            options,
        )
        self.assertEqual(gates["overall_status"], "passed")

        options["max_p95_latency_ms"] = 8_000
        failed = _evaluate_gates(
            {"tier1_pass_rate": 0.98},
            {
                "turn_count": 30,
                "p95_turn_latency_ms": 9_000,
            },
            options,
        )
        self.assertEqual(failed["overall_status"], "failed")
        self.assertEqual(
            failed["checks"]["p95_turn_latency_ms"]["status"],
            "failed",
        )
