import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from viva_evaluator.services.model_benchmark.contracts import (
    BenchmarkCase,
    BenchmarkImage,
    ModelSpec,
)
from viva_evaluator.services.model_benchmark.providers import (
    AnthropicCompatibleAdapter,
    GeminiDeveloperAdapter,
    OpenAICompatibleAdapter,
    ProviderError,
    VertexGeminiAdapter,
)
from viva_evaluator.services.model_benchmark.parsing import parse_json_response
from viva_evaluator.services.model_benchmark.registry import (
    BenchmarkConfigurationError,
    load_cases,
)
from viva_evaluator.services.model_benchmark.runner import (
    BenchmarkRunner,
    BudgetLedger,
    ResponseValidationError,
    calculate_list_price,
    validate_response,
)
from viva_evaluator.services.model_benchmark.scoring import (
    score_code_understanding,
    score_rubric_structure,
    score_visual_understanding,
    score_knowledge_preparation,
    score_answer_assessment,
    score_question_quality,
    score_session_reporting,
)


class _Response:
    def __init__(self, body, status_code=200, headers=None):
        self._body = body
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(body)

    def json(self):
        return self._body


def _spec(provider="groq", **overrides):
    values = {
        "id": "model-a",
        "provider": provider,
        "model": "provider-model",
        "api_key_env": "BENCHMARK_TEST_KEY",
        "endpoint": "https://example.test/chat",
        "billing_mode": "free_quota",
        "input_price_per_million_usd": 1.0,
        "output_price_per_million_usd": 2.0,
    }
    values.update(overrides)
    return ModelSpec(**values)


class ProviderAdapterTests(TestCase):
    def test_openai_compatible_parses_usage(self):
        session = Mock()
        session.post.return_value = _Response({
            "id": "req-1",
            "model": "exact-model-v1",
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        })
        adapter = OpenAICompatibleAdapter(session=session, max_retries=0)
        call = adapter.invoke(_spec(), BenchmarkCase("c1", "test", "hello"), "secret")
        self.assertEqual(call.response.text, "answer")
        self.assertEqual(call.response.total_tokens, 13)
        self.assertEqual(call.response.visible_output_tokens, 3)
        self.assertEqual(call.response.provider_model, "exact-model-v1")

    def test_openai_compatible_sends_image_as_data_url(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "diagram.png"
            image_path.write_bytes(b"png-bytes")
            session = Mock()
            session.post.return_value = _Response({
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            })
            adapter = OpenAICompatibleAdapter(session=session, max_retries=0)
            adapter.invoke(
                _spec(capabilities=("text", "vision")),
                BenchmarkCase(
                    "c1",
                    "visual_understanding",
                    "inspect",
                    images=(BenchmarkImage(str(image_path)),),
                    required_capabilities=("text", "vision"),
                ),
                "secret",
            )
            content = session.post.call_args.kwargs["json"]["messages"][-1]["content"]
            self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertEqual(content[-1], {"type": "text", "text": "inspect"})

    def test_anthropic_parser_selects_text_blocks(self):
        session = Mock()
        session.post.return_value = _Response({
            "id": "req-2",
            "model": "claude-version",
            "content": [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "visible"},
            ],
            "usage": {"input_tokens": 7, "output_tokens": 4},
            "stop_reason": "end_turn",
        })
        adapter = AnthropicCompatibleAdapter(session=session, max_retries=0)
        call = adapter.invoke(
            _spec(provider="anthropic_compatible"),
            BenchmarkCase("c1", "test", "hello"),
            "secret",
        )
        self.assertEqual(call.response.text, "visible")
        self.assertEqual(call.response.total_tokens, 11)

    def test_anthropic_sends_base64_image_block(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "diagram.png"
            image_path.write_bytes(b"png-bytes")
            session = Mock()
            session.post.return_value = _Response({
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            })
            adapter = AnthropicCompatibleAdapter(session=session, max_retries=0)
            adapter.invoke(
                _spec(provider="anthropic_compatible", capabilities=("text", "vision")),
                BenchmarkCase(
                    "c1",
                    "visual_understanding",
                    "inspect",
                    images=(BenchmarkImage(str(image_path)),),
                    required_capabilities=("text", "vision"),
                ),
                "secret",
            )
            content = session.post.call_args.kwargs["json"]["messages"][0]["content"]
            self.assertEqual(content[0]["type"], "image")
            self.assertEqual(content[0]["source"]["media_type"], "image/png")
            self.assertEqual(content[-1], {"type": "text", "text": "inspect"})

    def test_gemini_parser_reads_model_version(self):
        session = Mock()
        session.post.return_value = _Response({
            "modelVersion": "gemini-exact-001",
            "candidates": [{
                "content": {"parts": [{"text": "ok"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7,
            },
        })
        adapter = GeminiDeveloperAdapter(session=session, max_retries=0)
        call = adapter.invoke(
            _spec(provider="gemini", endpoint="https://example.test/{model}"),
            BenchmarkCase("c1", "test", "hello"),
            "secret",
        )
        self.assertEqual(call.response.provider_model, "gemini-exact-001")
        self.assertEqual(call.response.total_tokens, 7)

    @patch("AI_Evaluator_Backend.llm.get_llm")
    def test_vertex_gemini_uses_production_adc_client(self, get_llm):
        response = SimpleNamespace(
            text='{"status":"ok"}',
            model_version="gemini-exact-vertex-001",
            response_id="vertex-request",
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=SimpleNamespace(
                prompt_token_count=8,
                candidates_token_count=4,
                total_token_count=14,
                thoughts_token_count=2,
                cached_content_token_count=None,
            ),
        )
        generate_content = Mock(return_value=response)
        get_llm.return_value = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )
        adapter = VertexGeminiAdapter(max_retries=0)
        call = adapter.invoke(
            _spec(provider="vertex_gemini"),
            BenchmarkCase("c1", "test", "hello", system_prompt="system"),
            "project-id",
        )
        self.assertEqual(call.response.provider_model, "gemini-exact-vertex-001")
        self.assertEqual(call.response.total_tokens, 14)
        self.assertEqual(call.response.output_tokens, 6)
        self.assertEqual(call.response.visible_output_tokens, 4)
        self.assertEqual(call.response.reasoning_tokens, 2)
        self.assertEqual(call.response.raw_usage["thoughts_token_count"], 2)
        self.assertIn("system", generate_content.call_args.kwargs["contents"])

    @patch("AI_Evaluator_Backend.llm.get_llm")
    def test_vertex_gemini_sends_image_part_before_prompt(self, get_llm):
        response = SimpleNamespace(
            text='{"status":"ok"}',
            model_version="gemini-vision",
            response_id="vertex-request",
            candidates=[],
            usage_metadata=None,
        )
        generate_content = Mock(return_value=response)
        get_llm.return_value = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "diagram.png"
            image_path.write_bytes(b"png-bytes")
            VertexGeminiAdapter(max_retries=0).invoke(
                _spec(provider="vertex_gemini", capabilities=("text", "vision")),
                BenchmarkCase(
                    "c1",
                    "visual_understanding",
                    "inspect",
                    system_prompt="system",
                    images=(BenchmarkImage(str(image_path)),),
                    required_capabilities=("text", "vision"),
                ),
                "project-id",
            )
        contents = generate_content.call_args.kwargs["contents"]
        self.assertEqual(len(contents), 2)
        self.assertEqual(contents[-1], "system\n\ninspect")

    def test_failed_retry_reports_attempt_count(self):
        session = Mock()
        session.post.return_value = _Response(
            {"error": "busy"},
            status_code=503,
        )
        adapter = OpenAICompatibleAdapter(
            session=session,
            max_retries=2,
            sleeper=lambda _: None,
        )
        with self.assertRaises(ProviderError) as raised:
            adapter.invoke(
                _spec(),
                BenchmarkCase("c1", "test", "hello"),
                "secret",
            )
        self.assertEqual(raised.exception.attempt_count, 3)
        self.assertGreaterEqual(raised.exception.latency_ms, 0)


class BenchmarkRunnerTests(TestCase):
    def test_calculates_list_price(self):
        self.assertEqual(calculate_list_price(_spec(), 1000, 500), 0.002)

    def test_empty_or_malformed_expected_json_is_not_valid(self):
        case = BenchmarkCase(
            "c1",
            "test",
            "return json",
            expected_response_format="json",
        )
        with self.assertRaises(ResponseValidationError):
            validate_response(case, "")
        with self.assertRaises(ResponseValidationError):
            validate_response(case, "not json")
        self.assertTrue(validate_response(case, '{"ok": true}'))
        self.assertFalse(validate_response(case, '```json\n{"ok": true}\n```'))

    def test_recovers_json_surrounded_by_explanation(self):
        parsed, strict = parse_json_response(
            '```json\n{"ok": true}\n```\nExplanation follows.'
        )
        self.assertEqual(parsed, {"ok": True})
        self.assertFalse(strict)

    def test_does_not_recover_nested_object_from_truncated_json(self):
        value, strict = parse_json_response(
            '{"case_id":"c1","answers":[{"question_id":"q1","answer":true'
        )
        self.assertIsNone(value)
        self.assertFalse(strict)

    def test_dataset_can_load_prompt_from_sibling_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompt.txt").write_text("frozen prompt", encoding="utf-8")
            (root / "cases.jsonl").write_text(json.dumps({
                "case_id": "c1",
                "category": "rubric_understanding",
                "prompt_file": "prompt.txt",
            }) + "\n", encoding="utf-8")
            self.assertEqual(load_cases(root / "cases.jsonl")[0].prompt, "frozen prompt")

    def test_dataset_rejects_prompt_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cases.jsonl").write_text(json.dumps({
                "case_id": "c1",
                "category": "rubric_understanding",
                "prompt_file": "../outside.txt",
            }) + "\n", encoding="utf-8")
            with self.assertRaises(BenchmarkConfigurationError):
                load_cases(root / "cases.jsonl")

    def test_dataset_loads_image_and_required_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "diagram.png").write_bytes(b"image")
            (root / "cases.jsonl").write_text(json.dumps({
                "case_id": "c1",
                "category": "visual_understanding",
                "prompt": "inspect",
                "image_files": ["diagram.png"],
                "required_capabilities": ["text", "vision"],
            }) + "\n", encoding="utf-8")
            case = load_cases(root / "cases.jsonl")[0]
            self.assertEqual(Path(case.images[0].path), (root / "diagram.png").resolve())
            self.assertEqual(case.required_capabilities, ("text", "vision"))

    def test_dataset_rejects_image_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cases.jsonl").write_text(json.dumps({
                "case_id": "c1",
                "category": "visual_understanding",
                "prompt": "inspect",
                "image_files": ["../outside.png"],
            }) + "\n", encoding="utf-8")
            with self.assertRaises(BenchmarkConfigurationError):
                load_cases(root / "cases.jsonl")

    def test_paid_model_requires_explicit_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = BenchmarkRunner(output_path=Path(directory) / "results.jsonl")
            with self.assertRaisesRegex(RuntimeError, "marked paid"):
                runner.run(
                    models=[_spec(billing_mode="paid")],
                    cases=[BenchmarkCase("c1", "test", "hello")],
                    allow_paid=False,
                )

    def test_resume_skips_successful_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            output.write_text(json.dumps({
                "schema_version": 2,
                "status": "success",
                "model_id": "model-a",
                "case_id": "c1",
            }) + "\n", encoding="utf-8")
            runner = BenchmarkRunner(output_path=output)
            results = runner.run(
                models=[_spec()],
                cases=[BenchmarkCase("c1", "test", "hello")],
            )
            self.assertEqual(results, [])

    def test_runner_skips_model_without_required_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = BenchmarkRunner(output_path=Path(directory) / "results.jsonl")
            results = runner.run(
                models=[_spec(capabilities=("text",))],
                cases=[BenchmarkCase(
                    "c1",
                    "visual_understanding",
                    "inspect",
                    required_capabilities=("text", "vision"),
                )],
            )
            self.assertEqual(results, [])

    def test_resume_restores_paid_cap_debits(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            output.write_text(json.dumps({
                "status": "success",
                "model_id": "paid-a",
                "case_id": "old",
                "paid_cap_debit_usd": 2.25,
            }) + "\n", encoding="utf-8")
            ledger = BudgetLedger(paid_cap_usd=25)
            BenchmarkRunner(output_path=output, budget=ledger)
            self.assertEqual(ledger.paid_spend_usd, 2.25)

    @patch.dict(os.environ, {"BENCHMARK_TEST_KEY": "secret"})
    def test_result_never_contains_api_key(self):
        response = SimpleNamespace(
            text="ok",
            input_tokens=5,
            output_tokens=2,
            total_tokens=7,
            provider_model="exact",
            request_id="req",
            finish_reason="stop",
            raw_usage={},
        )
        adapter = SimpleNamespace(invoke=lambda *args: SimpleNamespace(
            response=response, attempt_count=1, latency_ms=10
        ))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            runner = BenchmarkRunner(
                output_path=output,
                adapter_factory=lambda *args, **kwargs: adapter,
                budget=BudgetLedger(),
            )
            runner.run(
                models=[_spec()],
                cases=[BenchmarkCase("c1", "test", "hello")],
            )
            stored = output.read_text(encoding="utf-8")
            self.assertNotIn("secret", stored)
            self.assertIn('"status": "success"', stored)


class RubricScoringTests(TestCase):
    def test_perfect_structure_scores_100(self):
        sections = [{
            "name": "Viva",
            "marks": 15,
            "criterion_weight_total_percent": 100,
            "criteria": [{
                "name": "Understanding",
                "weight_percent": 100,
                "descriptors": {
                    "weak": ["w"],
                    "satisfactory": ["s"],
                    "good": ["g"],
                    "excellent": ["e"],
                },
            }],
        }]
        bands = [
            {"name": "Weak", "minimum": 0, "maximum": 39, "source_grade_label": "E"},
        ]
        response = {
            "sections": sections,
            "performance_bands": bands,
            "represented_marks_total": 15,
        }
        gold = {"objective_checks": {
            "sections": [{
                "name": "Viva",
                "marks": 15,
                "criterion_weight_total_percent": 100,
                "criteria": [{"name": "Understanding", "weight_percent": 100}],
            }],
            "performance_bands": bands,
            "represented_marks_total": 15,
        }}
        metrics = score_rubric_structure(json.dumps(response), gold)
        self.assertEqual(metrics["overall_score"], 100.0)

    def test_hallucinated_criterion_reduces_precision(self):
        response = {
            "sections": [{
                "name": "Viva",
                "marks": 15,
                "criterion_weight_total_percent": 100,
                "criteria": [
                    {"name": "Understanding", "weight_percent": 100, "descriptors": {}},
                    {"name": "Invented", "weight_percent": 0, "descriptors": {}},
                ],
            }],
            "performance_bands": [],
            "represented_marks_total": 15,
        }
        gold = {"objective_checks": {
            "sections": [{
                "name": "Viva",
                "marks": 15,
                "criterion_weight_total_percent": 100,
                "criteria": [{"name": "Understanding", "weight_percent": 100}],
            }],
            "performance_bands": [],
            "represented_marks_total": 15,
        }}
        metrics = score_rubric_structure(json.dumps(response), gold)
        self.assertEqual(metrics["hallucinated_criterion_count"], 1)
        self.assertLess(metrics["criterion_precision"], 1.0)


class CodeUnderstandingScoringTests(TestCase):
    def setUp(self):
        self.gold = {
            "case_id": "code_test",
            "questions": [
                {
                    "question_id": "q1",
                    "answer": 3,
                    "answer_mode": "number",
                    "evidence": [
                        {"source_id": "B1", "line_start": 10, "line_end": 12}
                    ],
                },
                {
                    "question_id": "q2",
                    "answer": ["alpha", "beta"],
                    "answer_mode": "set",
                    "evidence": [
                        {"source_id": "F1", "line_start": 20, "line_end": 24}
                    ],
                },
            ],
        }

    def test_perfect_answers_and_evidence_score_100(self):
        response = {
            "case_id": "code_test",
            "answers": [
                {
                    "question_id": "q1",
                    "answer": 3,
                    "evidence": [
                        {"source_id": "B1", "line_start": 11, "line_end": 11}
                    ],
                },
                {
                    "question_id": "q2",
                    "answer": ["beta", "alpha"],
                    "evidence": [
                        {"source_id": "F1", "line_start": 21, "line_end": 23}
                    ],
                },
            ],
        }
        metrics = score_code_understanding(json.dumps(response), self.gold)
        self.assertEqual(metrics["overall_score"], 100.0)
        self.assertEqual(metrics["answer_accuracy"], 1.0)
        self.assertTrue(metrics["strict_json_compliance"])

    def test_wrong_answer_is_separate_from_evidence_accuracy(self):
        response = {
            "case_id": "code_test",
            "answers": [
                {
                    "question_id": "q1",
                    "answer": 4,
                    "evidence": [
                        {"source_id": "B1", "line_start": 10, "line_end": 12}
                    ],
                },
                {
                    "question_id": "q2",
                    "answer": ["alpha", "beta"],
                    "evidence": [
                        {"source_id": "F1", "line_start": 20, "line_end": 24}
                    ],
                },
            ],
        }
        metrics = score_code_understanding(json.dumps(response), self.gold)
        self.assertEqual(metrics["answer_accuracy"], 0.5)
        self.assertEqual(metrics["evidence_f1"], 1.0)
        self.assertEqual(metrics["overall_score"], 60.0)

    def test_reviewed_semantic_alias_is_accepted(self):
        self.gold["questions"][0]["answer_mode"] = "exact"
        self.gold["questions"][0]["accepted_answers"] = ["three"]
        response = {
            "case_id": "code_test",
            "answers": [
                {
                    "question_id": "q1",
                    "answer": "three",
                    "evidence": [
                        {"source_id": "B1", "line_start": 10, "line_end": 12}
                    ],
                },
                {
                    "question_id": "q2",
                    "answer": ["alpha", "beta"],
                    "evidence": [
                        {"source_id": "F1", "line_start": 20, "line_end": 24}
                    ],
                },
            ],
        }
        metrics = score_code_understanding(json.dumps(response), self.gold)
        self.assertEqual(metrics["answer_accuracy"], 1.0)

    def test_fenced_json_is_recoverable_but_not_strict(self):
        response = {
            "case_id": "code_test",
            "answers": [
                {"question_id": "q1", "answer": 3, "evidence": []},
                {"question_id": "q2", "answer": ["alpha", "beta"], "evidence": []},
            ],
        }
        metrics = score_code_understanding(
            f"```json\n{json.dumps(response)}\n```", self.gold
        )
        self.assertTrue(metrics["valid_json"])
        self.assertFalse(metrics["strict_json_compliance"])
        self.assertEqual(metrics["answer_accuracy"], 1.0)
        self.assertEqual(metrics["overall_score"], 80.0)

    def test_extra_answer_and_wrong_case_id_are_penalized(self):
        response = {
            "case_id": "wrong",
            "answers": [
                {
                    "question_id": "q1",
                    "answer": 3,
                    "evidence": [
                        {"source_id": "B1", "line_start": 10, "line_end": 12}
                    ],
                },
                {
                    "question_id": "q2",
                    "answer": ["alpha", "beta"],
                    "evidence": [
                        {"source_id": "F1", "line_start": 20, "line_end": 24}
                    ],
                },
                {"question_id": "invented", "answer": True, "evidence": []},
            ],
        }
        metrics = score_code_understanding(json.dumps(response), self.gold)
        self.assertEqual(metrics["unsupported_answer_count"], 1)
        self.assertFalse(metrics["case_id_correct"])
        self.assertEqual(metrics["overall_score"], 93.0)


class VisualUnderstandingScoringTests(TestCase):
    def setUp(self):
        self.gold = {
            "case_id": "visual_test",
            "visible_labels": [
                "User Query", "Dense Search", "Lexical Search", "k0=60"
            ],
            "questions": [
                {
                    "question_id": "q1",
                    "answer": ["Dense Search", "Lexical Search"],
                    "answer_mode": "set",
                    "evidence_labels": ["User Query", "Dense Search", "Lexical Search"],
                },
                {
                    "question_id": "q2",
                    "answer": 60,
                    "answer_mode": "number",
                    "evidence_labels": ["k0=60"],
                },
            ],
        }

    def test_perfect_visual_answers_score_100(self):
        response = {
            "case_id": "visual_test",
            "answers": [
                {
                    "question_id": "q1",
                    "answer": ["Lexical Search", "Dense Search"],
                    "evidence_labels": ["User Query", "Dense Search", "Lexical Search"],
                },
                {"question_id": "q2", "answer": 60, "evidence_labels": ["k0=60"]},
            ],
        }
        metrics = score_visual_understanding(json.dumps(response), self.gold)
        self.assertEqual(metrics["overall_score"], 100.0)
        self.assertEqual(metrics["answer_accuracy"], 1.0)
        self.assertEqual(metrics["evidence_label_f1"], 1.0)

    def test_hallucinated_label_reduces_grounding_precision(self):
        response = {
            "case_id": "visual_test",
            "answers": [
                {
                    "question_id": "q1",
                    "answer": ["Dense Search", "Lexical Search"],
                    "evidence_labels": [
                        "User Query", "Dense Search", "Lexical Search", "Invented Node"
                    ],
                },
                {"question_id": "q2", "answer": 60, "evidence_labels": ["k0=60"]},
            ],
        }
        metrics = score_visual_understanding(json.dumps(response), self.gold)
        self.assertEqual(metrics["answer_accuracy"], 1.0)
        self.assertLess(metrics["evidence_label_precision"], 1.0)
        self.assertLess(metrics["overall_score"], 100.0)


class KnowledgePreparationScoringTests(TestCase):
    def setUp(self):
        self.gold = {
            "case_id": "knowledge_test",
            "fact_ids": ["C1", "C2"],
            "alternative_ids": ["C3"],
            "limitation_ids": ["C4"],
            "reject_ids": ["C5", "C6"],
            "citation_map": {
                "C1": ["S1"],
                "C2": ["S1", "S2"],
                "C3": ["S2"],
                "C4": ["S2"],
            },
        }

    def test_perfect_knowledge_brief_scores_100(self):
        response = {
            "case_id": "knowledge_test",
            "fact_ids": ["C1", "C2"],
            "alternative_ids": ["C3"],
            "limitation_ids": ["C4"],
            "reject_ids": ["C5", "C6"],
            "citation_map": self.gold["citation_map"],
            "brief_claim_ids": ["C1", "C2", "C3", "C4"],
            "brief": "A concise evidence-backed technical brief.",
        }
        metrics = score_knowledge_preparation(json.dumps(response), self.gold)
        self.assertEqual(metrics["overall_score"], 100.0)
        self.assertEqual(metrics["claim_classification_accuracy"], 1.0)
        self.assertEqual(metrics["citation_f1"], 1.0)

    def test_unsupported_brief_claim_and_wrong_class_reduce_score(self):
        response = {
            "case_id": "knowledge_test",
            "fact_ids": ["C1", "C2", "C5"],
            "alternative_ids": ["C3"],
            "limitation_ids": ["C4"],
            "reject_ids": ["C6"],
            "citation_map": self.gold["citation_map"],
            "brief_claim_ids": ["C1", "C2", "C3", "C4", "C5"],
            "brief": "Includes one unsupported claim.",
        }
        metrics = score_knowledge_preparation(json.dumps(response), self.gold)
        self.assertLess(metrics["overall_score"], 100.0)
        self.assertEqual(metrics["unsupported_brief_claim_count"], 1)
        self.assertLess(metrics["claim_classification_accuracy"], 1.0)

    def test_fenced_json_is_recoverable_but_loses_strict_format_points(self):
        response = {
            "case_id": "knowledge_test",
            "fact_ids": ["C1", "C2"],
            "alternative_ids": ["C3"],
            "limitation_ids": ["C4"],
            "reject_ids": ["C5", "C6"],
            "citation_map": self.gold["citation_map"],
            "brief_claim_ids": ["C1", "C2", "C3", "C4"],
            "brief": "Complete brief.",
        }
        metrics = score_knowledge_preparation(
            f"```json\n{json.dumps(response)}\n```", self.gold
        )
        self.assertTrue(metrics["valid_json"])
        self.assertFalse(metrics["strict_json_compliance"])
        self.assertEqual(metrics["overall_score"], 95.0)


class AnswerAssessmentScoringTests(TestCase):
    def test_perfect_assessment_scores_100(self):
        gold = {
            "case_id": "answer_test",
            "assessments": [{
                "item_id": "A1",
                "triage": "SCORE",
                "criterion_id": "security",
                "bloom_alignment": "aligned",
                "decision": "strong",
                "score_ranges": {
                    "correctness": [8, 10],
                    "depth": [7, 9],
                    "consistency": [9, 10],
                    "overall_score": [8, 10],
                },
                "misconception_labels": [],
                "participant_ids": ["student-1"],
            }],
        }
        response = {
            "case_id": "answer_test",
            "assessments": [{
                "item_id": "A1",
                "triage": "SCORE",
                "criterion_id": "security",
                "bloom_alignment": "aligned",
                "decision": "strong",
                "correctness": 9,
                "depth": 8,
                "consistency": 9.5,
                "overall_score": 9,
                "misconception_labels": [],
                "participant_ids": ["student-1"],
                "rationale": "Supported by the supplied reference.",
            }],
        }
        metrics = score_answer_assessment(json.dumps(response), gold)
        self.assertEqual(metrics["overall_score"], 100.0)
        self.assertEqual(metrics["score_range_accuracy"], 1.0)

    def test_unscored_answer_requires_null_scores(self):
        gold = {
            "case_id": "answer_test",
            "assessments": [{
                "item_id": "A1",
                "triage": "CLARIFY",
                "criterion_id": "security",
                "bloom_alignment": "not_demonstrated",
                "decision": "unscored",
                "score_ranges": {
                    "correctness": None,
                    "depth": None,
                    "consistency": None,
                    "overall_score": None,
                },
                "misconception_labels": [],
                "participant_ids": ["student-1"],
            }],
        }
        response = {
            "case_id": "answer_test",
            "assessments": [{
                "item_id": "A1",
                "triage": "CLARIFY",
                "criterion_id": "security",
                "bloom_alignment": "not_demonstrated",
                "decision": "unscored",
                "correctness": 5,
                "depth": None,
                "consistency": None,
                "overall_score": None,
                "misconception_labels": [],
                "participant_ids": ["student-1"],
                "rationale": "The student could not hear the question.",
            }],
        }
        metrics = score_answer_assessment(json.dumps(response), gold)
        self.assertEqual(metrics["score_range_accuracy"], 0.75)
        self.assertLess(metrics["overall_score"], 100.0)


class QuestionQualityScoringTests(TestCase):
    def test_perfect_critic_review_scores_100(self):
        gold = {
            "case_id": "critic_test",
            "task_type": "critic",
            "reviews": [
                {"candidate_id": "C1", "verdict": "pass", "issue_labels": []},
                {"candidate_id": "C2", "verdict": "fail", "issue_labels": ["hallucination"]},
            ],
        }
        response = {
            "case_id": "critic_test",
            "reviews": [
                {"candidate_id": "C1", "verdict": "pass", "issue_labels": [], "rationale": "Supported."},
                {"candidate_id": "C2", "verdict": "fail", "issue_labels": ["hallucination"], "rationale": "Unsupported."},
            ],
        }
        metrics = score_question_quality(json.dumps(response), gold)
        self.assertEqual(metrics["overall_score"], 100.0)
        self.assertEqual(metrics["verdict_accuracy"], 1.0)

    def test_generation_penalizes_unsupported_source(self):
        gold = {
            "case_id": "generation_test",
            "task_type": "generation",
            "questions": [{
                "context_id": "G1",
                "target_bloom": "Apply",
                "socratic_intent": "probing_evidence",
                "source_chunk_ids": ["S1"],
                "required_keyword_groups": [["nonce"], ["chunk"]],
                "recent_questions": [],
                "prohibited_phrases": ["page 12"],
            }],
        }
        response = {
            "case_id": "generation_test",
            "questions": [{
                "context_id": "G1",
                "question_text": "How would you construct a unique nonce for every uploaded chunk?",
                "source_chunk_ids": ["S1", "invented"],
                "target_bloom": "Apply",
                "socratic_intent": "probing_evidence",
            }],
        }
        metrics = score_question_quality(json.dumps(response), gold)
        self.assertEqual(metrics["unsupported_source_count"], 1)
        self.assertLess(metrics["overall_score"], 100.0)


class SessionReportingScoringTests(TestCase):
    def test_exact_report_scores_100(self):
        gold = {
            "case_id": "report_test",
            "reports": [{
                "session_id": "S1",
                "participant_results": [{
                    "participant_id": "alice",
                    "final_score": 32,
                    "score_out_of": 40,
                    "answered_question_ids": ["Q1", "Q2"],
                    "criterion_scores": {"Technical": 24, "Communication": 8},
                }],
                "session_summary": {
                    "question_count": 3,
                    "scored_answer_count": 2,
                    "clarification_count": 1,
                    "examiner_override_count": 0,
                },
                "flags": [],
            }],
        }
        response = {"case_id": "report_test", "reports": gold["reports"]}
        metrics = score_session_reporting(json.dumps(response), gold)
        self.assertEqual(metrics["overall_score"], 100.0)

    def test_letter_grade_and_shared_group_score_are_penalized(self):
        gold = {
            "case_id": "report_test",
            "reports": [{
                "session_id": "G1",
                "participant_results": [
                    {"participant_id":"alice","final_score":19,"score_out_of":30,"answered_question_ids":["Q1"],"criterion_scores":{"Technical":19}},
                    {"participant_id":"john","final_score":25,"score_out_of":30,"answered_question_ids":["Q2"],"criterion_scores":{"Technical":25}},
                ],
                "session_summary": {"question_count":2,"scored_answer_count":2,"clarification_count":0,"examiner_override_count":0},
                "flags": [],
            }],
        }
        response = {"case_id":"report_test","reports":[{
            "session_id":"G1",
            "participant_results":[
                {"participant_id":"alice","final_score":22,"score_out_of":30,"grade":"B","answered_question_ids":["Q1"],"criterion_scores":{"Technical":22}},
                {"participant_id":"john","final_score":22,"score_out_of":30,"answered_question_ids":["Q2"],"criterion_scores":{"Technical":22}},
            ],
            "session_summary": gold["reports"][0]["session_summary"],
            "flags": [],
        }]}
        metrics = score_session_reporting(json.dumps(response), gold)
        self.assertEqual(metrics["forbidden_grade_field_count"], 1)
        self.assertLess(metrics["overall_score"], 100.0)
