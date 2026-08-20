import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")

from viva_evaluator.services.agents.critic import (
    CriticInput,
    _build_prompt,
    critique_question,
)
from viva_evaluator.services.agents.questioner import QuestionerInput
from viva_evaluator.services.agents.tier1_validator import Tier1Result
from viva_evaluator.services.pipeline.contracts import (
    QuestionCandidate,
    ValidatedQuestion,
)
from viva_evaluator.services.pipeline.exceptions import (
    QuestionGenerationUnavailableError,
)
from viva_evaluator.services.pipeline.evidence import (
    build_question_evidence_package,
)
from viva_evaluator.services.pipeline.stages.candidate_generation import (
    generate_question_candidate,
)
from viva_evaluator.services.pipeline.stages.question_validation import (
    _SAFE_FALLBACKS,
    validate_question_candidate,
)
from viva_evaluator.services.pipeline.stages.persistence import (
    _create_question,
    _assert_computed_next_question_safe,
    _assert_validated_question_safe,
    build_question_generation_audit,
)
from viva_evaluator.services.pipeline.presenter import (
    persisted_validation_metadata,
)
from viva_evaluator.services.evaluation.metrics import (
    compute_question_metrics,
    persisted_audit_to_result,
)
from viva_evaluator.services.rag.retrieval import _attach_chunk_evidence_ids


class EvidencePackageTests(TestCase):
    def test_chunk_evidence_ids_are_stable_for_the_same_scope(self):
        chunk = {
            "chunk_idx": 3,
            "text": "Authentication is isolated from project management.",
        }

        first = _attach_chunk_evidence_ids(
            [chunk],
            namespace="submission:abc",
        )[0]
        second = _attach_chunk_evidence_ids(
            [chunk],
            namespace="submission:abc",
        )[0]

        self.assertEqual(first["evidence_id"], second["evidence_id"])
        self.assertEqual(
            first["evidence_id"],
            "submission:abc:chunk:3",
        )

    def test_package_unifies_submission_module_kg_and_previous_answer(self):
        retrieval = {
            "chunks": [{
                "text": "The API is separated into authentication services.",
                "source": "report",
                "chunk_idx": 4,
            }],
            "contradicts_code_alerts": [{
                "source": "hardcoded token",
                "target": "secure token storage",
                "attrs": {"finding_detail": "Token appears in source."},
            }],
            "alternative_edges": [{
                "edge_type": "ALTERNATIVE_TO",
                "base_tech": "REST",
                "alternative": "GraphQL",
                "rationale": "Different query flexibility.",
            }],
            "depends_on_topics": ["django"],
        }
        modules = [{"text": "Service cohesion and coupling", "chunk_idx": 2}]

        package = build_question_evidence_package(
            retrieval=retrieval,
            module_chunks=modules,
            previous_answer="We separated authentication for isolation.",
        )

        evidence_types = {reference.evidence_type for reference in package.references}
        self.assertEqual(
            evidence_types,
            {
                "submission_chunk",
                "module_chunk",
                "kg_contradiction",
                "kg_alternative",
                "kg_dependency",
                "previous_answer",
            },
        )
        self.assertEqual(len(package.evidence_ids), len(set(package.evidence_ids)))
        self.assertTrue(retrieval["chunks"][0]["evidence_id"])
        self.assertTrue(modules[0]["evidence_id"])


class StructuredCandidateTests(TestCase):
    def _input(self):
        return QuestionerInput(
            criterion_name="Architecture",
            retrieved_chunks=[{
                "text": "The project separates authentication services.",
                "evidence_id": "submission:test:chunk:1",
            }],
            difficulty="medium",
            target_bloom="Analyze",
            socratic_intent="testing_connections",
        )

    @patch(
        "viva_evaluator.services.llm_service.llm_call",
        return_value={
            "question_text": (
                "In your project, how does the authentication service interact "
                "with the rest of your API during a failed request?"
            ),
            "source_reference_ids": ["invented:chunk:99"],
            "target_bloom": "Evaluate",
            "socratic_intent": "clarifying",
        },
    )
    def test_candidate_rejects_unknown_sources_and_strategy_mismatch(self, _llm):
        candidate = generate_question_candidate(self._input())

        self.assertIn("target_bloom_mismatch", candidate.schema_failures)
        self.assertIn("socratic_intent_mismatch", candidate.schema_failures)
        self.assertTrue(
            any(
                failure.startswith("unknown_source_reference_ids")
                for failure in candidate.schema_failures
            )
        )

    @patch(
        "viva_evaluator.services.pipeline.stages.question_validation."
        "validate_question",
        return_value=Tier1Result(
            passed=True,
            failures=[],
            similarity_to_recent=0.1,
            word_count=20,
        ),
    )
    def test_schema_failure_uses_safe_fallback_instead_of_leaking(self, _tier1):
        candidate = QuestionCandidate(
            question_text=(
                "In your project, how does the authentication service interact "
                "with the rest of your API during a failed request?"
            ),
            blooms_level="Analyze",
            difficulty="medium",
            socratic_intent="testing_connections",
            source_reference_ids=("invented:chunk:99",),
        )

        result = validate_question_candidate(
            self._input(),
            candidate,
            max_retries=0,
            enable_critic=False,
        )

        self.assertTrue(result.tier1_passed)
        self.assertEqual(result.validation_status, "safe_fallback")
        self.assertTrue(result.validation_degraded)
        self.assertTrue(result.fallback_used)
        self.assertNotEqual(result.question_text, candidate.question_text)


class CriticParityTests(TestCase):
    def _input_and_candidate(self):
        questioner_input = QuestionerInput(
            criterion_name="Architecture",
            retrieved_chunks=[{
                "text": "The project separates authentication services.",
                "source": "code",
                "evidence_id": "submission:test:chunk:1",
            }],
            module_chunks=[{
                "text": "Service cohesion and coupling",
                "evidence_id": "module:test:chunk:1",
            }],
            kg_signals={
                "chunks": [{
                    "text": "The project separates authentication services.",
                    "source": "code",
                    "evidence_id": "submission:test:chunk:1",
                }],
                "alternative_edges": [{
                    "base_tech": "REST",
                    "alternative": "GraphQL",
                    "edge_type": "ALTERNATIVE_TO",
                }],
            },
            difficulty="hard",
            target_bloom="Create",
            socratic_intent="exploring_alternatives",
        )
        package = build_question_evidence_package(
            retrieval=questioner_input.kg_signals,
            module_chunks=questioner_input.module_chunks,
        )
        questioner_input.evidence_package = package
        candidate = QuestionCandidate(
            question_text=(
                "In your project, how would you redesign the authentication "
                "boundary while comparing the supported REST alternative?"
            ),
            blooms_level="Create",
            difficulty="hard",
            socratic_intent="exploring_alternatives",
            source_reference_ids=("submission:test:chunk:1",),
        )
        return questioner_input, candidate, package

    @patch(
        "viva_evaluator.services.agents.critic.critique_question",
        return_value={
            "passed": True,
            "critique": "",
            "specificity_score": 0.9,
            "bloom_alignment_score": 0.9,
            "conversational_flow_score": 0.9,
            "boundary_check_score": 0.9,
            "source_reference_support_score": 0.9,
            "hallucination_flag": False,
        },
    )
    @patch(
        "viva_evaluator.services.pipeline.stages.question_validation."
        "validate_question",
        return_value=Tier1Result(
            passed=True,
            failures=[],
            similarity_to_recent=0.1,
            word_count=20,
        ),
    )
    def test_high_risk_candidate_requires_critic_with_identical_evidence(
        self,
        _tier1,
        critic_mock,
    ):
        questioner_input, candidate, package = self._input_and_candidate()

        result = validate_question_candidate(questioner_input, candidate)

        critic_mock.assert_called_once()
        critic_input = critic_mock.call_args.args[0]
        self.assertIs(critic_input.evidence_package, package)
        self.assertEqual(
            critic_input.source_reference_ids,
            ["submission:test:chunk:1"],
        )
        self.assertTrue(result.critic_passed)
        self.assertEqual(result.critic_scores["source_reference_support"], 0.9)

    def test_critic_prompt_contains_all_evidence_categories_and_cited_ids(self):
        questioner_input, candidate, package = self._input_and_candidate()

        prompt = _build_prompt(
            CriticInput(
                question_text=candidate.question_text,
                target_bloom=candidate.blooms_level,
                target_intent=candidate.socratic_intent,
                evidence_package=package,
                source_reference_ids=list(candidate.source_reference_ids),
            )
        )

        self.assertIn("STUDENT SUBMISSION EVIDENCE", prompt)
        self.assertIn("MODULE-BOUNDARY EVIDENCE", prompt)
        self.assertIn("KG ALTERNATIVE EVIDENCE", prompt)
        self.assertIn("submission:test:chunk:1", prompt)
        self.assertIn("SOURCE-REFERENCE SUPPORT", prompt)


class FailClosedValidationTests(TestCase):
    def _input(self, *, bloom="Create"):
        return QuestionerInput(
            criterion_name="Architecture",
            retrieved_chunks=[],
            difficulty="hard" if bloom in {"Evaluate", "Create"} else "medium",
            target_bloom=bloom,
            socratic_intent="probing_evidence",
            weak_grounding=True,
        )

    def _candidate(self, *, bloom="Create"):
        return QuestionCandidate(
            question_text=(
                "Thinking about your project, how would you redesign this "
                "part of your work to improve its reliability?"
            ),
            blooms_level=bloom,
            difficulty="hard" if bloom in {"Evaluate", "Create"} else "medium",
            candidate_hash="candidate-hash",
            socratic_intent="probing_evidence",
        )

    @patch(
        "viva_evaluator.services.agents.critic.llm_call",
        return_value={"passed": True},
    )
    def test_malformed_critic_output_is_unavailable_never_passed(self, _llm):
        result = critique_question(
            CriticInput(
                question_text=self._candidate().question_text,
                target_bloom="Create",
                target_intent="probing_evidence",
            )
        )

        self.assertFalse(result["passed"])
        self.assertTrue(result["_critic_unavailable"])
        self.assertIn("malformed", result["unavailable_reason"])

    @patch(
        "viva_evaluator.services.pipeline.stages.question_validation."
        "_critic_unavailable_policy",
        return_value="degraded_tier1",
    )
    @patch(
        "viva_evaluator.services.agents.critic.critique_question",
        return_value={
            "passed": False,
            "_critic_unavailable": True,
            "unavailable_reason": "critic_llm_call_failed",
            "critique": "Critic validation was unavailable.",
        },
    )
    def test_critic_outage_is_explicit_degraded_validation(
        self,
        _critic,
        _policy,
    ):
        result = validate_question_candidate(
            self._input(),
            self._candidate(),
        )

        self.assertTrue(result.tier1_passed)
        self.assertFalse(result.critic_passed)
        self.assertFalse(result.critic_available)
        self.assertTrue(result.validation_degraded)
        self.assertEqual(result.validation_status, "critic_unavailable")
        self.assertEqual(result.degradation_reason, "critic_llm_call_failed")

    @patch(
        "viva_evaluator.services.pipeline.stages.question_validation."
        "_critic_unavailable_policy",
        return_value="safe_fallback",
    )
    @patch(
        "viva_evaluator.services.agents.critic.critique_question",
        return_value={
            "passed": False,
            "_critic_unavailable": True,
            "unavailable_reason": "critic_llm_call_failed",
            "critique": "Critic validation was unavailable.",
        },
    )
    def test_critic_outage_can_be_routed_to_safe_fallback(
        self,
        _critic,
        _policy,
    ):
        result = validate_question_candidate(self._input(), self._candidate())

        self.assertEqual(result.validation_status, "safe_fallback")
        self.assertTrue(result.fallback_used)
        self.assertFalse(result.critic_available)

    @patch(
        "viva_evaluator.services.pipeline.stages.question_validation."
        "_critic_unavailable_policy",
        return_value="fail_closed",
    )
    @patch(
        "viva_evaluator.services.agents.critic.critique_question",
        return_value={
            "passed": False,
            "_critic_unavailable": True,
            "unavailable_reason": "critic_llm_call_failed",
            "critique": "Critic validation was unavailable.",
        },
    )
    def test_critic_outage_can_be_configured_fail_closed(
        self,
        _critic,
        _policy,
    ):
        with self.assertRaises(QuestionGenerationUnavailableError):
            validate_question_candidate(self._input(), self._candidate())

    @patch(
        "viva_evaluator.services.pipeline.stages.question_validation."
        "generate_question_candidate"
    )
    @patch(
        "viva_evaluator.services.agents.critic.critique_question",
        return_value={
            "passed": False,
            "critique": "The factual anchor is unsupported.",
            "specificity_score": 0.9,
            "bloom_alignment_score": 0.9,
            "conversational_flow_score": 0.9,
            "boundary_check_score": 0.9,
            "source_reference_support_score": 0.0,
            "hallucination_flag": True,
        },
    )
    def test_critic_rejection_exhaustion_uses_safe_fallback(
        self,
        critic_mock,
        generate_mock,
    ):
        generate_mock.return_value = self._candidate()

        result = validate_question_candidate(
            self._input(),
            self._candidate(),
        )

        self.assertEqual(critic_mock.call_count, 2)
        self.assertEqual(generate_mock.call_count, 1)
        self.assertEqual(result.validation_status, "safe_fallback")
        self.assertTrue(result.fallback_used)
        self.assertTrue(result.tier1_passed)
        self.assertIn("critic_rejected", result.degradation_reason)

    def test_every_bloom_fallback_has_one_tier1_valid_question(self):
        from viva_evaluator.services.agents.tier1_validator import validate_question

        for bloom, templates in _SAFE_FALLBACKS.items():
            with self.subTest(bloom=bloom):
                result = validate_question(templates[0], recent_questions=[])
                self.assertTrue(result.passed, result.failures)

    @patch(
        "viva_evaluator.services.pipeline.stages.question_validation."
        "validate_question",
        return_value=Tier1Result(
            passed=False,
            failures=["forced_failure"],
            similarity_to_recent=0.0,
            word_count=20,
        ),
    )
    def test_raises_when_generated_and_all_fallback_questions_are_invalid(
        self,
        _validate,
    ):
        with self.assertRaises(QuestionGenerationUnavailableError):
            validate_question_candidate(
                self._input(),
                self._candidate(),
                max_retries=0,
            )

    def test_persistence_guard_rejects_tier1_invalid_question(self):
        unsafe = ValidatedQuestion(
            question_text="",
            blooms_level="Analyze",
            difficulty="medium",
            tier1_passed=False,
            validation_status="rejected",
        )

        with self.assertRaises(QuestionGenerationUnavailableError):
            _assert_validated_question_safe(unsafe)

    def test_turn_persistence_guard_requires_an_accepted_validation_status(self):
        computation = {
            "session_complete": False,
            "next_question_payload": {
                "question_data": {
                    "question_text": "In your project, how does this part interact with the rest of your solution?",
                    "tier1_passed": True,
                },
            },
        }

        with self.assertRaises(QuestionGenerationUnavailableError):
            _assert_computed_next_question_safe(computation)


class SafeQuestionUnavailableViewTests(TestCase):
    def test_start_endpoint_maps_safe_question_failure_to_503(self):
        import django

        django.setup()
        from viva_evaluator.views.session_views import SessionStartView
        with (
            patch(
                "core.models.EvaluationSession.objects.get",
                return_value=SimpleNamespace(status="scheduled"),
            ),
            patch(
                "viva_evaluator.views.session_views."
                "_resolve_session_submission",
                return_value=object(),
            ),
            patch(
                "viva_evaluator.views.session_views."
                "_get_or_create_index_status",
                return_value=SimpleNamespace(status="ready"),
            ),
            patch(
                "viva_evaluator.services.pipeline.orchestrator."
                "VivaPipeline.start_session",
                side_effect=QuestionGenerationUnavailableError(),
            ),
        ):
            response = SessionStartView().post(
                SimpleNamespace(data={"session_id": "session-1"})
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data["code"], "safe_question_unavailable")


class ValidationAuditTests(TestCase):
    def _audit(self):
        return build_question_generation_audit(
            {
                "blooms_level": "Apply",
                "difficulty": "medium",
                "candidate_hash": "hash-1",
                "socratic_intent": "probing_evidence",
                "source_reference_ids": ["submission:test:chunk:1"],
                "tier1_passed": True,
                "tier1_failures": [],
                "schema_failures": [],
                "critic_ran": True,
                "critic_passed": True,
                "critic_available": True,
                "critic_scores": {"specificity": 0.9},
                "validation_status": "fully_validated",
                "validation_degraded": False,
                "fallback_used": False,
                "attempts": 2,
            },
            blooms_level="Apply",
            difficulty="medium",
        )

    def test_generation_audit_preserves_provenance_and_validation(self):
        audit = self._audit()

        self.assertEqual(audit["schema_version"], 3)
        self.assertEqual(audit["tts"]["status"], "disabled")
        self.assertEqual(audit["llm_telemetry"], {})
        self.assertEqual(audit["candidate_hash"], "hash-1")
        self.assertEqual(
            audit["source_reference_ids"],
            ["submission:test:chunk:1"],
        )
        self.assertTrue(audit["tier1"]["passed"])
        self.assertTrue(audit["critic"]["passed"])
        self.assertEqual(audit["validation"]["status"], "fully_validated")

    def test_generation_audit_persists_the_turn_telemetry_summary(self):
        audit = build_question_generation_audit(
            {
                "tier1_passed": True,
                "validation_status": "tier1_only_policy",
            },
            blooms_level="Analyze",
            difficulty="medium",
            llm_telemetry={
                "trace_id": "trace-1",
                "call_count": 2,
                "total_tokens": 120,
            },
        )

        self.assertEqual(audit["llm_telemetry"]["trace_id"], "trace-1")
        self.assertEqual(audit["llm_telemetry"]["total_tokens"], 120)

    def test_persisted_metadata_survives_resume_shape(self):
        audit = self._audit()
        question = SimpleNamespace(
            extension=SimpleNamespace(
                generation_audit=audit,
                validation_status="fully_validated",
                validation_degraded=False,
                fallback_used=False,
            )
        )

        metadata = persisted_validation_metadata(question)

        self.assertEqual(metadata["candidate_hash"], "hash-1")
        self.assertEqual(metadata["validation_status"], "fully_validated")
        self.assertEqual(
            metadata["source_reference_ids"],
            ["submission:test:chunk:1"],
        )

    def test_persisted_audits_feed_production_quality_metrics(self):
        audit = self._audit()
        extension = SimpleNamespace(
            generation_audit=audit,
            validation_status="fully_validated",
            validation_degraded=False,
            fallback_used=False,
            question=SimpleNamespace(
                id="question-1",
                question_text="In your project, how would you apply this design in a realistic failure scenario?",
                blooms_level="Apply",
            ),
        )

        metrics = compute_question_metrics(
            [persisted_audit_to_result(extension)]
        )

        self.assertEqual(metrics["fully_validated_rate"], 1.0)
        self.assertEqual(metrics["degraded_validation_rate"], 0.0)
        self.assertEqual(metrics["fallback_rate"], 0.0)
        self.assertEqual(metrics["source_attribution_rate"], 1.0)

    def test_question_creation_always_persists_audit_extension(self):
        import django

        django.setup()
        question = SimpleNamespace(id="question-1")
        session = SimpleNamespace(
            viva_questions=SimpleNamespace(count=lambda: 2),
        )
        with (
            patch(
                "core.models.VivaQuestion.objects.create",
                return_value=question,
            ),
            patch(
                "core.models.RubricCriteria.objects.filter"
            ) as criteria_filter,
            patch(
                "viva_evaluator.models.VivaQuestionExtension.objects.create"
            ) as extension_create,
        ):
            criteria_filter.return_value.first.return_value = None
            result = _create_question(
                session,
                question_text="In your project, how would you apply this design in a realistic failure scenario?",
                blooms_level="Apply",
                difficulty="medium",
                topic={
                    "topic_name": "Architecture",
                    "source_criteria_ids": ["missing-criterion"],
                },
                validation_data={
                    "tier1_passed": True,
                    "validation_status": "fully_validated",
                },
            )

        self.assertIs(result, question)
        extension_create.assert_called_once()
        create_kwargs = extension_create.call_args.kwargs
        self.assertIsNone(create_kwargs["criteria"])
        self.assertEqual(create_kwargs["validation_status"], "fully_validated")
        self.assertTrue(create_kwargs["generation_audit"]["tier1"]["passed"])

    def test_question_extension_exposes_queryable_audit_fields(self):
        import django

        django.setup()
        from viva_evaluator.models import VivaQuestionExtension

        field_names = {
            field.name for field in VivaQuestionExtension._meta.get_fields()
        }
        self.assertTrue(
            {
                "validation_status",
                "validation_degraded",
                "fallback_used",
                "generation_audit",
            }.issubset(field_names)
        )
