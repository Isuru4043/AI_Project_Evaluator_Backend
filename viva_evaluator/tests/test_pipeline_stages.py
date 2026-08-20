import os
import math
from concurrent.futures import Future
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")

from viva_evaluator.services.pipeline.contracts import (
    AnswerAssessment,
    FairnessVerdicts,
    QuestionCandidate,
)
from viva_evaluator.services.agents.questioner import QuestionerInput
from viva_evaluator.services.agents.fairness_review import (
    FairnessReviewInput,
    review_fairness,
)
from viva_evaluator.services.agents.tier1_validator import Tier1Result
from viva_evaluator.services.pipeline.session_state import SessionState
from viva_evaluator.services.pipeline.stages.ability_update import (
    record_scored_turn,
    update_topic_ability,
)
from viva_evaluator.services.pipeline.stages.answer_assessment import assess_answer
from viva_evaluator.services.pipeline.stages.fairness_adjustment import (
    apply_fairness_adjustments,
    plan_fairness_checks,
    resolve_fairness_futures,
    submit_fairness_checks,
)
from viva_evaluator.services.pipeline.stages.question_planning import (
    collect_topic_hints,
    plan_next_question,
)
from viva_evaluator.services.pipeline.stages.candidate_generation import (
    generate_question_candidate,
)
from viva_evaluator.services.pipeline.stages.question_validation import (
    validate_question_candidate,
)


class RecordingExecutor:
    def __init__(self):
        self.submitted = []

    def submit(self, function, *args, **kwargs):
        self.submitted.append(function.__name__)
        future = Future()
        if function.__name__ == "review_fairness":
            future.set_result(
                {
                    "consistency": {
                        "material": False,
                        "rationale": "wording only",
                    },
                    "charitable": {
                        "understanding_sound": True,
                        "rationale": "sound idea",
                    },
                    "self_correction": {
                        "is_correction": True,
                        "improved": True,
                        "rationale": "student corrected the earlier claim",
                    },
                }
            )
        else:
            future.set_exception(AssertionError("unexpected submitted function"))
        return future


class AnswerAssessmentStageTests(TestCase):
    @patch(
        "viva_evaluator.services.pipeline.stages.answer_assessment."
        "build_recent_transcript"
    )
    @patch("viva_evaluator.services.confidence.analyze_speech_confidence")
    @patch("viva_evaluator.services.agents.analyzer.analyze_answer")
    @patch("viva_evaluator.services.agents.response_triage.triage_response")
    @patch(
        "viva_evaluator.services.rag.retrieval.retrieve_hybrid_for_turn"
    )
    def test_assessment_returns_one_typed_scored_result(
        self,
        retrieve,
        triage,
        analyze,
        confidence,
        transcript,
    ):
        retrieve.return_value = {"chunks": [{"score": 0.8}]}
        triage.return_value = {"label": "ANSWER_ATTEMPT"}
        analyze.return_value = {
            "soft_score": 0.72,
            "correctness": {"score": 0.8},
        }
        confidence.return_value = {"flag": "high"}
        transcript.return_value = [{"question_text": "Earlier?", "answer_text": "Yes"}]
        marks = []

        result = assess_answer(
            session=SimpleNamespace(),
            submission=SimpleNamespace(),
            previous_question=SimpleNamespace(question_text="Why this design?"),
            student_answer="Because it separates responsibilities.",
            answered_topic={
                "topic_name": "Architecture",
                "topic_focus": "Boundaries",
            },
            speech_metrics={"pause_count": 1},
            clarification_allowed=True,
            marker=marks.append,
        )

        self.assertFalse(result.clarification_required)
        self.assertEqual(result.soft_score, 0.72)
        self.assertEqual(result.correctness, 0.8)
        self.assertEqual(result.speech_confidence, {"flag": "high"})
        self.assertEqual(
            marks,
            [
                "A:retrieval",
                "A.5:triage(parallel)",
                "B:analyzer(parallel)",
                "B.5:confidence",
            ],
        )

    @patch(
        "viva_evaluator.services.pipeline.stages.answer_assessment."
        "build_recent_transcript",
        return_value=[],
    )
    @patch("viva_evaluator.services.agents.analyzer.analyze_answer")
    @patch("viva_evaluator.services.agents.response_triage.triage_response")
    @patch(
        "viva_evaluator.services.rag.retrieval.retrieve_hybrid_for_turn",
        return_value={"chunks": []},
    )
    def test_clarification_result_does_not_contain_a_score(
        self,
        _retrieve,
        triage,
        analyze,
        _transcript,
    ):
        triage.return_value = {
            "label": "CONFUSED_BY_QUESTION",
            "rationale": "question unclear",
        }
        analyze.return_value = {
            "soft_score": 0.1,
            "correctness": {"score": 0.1},
        }

        result = assess_answer(
            session=SimpleNamespace(),
            submission=SimpleNamespace(),
            previous_question=SimpleNamespace(question_text="Question?"),
            student_answer="I do not understand.",
            answered_topic={"topic_name": "Topic", "topic_focus": ""},
            speech_metrics=None,
            clarification_allowed=True,
        )

        self.assertTrue(result.clarification_required)
        self.assertIsNone(result.analysis)
        self.assertIsNone(result.soft_score)
        self.assertIsNone(result.correctness)


class FairnessStageTests(TestCase):
    def _assessment(self):
        return AnswerAssessment(
            retrieval={"chunks": [{"score": 0.7}]},
            transcript_recent=[
                {"question_text": "Earlier?", "answer_text": "Earlier answer"}
            ],
            triage={"label": "ANSWER_ATTEMPT"},
            analysis={
                "soft_score": 0.5,
                "correctness": {"score": 0.5},
                "consistency": {"score": 0.8},
            },
            soft_score=0.5,
            correctness=0.5,
        )

    def test_charitable_agent_is_submitted_exactly_once(self):
        assessment = self._assessment()
        plan = plan_fairness_checks(
            assessment,
            answered_topic={
                "topic_name": "Topic",
                "source_criteria_ids": ["criterion-1"],
            },
            student_answer="An imprecise but sound answer",
        )
        executor = RecordingExecutor()

        futures = submit_fairness_checks(
            executor,
            plan=plan,
            assessment=assessment,
            previous_question=SimpleNamespace(question_text="Why?"),
            student_answer="An imprecise but sound answer",
            answered_topic={"topic_name": "Topic", "topic_focus": "Focus"},
        )
        verdicts = resolve_fairness_futures(futures)

        self.assertEqual(executor.submitted, ["review_fairness"])
        self.assertTrue(verdicts.charitable["understanding_sound"])

    def test_charitable_adjustment_only_raises_score(self):
        assessment = self._assessment()
        plan = plan_fairness_checks(
            assessment,
            answered_topic={
                "topic_name": "Topic",
                "source_criteria_ids": ["criterion-1"],
            },
            student_answer="An imprecise but sound answer",
        )

        adjusted = apply_fairness_adjustments(
            assessment,
            plan,
            FairnessVerdicts(
                charitable={
                    "understanding_sound": True,
                    "rationale": "understanding is sound",
                },
                self_correction={
                    "is_correction": False,
                    "improved": False,
                    "rationale": "not a correction",
                },
            ),
        )

        self.assertEqual(adjusted.soft_score, 0.65)
        self.assertTrue(adjusted.analysis["charitable"]["applied"])

    def test_self_correction_requires_signal_and_same_topic(self):
        assessment = self._assessment()
        assessment.transcript_recent[0].update({
            "topic_name": "Architecture",
            "source_criteria_ids": ["criterion-1"],
        })
        topic = {
            "topic_name": "Architecture",
            "source_criteria_ids": ["criterion-1"],
        }

        without_signal = plan_fairness_checks(
            assessment,
            answered_topic=topic,
            student_answer="The service uses a token.",
        )
        with_signal = plan_fairness_checks(
            assessment,
            answered_topic=topic,
            student_answer=(
                "Actually, I should have said the service validates a JWT."
            ),
        )
        unrelated = plan_fairness_checks(
            assessment,
            answered_topic={
                "topic_name": "Testing",
                "source_criteria_ids": ["criterion-2"],
            },
            student_answer="Actually, I should have said integration tests.",
        )

        self.assertFalse(without_signal.check_self_correction)
        self.assertTrue(with_signal.check_self_correction)
        self.assertFalse(unrelated.check_self_correction)

    def test_multiple_fairness_risks_still_submit_only_one_llm_call(self):
        assessment = self._assessment()
        assessment.analysis["consistency"] = {
            "score": 0.2,
            "evidence_quote": "Earlier I said sessions",
            "evidence_source": "transcript",
        }
        assessment.transcript_recent[0].update({
            "topic_name": "Architecture",
            "source_criteria_ids": ["criterion-1"],
        })
        topic = {
            "topic_name": "Architecture",
            "topic_focus": "Boundaries",
            "source_criteria_ids": ["criterion-1"],
        }
        plan = plan_fairness_checks(
            assessment,
            answered_topic=topic,
            student_answer=(
                "Actually, I mean the session token validates this request."
            ),
        )
        executor = RecordingExecutor()

        futures = submit_fairness_checks(
            executor,
            plan=plan,
            assessment=assessment,
            previous_question=SimpleNamespace(question_text="Why?"),
            student_answer=(
                "Actually, I mean the session token validates this request."
            ),
            answered_topic=topic,
        )

        self.assertEqual(
            plan.requested_checks,
            ("consistency", "charitable", "self_correction"),
        )
        self.assertEqual(len(futures), 1)
        self.assertEqual(executor.submitted, ["review_fairness"])

    @patch(
        "viva_evaluator.services.agents.fairness_review.llm_call",
        return_value={
            "consistency": None,
            "charitable": {
                "understanding_sound": True,
                "confidence": 0.9,
                "rationale": "The underlying idea matches the evidence.",
            },
            "self_correction": None,
        },
    )
    def test_batched_review_uses_one_fast_model_call(self, llm_call_mock):
        result = review_fairness(FairnessReviewInput(
            question_text="Why this boundary?",
            student_answer="It keeps failures away from the other service.",
            criterion_name="Architecture",
            retrieved_chunks=[{"text": "Services have separate boundaries."}],
            requested_checks=("charitable",),
        ))

        llm_call_mock.assert_called_once()
        self.assertEqual(
            llm_call_mock.call_args.kwargs["operation"],
            "fairness_review",
        )
        self.assertTrue(result["charitable"]["understanding_sound"])
        self.assertTrue(result["consistency"]["material"])


class AbilityStageTests(TestCase):
    def test_ability_and_coverage_updates_apply_to_every_topic_criterion(self):
        active = SessionState()
        group = active
        for criterion_id in ("criterion-1", "criterion-2"):
            active.get_or_init_bkt(criterion_id)
            active.get_or_init_coverage(criterion_id)

        topic = {"source_criteria_ids": ["criterion-1", "criterion-2"]}
        before = {
            criterion_id: active.bkt_states[criterion_id].p_lt
            for criterion_id in topic["source_criteria_ids"]
        }

        update_topic_ability(
            active_state=active,
            answered_topic=topic,
            soft_score=0.9,
            bloom_level="Analyze",
        )
        record_scored_turn(
            active_state=active,
            group_state=group,
            answered_topic=topic,
            correctness=0.8,
            soft_score=0.9,
        )

        for criterion_id in topic["source_criteria_ids"]:
            self.assertNotEqual(
                active.bkt_states[criterion_id].p_lt,
                before[criterion_id],
            )
            self.assertEqual(active.coverage[criterion_id].turns, 1)
            self.assertEqual(active.coverage[criterion_id].correct_turns, 1)
        self.assertEqual(group.total_turns, 1)
        self.assertEqual(group.soft_score_history, [0.9])


class QuestionPlanningStageTests(TestCase):
    def test_topic_hints_are_unique_and_follow_rubric_order(self):
        topic = {"source_criteria_ids": ["criterion-1", "criterion-2"]}
        rubric = [
            {"id": "criterion-1", "hints": ["Ask why", "Ask evidence"]},
            {"id": "criterion-2", "hints": ["Ask evidence", "Ask limits"]},
        ]

        self.assertEqual(
            collect_topic_hints(topic, rubric),
            ("Ask why", "Ask evidence", "Ask limits"),
        )

    @patch(
        "viva_evaluator.services.rag.retrieval.retrieve_module_materials",
        return_value=[{"text": "module boundary"}],
    )
    @patch(
        "viva_evaluator.services.rag.retrieval.retrieve_hybrid_for_turn",
        return_value={"chunks": [{"score": 0.9, "text": "service code"}]},
    )
    @patch(
        "viva_evaluator.services.agents.strategist.select_strategy",
        return_value={
            "bloom_level": "Create",
            "socratic_intent": "probing_evidence",
            "intent_prompt_hint": "Ask for concrete implementation evidence.",
            "rationale": "post-answer mastery supports Create",
        },
    )
    def test_planner_uses_updated_state_and_returns_full_generation_context(
        self,
        select_strategy_mock,
        hybrid_retrieval_mock,
        module_retrieval_mock,
    ):
        answered_topic = {
            "topic_name": "Architecture",
            "topic_focus": "Boundaries",
            "source_criteria_ids": ["criterion-1"],
            "suggested_questions": 1,
        }
        next_topic = {
            "topic_name": "Testing",
            "topic_focus": "Verification strategy",
            "source_criteria_ids": ["criterion-2"],
            "suggested_questions": 2,
        }
        state = SessionState(total_turns=1)
        completed_coverage = state.get_or_init_coverage("criterion-1")
        completed_coverage.correct_turns = 1
        completed_coverage.turns = 1
        state.get_or_init_bkt("criterion-1")
        next_coverage = state.get_or_init_coverage("criterion-2")
        next_ability = state.get_or_init_bkt("criterion-2")
        next_ability.mu = math.log(0.72 / 0.28)

        questions = MagicMock()
        questions.order_by.return_value.values_list.return_value.__getitem__.return_value = [
            "Recent question"
        ]
        session = SimpleNamespace(
            project=SimpleNamespace(id="project-1"),
            viva_questions=questions,
            max_total_questions=10,
        )
        rubric = [
            {"id": "criterion-1", "hints": ["Architecture hint"]},
            {
                "id": "criterion-2",
                "hints": ["Testing hint", "Ask for a failure case"],
            },
        ]

        result = plan_next_question(
            session=session,
            submission=SimpleNamespace(),
            topics=[answered_topic, next_topic],
            rubric=rubric,
            unified_state=state,
            intent_history=["clarifying"],
            adjusted_analysis={"correctness": {"score": 0.8}},
            speech_confidence={"flag": "high"},
            answered_topic=answered_topic,
            answered_retrieval={"chunks": [{"score": 0.7}]},
            student_answer="We use integration tests.",
        )

        strategist_input = select_strategy_mock.call_args.args[0]
        self.assertEqual(strategist_input.p_lt, 0.72)
        self.assertEqual(result.plan.topic.name, "Testing")
        self.assertEqual(result.plan.target_bloom, "Create")
        self.assertEqual(result.plan.difficulty, "hard")
        self.assertEqual(result.plan.question_number_in_topic, next_coverage.turns + 1)
        self.assertEqual(
            result.grounding.question_hints,
            ("Testing hint", "Ask for a failure case"),
        )
        self.assertEqual(result.grounding.recent_questions, ("Recent question",))
        self.assertEqual(result.grounding.module_chunks[0]["text"], "module boundary")
        self.assertIn(
            "submission_chunk",
            {
                reference.evidence_type
                for reference in result.grounding.evidence_package.references
            },
        )
        self.assertIn(
            "module_chunk",
            {
                reference.evidence_type
                for reference in result.grounding.evidence_package.references
            },
        )
        hybrid_retrieval_mock.assert_called_once()
        module_retrieval_mock.assert_called_once()


class QuestionGenerationStageTests(TestCase):
    def _input(self):
        return QuestionerInput(
            criterion_name="Architecture",
            retrieved_chunks=[{
                "text": "The project uses service boundaries.",
                "evidence_id": "submission:test:chunk:1",
            }],
            difficulty="hard",
            target_bloom="Create",
            socratic_intent="probing_evidence",
        )

    @patch(
        "viva_evaluator.services.pipeline.stages.question_validation."
        "validate_question"
    )
    @patch(
        "viva_evaluator.services.llm_service.llm_call",
        return_value={
            "question_text": (
                "In your architecture, how would you redesign the service "
                "boundary to remain reliable during partial failures?"
            ),
            "source_reference_ids": ["submission:test:chunk:1"],
            "target_bloom": "Create",
            "socratic_intent": "probing_evidence",
        },
    )
    def test_raw_generation_is_one_llm_call_without_validation(
        self,
        llm_call_mock,
        validate_mock,
    ):
        candidate = generate_question_candidate(self._input())

        llm_call_mock.assert_called_once()
        self.assertEqual(llm_call_mock.call_args.kwargs["model"], "reasoning")
        self.assertEqual(
            llm_call_mock.call_args.kwargs["operation"],
            "question_generation",
        )
        validate_mock.assert_not_called()
        self.assertEqual(candidate.blooms_level, "Create")
        self.assertEqual(candidate.difficulty, "hard")
        self.assertEqual(
            candidate.source_reference_ids,
            ("submission:test:chunk:1",),
        )
        self.assertEqual(candidate.schema_failures, ())
        self.assertEqual(len(candidate.candidate_hash), 64)

    @patch(
        "viva_evaluator.services.llm_service.llm_call",
        return_value={
            "question_text": (
                "In your architecture, how would you redesign the service "
                "boundary to remain reliable during partial failures?"
            ),
            "source_reference_ids": ["submission:test:chunk:1"],
            "target_bloom": "Create",
            "socratic_intent": "probing_evidence",
        },
    )
    def test_retry_uses_fast_question_repair_route(self, llm_call_mock):
        generate_question_candidate(
            self._input(),
            retry_reason="missing project anchor",
        )

        self.assertEqual(llm_call_mock.call_args.kwargs["model"], "fast")
        self.assertEqual(
            llm_call_mock.call_args.kwargs["operation"],
            "question_repair",
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
    def test_high_confidence_validation_skips_critic(self, validate_mock):
        questioner_input = self._input()
        questioner_input.target_bloom = "Analyze"
        questioner_input.difficulty = "medium"
        questioner_input.weak_grounding = True
        candidate = QuestionCandidate(
            question_text=(
                "In your architecture, how would you redesign the service "
                "boundary to remain reliable during partial failures?"
            ),
            blooms_level="Analyze",
            difficulty="medium",
            candidate_hash="candidate-1",
            socratic_intent="probing_evidence",
        )

        validated = validate_question_candidate(questioner_input, candidate)

        validate_mock.assert_called_once()
        self.assertTrue(validated.tier1_passed)
        self.assertIsNone(validated.critic_passed)
        self.assertEqual(validated.candidate_hash, "candidate-1")
