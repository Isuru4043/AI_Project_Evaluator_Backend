import os
import sys
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")

from viva_evaluator.services.pipeline import turn_pipeline
from viva_evaluator.services.pipeline.contracts import AnswerAssessment
from viva_evaluator.services.pipeline.session_state import SessionState
from viva_evaluator.services.pipeline.state_bundle import (
    SessionStateBundle,
    build_unified_state,
)
from viva_evaluator.services.pipeline.termination import (
    TerminationDecision,
    should_terminate,
)


class PipelineTerminationTests(TestCase):
    def _turn_fixture(self):
        topic = {
            "topic_name": "Architecture",
            "topic_focus": "Boundaries",
            "source_criteria_ids": ["criterion-1"],
            "suggested_questions": 2,
        }
        rubric = [
            {
                "id": "criterion-1",
                "name": "Architecture",
                "is_individual": False,
                "questions_to_ask": 2,
                "hints": [],
            }
        ]
        state = SessionState()
        state.get_or_init_coverage("criterion-1", questions_to_ask=2)
        state.get_or_init_bkt("criterion-1")
        bundle = SessionStateBundle(
            group_state=state,
            student_state=None,
            active_state=state,
            unified_state=build_unified_state(state, None, rubric),
            speaker_id="group",
            is_individual_topic=False,
        )
        assessment = AnswerAssessment(
            retrieval={"chunks": [{"score": 0.8}]},
            transcript_recent=[],
            triage={"label": "ANSWER_ATTEMPT"},
            analysis={
                "soft_score": 0.9,
                "correctness": {"score": 0.9},
                "depth": {"score": 0.9},
                "consistency": {"score": 0.9},
            },
            soft_score=0.9,
            correctness=0.9,
            speech_confidence={"flag": "high"},
        )
        questions = MagicMock()
        questions.filter.return_value.count.return_value = 1
        session = SimpleNamespace(
            id="session-1",
            project=SimpleNamespace(id="project-1"),
            viva_questions=questions,
            max_total_questions=10,
            save=MagicMock(),
        )
        question = SimpleNamespace(
            question_text="Why this architecture?",
            blooms_level="Analyze",
        )
        return topic, rubric, bundle, assessment, session, question

    def _run_turn(self, decision, *, examiner_paused=False, fake_core=False):
        topic, rubric, bundle, assessment, session, question = self._turn_fixture()
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(turn_pipeline, "load_rubric", return_value=rubric)
            )
            stack.enter_context(
                patch.object(turn_pipeline, "load_viva_topics", return_value=[topic])
            )
            stack.enter_context(
                patch.object(
                    turn_pipeline,
                    "resolve_answered_topic",
                    return_value=topic,
                )
            )
            stack.enter_context(
                patch.object(turn_pipeline, "load_state_bundle", return_value=bundle)
            )
            stack.enter_context(
                patch.object(turn_pipeline, "assess_answer", return_value=assessment)
            )
            stack.enter_context(
                patch(
                    "viva_evaluator.services.pipeline.termination.should_terminate",
                    return_value=decision,
                )
            )
            stack.enter_context(
                patch(
                    "viva_evaluator.services.pipeline.session_state.save_session_state"
                )
            )
            plan_mock = stack.enter_context(
                patch.object(turn_pipeline, "plan_next_question")
            )
            generate_mock = stack.enter_context(
                patch("viva_evaluator.services.agents.generate_anchored_question")
            )

            if fake_core:
                fake_models = ModuleType("core.models")
                fake_models.EvaluationSession = type(
                    "EvaluationSession",
                    (),
                    {"Status": SimpleNamespace(COMPLETED="completed")},
                )
                stack.enter_context(
                    patch.dict(sys.modules, {"core.models": fake_models})
                )

            result = turn_pipeline.process_answer_and_pick_next(
                session=session,
                submission=SimpleNamespace(),
                prev_question_obj=question,
                student_answer="It separates responsibilities.",
                examiner_paused=examiner_paused,
            )

        return result, session, plan_mock, generate_mock

    def test_termination_decision_is_pure_and_uses_explicit_hard_cap(self):
        state = SessionState(total_turns=1)

        decision = should_terminate(
            state,
            [],
            ai_question_count=4,
            hard_cap=4,
        )

        self.assertTrue(decision.should_end)
        self.assertTrue(decision.hard_cap_hit)

    def test_incorrect_answers_still_satisfy_attempt_coverage(self):
        state = SessionState(total_turns=2)
        coverage = state.get_or_init_coverage(
            "criterion-1",
            questions_to_ask=2,
        )
        coverage.turns = 2
        coverage.correct_turns = 0
        state.get_or_init_bkt("criterion-1")

        decision = should_terminate(
            state,
            [{
                "id": "criterion-1",
                "name": "Architecture",
                "questions_to_ask": 2,
            }],
            ai_question_count=2,
            hard_cap=10,
        )

        self.assertTrue(decision.coverage_met)
        self.assertFalse(decision.should_end)

    def test_per_concept_cap_ends_an_exhausted_small_rubric(self):
        state = SessionState(total_turns=5)
        coverage = state.get_or_init_coverage(
            "criterion-1",
            questions_to_ask=2,
        )
        coverage.turns = 5
        state.get_or_init_bkt("criterion-1")

        decision = should_terminate(
            state,
            [{
                "id": "criterion-1",
                "name": "Architecture",
                "questions_to_ask": 2,
            }],
            ai_question_count=5,
            hard_cap=10,
        )

        self.assertTrue(decision.should_end)
        self.assertIn("per-concept cap", decision.reason)

    def test_terminated_turn_does_not_plan_or_generate_question(self):
        decision = TerminationDecision(
            should_end=True,
            reason="hard cap",
            coverage_met=False,
            min_turns_met=True,
            bkt_converged=False,
            hard_cap_hit=True,
        )

        result, session, plan_mock, generate_mock = self._run_turn(
            decision,
            fake_core=True,
        )

        self.assertTrue(result["session_complete"])
        self.assertIn("_state_bundle", result)
        plan_mock.assert_not_called()
        generate_mock.assert_not_called()

    def test_paused_turn_updates_score_but_does_not_plan_or_generate(self):
        decision = TerminationDecision(
            should_end=False,
            reason="continue",
            coverage_met=False,
            min_turns_met=False,
            bkt_converged=False,
            hard_cap_hit=False,
        )

        result, _session, plan_mock, generate_mock = self._run_turn(
            decision,
            examiner_paused=True,
        )

        self.assertTrue(result["paused_by_examiner"])
        self.assertIsNone(result["next_question_payload"])
        plan_mock.assert_not_called()
        generate_mock.assert_not_called()
