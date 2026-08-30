import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import ANY, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")

from viva_evaluator.services.pipeline.orchestrator import VivaPipeline


class VivaPipelineOrchestratorTests(TestCase):
    def test_submit_computes_before_entering_persistence(self):
        events = []
        computation = {
            "analysis": {"reasoning": "sound"},
            "soft_score": 0.8,
            "speech_confidence": {},
            "session_complete": True,
        }
        answer = SimpleNamespace(id="answer-1")
        persisted = SimpleNamespace(duplicate=False, question=None, answer=answer)
        session = SimpleNamespace(examiner_paused=False)

        with (
            patch(
                "viva_evaluator.services.pipeline.orchestrator."
                "find_existing_answer",
                return_value=None,
            ),
            patch(
                "viva_evaluator.services.pipeline.orchestrator."
                "process_answer_and_pick_next",
                side_effect=lambda **kwargs: (
                    events.append("compute") or computation
                ),
            ),
            patch(
                "viva_evaluator.services.pipeline.orchestrator.persist_turn",
                side_effect=lambda **kwargs: (
                    events.append("persist") or persisted
                ),
            ),
            patch(
                "viva_evaluator.services.pipeline.orchestrator.present_turn",
                return_value={"answer_saved": True},
            ),
            patch(
                "viva_evaluator.services.pipeline.orchestrator."
                "_persist_answer_attribution",
                side_effect=lambda **kwargs: events.append("attribute"),
            ) as attribution_mock,
        ):
            response = VivaPipeline().submit_answer(
                session=session,
                submission=SimpleNamespace(),
                question=SimpleNamespace(),
                answer_text="It isolates service failures.",
                speech_metrics=None,
                speaker_id="group",
                student_profile=None,
            )

        self.assertEqual(events, ["compute", "persist", "attribute"])
        attribution_mock.assert_called_once_with(
            answer=answer,
            session=session,
            question=ANY,
            submitter_student_profile=None,
        )
        self.assertEqual(response, {"answer_saved": True})
        self.assertEqual(
            computation["llm_telemetry"]["trace_kind"],
            "answer_turn",
        )
        self.assertEqual(computation["llm_telemetry"]["call_count"], 0)

    def test_duplicate_retry_skips_all_ai_computation(self):
        session = SimpleNamespace(status="in_progress")
        question = SimpleNamespace(question_order=1)

        with (
            patch(
                "viva_evaluator.services.pipeline.orchestrator."
                "find_existing_answer",
                return_value=SimpleNamespace(),
            ),
            patch(
                "viva_evaluator.services.pipeline.orchestrator."
                "find_next_unanswered_question",
                return_value=None,
            ),
            patch(
                "viva_evaluator.services.pipeline.orchestrator."
                "process_answer_and_pick_next"
            ) as compute_mock,
        ):
            response = VivaPipeline().submit_answer(
                session=session,
                submission=SimpleNamespace(),
                question=question,
                answer_text="Repeated request",
                speech_metrics=None,
                speaker_id="group",
                student_profile=None,
            )

        compute_mock.assert_not_called()
        self.assertTrue(response["duplicate_ignored"])
