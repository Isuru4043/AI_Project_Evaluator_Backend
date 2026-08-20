import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")

from viva_evaluator.services.pipeline import turn_pipeline
from viva_evaluator.services.pipeline.contracts import (
    AnswerAssessment,
    FairnessAdjustedAssessment,
)
from viva_evaluator.services.pipeline.context import (
    grounding_is_weak,
    load_viva_topics,
    resolve_answered_topic,
)
from viva_evaluator.services.pipeline.session_state import SessionState
from viva_evaluator.services.pipeline.state_bundle import build_unified_state
from viva_evaluator.services.pipeline.state_bundle import SessionStateBundle
from viva_evaluator.services.pipeline.termination import TerminationDecision
from viva_evaluator.services.pipeline.turn_pipeline import (
    _bloom_to_difficulty,
    pick_next_topic,
)


class PipelineCharacterizationTests(TestCase):
    def test_grounding_uses_best_retrieval_score(self):
        self.assertTrue(grounding_is_weak([]))
        self.assertTrue(grounding_is_weak([{"score": 0.29}]))
        self.assertFalse(
            grounding_is_weak([{"score": 0.12}, {"score": 0.30}])
        )

    def test_grouped_topics_are_used_without_rebuilding_rubric(self):
        topics = [{"topic_name": "Architecture"}]
        session = SimpleNamespace(
            grouping_cache=SimpleNamespace(
                grouped_criteria={"viva_topics": topics}
            )
        )

        self.assertIs(load_viva_topics(session), topics)

    def test_answered_topic_uses_persisted_topic_name(self):
        architecture = {
            "topic_name": "Architecture",
            "source_criteria_ids": ["criterion-1"],
            "suggested_questions": 2,
            "topic_focus": "Boundaries",
        }
        question = SimpleNamespace(viva_topic_name="Architecture")

        self.assertIs(
            resolve_answered_topic(question, [architecture]),
            architecture,
        )

    def test_unified_state_routes_individual_criteria_to_student(self):
        group = SessionState(total_turns=2, intent_history=["clarifying"])
        student = SessionState()
        group_ability = group.get_or_init_bkt("group-criterion")
        student_ability = student.get_or_init_bkt("individual-criterion")
        group.get_or_init_bkt("individual-criterion")
        group_coverage = group.get_or_init_coverage("group-criterion")
        student_coverage = student.get_or_init_coverage("individual-criterion")
        group.get_or_init_coverage("individual-criterion")

        unified = build_unified_state(
            group,
            student,
            [
                {"id": "group-criterion", "is_individual": False},
                {"id": "individual-criterion", "is_individual": True},
            ],
        )

        self.assertIs(unified.bkt_states["group-criterion"], group_ability)
        self.assertIs(
            unified.bkt_states["individual-criterion"], student_ability
        )
        self.assertIs(unified.coverage["group-criterion"], group_coverage)
        self.assertIs(
            unified.coverage["individual-criterion"], student_coverage
        )
        self.assertEqual(unified.total_turns, 2)
        self.assertEqual(unified.intent_history, ["clarifying"])

    def test_topic_selection_prefers_first_uncovered_topic(self):
        state = SessionState(total_turns=1)
        first = state.get_or_init_coverage("criterion-1")
        first.turns = 2
        first.correct_turns = 2
        state.get_or_init_coverage("criterion-2")
        topics = [
            {
                "topic_name": "Completed",
                "source_criteria_ids": ["criterion-1"],
                "suggested_questions": 2,
            },
            {
                "topic_name": "Next",
                "source_criteria_ids": ["criterion-2"],
                "suggested_questions": 2,
            },
        ]

        selected = pick_next_topic(
            topics,
            state,
            SimpleNamespace(max_total_questions=10),
        )

        self.assertEqual(selected["topic_name"], "Next")

    def test_weak_first_topic_does_not_block_breadth_coverage(self):
        state = SessionState(total_turns=1)
        first = state.get_or_init_coverage(
            "criterion-1",
            questions_to_ask=3,
        )
        first.turns = 1
        first.correct_turns = 0
        state.get_or_init_bkt("criterion-1")
        state.get_or_init_coverage("criterion-2", questions_to_ask=3)
        state.get_or_init_bkt("criterion-2")
        topics = [
            {
                "topic_name": "Weak first topic",
                "source_criteria_ids": ["criterion-1"],
                "suggested_questions": 3,
            },
            {
                "topic_name": "Unattempted topic",
                "source_criteria_ids": ["criterion-2"],
                "suggested_questions": 3,
            },
        ]

        selected = pick_next_topic(
            topics,
            state,
            SimpleNamespace(max_total_questions=10),
        )

        self.assertEqual(selected["topic_name"], "Unattempted topic")

    def test_adaptive_revisit_prefers_lowest_mastery_after_coverage(self):
        state = SessionState(total_turns=4)
        for criterion_id in ("criterion-1", "criterion-2"):
            coverage = state.get_or_init_coverage(
                criterion_id,
                questions_to_ask=1,
            )
            coverage.turns = 2
            ability = state.get_or_init_bkt(criterion_id)
            ability.turns = 2
            ability.sigma2 = 0.8

        state.bkt_states["criterion-1"].mu = 0.2
        state.bkt_states["criterion-2"].mu = -1.5
        topics = [
            {
                "topic_name": "Stronger",
                "source_criteria_ids": ["criterion-1"],
            },
            {
                "topic_name": "Weaker",
                "source_criteria_ids": ["criterion-2"],
            },
        ]

        selected = pick_next_topic(
            topics,
            state,
            SimpleNamespace(max_total_questions=10),
        )

        self.assertEqual(selected["topic_name"], "Weaker")

    def test_current_bloom_to_difficulty_mapping_is_preserved(self):
        self.assertEqual(_bloom_to_difficulty("Remember"), "easy")
        self.assertEqual(_bloom_to_difficulty("Analyze"), "medium")
        self.assertEqual(_bloom_to_difficulty("Create"), "hard")

    @patch(
        "viva_evaluator.services.pipeline.session_state.save_session_state"
    )
    @patch.object(turn_pipeline, "assess_answer")
    @patch.object(turn_pipeline, "load_state_bundle")
    @patch.object(turn_pipeline, "resolve_answered_topic")
    @patch.object(turn_pipeline, "load_viva_topics")
    @patch.object(turn_pipeline, "load_rubric")
    def test_public_pipeline_returns_stable_clarification_shape(
        self,
        load_rubric_mock,
        load_topics_mock,
        resolve_topic_mock,
        load_bundle_mock,
        assess_mock,
        save_state_mock,
    ):
        topic = {
            "topic_name": "Architecture",
            "topic_focus": "Boundaries",
            "source_criteria_ids": ["criterion-1"],
            "suggested_questions": 2,
        }
        group_state = SessionState(clarification_streak=0)
        load_rubric_mock.return_value = [
            {
                "id": "criterion-1",
                "is_individual": False,
                "questions_to_ask": 2,
            }
        ]
        load_topics_mock.return_value = [topic]
        resolve_topic_mock.return_value = topic
        load_bundle_mock.return_value = SimpleNamespace(
            group_state=group_state,
            student_state=None,
            active_state=group_state,
            unified_state=group_state,
        )
        assess_mock.return_value = AnswerAssessment(
            retrieval={"chunks": []},
            transcript_recent=[],
            triage={
                "label": "GARBLED_TRANSCRIPTION",
                "rationale": "audio was unclear",
            },
            clarification_required=True,
            is_restate=True,
        )
        session = SimpleNamespace(id="session-1", project=SimpleNamespace())
        question = SimpleNamespace(
            question_text="Why this architecture?",
            blooms_level="Analyze",
        )

        result = turn_pipeline.process_answer_and_pick_next(
            session=session,
            submission=SimpleNamespace(),
            prev_question_obj=question,
            student_answer="garbled audio",
            speech_metrics={"pause_count": 1},
        )

        self.assertTrue(result["clarification"])
        self.assertEqual(result["clarification_attempt"], 1)
        self.assertEqual(
            result["clarified_question_payload"]["question_data"]["question_text"],
            "Why this architecture?",
        )
        self.assertEqual(save_state_mock.call_count, 0)
        self.assertIs(result["_state_bundle"].group_state, group_state)

    def test_public_scored_turn_plans_from_post_answer_mastery(self):
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
                "hints": ["Ask for concrete architectural evidence"],
            }
        ]
        group_state = SessionState()
        group_state.get_or_init_coverage("criterion-1", questions_to_ask=2)
        ability = group_state.get_or_init_bkt("criterion-1")
        mastery_before_answer = ability.p_lt
        bundle = SessionStateBundle(
            group_state=group_state,
            student_state=None,
            active_state=group_state,
            unified_state=build_unified_state(group_state, None, rubric),
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
        adjusted_analysis = {
            **assessment.analysis,
            "soft_score": 0.95,
            "charitable": {"applied": True},
        }
        adjusted_assessment = FairnessAdjustedAssessment(
            analysis=adjusted_analysis,
            soft_score=0.95,
            correctness=0.9,
        )
        questions = MagicMock()
        questions.order_by.return_value.values_list.return_value.__getitem__.return_value = []
        questions.filter.return_value.count.return_value = 1
        session = SimpleNamespace(
            id="session-1",
            project=SimpleNamespace(id="project-1"),
            viva_questions=questions,
            max_total_questions=10,
        )
        question = SimpleNamespace(
            question_text="Why this architecture?",
            blooms_level="Analyze",
        )
        strategy = {
            "bloom_level": "Create",
            "socratic_intent": "probing_evidence",
            "intent_prompt_hint": "Ask for specific evidence.",
            "rationale": "characterization",
        }
        generated_question = {
            "question_text": "What evidence supports that boundary?",
            "blooms_level": "Create",
            "difficulty": "hard",
            "attempts": 1,
        }
        continue_decision = TerminationDecision(
            should_end=False,
            reason="continue",
            coverage_met=False,
            min_turns_met=False,
            bkt_converged=False,
            hard_cap_hit=False,
        )

        with (
            patch.object(turn_pipeline, "load_rubric", return_value=rubric),
            patch.object(turn_pipeline, "load_viva_topics", return_value=[topic]),
            patch.object(turn_pipeline, "resolve_answered_topic", return_value=topic),
            patch.object(turn_pipeline, "load_state_bundle", return_value=bundle),
            patch.object(turn_pipeline, "assess_answer", return_value=assessment),
            patch.object(
                turn_pipeline,
                "apply_fairness_adjustments",
                return_value=adjusted_assessment,
            ),
            patch(
                "viva_evaluator.services.agents.strategist.select_strategy",
                return_value=strategy,
            ) as select_strategy_mock,
            patch(
                "viva_evaluator.services.agents.generate_anchored_question",
                return_value=generated_question,
            ) as generate_question_mock,
            patch(
                "viva_evaluator.services.rag.retrieval.retrieve_module_materials",
                return_value=[],
            ),
            patch(
                "viva_evaluator.services.pipeline.termination.should_terminate",
                return_value=continue_decision,
            ),
            patch(
                "viva_evaluator.services.pipeline.session_state.save_session_state"
            ),
        ):
            result = turn_pipeline.process_answer_and_pick_next(
                session=session,
                submission=SimpleNamespace(),
                prev_question_obj=question,
                student_answer="It separates responsibilities.",
        )

        strategist_input = select_strategy_mock.call_args.args[0]
        self.assertNotEqual(
            group_state.bkt_states["criterion-1"].p_lt,
            mastery_before_answer,
        )
        self.assertEqual(
            strategist_input.p_lt,
            group_state.bkt_states["criterion-1"].p_lt,
        )
        self.assertIs(strategist_input.analysis, adjusted_analysis)
        questioner_input = generate_question_mock.call_args.args[0]
        self.assertEqual(questioner_input.target_bloom, "Create")
        self.assertEqual(questioner_input.socratic_intent, "probing_evidence")
        self.assertEqual(
            questioner_input.intent_prompt_hint,
            "Ask for specific evidence.",
        )
        self.assertEqual(
            questioner_input.question_hints,
            ["Ask for concrete architectural evidence"],
        )
        self.assertFalse(result["session_complete"])
        self.assertEqual(
            result["next_question_payload"]["p_lt"],
            group_state.bkt_states["criterion-1"].p_lt,
        )
        self.assertEqual(
            result["next_question_payload"]["question_data"],
            generated_question,
        )
