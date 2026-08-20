"""Shared application service for initial and subsequent viva questions."""

from typing import Dict, Optional

from viva_evaluator.services.agents.questioner import QuestionerInput
from viva_evaluator.services.llm_telemetry import collect_llm_telemetry
from viva_evaluator.services.pipeline.context import load_rubric, load_viva_topics
from viva_evaluator.services.pipeline.presenter import (
    build_detailed_analysis,
    build_rubric_payload,
    present_duplicate,
    present_opening_question,
    present_resumed_session,
    present_turn,
)
from viva_evaluator.services.pipeline.session_state import load_session_state
from viva_evaluator.services.pipeline.state_bundle import (
    SessionStateBundle,
    build_unified_state,
)
from viva_evaluator.services.pipeline.stages import (
    generate_question_candidate,
    plan_next_question,
    validate_question_candidate,
)
from viva_evaluator.services.pipeline.stages.persistence import (
    find_existing_answer,
    find_next_unanswered_question,
    persist_opening_question,
    persist_turn,
)
from viva_evaluator.services.pipeline.turn_pipeline import (
    process_answer_and_pick_next,
)
from viva_evaluator.services.tts import bind_question_tts_audit


class VivaPipelineInputError(ValueError):
    """A client-correctable pipeline precondition failure."""


class VivaPipeline:
    """Coordinate pure/AI stages first, then one atomic persistence stage."""

    def start_session(self, *, session, submission) -> Dict:
        latest_question = session.viva_questions.order_by("question_order").last()
        if session.status == "in_progress" and latest_question is not None:
            return present_resumed_session(session, latest_question)

        with collect_llm_telemetry(
            trace_kind="opening_question",
            session_id=str(getattr(session, "id", "")),
        ) as telemetry:
            rubric = load_rubric(session.project)
            if not rubric:
                raise VivaPipelineInputError(
                    "No rubric configured for this project. Please add rubric "
                    "criteria before starting a viva session."
                )

            topics = load_viva_topics(session)
            if not topics:
                raise VivaPipelineInputError(
                    "No viva topics could be built from the rubric."
                )

            group_state = load_session_state(session, speaker_id="group")
            for criterion in rubric:
                criterion_id = str(criterion["id"])
                group_state.get_or_init_coverage(
                    criterion_id,
                    questions_to_ask=int(criterion["questions_to_ask"]),
                )
                group_state.get_or_init_bkt(criterion_id)

            unified_state = build_unified_state(group_state, None, rubric)
            state_bundle = SessionStateBundle(
                group_state=group_state,
                student_state=None,
                active_state=group_state,
                unified_state=unified_state,
                speaker_id="group",
                is_individual_topic=False,
            )
            planned = plan_next_question(
                session=session,
                submission=submission,
                topics=topics,
                rubric=rubric,
                unified_state=unified_state,
                intent_history=group_state.intent_history,
                adjusted_analysis={},
                speech_confidence={},
            )
            group_state.intent_history.append(planned.plan.socratic_intent)
            group_state.intent_history = group_state.intent_history[-30:]

            question_input = _questioner_input_from_plan(
                session=session,
                planned=planned,
            )
            candidate = generate_question_candidate(question_input)
            validated = validate_question_candidate(question_input, candidate)
            telemetry_summary = telemetry.snapshot()

        persisted = persist_opening_question(
            session,
            planned,
            validated,
            state_bundle,
            llm_telemetry=telemetry_summary,
        )
        if persisted.duplicate:
            return present_resumed_session(session, persisted.question)
        bind_question_tts_audit(persisted.question)
        return present_opening_question(
            session,
            persisted.question,
            planned,
            validated,
        )

    def submit_answer(
        self,
        *,
        session,
        submission,
        question,
        answer_text: str,
        speech_metrics: Optional[Dict],
        speaker_id: str,
        student_profile,
    ) -> Dict:
        existing = find_existing_answer(question, student_profile)
        if existing is not None:
            next_question = find_next_unanswered_question(session, question)
            return present_duplicate(session, next_question)

        with collect_llm_telemetry(
            trace_kind="answer_turn",
            session_id=str(getattr(session, "id", "")),
            question_id=str(getattr(question, "id", "")),
        ) as telemetry:
            computation = process_answer_and_pick_next(
                session=session,
                submission=submission,
                prev_question_obj=question,
                student_answer=answer_text,
                speech_metrics=speech_metrics,
                speaker_id=speaker_id,
                examiner_paused=session.examiner_paused,
            )
            computation["llm_telemetry"] = telemetry.snapshot()

        detailed_analysis = None
        if not computation.get("clarification"):
            rubric_payload = build_rubric_payload(computation)
            detailed_analysis = build_detailed_analysis(
                computation,
                rubric_payload,
            )

        persisted = persist_turn(
            session=session,
            previous_question=question,
            student_profile=student_profile,
            answer_text=answer_text,
            computation=computation,
            detailed_analysis=detailed_analysis,
            deduplication_key=(
                f"student:{student_profile.id}"
                if student_profile is not None
                else "group"
            ),
        )
        if persisted.duplicate:
            return present_duplicate(session, persisted.question)
        if persisted.question is not None:
            bind_question_tts_audit(persisted.question)
        return present_turn(computation, persisted)


def _questioner_input_from_plan(*, session, planned) -> QuestionerInput:
    plan = planned.plan
    grounding = planned.grounding
    return QuestionerInput(
        criterion_name=plan.topic.name,
        criterion_description=plan.topic.focus,
        retrieved_chunks=grounding.retrieval.get("chunks") or [],
        module_chunks=list(grounding.module_chunks),
        kg_signals=grounding.retrieval,
        difficulty=plan.difficulty,
        target_bloom=plan.target_bloom,
        socratic_intent=plan.socratic_intent,
        intent_prompt_hint=plan.intent_prompt_hint,
        question_hints=list(grounding.question_hints),
        recent_questions=list(grounding.recent_questions),
        is_first_question=True,
        question_number_in_criterion=plan.question_number_in_topic,
        weak_grounding=grounding.weak_grounding,
        session_id=str(session.id),
        evidence_package=grounding.evidence_package,
    )
