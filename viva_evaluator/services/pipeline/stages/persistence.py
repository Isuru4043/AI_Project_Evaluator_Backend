"""Atomic database commits for opening questions and completed viva turns.

All retrieval and LLM work must finish before entering these functions.  The
transaction is deliberately short and contains only deterministic ORM writes.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from django.db import transaction
from django.utils import timezone

from viva_evaluator.services.pipeline.exceptions import (
    QuestionGenerationUnavailableError,
)


_PERSISTABLE_VALIDATION_STATUSES = {
    "fully_validated",
    "tier1_only_policy",
    "critic_unavailable",
    "safe_fallback",
}


@dataclass(frozen=True)
class PersistenceResult:
    answer: Optional[Any] = None
    question: Optional[Any] = None
    duplicate: bool = False


def find_existing_answer(question, student_profile):
    """Return an existing answer for the same question/speaker, if present."""
    queryset = question.answers.all()
    if student_profile is None:
        queryset = queryset.filter(student__isnull=True)
    else:
        queryset = queryset.filter(student=student_profile)
    return queryset.order_by("answered_at").first()


def find_next_unanswered_question(session, question):
    return (
        session.viva_questions.filter(
            question_order__gt=question.question_order,
            answers__isnull=True,
        )
        .order_by("question_order")
        .first()
    )


def persist_opening_question(
    session,
    planned_question,
    validated_question,
    state_bundle,
    llm_telemetry: Optional[Dict] = None,
):
    """Commit initial state, question, extension, and lifecycle atomically."""
    from core.models import EvaluationSession

    _assert_validated_question_safe(validated_question)

    with transaction.atomic():
        locked_session = EvaluationSession.objects.select_for_update().get(
            pk=session.pk
        )
        existing = locked_session.viva_questions.order_by("question_order").last()
        if existing is not None:
            _copy_session_fields(session, locked_session)
            return PersistenceResult(question=existing, duplicate=True)

        plan = planned_question.plan
        question = _create_question(
            locked_session,
            question_text=validated_question.question_text,
            blooms_level=validated_question.blooms_level,
            difficulty=validated_question.difficulty,
            topic=plan.topic.to_pipeline_dict(),
            validation_data=validated_question.to_legacy_dict(),
            llm_telemetry=llm_telemetry,
        )
        _save_state_bundle(locked_session, state_bundle)
        locked_session.status = EvaluationSession.Status.IN_PROGRESS
        update_fields = ["status"]
        if locked_session.actual_start is None:
            locked_session.actual_start = timezone.now()
            update_fields.append("actual_start")
        locked_session.save(update_fields=update_fields)

        _copy_session_fields(session, locked_session)
        return PersistenceResult(question=question)


def persist_turn(
    *,
    session,
    previous_question,
    student_profile,
    answer_text: str,
    computation: Dict,
    detailed_analysis: Optional[Dict] = None,
    deduplication_key: Optional[str] = None,
) -> PersistenceResult:
    """Commit one computed turn in one short transaction.

    The session/question locks plus the database speaker uniqueness constraint
    provide defense in depth for concurrent retries.
    """
    from core.models import EvaluationSession, VivaAnswer, VivaQuestion
    from viva_evaluator.models import VivaAnswerExtension

    _assert_computed_next_question_safe(computation)

    with transaction.atomic():
        locked_session = EvaluationSession.objects.select_for_update().get(
            pk=session.pk
        )
        locked_question = VivaQuestion.objects.select_for_update().get(
            pk=previous_question.pk,
            session=locked_session,
        )

        duplicate = find_existing_answer(locked_question, student_profile)
        if duplicate is not None:
            next_question = find_next_unanswered_question(
                locked_session,
                locked_question,
            )
            _copy_session_fields(session, locked_session)
            return PersistenceResult(
                answer=duplicate,
                question=next_question,
                duplicate=True,
            )

        is_clarification = bool(computation.get("clarification"))
        soft_score = computation.get("soft_score")
        answer = VivaAnswer.objects.create(
            question=locked_question,
            student=student_profile,
            deduplication_key=deduplication_key,
            transcribed_answer=answer_text,
            ai_answer_score=(
                None if is_clarification else round(float(soft_score) * 10.0, 2)
            ),
        )

        if not is_clarification:
            analysis = computation.get("analysis") or {}
            VivaAnswerExtension.objects.create(
                answer=answer,
                llm_score=round(float(soft_score) * 10.0, 2),
                llm_reasoning=analysis.get("reasoning", "") or "",
                next_difficulty_signal=_difficulty_signal(float(soft_score)),
                detailed_ai_analysis=detailed_analysis or {},
            )

        _save_state_bundle(locked_session, computation.get("_state_bundle"))

        next_question = None
        if is_clarification:
            payload = computation["clarified_question_payload"]
            question_data = payload["question_data"]
            next_question = _create_question(
                locked_session,
                question_text=question_data["question_text"],
                blooms_level=question_data.get(
                    "blooms_level",
                    payload["bloom_level"],
                ),
                difficulty=question_data.get(
                    "difficulty",
                    payload["difficulty"],
                ),
                topic=payload["topic"],
                validation_data=question_data,
                llm_telemetry=computation.get("llm_telemetry"),
            )
        elif computation.get("session_complete"):
            locked_session.status = EvaluationSession.Status.COMPLETED
            locked_session.save(update_fields=["status"])
        elif not computation.get("paused_by_examiner"):
            payload = computation.get("next_question_payload")
            if payload:
                question_data = payload["question_data"]
                next_question = _create_question(
                    locked_session,
                    question_text=question_data["question_text"],
                    blooms_level=question_data.get(
                        "blooms_level",
                        payload["bloom_level"],
                    ),
                    difficulty=question_data.get(
                        "difficulty",
                        payload["difficulty"],
                    ),
                    topic=payload["topic"],
                    validation_data=question_data,
                    llm_telemetry=computation.get("llm_telemetry"),
                )

        _copy_session_fields(session, locked_session)
        return PersistenceResult(answer=answer, question=next_question)


def _create_question(
    session,
    *,
    question_text: str,
    blooms_level: str,
    difficulty: str,
    topic: Dict,
    validation_data: Optional[Dict] = None,
    llm_telemetry: Optional[Dict] = None,
):
    from core.models import RubricCriteria, VivaQuestion
    from viva_evaluator.models import VivaQuestionExtension

    question = VivaQuestion.objects.create(
        session=session,
        question_text=question_text,
        blooms_level=blooms_level,
        question_order=session.viva_questions.count() + 1,
        question_source=VivaQuestion.QuestionSource.AI,
        viva_topic_name=topic.get("topic_name") or "General",
        source_criteria_ids=topic.get("source_criteria_ids") or [],
    )

    criterion_ids = topic.get("source_criteria_ids") or []
    criterion = (
        RubricCriteria.objects.filter(pk=criterion_ids[0]).first()
        if criterion_ids
        else None
    )
    audit = build_question_generation_audit(
        validation_data or {},
        blooms_level=blooms_level,
        difficulty=difficulty,
        llm_telemetry=llm_telemetry,
    )
    VivaQuestionExtension.objects.create(
        question=question,
        criteria=criterion,
        difficulty_level=difficulty,
        validation_status=audit["validation"]["status"],
        validation_degraded=audit["validation"]["degraded"],
        fallback_used=audit["validation"]["fallback_used"],
        generation_audit=audit,
    )
    return question


def build_question_generation_audit(
    question_data: Dict,
    *,
    blooms_level: str,
    difficulty: str,
    llm_telemetry: Optional[Dict] = None,
) -> Dict:
    """Build the versioned, JSON-safe audit record stored with a question."""
    critic_passed = question_data.get("critic_passed")
    tts_metadata = dict(question_data.get("tts_metadata") or {})
    tts_metadata.setdefault(
        "status",
        str(question_data.get("tts_status") or "disabled"),
    )
    return {
        "schema_version": 3,
        "candidate_hash": str(question_data.get("candidate_hash") or ""),
        "target_bloom": str(
            question_data.get("blooms_level") or blooms_level or ""
        ),
        "difficulty": str(question_data.get("difficulty") or difficulty or ""),
        "socratic_intent": str(
            question_data.get("socratic_intent") or ""
        ),
        "source_reference_ids": [
            str(reference_id)
            for reference_id in (
                question_data.get("source_reference_ids") or []
            )
        ],
        "tier1": {
            "passed": question_data.get("tier1_passed") is True,
            "failures": [
                str(failure)
                for failure in (question_data.get("tier1_failures") or [])
            ],
            "schema_failures": [
                str(failure)
                for failure in (question_data.get("schema_failures") or [])
            ],
        },
        "critic": {
            "ran": bool(
                question_data.get("critic_ran")
                or critic_passed is not None
            ),
            "passed": critic_passed,
            "available": question_data.get("critic_available", True) is True,
            "critique": str(question_data.get("critic_critique") or ""),
            "scores": dict(question_data.get("critic_scores") or {}),
        },
        "validation": {
            "status": str(
                question_data.get("validation_status") or "not_applicable"
            ),
            "degraded": question_data.get("validation_degraded") is True,
            "degradation_reason": str(
                question_data.get("degradation_reason") or ""
            ),
            "fallback_used": question_data.get("fallback_used") is True,
        },
        "attempts": max(0, int(question_data.get("attempts", 0) or 0)),
        "llm_telemetry": dict(
            llm_telemetry
            or question_data.get("llm_telemetry")
            or {}
        ),
        "tts": tts_metadata,
    }


def _assert_validated_question_safe(validated_question) -> None:
    if (
        not str(validated_question.question_text or "").strip()
        or validated_question.tier1_passed is not True
        or validated_question.validation_status
        not in _PERSISTABLE_VALIDATION_STATUSES
    ):
        raise QuestionGenerationUnavailableError()


def _assert_computed_next_question_safe(computation: Dict) -> None:
    payload = None
    if computation.get("clarification"):
        payload = computation.get("clarified_question_payload")
    elif (
        not computation.get("session_complete")
        and not computation.get("paused_by_examiner")
    ):
        payload = computation.get("next_question_payload")

    if payload is None:
        return
    question_data = payload.get("question_data") or {}
    if (
        not str(question_data.get("question_text") or "").strip()
        or question_data.get("tier1_passed") is not True
        or question_data.get("validation_status")
        not in _PERSISTABLE_VALIDATION_STATUSES
    ):
        raise QuestionGenerationUnavailableError()


def _save_state_bundle(session, bundle) -> None:
    if bundle is None:
        return

    raw = session.bkt_state_json or {}
    if "bkt_states" in raw or "total_turns" in raw:
        raw = {"group": raw}
    raw["group"] = bundle.group_state.to_dict()
    if bundle.student_state is not None:
        raw[bundle.speaker_id] = bundle.student_state.to_dict()
    session.bkt_state_json = raw
    session.save(update_fields=["bkt_state_json"])


def _difficulty_signal(soft_score: float) -> str:
    if soft_score < 0.4:
        return "lower"
    if soft_score < 0.7:
        return "same"
    return "higher"


def _copy_session_fields(target, source) -> None:
    target.status = source.status
    target.actual_start = source.actual_start
    target.bkt_state_json = source.bkt_state_json
