"""Answer assessment stage: retrieval, triage, analysis, and confidence."""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

from viva_evaluator.services.pipeline.context import build_recent_transcript
from viva_evaluator.services.pipeline.contracts import AnswerAssessment
from viva_evaluator.services.llm_telemetry import submit_with_telemetry_context


StageMarker = Optional[Callable[[str], None]]


def _mark(marker: StageMarker, label: str) -> None:
    if marker:
        marker(label)


def assess_answer(
    *,
    session,
    submission,
    previous_question,
    student_answer: str,
    answered_topic: Dict,
    speech_metrics: Optional[Dict],
    clarification_allowed: bool,
    marker: StageMarker = None,
) -> AnswerAssessment:
    """Assess an answer without mutating session state or writing to the DB."""
    from viva_evaluator.services.agents.analyzer import AnalyzerInput, analyze_answer
    from viva_evaluator.services.agents.response_triage import (
        CLARIFY_LABELS,
        RESTATE_LABELS,
        TriageInput,
        triage_response,
    )
    from viva_evaluator.services.confidence import analyze_speech_confidence
    from viva_evaluator.services.rag.retrieval import retrieve_hybrid_for_turn

    retrieval = retrieve_hybrid_for_turn(
        submission=submission,
        criterion_name=answered_topic["topic_name"],
        criterion_description=answered_topic["topic_focus"],
        last_answer=student_answer,
        top_k=3,
    )
    _mark(marker, "A:retrieval")

    transcript_recent = build_recent_transcript(session)
    triage_input = TriageInput(
        question_text=previous_question.question_text,
        student_answer=student_answer,
        is_spoken=bool(speech_metrics),
    )
    analyzer_input = AnalyzerInput(
        question_text=previous_question.question_text,
        student_answer=student_answer,
        topic_name=answered_topic["topic_name"],
        topic_focus=answered_topic["topic_focus"],
        retrieved_chunks=retrieval["chunks"],
        contradicts_code_alerts=(
            retrieval.get("contradicts_code_alerts") or []
        ),
        transcript_recent=transcript_recent,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        triage_future = submit_with_telemetry_context(
            executor,
            triage_response,
            triage_input,
        )
        analyzer_future = submit_with_telemetry_context(
            executor,
            analyze_answer,
            analyzer_input,
        )

        triage = triage_future.result()
        _mark(marker, "A.5:triage(parallel)")

        gate_labels = CLARIFY_LABELS | RESTATE_LABELS
        if clarification_allowed and triage["label"] in gate_labels:
            return AnswerAssessment(
                retrieval=retrieval,
                transcript_recent=transcript_recent,
                triage=triage,
                clarification_required=True,
                is_restate=triage["label"] in RESTATE_LABELS,
            )

        analysis = analyzer_future.result()
        _mark(marker, "B:analyzer(parallel)")

    soft_score = float(analysis.get("soft_score", 0.5))
    correctness = float(
        (analysis.get("correctness") or {}).get("score", 0.5)
    )
    confidence = analyze_speech_confidence(
        answer_text=student_answer,
        speech_metrics=speech_metrics,
    )
    _mark(marker, "B.5:confidence")

    return AnswerAssessment(
        retrieval=retrieval,
        transcript_recent=transcript_recent,
        triage=triage,
        analysis=analysis,
        soft_score=soft_score,
        correctness=correctness,
        speech_confidence=confidence,
    )
