"""Conditional fairness checks and authoritative score adjustment."""

import re
from concurrent.futures import Executor, Future
from typing import Callable, Dict, Optional

from viva_evaluator.services.pipeline.contracts import (
    AnswerAssessment,
    FairnessAdjustedAssessment,
    FairnessCheckPlan,
    FairnessVerdicts,
)
from viva_evaluator.services.llm_telemetry import submit_with_telemetry_context


CHARITABLE_BAND = (0.45, 0.58)
CHARITABLE_FLOOR = 0.65
CONSISTENCY_REVIEW_THRESHOLD = 0.35
CONSISTENCY_NEUTRAL = 0.80
SELF_CORRECTION_TRIGGER_MAX = 0.65
SELF_CORRECTION_FLOOR = 0.65
MIN_CHARITABLE_ANSWER_WORDS = 4
_CORRECTION_SIGNAL = re.compile(
    r"\b(?:actually|correction|i mean|rather|let me correct|to clarify|"
    r"instead|sorry|wait|more precisely|i should have said)\b",
    re.IGNORECASE,
)


def plan_fairness_checks(
    assessment: AnswerAssessment,
    *,
    answered_topic: Optional[Dict] = None,
    student_answer: str = "",
) -> FairnessCheckPlan:
    """Determine which conditional rescue checks are required."""
    if assessment.analysis is None:
        raise ValueError("A scored assessment is required for fairness planning.")

    consistency = assessment.analysis.get("consistency") or {}
    consistency_score = float(consistency.get("score", 1.0))
    correctness = float(assessment.correctness)
    soft_score = float(assessment.soft_score)
    prior_answers = [
        pair
        for pair in assessment.transcript_recent
        if str(pair.get("answer_text") or "").strip()
    ]
    consistency_source = str(consistency.get("evidence_source") or "")
    check_consistency = (
        consistency_score < CONSISTENCY_REVIEW_THRESHOLD
        and bool(str(consistency.get("evidence_quote") or "").strip())
        and (bool(prior_answers) or consistency_source == "retrieved")
    )

    answer_word_count = len(str(student_answer or "").split())
    check_charitable = (
        CHARITABLE_BAND[0] <= correctness <= CHARITABLE_BAND[1]
        and soft_score < CHARITABLE_FLOOR
        and answer_word_count >= MIN_CHARITABLE_ANSWER_WORDS
        and bool(assessment.retrieval.get("chunks") or [])
    )

    previous_answer = ""
    has_correction_signal = bool(_CORRECTION_SIGNAL.search(student_answer or ""))
    check_self_correction = (
        soft_score < SELF_CORRECTION_TRIGGER_MAX
        and has_correction_signal
        and answered_topic is not None
    )
    if check_self_correction and answered_topic is not None:
        for pair in reversed(assessment.transcript_recent):
            if (
                pair.get("answer_text")
                and _same_topic(pair, answered_topic)
            ):
                previous_answer = str(pair["answer_text"])
                break
        if not previous_answer:
            check_self_correction = False

    requested_checks = tuple(
        check_name
        for check_name, enabled in (
            ("consistency", check_consistency),
            ("charitable", check_charitable),
            ("self_correction", check_self_correction),
        )
        if enabled
    )
    routing_reasons = []
    if check_consistency:
        routing_reasons.append("low_consistency_with_evidence")
    if check_charitable:
        routing_reasons.append("borderline_grounded_answer")
    if check_self_correction:
        routing_reasons.append("same_topic_explicit_correction")

    return FairnessCheckPlan(
        check_consistency=check_consistency,
        check_charitable_interpretation=check_charitable,
        check_self_correction=check_self_correction,
        consistency_score=consistency_score,
        previous_answer=previous_answer,
        requested_checks=requested_checks,
        routing_reasons=tuple(routing_reasons),
        max_llm_calls=1,
    )


def submit_fairness_checks(
    executor: Executor,
    *,
    plan: FairnessCheckPlan,
    assessment: AnswerAssessment,
    previous_question,
    student_answer: str,
    answered_topic: Dict,
) -> Dict[str, Future]:
    """Submit all selected checks in one bounded structured review."""
    futures: Dict[str, Future] = {}

    if not plan.requested_checks or plan.max_llm_calls < 1:
        return futures

    from viva_evaluator.services.agents.fairness_review import (
        FairnessReviewInput,
        review_fairness,
    )

    consistency = assessment.analysis.get("consistency") or {}
    futures["fairness_review"] = submit_with_telemetry_context(
        executor,
        review_fairness,
        FairnessReviewInput(
            question_text=previous_question.question_text,
            student_answer=student_answer,
            criterion_name=answered_topic["topic_name"],
            criterion_description=answered_topic.get("topic_focus") or "",
            retrieved_chunks=assessment.retrieval.get("chunks") or [],
            transcript_recent=assessment.transcript_recent,
            previous_answer=plan.previous_answer,
            consistency_evidence=consistency.get("evidence_quote", ""),
            requested_checks=plan.requested_checks,
        ),
    )

    return futures


def resolve_fairness_futures(futures: Dict[str, Future]) -> FairnessVerdicts:
    """Resolve submitted checks into a stable stage contract."""
    review_future = futures.get("fairness_review")
    if review_future is None:
        return FairnessVerdicts()
    results = review_future.result()
    return FairnessVerdicts(
        consistency=results.get("consistency"),
        charitable=results.get("charitable"),
        self_correction=results.get("self_correction"),
    )


def apply_fairness_adjustments(
    assessment: AnswerAssessment,
    plan: FairnessCheckPlan,
    verdicts: FairnessVerdicts,
    marker: Optional[Callable[[str], None]] = None,
) -> FairnessAdjustedAssessment:
    """Apply fairness verdicts asymmetrically: they may only rescue scores."""
    from viva_evaluator.services.agents.analyzer import recompute_soft_score

    if assessment.analysis is None:
        raise ValueError("A scored assessment is required for fairness adjustment.")

    analysis = assessment.analysis
    soft_score = float(assessment.soft_score)
    correctness = float(assessment.correctness)
    analysis["fairness_routing"] = {
        "requested_checks": list(plan.requested_checks),
        "routing_reasons": list(plan.routing_reasons),
        "llm_calls": 1 if plan.requested_checks else 0,
        "max_llm_calls": plan.max_llm_calls,
    }

    if plan.check_consistency and verdicts.consistency is not None:
        verdict = verdicts.consistency
        if not verdict["material"]:
            analysis["consistency"]["score"] = max(
                plan.consistency_score,
                CONSISTENCY_NEUTRAL,
            )
            analysis["consistency_adjustment"] = {
                "applied": True,
                "original": round(plan.consistency_score, 4),
                "rationale": verdict["rationale"],
            }
            soft_score = recompute_soft_score(analysis)
            analysis["soft_score"] = soft_score
        else:
            analysis["consistency_adjustment"] = {
                "applied": False,
                "material": True,
                "rationale": verdict["rationale"],
            }
        if marker:
            marker("B.1:consistency")

    if plan.check_charitable_interpretation and verdicts.charitable is not None:
        charitable = verdicts.charitable
        if charitable["understanding_sound"] and soft_score < CHARITABLE_FLOOR:
            original_soft = soft_score
            soft_score = CHARITABLE_FLOOR
            analysis["charitable"] = {
                "applied": True,
                "original_soft": round(original_soft, 4),
                "adjusted_soft": CHARITABLE_FLOOR,
                "rationale": charitable["rationale"],
            }
            analysis["soft_score"] = soft_score
        else:
            analysis["charitable"] = {
                "applied": False,
                "rationale": charitable["rationale"],
            }

    if plan.check_self_correction and verdicts.self_correction is not None:
        self_correction = verdicts.self_correction
        if (
            self_correction["is_correction"]
            and self_correction["improved"]
            and soft_score < SELF_CORRECTION_FLOOR
        ):
            original_soft = soft_score
            soft_score = SELF_CORRECTION_FLOOR
            analysis["self_correction"] = {
                "applied": True,
                "original_soft": round(original_soft, 4),
                "adjusted_soft": SELF_CORRECTION_FLOOR,
                "rationale": self_correction["rationale"],
            }
            analysis["soft_score"] = soft_score
        else:
            analysis["self_correction"] = {
                "applied": False,
                "rationale": self_correction["rationale"],
            }

    return FairnessAdjustedAssessment(
        analysis=analysis,
        soft_score=soft_score,
        correctness=correctness,
    )


def _same_topic(transcript_pair: Dict, answered_topic: Dict) -> bool:
    current_ids = {
        str(criterion_id)
        for criterion_id in answered_topic.get("source_criteria_ids") or []
    }
    previous_ids = {
        str(criterion_id)
        for criterion_id in transcript_pair.get("source_criteria_ids") or []
    }
    if current_ids and previous_ids and current_ids.intersection(previous_ids):
        return True
    previous_name = str(transcript_pair.get("topic_name") or "").strip().lower()
    current_name = str(answered_topic.get("topic_name") or "").strip().lower()
    return bool(previous_name and current_name and previous_name == current_name)
