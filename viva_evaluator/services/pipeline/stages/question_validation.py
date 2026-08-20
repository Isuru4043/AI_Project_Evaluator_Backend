"""Fail-closed Tier-1 and Critic validation for viva questions."""

import logging
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from django.conf import settings

from viva_evaluator.services.agents.tier1_validator import (
    Tier1Result,
    validate_question,
)
from viva_evaluator.services.pipeline.contracts import (
    QuestionCandidate,
    ValidatedQuestion,
)
from viva_evaluator.services.pipeline.evidence import (
    ensure_question_evidence_package,
)
from viva_evaluator.services.pipeline.exceptions import (
    QuestionGenerationUnavailableError,
)
from viva_evaluator.services.pipeline.stages.candidate_generation import (
    candidate_hash,
    generate_question_candidate,
)
from viva_evaluator.services.tts import (
    discard_speculative_tts,
    finalize_question_tts,
    start_speculative_tts,
)


logger = logging.getLogger(__name__)

CRITIC_MAX_RETRIES = 1


@dataclass(frozen=True)
class _CriticValidationOutcome:
    candidate: QuestionCandidate
    tier1: Tier1Result
    passed: bool
    unavailable: bool
    critique: str
    scores: Dict
    regeneration_attempts: int
    unavailable_reason: str = ""


def validate_question_candidate(
    questioner_input,
    candidate: QuestionCandidate,
    *,
    max_retries: int = 1,
    enable_critic: bool = True,
) -> ValidatedQuestion:
    """Return only a Tier-1-valid candidate or a verified safe fallback.

    Critic service failure is represented explicitly as degraded validation.
    A genuine Critic quality rejection is never accepted: after its bounded
    retry is exhausted, validation switches to a deterministic fallback.
    """
    current = candidate
    tier1 = _validate_candidate_tier1(questioner_input, current)
    attempts = 1

    for _ in range(max(0, max_retries)):
        if tier1.passed:
            break
        logger.info(
            "question validation: Tier 1 failed (%s); regenerating.",
            tier1.reason_string(),
        )
        current = generate_question_candidate(
            questioner_input,
            retry_reason=tier1.reason_string(),
            fallback_question=current.question_text,
        )
        tier1 = _validate_candidate_tier1(questioner_input, current)
        attempts += 1

    if not tier1.passed:
        return _attach_tts(
            _safe_fallback(
                questioner_input,
                attempts=attempts,
                reason=f"tier1_rejected:{tier1.reason_string()}",
                critic_available=True,
            ),
            None,
        )

    # The final quality decision is still pending, but Tier 1 has established
    # that this is a viable candidate. Speech can now run beside the Critic.
    speculative_tts = start_speculative_tts(
        current.question_text,
        current.candidate_hash,
    )

    evidence_package = ensure_question_evidence_package(questioner_input)
    high_confidence_tier1 = (
        18 <= tier1.word_count <= 45
        and tier1.similarity_to_recent < 0.65
        and not questioner_input.clarify_mode
        and not _requires_critic(
            questioner_input,
            current,
            evidence_package,
        )
    )

    if not enable_critic or high_confidence_tier1:
        return _attach_tts(
            _validated_from_candidate(
                current,
                tier1,
                attempts=attempts,
                validation_status="tier1_only_policy",
            ),
            speculative_tts,
        )

    try:
        critic = _run_critic_validation(questioner_input, current, tier1)
    except Exception:
        discard_speculative_tts(speculative_tts)
        raise
    attempts += critic.regeneration_attempts

    if critic.unavailable:
        logger.warning(
            "question validation degraded because Critic is unavailable: %s",
            critic.unavailable_reason,
        )
        unavailable_reason = (
            critic.unavailable_reason or "critic_validation_unavailable"
        )
        unavailable_policy = _critic_unavailable_policy()
        if unavailable_policy == "safe_fallback":
            return _attach_tts(
                _safe_fallback(
                    questioner_input,
                    attempts=attempts,
                    reason=f"critic_unavailable:{unavailable_reason}",
                    critic_available=False,
                ),
                speculative_tts,
            )
        if unavailable_policy == "fail_closed":
            discard_speculative_tts(speculative_tts)
            raise QuestionGenerationUnavailableError(
                "Critic validation is unavailable. Please retry the request."
            )
        return _attach_tts(
            _validated_from_candidate(
                critic.candidate,
                critic.tier1,
                attempts=attempts,
                critic_passed=False,
                critic_critique=critic.critique,
                critic_scores={},
                validation_status="critic_unavailable",
                validation_degraded=True,
                degradation_reason=unavailable_reason,
                critic_available=False,
            ),
            speculative_tts,
        )

    if critic.passed:
        return _attach_tts(
            _validated_from_candidate(
                critic.candidate,
                critic.tier1,
                attempts=attempts,
                critic_passed=True,
                critic_scores=critic.scores,
                validation_status="fully_validated",
            ),
            speculative_tts,
        )

    return _attach_tts(
        _safe_fallback(
            questioner_input,
            attempts=attempts,
            reason=f"critic_rejected:{critic.critique or 'quality_threshold_failed'}",
            critic_available=True,
        ),
        speculative_tts,
    )


def _attach_tts(validated: ValidatedQuestion, speculative_ticket) -> ValidatedQuestion:
    """Attach JSON-safe non-blocking audio state to the accepted question."""
    try:
        metadata = finalize_question_tts(
            validated.question_text,
            validated.candidate_hash,
            speculative_ticket,
        )
    except Exception:
        # Voice quality is optional; it must never turn a valid question into a
        # failed request or delay delivery of its text.
        logger.exception("failed to finalize speculative question TTS")
        metadata = {"enabled": True, "status": "failed"}
    status = str(metadata.get("status") or "failed")
    if status not in {"disabled", "pending", "ready", "failed"}:
        status = "failed"
    return replace(
        validated,
        tts_status=status,
        tts_metadata=dict(metadata),
    )


def _validated_from_candidate(
    candidate: QuestionCandidate,
    tier1: Tier1Result,
    *,
    attempts: int,
    critic_passed: Optional[bool] = None,
    critic_critique: str = "",
    critic_scores: Optional[Dict] = None,
    validation_status="tier1_only_policy",
    validation_degraded: bool = False,
    degradation_reason: str = "",
    critic_available: bool = True,
) -> ValidatedQuestion:
    validated = ValidatedQuestion(
        question_text=candidate.question_text,
        blooms_level=candidate.blooms_level,
        difficulty=candidate.difficulty,
        tier1_passed=tier1.passed,
        tier1_failures=tuple(tier1.failures),
        critic_passed=critic_passed,
        critic_critique=critic_critique,
        critic_scores=critic_scores or {},
        attempts=attempts,
        candidate_hash=candidate.candidate_hash,
        socratic_intent=candidate.socratic_intent,
        source_reference_ids=candidate.source_reference_ids,
        schema_failures=candidate.schema_failures,
        validation_status=validation_status,
        validation_degraded=validation_degraded,
        degradation_reason=degradation_reason[:300],
        fallback_used=False,
        critic_available=critic_available,
    )
    _log_validation_outcome(validated)
    return validated


def _run_critic_validation(
    questioner_input,
    candidate: QuestionCandidate,
    tier1: Tier1Result,
) -> _CriticValidationOutcome:
    from viva_evaluator.services.agents.critic import CriticInput, critique_question

    evidence_package = ensure_question_evidence_package(questioner_input)
    current = candidate
    current_tier1 = tier1
    regeneration_attempts = 0
    last_critique = ""
    last_scores: Dict = {}

    for attempt_index in range(CRITIC_MAX_RETRIES + 1):
        result = critique_question(
            CriticInput(
                question_text=current.question_text,
                target_bloom=current.blooms_level,
                target_intent=(
                    questioner_input.socratic_intent
                    or _intent_label_from_kg(questioner_input.kg_signals)
                ),
                evidence_package=evidence_package,
                source_reference_ids=list(current.source_reference_ids),
                retrieved_chunks=questioner_input.retrieved_chunks,
                module_chunks=questioner_input.module_chunks,
                student_last_answer=questioner_input.previous_answer,
            )
        )

        if result.get("_critic_unavailable") is True:
            return _CriticValidationOutcome(
                candidate=current,
                tier1=current_tier1,
                passed=False,
                unavailable=True,
                critique=result.get("critique", "") or "",
                scores={},
                regeneration_attempts=regeneration_attempts,
                unavailable_reason=(
                    result.get("unavailable_reason")
                    or "critic_validation_unavailable"
                ),
            )

        last_critique = result.get("critique", "") or ""
        last_scores = _critic_scores(result)
        if result.get("passed") is True:
            return _CriticValidationOutcome(
                candidate=current,
                tier1=current_tier1,
                passed=True,
                unavailable=False,
                critique="",
                scores=last_scores,
                regeneration_attempts=regeneration_attempts,
            )

        if attempt_index >= CRITIC_MAX_RETRIES:
            break

        retry = generate_question_candidate(
            questioner_input,
            retry_reason=f"critic feedback: {last_critique}",
            fallback_question=current.question_text,
        )
        retry_tier1 = _validate_candidate_tier1(questioner_input, retry)
        regeneration_attempts += 1
        if not retry_tier1.passed:
            last_critique = (
                f"Critic retry failed Tier 1: {retry_tier1.reason_string()}"
            )
            break
        current = retry
        current_tier1 = retry_tier1

    return _CriticValidationOutcome(
        candidate=current,
        tier1=current_tier1,
        passed=False,
        unavailable=False,
        critique=last_critique,
        scores=last_scores,
        regeneration_attempts=regeneration_attempts,
    )


def _critic_scores(result: Dict) -> Dict:
    return {
        "specificity": result.get("specificity_score", 0.0),
        "bloom_alignment": result.get("bloom_alignment_score", 0.0),
        "boundary_check": result.get("boundary_check_score", 0.0),
        "hallucination": result.get("hallucination_flag", False),
        "conversational_flow": result.get(
            "conversational_flow_score",
            0.0,
        ),
        "source_reference_support": result.get(
            "source_reference_support_score",
            0.0,
        ),
    }


_SAFE_FALLBACKS = {
    "Remember": (
        "Thinking about your project, what key decision did you make for this part of the work?",
        "In your project, what was the main purpose of this part of your work?",
        "From your project work, what important choice can you recall about this area?",
    ),
    "Understand": (
        "Thinking about your project, how would you explain the purpose of this part of your work?",
        "In your project, how would you describe this part of the work in your own words?",
        "Based on your project, how would you explain why this area matters to the overall solution?",
    ),
    "Apply": (
        "Thinking about your project, how would you apply this part of your work in a realistic scenario?",
        "In your project, how would you use this part of the work to handle a practical problem?",
        "Based on your project experience, how would you apply this idea in a new situation?",
    ),
    "Analyze": (
        "Thinking about your project, how does this part of your work interact with the rest of your solution?",
        "In your project, what relationship between this part and another component most affects its behavior?",
        "Based on your project, how would you break down the factors that shaped this part of your work?",
    ),
    "Evaluate": (
        "Thinking about your project, what evidence best supports your most important decision in this part of your work?",
        "In your project, which decision in this area is strongest when judged against its trade-offs?",
        "Based on your project, how would you justify the quality of your main decision in this area?",
    ),
    "Create": (
        "Thinking about your project, how would you redesign this part of your work to improve its reliability?",
        "In your project, what new design would you propose to improve this part of your solution?",
        "Based on your project, how would you create a better version of this part while preserving its purpose?",
    ),
}


def _safe_fallback(
    questioner_input,
    *,
    attempts: int,
    reason: str,
    critic_available: bool,
) -> ValidatedQuestion:
    """Choose the first deterministic template that independently passes Tier 1."""
    target_bloom = questioner_input.target_bloom or "Analyze"
    templates = _SAFE_FALLBACKS.get(target_bloom, _SAFE_FALLBACKS["Analyze"])
    for question_text in templates:
        tier1 = validate_question(
            question_text,
            recent_questions=questioner_input.recent_questions,
        )
        if not tier1.passed:
            continue
        logger.warning("using deterministic safe question fallback: %s", reason)
        validated = ValidatedQuestion(
            question_text=question_text,
            blooms_level=target_bloom,
            difficulty=questioner_input.difficulty,
            tier1_passed=True,
            tier1_failures=(),
            critic_passed=None,
            critic_critique="",
            critic_scores={},
            attempts=attempts,
            candidate_hash=candidate_hash(question_text),
            socratic_intent=(
                questioner_input.socratic_intent or "general_probe"
            ),
            source_reference_ids=(),
            schema_failures=(),
            validation_status="safe_fallback",
            validation_degraded=True,
            degradation_reason=reason[:300],
            fallback_used=True,
            critic_available=critic_available,
        )
        _log_validation_outcome(validated)
        return validated

    logger.error("all deterministic safe fallbacks failed Tier 1: %s", reason)
    raise QuestionGenerationUnavailableError()


def _critic_unavailable_policy() -> str:
    policy = str(
        getattr(
            settings,
            "VIVA_QUESTION_CRITIC_UNAVAILABLE_POLICY",
            "degraded_tier1",
        )
    ).strip().lower()
    if policy in {"degraded_tier1", "safe_fallback", "fail_closed"}:
        return policy
    logger.error(
        "invalid VIVA_QUESTION_CRITIC_UNAVAILABLE_POLICY=%r; using fail_closed",
        policy,
    )
    return "fail_closed"


def _log_validation_outcome(validated: ValidatedQuestion) -> None:
    logger.info(
        "question_validation_outcome status=%s degraded=%s fallback=%s "
        "critic_available=%s tier1_passed=%s attempts=%d source_refs=%d",
        validated.validation_status,
        validated.validation_degraded,
        validated.fallback_used,
        validated.critic_available,
        validated.tier1_passed,
        validated.attempts,
        len(validated.source_reference_ids),
    )


def _intent_label_from_kg(kg_signals: Optional[Dict]) -> str:
    if not kg_signals:
        return "general_probe"
    if kg_signals.get("contradicts_code_alerts"):
        return "challenge_contradiction"
    if kg_signals.get("depends_on_topics"):
        return "exploring_alternatives"
    return "general_probe"


def _validate_candidate_tier1(questioner_input, candidate) -> Tier1Result:
    result = validate_question(
        candidate.question_text,
        recent_questions=questioner_input.recent_questions,
    )
    failures = list(_candidate_schema_failures(questioner_input, candidate))
    failures.extend(result.failures)
    return Tier1Result(
        passed=not failures,
        failures=failures,
        similarity_to_recent=result.similarity_to_recent,
        word_count=result.word_count,
    )


def _candidate_schema_failures(questioner_input, candidate) -> Tuple[str, ...]:
    package = ensure_question_evidence_package(questioner_input)
    failures = list(candidate.schema_failures)
    expected_bloom = questioner_input.target_bloom or candidate.blooms_level
    expected_intent = questioner_input.socratic_intent or "general_probe"

    if candidate.blooms_level != expected_bloom:
        failures.append("target_bloom_mismatch")
    if candidate.socratic_intent != expected_intent:
        failures.append("socratic_intent_mismatch")

    available_ids = set(package.evidence_ids)
    unknown_ids = sorted(set(candidate.source_reference_ids) - available_ids)
    if unknown_ids:
        failures.append(
            "unknown_source_reference_ids:" + ",".join(unknown_ids)
        )
    if (
        not package.weak_grounding
        and package.references
        and not candidate.source_reference_ids
    ):
        failures.append("missing_source_reference_ids")
    return tuple(dict.fromkeys(failures))


def _requires_critic(questioner_input, candidate, evidence_package) -> bool:
    """Return whether this candidate carries risks that require Tier 2."""
    if candidate.blooms_level in {"Evaluate", "Create"}:
        return True
    if candidate.socratic_intent in {
        "challenge_contradiction",
        "exploring_alternatives",
    }:
        return True
    if questioner_input.clarify_mode:
        return True

    cited_references = tuple(
        reference
        for reference_id in candidate.source_reference_ids
        if (reference := evidence_package.get(reference_id)) is not None
    )
    if cited_references and not evidence_package.weak_grounding:
        return True
    if any(
        reference.evidence_type
        in {
            "module_chunk",
            "kg_contradiction",
            "kg_alternative",
            "kg_dependency",
        }
        for reference in cited_references
    ):
        return True
    return any(
        reference.evidence_type == "submission_chunk"
        and reference.metadata.get("source") == "code"
        for reference in cited_references
    )
