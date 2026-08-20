"""Raw LLM candidate generation without validation or persistence."""

import hashlib
from typing import Optional, cast

from viva_evaluator.services.pipeline.contracts import (
    BloomLevel,
    Difficulty,
    QuestionCandidate,
)


def candidate_hash(question_text: str) -> str:
    """Stable identity used by validation and speculative TTS consumers."""
    return hashlib.sha256(question_text.encode("utf-8")).hexdigest()


def generate_question_candidate(
    questioner_input,
    *,
    retry_reason: Optional[str] = None,
    fallback_question: str = "",
) -> QuestionCandidate:
    """Make exactly one Questioner LLM call and return its unvalidated output."""
    from viva_evaluator.services.agents.questioner import (
        DIFFICULTY_TO_BLOOMS,
        _build_prompt,
    )
    from viva_evaluator.services.llm_service import llm_call
    from viva_evaluator.services.pipeline.evidence import (
        ensure_question_evidence_package,
    )

    evidence_package = ensure_question_evidence_package(questioner_input)
    blooms = questioner_input.target_bloom or DIFFICULTY_TO_BLOOMS.get(
        questioner_input.difficulty,
        "Analyze",
    )
    prompt = _build_prompt(questioner_input, blooms, retry_reason=retry_reason)
    is_repair = bool(retry_reason)
    response = llm_call(
        prompt,
        model="fast" if is_repair else "reasoning",
        expect_json=True,
        fallback={
            "question_text": fallback_question,
            "source_reference_ids": [],
            "target_bloom": blooms,
            "socratic_intent": (
                questioner_input.socratic_intent or "general_probe"
            ),
        },
        operation="question_repair" if is_repair else "question_generation",
    )
    if not isinstance(response, dict):
        response = {}

    schema_failures = []
    raw_question = response.get("question_text")
    if not isinstance(raw_question, str):
        schema_failures.append("malformed_question_text")
        question_text = ""
    else:
        question_text = raw_question.strip()
        if not question_text:
            schema_failures.append("empty_question_text")

    raw_reference_ids = response.get("source_reference_ids")
    if not isinstance(raw_reference_ids, list) or not all(
        isinstance(reference_id, str) for reference_id in raw_reference_ids
    ):
        schema_failures.append("malformed_source_reference_ids")
        source_reference_ids = ()
    else:
        source_reference_ids = tuple(
            dict.fromkeys(
                reference_id.strip()
                for reference_id in raw_reference_ids
                if reference_id.strip()
            )
        )

    planned_intent = questioner_input.socratic_intent or "general_probe"
    if response.get("target_bloom") != blooms:
        schema_failures.append("target_bloom_mismatch")
    if response.get("socratic_intent") != planned_intent:
        schema_failures.append("socratic_intent_mismatch")

    available_ids = set(evidence_package.evidence_ids)
    unknown_ids = sorted(set(source_reference_ids) - available_ids)
    if unknown_ids:
        schema_failures.append(
            "unknown_source_reference_ids:" + ",".join(unknown_ids)
        )
    if (
        not evidence_package.weak_grounding
        and evidence_package.references
        and not source_reference_ids
    ):
        schema_failures.append("missing_source_reference_ids")

    return QuestionCandidate(
        question_text=question_text,
        blooms_level=cast(BloomLevel, blooms),
        difficulty=cast(Difficulty, questioner_input.difficulty),
        raw_response=response,
        candidate_hash=candidate_hash(question_text),
        socratic_intent=planned_intent,
        source_reference_ids=source_reference_ids,
        schema_failures=tuple(schema_failures),
    )
