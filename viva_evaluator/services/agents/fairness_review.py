"""One structured, conditional fairness review for a scored answer."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from viva_evaluator.services.llm_service import llm_call
from viva_evaluator.services.rag.retrieval import format_chunks_for_prompt


logger = logging.getLogger(__name__)


@dataclass
class FairnessReviewInput:
    question_text: str
    student_answer: str
    criterion_name: str
    criterion_description: str = ""
    retrieved_chunks: List[Dict] = field(default_factory=list)
    transcript_recent: List[Dict] = field(default_factory=list)
    previous_answer: str = ""
    consistency_evidence: str = ""
    requested_checks: Tuple[str, ...] = ()


def review_fairness(inp: FairnessReviewInput) -> Dict:
    """Run all selected rescue checks in at most one fast-model call."""
    requested = tuple(dict.fromkeys(inp.requested_checks))
    if not requested:
        return _neutral_result()

    try:
        response = llm_call(
            prompt=_build_prompt(inp, requested),
            model="fast",
            expect_json=True,
            max_retries=1,
            fallback=_neutral_result(unavailable=True),
            operation="fairness_review",
        )
    except Exception as exc:
        logger.warning("fairness_review failed (%s); keeping original score", exc)
        return _neutral_result(unavailable=True)

    if not isinstance(response, dict):
        return _neutral_result(unavailable=True)
    return _normalize_result(response, requested)


def _build_prompt(inp: FairnessReviewInput, requested: Tuple[str, ...]) -> str:
    sources = format_chunks_for_prompt(inp.retrieved_chunks, max_chars=1400)
    transcript_lines = []
    for turn in inp.transcript_recent[-3:]:
        answer = str(turn.get("answer_text") or "").strip()
        if not answer:
            continue
        transcript_lines.append(
            f"Topic: {turn.get('topic_name') or 'unknown'}\n"
            f"Q: {str(turn.get('question_text') or '')[:180]}\n"
            f"A: {answer[:220]}"
        )
    transcript = "\n\n".join(transcript_lines) or "(none)"
    checks = []
    if "consistency" in requested:
        checks.append(
            "consistency: decide whether the flagged inconsistency is a real "
            "material contradiction or merely different wording."
        )
    if "charitable" in requested:
        checks.append(
            "charitable: decide whether the borderline answer demonstrates "
            "sound underlying understanding despite imprecise wording."
        )
    if "self_correction" in requested:
        checks.append(
            "self_correction: decide whether the current answer explicitly "
            "corrects the prior same-topic answer and meaningfully improves it."
        )

    return f"""You are a fair academic viva examiner performing a bounded rescue review.
The original automated score may only be RAISED by this review, never lowered.
Evaluate ONLY these requested checks:
- {chr(10).join(checks)}

CURRENT TOPIC: {inp.criterion_name}
TOPIC DESCRIPTION: {inp.criterion_description or '(none)'}
CURRENT QUESTION: {inp.question_text}
CURRENT ANSWER: {inp.student_answer}

PREVIOUS SAME-TOPIC ANSWER (only relevant to self_correction):
{inp.previous_answer or '(none)'}

FLAGGED CONSISTENCY EVIDENCE:
{inp.consistency_evidence or '(none)'}

RECENT TRANSCRIPT:
{transcript}

RETRIEVED SUBMISSION EVIDENCE:
{sources}

Return ONLY JSON. For checks that were not requested, use null.
{{
  "consistency": {{
    "material": true,
    "confidence": 0.0,
    "rationale": "one short sentence"
  }},
  "charitable": {{
    "understanding_sound": false,
    "confidence": 0.0,
    "rationale": "one short sentence"
  }},
  "self_correction": {{
    "is_correction": false,
    "improved": false,
    "confidence": 0.0,
    "rationale": "one short sentence"
  }}
}}
"""


def _normalize_result(response: Dict, requested: Tuple[str, ...]) -> Dict:
    neutral = _neutral_result()
    if "consistency" in requested and isinstance(response.get("consistency"), dict):
        value = response["consistency"]
        neutral["consistency"] = {
            "material": value.get("material") is not False,
            "confidence": _confidence(value.get("confidence")),
            "rationale": str(value.get("rationale") or "")[:300],
        }
    if "charitable" in requested and isinstance(response.get("charitable"), dict):
        value = response["charitable"]
        neutral["charitable"] = {
            "understanding_sound": value.get("understanding_sound") is True,
            "confidence": _confidence(value.get("confidence")),
            "rationale": str(value.get("rationale") or "")[:300],
        }
    if "self_correction" in requested and isinstance(response.get("self_correction"), dict):
        value = response["self_correction"]
        neutral["self_correction"] = {
            "is_correction": value.get("is_correction") is True,
            "improved": value.get("improved") is True,
            "confidence": _confidence(value.get("confidence")),
            "rationale": str(value.get("rationale") or "")[:300],
        }
    return neutral


def _neutral_result(*, unavailable: bool = False) -> Dict:
    suffix = " review unavailable" if unavailable else " not requested"
    return {
        "consistency": {
            "material": True,
            "confidence": 0.0,
            "rationale": "consistency" + suffix,
        },
        "charitable": {
            "understanding_sound": False,
            "confidence": 0.0,
            "rationale": "charitable" + suffix,
        },
        "self_correction": {
            "is_correction": False,
            "improved": False,
            "confidence": 0.0,
            "rationale": "self-correction" + suffix,
        },
    }


def _confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
