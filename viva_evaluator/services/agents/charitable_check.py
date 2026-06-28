"""
Charitable interpretation agent (A2) — fairness rescue for borderline answers.

PURPOSE:
    A human examiner gives credit when a student clearly UNDERSTANDS a concept
    but expresses it with imprecise wording or their own terminology. The
    Analyzer scores against the rubric text, so a fundamentally-correct idea in
    sloppy language can land in a borderline band and unfairly drag the score
    (and therefore mastery) down.

    This agent runs ONLY for borderline answers and asks one focused question:
    "Does the answer demonstrate correct underlying understanding, even if the
    wording is imprecise?"

SAFETY / FAIRNESS:
    - Asymmetric: it can only RAISE a borderline score, never lower it.
    - Bounded: only invoked inside a narrow correctness band, so it adds at
      most one LLM call per turn and cannot rewrite confident scores.
    - Auditable: returns a rationale; the pipeline records the original score.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from viva_evaluator.services.llm_service import llm_call
from viva_evaluator.services.rag.retrieval import format_chunks_for_prompt

logger = logging.getLogger(__name__)


@dataclass
class CharitableInput:
    question_text: str
    student_answer: str
    criterion_name: str
    criterion_description: str = ''
    retrieved_chunks: List[Dict] = field(default_factory=list)


def assess_understanding(inp: CharitableInput) -> Dict:
    """
    Judge whether a borderline answer reflects sound underlying understanding.

    Returns:
        {
            'understanding_sound': bool,
            'confidence':          0..1,
            'rationale':           short string,
        }
    """
    prompt = _build_prompt(inp)
    try:
        response = llm_call(
            prompt=prompt,
            model='fast',
            expect_json=True,
            max_retries=1,
            fallback={
                'understanding_sound': False,   # safe default: no rescue
                'confidence': 0.5,
                'rationale': 'charitable check unavailable; no adjustment',
            },
        )
    except Exception as exc:
        # Never let a fairness rescue failure block the turn. On error, simply
        # don't apply any adjustment (the original score stands).
        logger.warning('charitable_check failed (%s); no adjustment', exc)
        return {'understanding_sound': False, 'confidence': 0.5,
                'rationale': 'charitable error; no adjustment'}

    if not isinstance(response, dict):
        return {'understanding_sound': False, 'confidence': 0.5,
                'rationale': 'charitable parse failed; no adjustment'}

    try:
        confidence = max(0.0, min(1.0, float(response.get('confidence', 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    result = {
        'understanding_sound': bool(response.get('understanding_sound', False)),
        'confidence':          confidence,
        'rationale':           str(response.get('rationale', ''))[:300],
    }
    logger.info('charitable_check: sound=%s conf=%.2f (%s)',
                result['understanding_sound'], result['confidence'],
                result['rationale'][:80])
    return result


def _build_prompt(inp: CharitableInput) -> str:
    sources_block = format_chunks_for_prompt(inp.retrieved_chunks, max_chars=1500)

    return f"""You are an experienced, fair viva examiner. A student's answer scored in
the BORDERLINE range. Your job is to decide whether the student actually
UNDERSTANDS the concept but simply expressed it with weak wording or their own
informal terminology — in which case they deserve credit.

RUBRIC CRITERION:
Name: {inp.criterion_name}
Description: {inp.criterion_description or '(no description)'}

QUESTION ASKED:
{inp.question_text}

STUDENT'S ANSWER (verbatim):
{inp.student_answer}

RELEVANT SOURCES from the student's submission:
{sources_block}

Decide: does the answer demonstrate CORRECT underlying understanding of the
concept, even if the terminology is imprecise, informal, or non-standard?

Examples of where the answer is SOUND despite wording:
  - "I scrambled the data so the server can't read it" = understands encryption.
  - "the app remembers who you are between pages" = understands sessions.

Be honest: if the answer is actually wrong, confused, or missing the core idea,
say understanding_sound = false. Do NOT reward fluent-sounding but incorrect
answers. This check exists only to avoid penalising correct ideas for poor
phrasing — not to inflate wrong answers.

Respond ONLY with valid JSON:
{{
    "understanding_sound": false,
    "confidence": 0.0,
    "rationale": "<one short sentence>"
}}
"""
