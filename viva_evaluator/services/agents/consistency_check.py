"""
Consistency-classifier agent (A3) — material vs superficial inconsistency.

PURPOSE:
    The Analyzer flags low "consistency" when an answer seems to clash with
    something the student said earlier or wrote in their report. But not every
    clash is a real contradiction — often it's just different wording for the
    same idea. Penalising a student (and dropping their Bloom level) for clumsy
    rephrasing is unfair.

    This agent decides whether a detected inconsistency is:
      - MATERIAL    : a genuine contradiction worth probing → keep the penalty.
      - SUPERFICIAL : the same idea phrased differently       → neutralise it.

SAFETY / FAIRNESS:
    - Asymmetric: it can only SUPPRESS a consistency penalty, never invent one.
    - Bounded: only invoked when consistency is already flagged as low.
    - Auditable: returns a rationale recorded on the analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from viva_evaluator.services.llm_service import llm_call

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyInput:
    question_text: str
    student_answer: str
    transcript_recent: List[Dict] = field(default_factory=list)  # last Q/A pairs
    consistency_evidence: str = ''   # the Analyzer's consistency evidence_quote


def classify_inconsistency(inp: ConsistencyInput) -> Dict:
    """
    Decide whether a flagged inconsistency is material or superficial.

    Returns:
        {
            'material':   bool,   # True = real contradiction, keep penalty
            'confidence': 0..1,
            'rationale':  short string,
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
                'material': True,   # safe default: keep the penalty
                'confidence': 0.5,
                'rationale': 'consistency check unavailable; penalty kept',
            },
        )
    except Exception as exc:
        logger.warning('consistency_check failed (%s); keeping penalty', exc)
        return {'material': True, 'confidence': 0.5,
                'rationale': 'consistency error; penalty kept'}

    if not isinstance(response, dict):
        return {'material': True, 'confidence': 0.5,
                'rationale': 'consistency parse failed; penalty kept'}

    try:
        confidence = max(0.0, min(1.0, float(response.get('confidence', 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    result = {
        'material':   bool(response.get('material', True)),
        'confidence': confidence,
        'rationale':  str(response.get('rationale', ''))[:300],
    }
    logger.info('consistency_check: material=%s conf=%.2f (%s)',
                result['material'], result['confidence'], result['rationale'][:80])
    return result


def _build_prompt(inp: ConsistencyInput) -> str:
    transcript_block = ''
    if inp.transcript_recent:
        snippets = []
        for turn in inp.transcript_recent[-3:]:
            snippets.append(
                f"  Q: {turn.get('question_text', '')[:160]}\n"
                f"  A: {turn.get('answer_text', '')[:160]}"
            )
        transcript_block = "\n".join(snippets)

    evidence_block = ''
    if inp.consistency_evidence:
        evidence_block = (
            f"\nThe automated scorer flagged this as the conflicting excerpt:\n"
            f'"{inp.consistency_evidence}"\n'
        )

    return f"""You are a fair viva examiner. An automated check flagged the student's
latest answer as possibly INCONSISTENT with what they said earlier or wrote in
their report. Decide whether this is a REAL contradiction or just different
wording for the same idea.

CURRENT QUESTION:
{inp.question_text}

CURRENT ANSWER:
{inp.student_answer}

EARLIER CONVERSATION:
{transcript_block or '(none)'}
{evidence_block}
Decide:
- "material": true  → a genuine contradiction (the two statements cannot both
  be true; the student actually changed or contradicted a factual claim).
- "material": false → SUPERFICIAL: the same underlying idea expressed with
  different words, a synonym, a more/less precise phrasing, or harmless
  rewording. No real conflict.

Examples of SUPERFICIAL (material=false):
  - earlier "a token logs the user in", now "JWT handles authentication".
  - earlier "the database", now "the SQLite file".

Examples of MATERIAL (material=true):
  - earlier "I hash passwords with bcrypt", now "I store passwords in plain text".

Respond ONLY with valid JSON:
{{
    "material": true,
    "confidence": 0.0,
    "rationale": "<one short sentence>"
}}
"""
