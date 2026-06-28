"""
Self-correction agent (A4) — credit a student's best demonstrated understanding.

PURPOSE:
    Students often stumble on a first attempt, then correct or substantially
    improve it ("actually, wait — it's asymmetric, RSA uses a public/private
    key pair"). A human examiner credits the FINAL, corrected understanding,
    not the initial fumble. Scoring each turn in isolation can unfairly hold
    the earlier dip against them.

    This agent compares the student's CURRENT answer with their PREVIOUS answer
    (same line of questioning) and decides whether the current one is a genuine
    self-correction that IMPROVES on the earlier attempt.

SAFETY / FAIRNESS:
    - Asymmetric: can only RAISE the current score (credit the recovery),
      never lower it.
    - Bounded: only runs when there IS a previous answer and the current score
      left room to rescue.
    - Auditable: returns a rationale recorded on the analysis.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from viva_evaluator.services.llm_service import llm_call

logger = logging.getLogger(__name__)


@dataclass
class SelfCorrectionInput:
    question_text: str
    current_answer: str
    previous_answer: str


def assess_self_correction(inp: SelfCorrectionInput) -> Dict:
    """
    Decide whether the current answer is a self-correction that improves on the
    previous one.

    Returns:
        {
            'is_correction': bool,   # current revises/builds on the previous
            'improved':      bool,   # current is meaningfully better
            'confidence':    0..1,
            'rationale':     short string,
        }
    """
    if not (inp.previous_answer or '').strip():
        return {'is_correction': False, 'improved': False, 'confidence': 1.0,
                'rationale': 'no previous answer to compare'}

    prompt = _build_prompt(inp)
    try:
        response = llm_call(
            prompt=prompt,
            model='fast',
            expect_json=True,
            max_retries=1,
            fallback={'is_correction': False, 'improved': False,
                      'confidence': 0.5, 'rationale': 'self-correction check unavailable'},
        )
    except Exception as exc:
        logger.warning('self_correction check failed (%s); no credit', exc)
        return {'is_correction': False, 'improved': False, 'confidence': 0.5,
                'rationale': 'self-correction error; no credit'}

    if not isinstance(response, dict):
        return {'is_correction': False, 'improved': False, 'confidence': 0.5,
                'rationale': 'self-correction parse failed; no credit'}

    try:
        confidence = max(0.0, min(1.0, float(response.get('confidence', 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    result = {
        'is_correction': bool(response.get('is_correction', False)),
        'improved':      bool(response.get('improved', False)),
        'confidence':    confidence,
        'rationale':     str(response.get('rationale', ''))[:300],
    }
    logger.info('self_correction: correction=%s improved=%s conf=%.2f (%s)',
                result['is_correction'], result['improved'], result['confidence'],
                result['rationale'][:80])
    return result


def _build_prompt(inp: SelfCorrectionInput) -> str:
    return f"""You are a fair viva examiner. Compare the student's CURRENT answer with
their PREVIOUS answer on the same topic, and decide whether the current answer
is a SELF-CORRECTION that improves their demonstrated understanding.

QUESTION:
{inp.question_text}

PREVIOUS ANSWER (earlier attempt):
{inp.previous_answer}

CURRENT ANSWER (latest):
{inp.current_answer}

Decide:
- "is_correction": true if the current answer revises, fixes, or builds on the
  previous one (e.g. corrects a mistake, adds the missing key idea, clarifies a
  confused earlier point).
- "improved": true if the current answer demonstrates MEANINGFULLY better
  understanding than the previous one.

Set both false if the current answer is unrelated, the same quality, or worse.
Do NOT reward simply restating the previous answer.

Respond ONLY with valid JSON:
{{
    "is_correction": false,
    "improved": false,
    "confidence": 0.0,
    "rationale": "<one short sentence>"
}}
"""
