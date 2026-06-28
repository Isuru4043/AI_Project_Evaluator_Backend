"""
Response Triage agent — fairness gate before scoring.

PURPOSE:
    A student's weak or empty response can mean two very different things:
      (a) they don't KNOW the answer        → a genuine knowledge gap, score it.
      (b) they didn't UNDERSTAND the question → a clarity problem (often the
          AI's fault), which must NOT be scored against the student.

    This agent classifies the response so the pipeline can decide whether to
    score it (normal flow) or suspend scoring and re-ask a clearer question.

DESIGN (matches the project's other agents):
    - One cheap `fast`-model LLM call returning structured JSON.
    - It ONLY classifies. It never decides what happens next — the pipeline's
      deterministic gate does that, with bounded retries and full logging.

SAFETY / FAIRNESS:
    - Asymmetric: a confusion verdict can only SUSPEND a penalty, never add
      credit.
    - Bounded: the pipeline caps consecutive clarifications so a student can't
      stall indefinitely by feigning confusion.
    - Auditable: the label + rationale are returned for the examiner trail.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from viva_evaluator.services.llm_service import llm_call

logger = logging.getLogger(__name__)


# Labels
LABEL_ANSWER_ATTEMPT = 'ANSWER_ATTEMPT'        # engaged with the question → score normally
LABEL_CONFUSED       = 'CONFUSED_BY_QUESTION'  # signalled the question was unclear → re-ask
LABEL_NON_ANSWER     = 'NON_ANSWER'            # empty / "I don't know how to start" → re-ask simpler
LABEL_GARBLED        = 'GARBLED_TRANSCRIPTION' # A5: speech-to-text garble → ask to restate

# Labels that should trigger a clarification (no scoring) if budget remains.
CLARIFY_LABELS = {LABEL_CONFUSED, LABEL_NON_ANSWER}
# Labels that should trigger a verbatim re-ask ("please restate") if budget remains.
RESTATE_LABELS = {LABEL_GARBLED}


@dataclass
class TriageInput:
    question_text: str
    student_answer: str
    # Optional: the just-asked question's Critic specificity/clarity score, if
    # available. A low value is corroborating evidence the question was unclear.
    question_clarity_score: Optional[float] = None
    # A5: True when the answer came from speech-to-text, so a GARBLED verdict
    # (transcription artifact) is plausible. For typed answers this stays off.
    is_spoken: bool = False


def triage_response(inp: TriageInput) -> Dict:
    """
    Classify a student's response.

    Returns:
        {
            'label':      one of ANSWER_ATTEMPT | CONFUSED_BY_QUESTION | NON_ANSWER,
            'confidence': 0..1,
            'rationale':  short string for the audit log,
        }
    """
    answer = (inp.student_answer or '').strip()

    # Fast path: truly empty answer needs no LLM call.
    if not answer:
        return {
            'label':      LABEL_NON_ANSWER,
            'confidence': 1.0,
            'rationale':  'empty response',
        }

    prompt = _build_prompt(inp)
    try:
        response = llm_call(
            prompt=prompt,
            model='fast',
            expect_json=True,
            max_retries=1,
            fallback={
                'label':      LABEL_ANSWER_ATTEMPT,   # safe default: score it normally
                'confidence': 0.5,
                'rationale':  'triage unavailable; defaulting to score',
            },
        )
    except Exception as exc:
        # A fairness check must never block the turn. On any failure (incl.
        # quota), default to scoring normally — the neutral, non-penalising path.
        logger.warning('response_triage failed (%s); defaulting to ANSWER_ATTEMPT', exc)
        return {'label': LABEL_ANSWER_ATTEMPT, 'confidence': 0.5,
                'rationale': 'triage error; defaulting to score'}

    if not isinstance(response, dict):
        return {
            'label': LABEL_ANSWER_ATTEMPT,
            'confidence': 0.5,
            'rationale': 'triage parse failed; defaulting to score',
        }

    label = str(response.get('label', LABEL_ANSWER_ATTEMPT)).strip().upper()
    if label not in (LABEL_ANSWER_ATTEMPT, LABEL_CONFUSED, LABEL_NON_ANSWER, LABEL_GARBLED):
        label = LABEL_ANSWER_ATTEMPT
    # Only allow GARBLED for spoken answers; otherwise treat as a real attempt.
    if label == LABEL_GARBLED and not inp.is_spoken:
        label = LABEL_ANSWER_ATTEMPT

    try:
        confidence = max(0.0, min(1.0, float(response.get('confidence', 0.6))))
    except (TypeError, ValueError):
        confidence = 0.6

    result = {
        'label':      label,
        'confidence': confidence,
        'rationale':  str(response.get('rationale', ''))[:300],
    }
    logger.info('response_triage: label=%s conf=%.2f (%s)',
                result['label'], result['confidence'], result['rationale'][:80])
    return result


def _build_prompt(inp: TriageInput) -> str:
    clarity_note = ''
    if inp.question_clarity_score is not None and inp.question_clarity_score < 0.5:
        clarity_note = (
            "\nNOTE: an automated check already rated the QUESTION as low in "
            "specificity/clarity, so confusion is plausibly the question's fault.\n"
        )

    garbled_option = ''
    if inp.is_spoken:
        garbled_option = (
            '\n- "GARBLED_TRANSCRIPTION": the answer came from speech-to-text and '
            'reads as incoherent or scrambled in a way that looks like a MISHEARING '
            '(nonsense word sequences, broken fragments), not a genuine attempt. '
            'These should NOT be scored; ask the student to restate.\n'
        )

    return f"""You are helping run a fair oral exam. Classify the STUDENT'S RESPONSE
to decide whether it should be scored or whether the question should be re-asked.

QUESTION ASKED:
"{inp.question_text}"

STUDENT'S RESPONSE:
"{inp.student_answer}"
{clarity_note}
Choose exactly ONE label:

- "ANSWER_ATTEMPT": the student engaged with the question being asked, even if
  the answer is wrong, partial, or uncertain. This includes "I think it's X" or
  a genuine wrong attempt. These SHOULD be scored.

- "CONFUSED_BY_QUESTION": the student signalled they did not understand the
  QUESTION itself — e.g. "what do you mean?", "can you rephrase?", "I don't
  understand the question", or they clearly answered a DIFFERENT question than
  the one asked. These should NOT be scored; the question should be re-asked.

- "NON_ANSWER": no real attempt and no clear signal — e.g. silence, "I don't
  know where to start", or a request to repeat. Treat as needing a simpler
  re-ask, not a score.
{garbled_option}
CRITICAL DISTINCTION:
  "I don't know the answer" / "I'm not sure, maybe it's caching" = ANSWER_ATTEMPT
    (they understood the question but lack the knowledge — score it).
  "I don't understand what you're asking" = CONFUSED_BY_QUESTION
    (the question was unclear — do not penalise).

Respond ONLY with valid JSON:
{{
    "label": "ANSWER_ATTEMPT",
    "confidence": 0.0,
    "rationale": "<one short sentence>"
}}
"""
