"""
Strategist agent — picks the next question's Bloom level and Socratic intent.

DESIGN:
    Pure Python decision logic — no LLM call. Reads the BKT mastery state,
    the latest 3D rubric output, KG signals, and conversation history.

OUTPUT:
    {
        'bloom_level':    'Understand' | 'Apply' | 'Analyze' | 'Evaluate' | 'Create',
        'socratic_intent': one of 7 intents below,
        'rationale':       short explanation for the audit log,
        'kg_edge_used':    optional dict (when intent=exploring_alternatives),
    }

INTENT TAXONOMY (from v3 spec, Phase 3 step 3.5):
    - clarifying            : Depth < 0.4 OR vague terminology
    - probing_assumptions   : Analyzer flagged a revealed_assumption
    - probing_evidence      : Claim made without grounding
    - exploring_alternatives: KG has alt edge (T1/T2) AND topic available
    - challenge_contradiction: contradicts_code_flag set (HIGHEST priority)
    - testing_connections   : Consistency < 0.4
    - reassure_and_clarify  : (deferred — needs spoken confidence flag in Week 6)

REPETITION PREVENTION:
    If the same intent appeared >= 3 times in the last 4 turns, force a
    different intent. Real examiners vary their approach.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from viva_evaluator.services.bkt.ability_engine import bloom_target_for_ability

logger = logging.getLogger(__name__)


# =============================================================================
# Intents (v3 7-intent taxonomy minus Week-6 reassure_and_clarify)
# =============================================================================

INTENT_CLARIFYING             = 'clarifying'
INTENT_PROBING_ASSUMPTIONS    = 'probing_assumptions'
INTENT_PROBING_EVIDENCE       = 'probing_evidence'
INTENT_EXPLORING_ALTERNATIVES = 'exploring_alternatives'
INTENT_CHALLENGE_CONTRADICTION = 'challenge_contradiction'
INTENT_TESTING_CONNECTIONS    = 'testing_connections'
INTENT_REASSURE_AND_CLARIFY   = 'reassure_and_clarify'   # Week 6

INTENT_PROMPTS = {
    INTENT_CLARIFYING:
        "Ask the student to define or clarify a vague term they used.",
    INTENT_PROBING_ASSUMPTIONS:
        "Surface the implicit assumption they made and ask them to defend it.",
    INTENT_PROBING_EVIDENCE:
        "Push them for the SPECIFIC code or design evidence behind their claim.",
    INTENT_EXPLORING_ALTERNATIVES:
        "Reference an alternative technology and ask why they chose theirs over it.",
    INTENT_CHALLENGE_CONTRADICTION:
        "Confront the student with the report-vs-code contradiction directly.",
    INTENT_TESTING_CONNECTIONS:
        "Ask how their component interacts with another part of their system.",
    INTENT_REASSURE_AND_CLARIFY:
        "Acknowledge a partially correct answer and gently ask them to expand.",
}


# =============================================================================
# Inputs
# =============================================================================

@dataclass
class StrategistInput:
    p_lt: float                                      # current BKT mastery
    analysis: Dict                                   # output of Analyzer.analyze_answer()
    kg_signals: Optional[Dict] = None                # output of retrieve_hybrid_for_turn()
    intent_history: List[str] = field(default_factory=list)   # last N intents used
    consistency_score: Optional[float] = None        # = analysis['consistency']['score'] convenience
    speech_confidence: Optional[str] = None          # 'low' | 'medium' | 'high' | None


# =============================================================================
# Public API
# =============================================================================

def select_strategy(inp: StrategistInput) -> Dict:
    """
    Decide what kind of question to ask next.

    Returns:
        {
            'bloom_level': str,
            'socratic_intent': str,
            'intent_prompt_hint': str,    # one-line guidance for the Questioner
            'rationale': str,
            'kg_edge_used': Optional[dict],
        }
    """
    analysis = inp.analysis or {}
    correctness = (analysis.get('correctness') or {}).get('score', 0.5)
    depth = (analysis.get('depth') or {}).get('score', 0.5)
    consistency = (
        inp.consistency_score
        if inp.consistency_score is not None
        else (analysis.get('consistency') or {}).get('score', 1.0)
    )
    contradicts_flag = analysis.get('contradicts_code_flag', False)
    revealed_assumption = (analysis.get('revealed_assumption') or '').strip()

    kg = inp.kg_signals or {}
    contradicts_alerts = kg.get('contradicts_code_alerts') or []
    has_kg_alt = _has_alternative_edge(kg)

    # ----- 1) Bloom level: ability-driven baseline --------------------------
    bloom = bloom_target_for_ability(inp.p_lt)
    rationale_parts = [f'ability P(mastery)={inp.p_lt:.2f} → bloom={bloom}']

    # ----- 2) Apply overrides per v3 spec -----------------------------------

    # Override A: ACTIVE CONTRADICTS_CODE alert OR student failed to acknowledge
    intent = None
    kg_edge_used = None

    # Highest priority: student missed a contradiction the system flagged
    if contradicts_flag and contradicts_alerts:
        intent = INTENT_CHALLENGE_CONTRADICTION
        bloom = 'Evaluate'
        kg_edge_used = contradicts_alerts[0]
        rationale_parts.append('contradicts_code_flag set → challenge_contradiction')

    # Override B: low consistency → drop bloom, test connections
    elif consistency is not None and consistency < 0.4:
        intent = INTENT_TESTING_CONNECTIONS
        # Drop one Bloom level
        bloom = _drop_bloom_one_level(bloom)
        rationale_parts.append(
            f'consistency={consistency:.2f} < 0.4 → testing_connections, bloom dropped to {bloom}'
        )

    # Override C (Week 6): low spoken confidence + decent correctness
    # → reassure the student rather than challenge them. The flag does NOT
    # change scoring (asymmetric multimodal handling); it only changes the
    # PHRASING strategy.
    elif (
        inp.speech_confidence == 'low'
        and correctness >= 0.6
    ):
        intent = INTENT_REASSURE_AND_CLARIFY
        # Don't escalate Bloom when student is hesitant
        bloom = _drop_bloom_one_level(bloom)
        rationale_parts.append(
            f'speech_confidence=low + correctness={correctness:.2f} '
            f'→ reassure_and_clarify, bloom={bloom}'
        )

    # Standard intent selection (no overrides)
    else:
        intent, kg_edge_used = _pick_standard_intent(
            depth=depth,
            revealed_assumption=revealed_assumption,
            has_kg_alt=has_kg_alt,
            kg_signals=kg,
            correctness=correctness,
            analysis=analysis,
        )
        rationale_parts.append(f'standard intent={intent}')

    # ----- 3) Repetition prevention -----------------------------------------
    intent = _avoid_repetition(intent, inp.intent_history)
    if intent != _last_intent(inp.intent_history):
        rationale_parts.append('rotation applied' if _last_intent(inp.intent_history) else '')

    return {
        'bloom_level':       bloom,
        'socratic_intent':   intent,
        'intent_prompt_hint': INTENT_PROMPTS.get(intent, ''),
        'rationale':         '; '.join(p for p in rationale_parts if p),
        'kg_edge_used':      kg_edge_used,
    }


# =============================================================================
# Internals
# =============================================================================

_BLOOM_ORDER = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']


def _drop_bloom_one_level(current: str) -> str:
    if current not in _BLOOM_ORDER:
        return 'Understand'
    idx = _BLOOM_ORDER.index(current)
    return _BLOOM_ORDER[max(0, idx - 1)]


def _has_alternative_edge(kg: Dict) -> bool:
    """True if the KG retrieval surfaced a usable alt edge for the current topic."""
    # Any depends_on_topics implies the KG has data; for a stronger signal we
    # would peek at edge types but the v3 design treats availability as binary.
    return bool(kg.get('kg_available_for_topic')) and bool(kg.get('depends_on_topics'))


def _pick_standard_intent(
    depth: float,
    revealed_assumption: str,
    has_kg_alt: bool,
    kg_signals: Dict,
    correctness: float,
    analysis: Dict,
) -> tuple:
    """
    Standard intent ranking when no high-priority overrides apply.
    Returns (intent, kg_edge_used).
    """
    # Priority 1: assumption was revealed → probe it
    if revealed_assumption:
        return INTENT_PROBING_ASSUMPTIONS, None

    # Priority 2: low depth → ask for clarification or evidence
    if depth < 0.4:
        # Pick "evidence" if the answer was confidently asserted (high correctness)
        # otherwise "clarifying" because student seems hesitant
        if correctness >= 0.6:
            return INTENT_PROBING_EVIDENCE, None
        return INTENT_CLARIFYING, None

    # Priority 3: KG has alternatives + topic available → probe alternatives
    if has_kg_alt:
        # Surface the first depends_on_topic as the implied "your tech" anchor
        edge_used = {
            'topics': kg_signals.get('depends_on_topics', [])[:5],
            'note':   'KG-driven alternative probing',
        }
        return INTENT_EXPLORING_ALTERNATIVES, edge_used

    # Priority 4: default — test connections (relates this answer to system context)
    return INTENT_TESTING_CONNECTIONS, None


def _avoid_repetition(candidate: str, history: List[str]) -> str:
    """
    If the candidate intent appeared >= 3 times in the last 4 turns,
    rotate to a different intent.
    """
    if not history:
        return candidate

    recent = history[-4:]
    count = recent.count(candidate)
    if count < 3:
        return candidate

    # Pick the first non-matching intent — preference order keeps us close to
    # the original intent's purpose without exact repeat.
    for fallback in [
        INTENT_TESTING_CONNECTIONS,
        INTENT_PROBING_EVIDENCE,
        INTENT_CLARIFYING,
        INTENT_PROBING_ASSUMPTIONS,
        INTENT_EXPLORING_ALTERNATIVES,
    ]:
        if fallback != candidate:
            logger.info('Strategist: rotating intent %s → %s (history=%s)',
                        candidate, fallback, recent)
            return fallback
    return candidate


def _last_intent(history: List[str]) -> Optional[str]:
    return history[-1] if history else None
