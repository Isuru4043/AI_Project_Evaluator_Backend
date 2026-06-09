"""
Soft-evidence Bayesian Knowledge Tracing engine.

CORE EQUATIONS (from v3 spec, Phase 3 step 3.4):

    Likelihoods given soft score s ∈ [0, 1]:
        P(s | mastered)     = (1 − P_S) × s + P_S × (1 − s)
        P(s | not_mastered) = P_G × s + (1 − P_G) × (1 − s)

    Posterior after observing s:
        P(L_t | s) = P(L_{t-1}) × P(s | M)
                     ─────────────────────────────────────────────
                     P(L_{t-1}) × P(s | M) + (1 − P(L_{t-1})) × P(s | ¬M)

    Transition (learning between turns — small for assessment context):
        P(L_{t+1}) = P(L_t | s) + (1 − P(L_t | s)) × P_T

When s = 1.0 or 0.0, this reduces exactly to standard binary BKT.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# =============================================================================
# Defaults — overridable per criterion in Phase 0 (examiner setup).
# =============================================================================

BKT_DEFAULTS = {
    'p_l0':  0.30,   # initial mastery probability
    'p_t':   0.05,   # learning transition (LOW for assessment)
    'p_g':   0.20,   # guess probability
    'p_s':   0.10,   # slip probability
}


# =============================================================================
# State container — one per concept (typically one per rubric criterion)
# =============================================================================

@dataclass
class BktState:
    """
    Per-concept BKT state. Held in session memory and persisted between
    turns via EvaluationSession.bkt_state_json.
    """
    concept_id: str                          # criterion UUID or name
    p_lt: float = BKT_DEFAULTS['p_l0']        # current mastery probability
    p_t:  float = BKT_DEFAULTS['p_t']
    p_g:  float = BKT_DEFAULTS['p_g']
    p_s:  float = BKT_DEFAULTS['p_s']
    history: List[float] = field(default_factory=list)   # full P(L_t) trajectory
    delta_last3: List[float] = field(default_factory=list)  # for convergence check

    def to_dict(self) -> Dict:
        return {
            'concept_id':  self.concept_id,
            'p_lt':        self.p_lt,
            'p_t':         self.p_t,
            'p_g':         self.p_g,
            'p_s':         self.p_s,
            'history':     list(self.history),
            'delta_last3': list(self.delta_last3),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BktState':
        return cls(
            concept_id=data.get('concept_id', ''),
            p_lt=data.get('p_lt', BKT_DEFAULTS['p_l0']),
            p_t=data.get('p_t', BKT_DEFAULTS['p_t']),
            p_g=data.get('p_g', BKT_DEFAULTS['p_g']),
            p_s=data.get('p_s', BKT_DEFAULTS['p_s']),
            history=list(data.get('history', [])),
            delta_last3=list(data.get('delta_last3', [])),
        )

    def is_converged(self, threshold: float = 0.05) -> bool:
        """std-dev of last 3 deltas < threshold → mastery estimate has stabilized."""
        if len(self.delta_last3) < 3:
            return False
        mean = sum(self.delta_last3) / len(self.delta_last3)
        var = sum((d - mean) ** 2 for d in self.delta_last3) / len(self.delta_last3)
        std = var ** 0.5
        return std < threshold


# =============================================================================
# Public API — the BKT update step.
# =============================================================================

def update_bkt(state: BktState, soft_score: float) -> BktState:
    """
    Apply one soft-evidence BKT update in place. Mutates and returns the state.

    Args:
        state:      BktState for the concept being updated.
        soft_score: continuous evidence score in [0, 1] from the Analyzer's
                    weighted 3D rubric.

    Returns:
        The same state object (mutated). Updates p_lt, history, delta_last3.
    """
    # Clamp inputs
    s = max(0.0, min(1.0, float(soft_score)))
    p_prev = max(0.0, min(1.0, state.p_lt))

    # Likelihoods of the observation given mastery / non-mastery
    p_s_given_mastered     = (1.0 - state.p_s) * s + state.p_s * (1.0 - s)
    p_s_given_not_mastered = state.p_g * s + (1.0 - state.p_g) * (1.0 - s)

    # Posterior after observation
    numerator = p_prev * p_s_given_mastered
    denominator = numerator + (1.0 - p_prev) * p_s_given_not_mastered

    if denominator <= 0:
        # Defensive — shouldn't happen with bounded inputs, but guard against div-by-zero
        posterior = p_prev
    else:
        posterior = numerator / denominator

    # Apply learning transition (P_T)
    p_new = posterior + (1.0 - posterior) * state.p_t
    p_new = max(0.0, min(1.0, p_new))

    # Track delta and history
    delta = p_new - p_prev
    state.delta_last3.append(delta)
    if len(state.delta_last3) > 3:
        state.delta_last3 = state.delta_last3[-3:]

    state.history.append(p_new)
    state.p_lt = p_new
    return state


# =============================================================================
# Bloom target lookup — Strategist consumes this to pick question difficulty.
# =============================================================================

def bloom_target_for_mastery(p_lt: float) -> str:
    """
    Map current mastery probability to a target Bloom level.

    From v3 spec, Phase 3 step 3.5:
        P(L_t) range          Bloom target
        < 0.35                Remember / Understand → 'Understand'
        0.35 ≤ P(L_t) < 0.55  Apply / Analyze       → 'Apply'
        0.55 ≤ P(L_t) < 0.75  Analyze / Evaluate    → 'Analyze'
        ≥ 0.75                Evaluate / Create     → 'Evaluate'

    The Strategist may override with 'Evaluate' if a CONTRADICTS_CODE alert
    is active, regardless of mastery — that's handled in strategist.py, not
    here.
    """
    if p_lt < 0.35:
        return 'Understand'
    if p_lt < 0.55:
        return 'Apply'
    if p_lt < 0.75:
        return 'Analyze'
    return 'Evaluate'
