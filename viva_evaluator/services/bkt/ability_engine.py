"""
Bayesian ability engine — difficulty-aware mastery estimation for viva.

WHY THIS EXISTS (vs Bayesian Knowledge Tracing / BKT):
    BKT comes from intelligent tutoring systems: it tracks a student *learning*
    a skill over many attempts (hence its P_T "learning transition" term). A
    viva is the opposite — we *measure* a fixed, pre-existing ability in a
    single sitting, with very few observations per concept.

    The correct model family for adaptive *assessment* is Item Response Theory
    (IRT) / Computerized Adaptive Testing (CAT). This engine implements an
    online Bayesian Rasch-style estimator:

        - Each concept has a latent ability θ with a Gaussian belief N(μ, σ²).
        - Each question has a difficulty b (mapped from its Bloom level).
        - Expected performance is logistic: p = sigmoid(μ − b).
        - We update μ and σ² after each answer using the continuous soft_score
          as evidence (online Bayesian logistic / Kalman-style update).

    Two advantages over BKT for this use case:
        1. DIFFICULTY-AWARE (fairer): nailing a hard (Evaluate) question raises
           ability more than nailing an easy (Understand) one; missing an easy
           one costs more than missing a hard one — exactly how a human
           examiner weights evidence. BKT ignores difficulty entirely.
        2. UNCERTAINTY (honest stopping): σ shrinks as evidence accumulates, so
           "we are now confident about this concept" is a real statistical
           statement (σ below a threshold) rather than a noisy
           std-of-last-3-deltas heuristic.

INTERFACE COMPATIBILITY:
    AbilityState intentionally exposes `.p_lt` (mastery in [0, 1] = sigmoid(μ))
    and `.history` so the Strategist, termination logic, reporting, and API
    payloads keep working unchanged — they already speak "mastery in [0, 1]".
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List


# =============================================================================
# Configuration
# =============================================================================

# Initial belief. INITIAL_MU = logit(0.30) keeps the same "start skeptical"
# prior the old BKT used (P_L0 = 0.30). INITIAL_SIGMA2 = 1.0 is a wide,
# uninformative prior that narrows as answers arrive.
P_L0 = 0.30
INITIAL_MU = math.log(P_L0 / (1.0 - P_L0))   # ≈ -0.847
INITIAL_SIGMA2 = 1.0

# Posterior standard-deviation threshold below which a concept's ability is
# considered "measured precisely enough" (convergence / stop probing).
ABILITY_SD_THRESHOLD = 0.70

# Bloom level → item difficulty (b), in the same logit units as ability μ.
# Higher = harder. A correct answer to a high-b item is strong evidence of
# high ability.
BLOOM_DIFFICULTY = {
    'Remember':   -1.2,
    'Understand': -0.8,
    'Apply':      -0.2,
    'Analyze':     0.4,
    'Evaluate':    1.0,
    'Create':      1.4,
}
DEFAULT_DIFFICULTY = 0.0


def _sigmoid(x: float) -> float:
    # Clamp to avoid overflow on extreme logits.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# =============================================================================
# State container — one per concept (rubric criterion)
# =============================================================================

@dataclass
class AbilityState:
    """
    Per-concept Bayesian ability belief. Persisted between turns inside
    EvaluationSession.bkt_state_json (same column as before).
    """
    concept_id: str
    mu: float = INITIAL_MU                     # posterior mean of latent ability
    sigma2: float = INITIAL_SIGMA2             # posterior variance (uncertainty)
    history: List[float] = field(default_factory=list)   # mastery trajectory [0,1]
    turns: int = 0

    # --- mastery view (everything downstream speaks "mastery in [0,1]") -----

    @property
    def p_lt(self) -> float:
        """Posterior-mean mastery in [0, 1] = sigmoid(μ)."""
        return _sigmoid(self.mu)

    @property
    def sigma(self) -> float:
        return math.sqrt(max(self.sigma2, 0.0))

    def is_converged(self, sd_threshold: float = ABILITY_SD_THRESHOLD) -> bool:
        """
        Converged when we have at least two observations AND the posterior
        standard deviation has dropped below the threshold (ability measured
        precisely). The per-concept MAX_TURNS backstop in termination.py
        guarantees we never loop forever even if SD stays high.
        """
        return self.turns >= 2 and self.sigma < sd_threshold

    # --- serialization ------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            'concept_id': self.concept_id,
            'mu':         self.mu,
            'sigma2':     self.sigma2,
            'history':    list(self.history),
            'turns':      self.turns,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AbilityState':
        """
        Tolerant loader. Reads the new ability schema, but also migrates any
        legacy BKT state (which stored 'p_lt') so old sessions don't break.
        """
        if 'mu' in data:
            return cls(
                concept_id=data.get('concept_id', ''),
                mu=float(data.get('mu', INITIAL_MU)),
                sigma2=float(data.get('sigma2', INITIAL_SIGMA2)),
                history=list(data.get('history', [])),
                turns=int(data.get('turns', len(data.get('history', [])))),
            )

        # Legacy BKT row → seed μ from the old mastery probability.
        legacy_p = float(data.get('p_lt', P_L0))
        legacy_p = min(max(legacy_p, 1e-4), 1 - 1e-4)
        return cls(
            concept_id=data.get('concept_id', ''),
            mu=math.log(legacy_p / (1.0 - legacy_p)),
            sigma2=INITIAL_SIGMA2,
            history=list(data.get('history', [])),
            turns=len(data.get('history', [])),
        )


# =============================================================================
# Public API — the Bayesian update step
# =============================================================================

def update_ability(
    state: AbilityState,
    soft_score: float,
    bloom_level: str = 'Analyze',
) -> AbilityState:
    """
    Apply one online Bayesian (Kalman-style) update for a logistic/Rasch model.
    Mutates and returns the state.

    Args:
        state:       AbilityState for the answered concept.
        soft_score:  continuous evidence in [0, 1] from the Analyzer's 3D rubric.
        bloom_level: the *answered* question's Bloom level → item difficulty b.

    Update equations (online Bayesian logistic regression, single item):
        b      = difficulty(bloom_level)
        p      = sigmoid(μ − b)                # expected performance
        I      = p · (1 − p)                   # Fisher information of the item
        σ²_new = 1 / (1/σ² + I)                # precisions add → uncertainty falls
        μ_new  = μ + σ²_new · (s − p)          # mean moves toward the evidence

    Intuition:
        - (s − p) is the surprise: doing better than expected pushes ability up.
        - Because p depends on difficulty b, the same soft_score moves ability
          by different amounts for easy vs hard questions (the fairness gain).
        - σ² always shrinks, so the estimate gets more confident each turn.
    """
    s = max(0.0, min(1.0, float(soft_score)))
    b = BLOOM_DIFFICULTY.get(bloom_level, DEFAULT_DIFFICULTY)

    mu = state.mu
    var = max(state.sigma2, 1e-6)

    p = _sigmoid(mu - b)
    info = max(p * (1.0 - p), 1e-6)

    new_var = 1.0 / (1.0 / var + info)
    new_mu = mu + new_var * (s - p)

    state.mu = new_mu
    state.sigma2 = new_var
    state.turns += 1
    state.history.append(round(_sigmoid(new_mu), 4))
    return state


# =============================================================================
# Next-question difficulty — CAT-style target near current ability.
# =============================================================================

def bloom_target_for_ability(p_lt: float) -> str:
    """
    Pick the next Bloom level to target. CAT theory says to ask items near the
    current ability estimate (maximally informative). We keep the same mastery
    thresholds the project already used so behaviour stays comparable; the real
    upgrade is in how ability is *estimated*, not in this mapping.
    """
    if p_lt < 0.35:
        return 'Understand'
    if p_lt < 0.55:
        return 'Apply'
    if p_lt < 0.75:
        return 'Analyze'
    return 'Evaluate'
