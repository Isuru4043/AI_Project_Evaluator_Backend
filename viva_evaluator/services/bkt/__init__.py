"""
Bayesian Knowledge Tracing (BKT) — calibrated for ASSESSMENT, not tutoring.

Standard BKT was designed for intelligent tutoring systems where the learner
acquires knowledge between turns. A viva is the opposite: the student already
holds whatever knowledge they have, and the session reveals what's there.

Calibration choices (from the v3 spec):
    P(L₀)   = 0.30   (slight presumption of non-demonstrated mastery)
    P(T)    = 0.05   (low; viva reveals existing knowledge)
    P(G)    = 0.20   (literature standard)
    P(S)    = 0.10   (literature standard)

Soft evidence: instead of binary correct/incorrect, the Analyzer feeds in a
continuous score s ∈ [0, 1] from the 3D rubric weighted average.
"""

from viva_evaluator.services.bkt.bkt_engine import (
    BKT_DEFAULTS,
    BktState,
    update_bkt,
    bloom_target_for_mastery,
)

__all__ = [
    'BKT_DEFAULTS',
    'BktState',
    'update_bkt',
    'bloom_target_for_mastery',
]
