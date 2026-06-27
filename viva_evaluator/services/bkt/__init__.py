"""
Mastery estimation for ASSESSMENT (not tutoring).

A viva measures a student's pre-existing ability in a single sitting with few
observations per concept. The appropriate model family is Item Response Theory
(IRT) / Computerized Adaptive Testing (CAT), not Bayesian Knowledge Tracing
(BKT, which is built for tracking *learning* over many attempts).

This package provides `ability_engine`: an online Bayesian Rasch-style
estimator that is difficulty-aware (a correct hard answer counts more than a
correct easy one) and reports calibrated uncertainty (σ) used as the
convergence / stopping signal.

NOTE: the package directory is still named `bkt` for import stability
(session state, pipeline, and strategist import from here). The legacy BKT
engine itself has been removed.
"""

from viva_evaluator.services.bkt.ability_engine import (
    AbilityState,
    update_ability,
    bloom_target_for_ability,
    ABILITY_SD_THRESHOLD,
)

__all__ = [
    'AbilityState',
    'update_ability',
    'bloom_target_for_ability',
    'ABILITY_SD_THRESHOLD',
]
