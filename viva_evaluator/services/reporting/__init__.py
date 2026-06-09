"""
Post-viva reporting — produces the dissertation-grade artifact for examiners.

WHAT IT BUILDS:
    - BKT trajectories (P(L_t) over turns, per criterion)
    - 3D rubric aggregation (mean Correctness/Depth/Consistency per criterion)
    - Weighted final grade bracket (rubric means × Phase 0 weights)
    - Authorship verification section (CONTRADICTS_CODE turns extracted)
    - Knowledge audit (Tier 2/3 KG edges actually used during the session)
    - Weak-area review (every turn with Correctness < 0.4)

OUTPUT:
    The full report is a structured dict; examiners consume it via
    GET /api/viva/sessions/<id>/report/. Charts are returned as
    base64-encoded PNGs so the frontend can render them inline.
"""

from viva_evaluator.services.reporting.post_viva_report import (
    generate_post_viva_report,
)

__all__ = ['generate_post_viva_report']
