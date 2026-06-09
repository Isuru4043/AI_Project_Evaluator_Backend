"""
Termination logic — three formal conditions + 25-turn hard cap.

From v3 spec, Phase 3 step 3.8:

    Loop terminates when ALL THREE are simultaneously true:
        1. Rubric coverage   : every criterion has received >= questions_to_ask
                               turns where Correctness > 0.3
        2. Min session length: total_turns >= MIN_TOTAL_TURNS
        3. BKT convergence   : for every concept C, either
                                  std(delta_last3) < 0.05  (mastery stable)
                               OR concept has reached MAX_TURNS_PER_CONCEPT

    Hard cap: if total_turns >= HARD_TURN_CAP, terminate regardless.

These limits are per the v3 spec. They can be made examiner-configurable in
Phase 0 polish; for the FYP we hard-code the literature defaults.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from viva_evaluator.services.pipeline.session_state import SessionState

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration — v3 spec defaults
# =============================================================================

MIN_TOTAL_TURNS         = 6     # default 10 in spec; lowered for FYP rubrics with fewer criteria
MAX_TURNS_PER_CONCEPT   = 5
BKT_CONVERGENCE_THRESHOLD = 0.05
HARD_TURN_CAP           = 25
WEAK_MASTERY_THRESHOLD  = 0.40   # criteria below this get extra turns until cap


# =============================================================================
# Result type
# =============================================================================

@dataclass
class TerminationDecision:
    should_end: bool
    reason: str
    coverage_met: bool
    min_turns_met: bool
    bkt_converged: bool
    hard_cap_hit: bool


# =============================================================================
# Public API
# =============================================================================

def should_terminate(
    state: SessionState,
    all_criteria: List[Dict],
) -> TerminationDecision:
    """
    Evaluate the three termination conditions plus hard cap.

    Args:
        state:        Current session state (BKT + coverage + total_turns).
        all_criteria: List of criterion dicts in rubric order. Each must have
                      'id' and 'questions_to_ask'.

    Returns:
        TerminationDecision flagging which conditions hold.
    """
    # ----- Hard cap (overrides everything) ----------------------------------
    if state.total_turns >= HARD_TURN_CAP:
        return TerminationDecision(
            should_end=True,
            reason=f'hard_cap reached ({HARD_TURN_CAP} turns)',
            coverage_met=False,
            min_turns_met=True,
            bkt_converged=False,
            hard_cap_hit=True,
        )

    # ----- Condition 1: rubric coverage ------------------------------------
    coverage_met, coverage_reason = _check_coverage(state, all_criteria)

    # ----- Condition 2: min total turns ------------------------------------
    min_turns_met = state.total_turns >= MIN_TOTAL_TURNS

    # ----- Condition 3: BKT convergence -------------------------------------
    bkt_converged, conv_reason = _check_bkt_convergence(state, all_criteria)

    all_met = coverage_met and min_turns_met and bkt_converged
    if all_met:
        reason = (
            f'coverage_met=True, min_turns_met=True ({state.total_turns}/{MIN_TOTAL_TURNS}), '
            f'bkt_converged=True'
        )
    else:
        reason_parts = []
        if not coverage_met: reason_parts.append(f'coverage: {coverage_reason}')
        if not min_turns_met: reason_parts.append(
            f'turns {state.total_turns}/{MIN_TOTAL_TURNS}'
        )
        if not bkt_converged: reason_parts.append(f'bkt: {conv_reason}')
        reason = '; '.join(reason_parts)

    return TerminationDecision(
        should_end=all_met,
        reason=reason,
        coverage_met=coverage_met,
        min_turns_met=min_turns_met,
        bkt_converged=bkt_converged,
        hard_cap_hit=False,
    )


# =============================================================================
# Internals
# =============================================================================

def _check_coverage(state: SessionState, all_criteria: List[Dict]) -> tuple:
    """
    Every criterion must have at minimum its questions_to_ask turns,
    AND criteria with weak mastery (P_Lt < 0.4) need either more turns
    until they hit MAX_TURNS_PER_CONCEPT.
    """
    incomplete = []
    for crit in all_criteria:
        crit_id = str(crit['id'])
        cov = state.coverage.get(crit_id)
        required = int(crit.get('questions_to_ask', 3))

        turns = cov.turns if cov else 0
        correct_turns = cov.correct_turns if cov else 0

        if correct_turns < required:
            incomplete.append(f"{crit.get('name', crit_id)}: {correct_turns}/{required}")
            continue

        # Weak mastery → extend up to MAX_TURNS_PER_CONCEPT
        bkt = state.bkt_states.get(crit_id)
        if bkt and bkt.p_lt < WEAK_MASTERY_THRESHOLD and turns < MAX_TURNS_PER_CONCEPT:
            incomplete.append(
                f"{crit.get('name', crit_id)}: weak mastery {bkt.p_lt:.2f}, "
                f"turns {turns}/{MAX_TURNS_PER_CONCEPT}"
            )

    if incomplete:
        return False, '; '.join(incomplete[:3]) + (
            f' (+{len(incomplete) - 3} more)' if len(incomplete) > 3 else ''
        )
    return True, 'all criteria covered'


def _check_bkt_convergence(state: SessionState, all_criteria: List[Dict]) -> tuple:
    """
    Every concept must satisfy at least one of:
      - std(delta_last3) < BKT_CONVERGENCE_THRESHOLD
      - turns >= MAX_TURNS_PER_CONCEPT
    """
    not_converged = []
    for crit in all_criteria:
        crit_id = str(crit['id'])
        bkt = state.bkt_states.get(crit_id)
        cov = state.coverage.get(crit_id)
        turns = cov.turns if cov else 0

        if turns >= MAX_TURNS_PER_CONCEPT:
            continue   # max-turns satisfies the convergence guarantee

        if not bkt or len(bkt.delta_last3) < 3:
            not_converged.append(
                f"{crit.get('name', crit_id)}: insufficient history "
                f"(turns={turns})"
            )
            continue

        if not bkt.is_converged(BKT_CONVERGENCE_THRESHOLD):
            not_converged.append(
                f"{crit.get('name', crit_id)}: not converged "
                f"(P_Lt={bkt.p_lt:.2f}, turns={turns})"
            )

    if not_converged:
        return False, '; '.join(not_converged[:3]) + (
            f' (+{len(not_converged) - 3} more)' if len(not_converged) > 3 else ''
        )
    return True, 'all concepts converged or hit max turns'
