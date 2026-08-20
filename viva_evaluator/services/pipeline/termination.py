"""
Termination logic — three formal conditions + 25-turn hard cap.

From v3 spec, Phase 3 step 3.8:

    Loop terminates when ALL THREE are simultaneously true:
        1. Rubric coverage   : every criterion has received >= questions_to_ask
                               attempted questions
        2. Min session length: total_turns >= MIN_TOTAL_TURNS
        3. BKT convergence   : for every concept C, either
                                  posterior SD σ < ABILITY_SD_THRESHOLD
                                  (ability measured precisely)
                               OR concept has reached MAX_TURNS_PER_CONCEPT

    Backstops:
        - If every concept reaches MAX_TURNS_PER_CONCEPT after coverage, end
          even when a very small rubric cannot reach MIN_TOTAL_TURNS.
        - If total_turns >= HARD_TURN_CAP, terminate regardless.

These limits are per the v3 spec. They can be made examiner-configurable in
Phase 0 polish; for the FYP we hard-code the literature defaults.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    *,
    ai_question_count: Optional[int] = None,
    hard_cap: int = HARD_TURN_CAP,
) -> TerminationDecision:
    """
    Evaluate the three termination conditions plus hard cap.

    Args:
        state:        Current session state (BKT + coverage + total_turns).
        all_criteria:     Criterion dictionaries in rubric order.
        ai_question_count: Number of persisted AI questions in the session.
                           Defaults to state.total_turns for non-DB callers.
        hard_cap:         Examiner-configured maximum question count.

    Returns:
        TerminationDecision flagging which conditions hold.
    """
    # ----- Hard cap (overrides everything) ----------------------------------
    ai_turns = state.total_turns if ai_question_count is None else ai_question_count
    if ai_turns >= hard_cap:
        return TerminationDecision(
            should_end=True,
            reason=f'hard_cap reached ({hard_cap} turns)',
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

    capacity_exhausted = bool(all_criteria) and all(
        (
            state.coverage.get(str(criterion["id"])).turns
            if state.coverage.get(str(criterion["id"]))
            else 0
        ) >= MAX_TURNS_PER_CONCEPT
        for criterion in all_criteria
    )
    all_met = coverage_met and min_turns_met and bkt_converged
    should_end = all_met or (coverage_met and capacity_exhausted)
    if all_met:
        reason = (
            f'coverage_met=True, min_turns_met=True ({state.total_turns}/{MIN_TOTAL_TURNS}), '
            f'bkt_converged=True'
        )
    elif coverage_met and capacity_exhausted:
        reason = (
            "coverage_met=True; all concepts reached their per-concept "
            f"cap ({MAX_TURNS_PER_CONCEPT})"
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
        should_end=should_end,
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
    Every criterion must have its configured minimum attempted questions.

    Correctness is deliberately excluded: a wrong but genuine answer is still
    an assessment attempt. Weak mastery and uncertainty are handled separately
    by the convergence/revisit rule below.
    """
    incomplete = []
    for crit in all_criteria:
        crit_id = str(crit['id'])
        cov = state.coverage.get(crit_id)
        required = int(crit.get('questions_to_ask', 3))

        turns = cov.turns if cov else 0
        if turns < required:
            incomplete.append(
                f"{crit.get('name', crit_id)}: attempts {turns}/{required}"
            )

    if incomplete:
        return False, '; '.join(incomplete[:3]) + (
            f' (+{len(incomplete) - 3} more)' if len(incomplete) > 3 else ''
        )
    return True, 'all criteria covered'


def _check_bkt_convergence(state: SessionState, all_criteria: List[Dict]) -> tuple:
    """
    Every concept must satisfy at least one of:
      - posterior ability uncertainty is converged and mastery is not weak
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

        if not bkt or bkt.turns < 2:
            not_converged.append(
                f"{crit.get('name', crit_id)}: insufficient history "
                f"(turns={turns})"
            )
            continue

        if bkt.p_lt < WEAK_MASTERY_THRESHOLD:
            not_converged.append(
                f"{crit.get('name', crit_id)}: weak mastery revisit "
                f"(P_Lt={bkt.p_lt:.2f}, turns={turns}/"
                f"{MAX_TURNS_PER_CONCEPT})"
            )
            continue

        if not bkt.is_converged():
            not_converged.append(
                f"{crit.get('name', crit_id)}: not converged "
                f"(P_Lt={bkt.p_lt:.2f}, σ={bkt.sigma:.2f}, turns={turns})"
            )

    if not_converged:
        return False, '; '.join(not_converged[:3]) + (
            f' (+{len(not_converged) - 3} more)' if len(not_converged) > 3 else ''
        )
    return True, 'all concepts converged or hit max turns'
