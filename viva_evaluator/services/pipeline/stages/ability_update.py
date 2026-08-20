"""In-memory ability and coverage transitions for a scored turn."""

from typing import Dict


def update_topic_ability(
    *,
    active_state,
    answered_topic: Dict,
    soft_score: float,
    bloom_level: str,
) -> None:
    """Apply the difficulty-aware Bayesian update to every source criterion."""
    from viva_evaluator.services.bkt.ability_engine import update_ability

    for criterion_id in answered_topic["source_criteria_ids"]:
        ability_state = active_state.get_or_init_bkt(str(criterion_id))
        update_ability(
            ability_state,
            soft_score,
            bloom_level=bloom_level,
        )


def record_scored_turn(
    *,
    active_state,
    group_state,
    answered_topic: Dict,
    correctness: float,
    soft_score: float,
) -> None:
    """Update coverage and global turn counters without persisting them."""
    for criterion_id in answered_topic["source_criteria_ids"]:
        coverage = active_state.coverage[str(criterion_id)]
        coverage.turns += 1
        coverage.sum_correctness += correctness
        if correctness >= 0.3:
            coverage.correct_turns += 1

    group_state.total_turns += 1
    group_state.soft_score_history.append(round(soft_score, 4))

