"""Deterministic, attempt-aware next-topic selection."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class _TopicStats:
    topic: Dict
    index: int
    attempts: int
    max_criterion_attempts: int
    required_attempts: int
    mastery: float
    uncertainty: float
    needs_adaptive_revisit: bool


def pick_next_topic(topics: List[Dict], state, session) -> Optional[Dict]:
    """Choose coverage breadth first, then a bounded adaptive revisit.

    Selection has three explicit layers:

    1. Give every topic one attempted question before revisiting any topic.
    2. Satisfy rubric minimum attempts using ``coverage.turns`` (never
       ``correct_turns``), spreading questions across under-covered topics.
    3. Revisit weak or uncertain topics, ordered by lowest mastery, highest
       uncertainty, then fewest attempts.

    A topic is never selected once any criterion it covers has reached the
    per-concept cap. Grouped criteria normally move together; the strict check
    prevents inconsistent legacy state from pushing one concept past the cap.
    """
    from viva_evaluator.services.pipeline.termination import (
        MAX_TURNS_PER_CONCEPT,
        MIN_TOTAL_TURNS,
    )

    try:
        session_cap = int(session.max_total_questions)
    except (AttributeError, TypeError, ValueError):
        session_cap = 0
    if session_cap > 0 and state.total_turns >= session_cap:
        return None

    stats = [
        topic_stats
        for index, topic in enumerate(topics)
        if (
            topic_stats := _build_topic_stats(
                topic,
                index,
                state,
            )
        ) is not None
        and topic_stats.max_criterion_attempts < MAX_TURNS_PER_CONCEPT
    ]
    if not stats:
        return None

    # Breadth guarantee: a weak answer on the first topic cannot prevent the
    # remaining rubric topics from receiving their first question.
    unattempted = [item for item in stats if item.attempts == 0]
    if unattempted:
        return min(unattempted, key=lambda item: item.index).topic

    # Required coverage is attempt based and spread proportionally. This keeps
    # the selector from exhausting one topic before visiting the next.
    under_covered = [
        item for item in stats if item.attempts < item.required_attempts
    ]
    if under_covered:
        return min(
            under_covered,
            key=lambda item: (
                item.attempts / max(1, item.required_attempts),
                item.attempts,
                item.index,
            ),
        ).topic

    adaptive = [item for item in stats if item.needs_adaptive_revisit]
    if adaptive:
        return min(
            adaptive,
            key=lambda item: (
                item.mastery,
                -item.uncertainty,
                item.attempts,
                item.index,
            ),
        ).topic

    # If coverage and ability have converged unusually quickly, the formal
    # minimum session length may still require another question.
    if state.total_turns < MIN_TOTAL_TURNS:
        return min(
            stats,
            key=lambda item: (item.attempts, item.index),
        ).topic
    return None


def _build_topic_stats(
    topic: Dict,
    index: int,
    state,
) -> Optional[_TopicStats]:
    from viva_evaluator.services.pipeline.termination import (
        WEAK_MASTERY_THRESHOLD,
    )

    criterion_ids = tuple(
        str(criterion_id)
        for criterion_id in (topic.get("source_criteria_ids") or [])
    )
    if not criterion_ids:
        return None

    attempt_counts = []
    requirements = []
    mastery_values = []
    uncertainty_values = []
    needs_revisit = False

    for criterion_id in criterion_ids:
        coverage = state.coverage.get(criterion_id)
        attempts = max(0, int(coverage.turns if coverage else 0))
        attempt_counts.append(attempts)
        requirements.append(
            max(1, int(coverage.questions_to_ask if coverage else 1))
        )

        ability = state.bkt_states.get(criterion_id)
        if ability is None:
            mastery = 0.0
            uncertainty = float("inf")
            needs_revisit = True
        else:
            mastery = float(ability.p_lt)
            uncertainty = float(ability.sigma)
            needs_revisit = needs_revisit or (
                mastery < WEAK_MASTERY_THRESHOLD
                or not ability.is_converged()
            )
        mastery_values.append(mastery)
        uncertainty_values.append(uncertainty)

    return _TopicStats(
        topic=topic,
        index=index,
        attempts=min(attempt_counts),
        max_criterion_attempts=max(attempt_counts),
        required_attempts=max(requirements),
        mastery=min(mastery_values),
        uncertainty=max(uncertainty_values),
        needs_adaptive_revisit=needs_revisit,
    )
