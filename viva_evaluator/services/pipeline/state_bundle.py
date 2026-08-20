"""Group/individual state routing for a viva turn."""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

from viva_evaluator.services.pipeline.session_state import (
    SessionState,
    load_session_state,
)


@dataclass
class SessionStateBundle:
    group_state: SessionState
    student_state: Optional[SessionState]
    active_state: SessionState
    unified_state: SessionState
    speaker_id: str
    is_individual_topic: bool

    def rebuild_unified(self, rubric: List[Dict]) -> SessionState:
        self.unified_state = build_unified_state(
            self.group_state,
            self.student_state,
            rubric,
        )
        return self.unified_state


def build_unified_state(
    group_state: SessionState,
    student_state: Optional[SessionState],
    rubric: List[Dict],
) -> SessionState:
    """Build the state view used by topic selection and termination."""
    unified = SessionState()
    unified.total_turns = group_state.total_turns
    unified.intent_history = list(group_state.intent_history)
    unified.clarification_streak = group_state.clarification_streak
    unified.soft_score_history = list(group_state.soft_score_history)

    for criterion in rubric:
        criterion_id = str(criterion["id"])
        if criterion.get("is_individual", False) and student_state:
            unified.bkt_states[criterion_id] = student_state.bkt_states.get(
                criterion_id,
                group_state.bkt_states.get(criterion_id),
            )
            unified.coverage[criterion_id] = student_state.coverage.get(
                criterion_id,
                group_state.coverage.get(criterion_id),
            )
        else:
            unified.bkt_states[criterion_id] = group_state.bkt_states.get(
                criterion_id
            )
            unified.coverage[criterion_id] = group_state.coverage.get(criterion_id)
    return unified


def load_state_bundle(
    session,
    rubric: List[Dict],
    answered_topic: Dict,
    speaker_id: str,
) -> SessionStateBundle:
    """Load and route group/student state exactly once for the current turn."""
    group_state = load_session_state(session, speaker_id="group")
    student_state = (
        load_session_state(session, speaker_id=speaker_id)
        if speaker_id != "group"
        else None
    )

    for criterion in rubric:
        criterion_id = str(criterion["id"])
        group_state.get_or_init_coverage(
            criterion_id,
            questions_to_ask=int(criterion["questions_to_ask"]),
        )
        group_state.get_or_init_bkt(criterion_id)

    answered_criterion_ids = [
        str(criterion_id)
        for criterion_id in answered_topic.get("source_criteria_ids", [])
    ]
    is_individual_topic = False
    for criterion in rubric:
        if str(criterion["id"]) in answered_criterion_ids:
            is_individual_topic = criterion.get("is_individual", False)
            break

    active_state = group_state
    if student_state and is_individual_topic:
        active_state = student_state
        for criterion_id in answered_criterion_ids:
            if (
                criterion_id not in active_state.bkt_states
                and criterion_id in group_state.bkt_states
            ):
                active_state.bkt_states[criterion_id] = copy.deepcopy(
                    group_state.bkt_states[criterion_id]
                )
            if (
                criterion_id not in active_state.coverage
                and criterion_id in group_state.coverage
            ):
                active_state.coverage[criterion_id] = copy.deepcopy(
                    group_state.coverage[criterion_id]
                )

            active_state.get_or_init_coverage(criterion_id)
            active_state.get_or_init_bkt(criterion_id)

    return SessionStateBundle(
        group_state=group_state,
        student_state=student_state,
        active_state=active_state,
        unified_state=build_unified_state(group_state, student_state, rubric),
        speaker_id=speaker_id,
        is_individual_topic=is_individual_topic,
    )
