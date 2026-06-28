"""
SessionState — typed view over per-session memory.

Persisted on EvaluationSession.bkt_state_json (added in migration 0007).
Holds:
    - bkt_states: Dict[criterion_id → AbilityState dict]
    - intent_history: list of intents used (anti-repetition)
    - rubric_coverage: per-criterion turn counts and avg correctness
    - last_soft_scores: rolling list (for diagnostics)

This is a TYPED STRUCTURE around a JSON blob, NOT a Django model.
We chose JSON over a separate table to keep migrations simple — every
session has exactly one state, lifecycle == session lifecycle.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from viva_evaluator.services.bkt.ability_engine import AbilityState

logger = logging.getLogger(__name__)


# =============================================================================
# State container
# =============================================================================

@dataclass
class CriterionCoverage:
    criterion_id: str
    turns: int = 0
    correct_turns: int = 0          # turns with correctness >= 0.3
    sum_correctness: float = 0.0
    questions_to_ask: int = 3       # min turns required (set from rubric)

    @property
    def avg_correctness(self) -> float:
        return self.sum_correctness / self.turns if self.turns else 0.0

    def to_dict(self) -> Dict:
        return {
            'criterion_id':    self.criterion_id,
            'turns':           self.turns,
            'correct_turns':   self.correct_turns,
            'sum_correctness': self.sum_correctness,
            'questions_to_ask': self.questions_to_ask,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CriterionCoverage':
        return cls(
            criterion_id=data.get('criterion_id', ''),
            turns=data.get('turns', 0),
            correct_turns=data.get('correct_turns', 0),
            sum_correctness=data.get('sum_correctness', 0.0),
            questions_to_ask=data.get('questions_to_ask', 3),
        )


@dataclass
class SessionState:
    """Top-level session memory persisted between turns."""

    bkt_states: Dict[str, AbilityState] = field(default_factory=dict)
    coverage: Dict[str, CriterionCoverage] = field(default_factory=dict)
    intent_history: List[str] = field(default_factory=list)
    soft_score_history: List[float] = field(default_factory=list)
    total_turns: int = 0
    # A1 Response Triage: number of CONSECUTIVE clarification re-asks. Bounded
    # so a student cannot stall indefinitely. Reset to 0 on any scored turn.
    clarification_streak: int = 0

    # ------------------------------------------------------------------
    # BKT helpers
    # ------------------------------------------------------------------

    def get_or_init_bkt(self, criterion_id: str) -> AbilityState:
        if criterion_id not in self.bkt_states:
            self.bkt_states[criterion_id] = AbilityState(concept_id=criterion_id)
        return self.bkt_states[criterion_id]

    # ------------------------------------------------------------------
    # Coverage helpers
    # ------------------------------------------------------------------

    def get_or_init_coverage(self, criterion_id: str, questions_to_ask: int = 3) -> CriterionCoverage:
        cov = self.coverage.get(criterion_id)
        if cov is None:
            cov = CriterionCoverage(
                criterion_id=criterion_id,
                questions_to_ask=questions_to_ask,
            )
            self.coverage[criterion_id] = cov
        else:
            # Keep questions_to_ask in sync with the current rubric value
            cov.questions_to_ask = questions_to_ask
        return cov

    def record_turn(
        self,
        criterion_id: str,
        soft_score: float,
        correctness_score: float,
        intent: str,
    ) -> None:
        """Bump counters after a turn is processed."""
        self.total_turns += 1

        cov = self.get_or_init_coverage(criterion_id)
        cov.turns += 1
        cov.sum_correctness += correctness_score
        if correctness_score >= 0.3:
            cov.correct_turns += 1

        self.soft_score_history.append(round(soft_score, 4))
        if intent:
            self.intent_history.append(intent)
            # Cap history to reasonable size for prompt context
            if len(self.intent_history) > 30:
                self.intent_history = self.intent_history[-30:]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            'bkt_states':         {k: v.to_dict() for k, v in self.bkt_states.items()},
            'coverage':           {k: v.to_dict() for k, v in self.coverage.items()},
            'intent_history':     list(self.intent_history),
            'soft_score_history': list(self.soft_score_history),
            'total_turns':        self.total_turns,
            'clarification_streak': self.clarification_streak,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'SessionState':
        data = data or {}
        return cls(
            bkt_states={k: AbilityState.from_dict(v) for k, v in (data.get('bkt_states') or {}).items()},
            coverage={k: CriterionCoverage.from_dict(v) for k, v in (data.get('coverage') or {}).items()},
            intent_history=list(data.get('intent_history') or []),
            soft_score_history=list(data.get('soft_score_history') or []),
            total_turns=int(data.get('total_turns') or 0),
            clarification_streak=int(data.get('clarification_streak') or 0),
        )


# =============================================================================
# Persistence helpers (read/write EvaluationSession.bkt_state_json)
# =============================================================================

def load_session_state(session) -> SessionState:
    """Load (or create empty) state from EvaluationSession.bkt_state_json."""
    raw = getattr(session, 'bkt_state_json', None) or {}
    return SessionState.from_dict(raw)


def save_session_state(session, state: SessionState) -> None:
    """Persist state back to the session row."""
    session.bkt_state_json = state.to_dict()
    session.save(update_fields=['bkt_state_json'])
