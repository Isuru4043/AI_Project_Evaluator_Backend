"""
Pipeline — orchestrates a single viva turn end-to-end.

Replaces the legacy session_manager.py glue with a clean state machine:
    1. Load session memory (BKT states, transcript, intent history)
    2. Hybrid retrieval (FAISS + KG)
    3. Analyzer agent → 3D rubric + soft score
    4. BKT update for the active criterion
    5. Strategist agent → next Bloom level + intent
    6. Termination check
    7. Questioner agent → next question
    8. Save updated session state

session_state.py: typed view over EvaluationSession.bkt_state_json + intent history.
termination.py:    three-condition termination logic + 25-turn hard cap.
turn_pipeline.py:  the high-level run_turn() orchestrator (used by views).
"""

from viva_evaluator.services.pipeline.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from viva_evaluator.services.pipeline.termination import (
    should_terminate,
    TerminationDecision,
)
from viva_evaluator.services.pipeline.turn_pipeline import (
    process_answer_and_pick_next,
    load_rubric,
    pick_next_criterion,
)

__all__ = [
    'SessionState',
    'load_session_state',
    'save_session_state',
    'should_terminate',
    'TerminationDecision',
    'process_answer_and_pick_next',
    'load_rubric',
    'pick_next_criterion',
]
