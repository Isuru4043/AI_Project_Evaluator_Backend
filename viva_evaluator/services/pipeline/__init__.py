"""
Pipeline — orchestrates a single viva turn end-to-end.

Authoritative turn order:
    assessment → fairness → ability → termination → planning → generation
    → persistence

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
from viva_evaluator.services.pipeline.context import (
    load_rubric,
    load_viva_topics,
)
from viva_evaluator.services.pipeline.contracts import (
    AnswerAssessment,
    EvidenceReference,
    FairnessAdjustedAssessment,
    NextQuestionPlan,
    PlannedQuestion,
    QuestionGroundingContext,
    QuestionEvidencePackage,
    ValidatedQuestion,
    VivaTopicRef,
)
from viva_evaluator.services.pipeline.turn_pipeline import (
    process_answer_and_pick_next,
)
from viva_evaluator.services.pipeline.topic_selector import pick_next_topic
from viva_evaluator.services.pipeline.stages.question_planning import (
    plan_next_question,
)
from viva_evaluator.services.pipeline.orchestrator import (
    VivaPipeline,
    VivaPipelineInputError,
)

__all__ = [
    'SessionState',
    'load_session_state',
    'save_session_state',
    'should_terminate',
    'TerminationDecision',
    'AnswerAssessment',
    'EvidenceReference',
    'FairnessAdjustedAssessment',
    'NextQuestionPlan',
    'PlannedQuestion',
    'QuestionGroundingContext',
    'QuestionEvidencePackage',
    'ValidatedQuestion',
    'VivaTopicRef',
    'process_answer_and_pick_next',
    'load_rubric',
    'pick_next_topic',
    'plan_next_question',
    'load_viva_topics',
    'VivaPipeline',
    'VivaPipelineInputError',
]
