"""Typed contracts shared by the viva pipeline stages.

The orchestration layer still returns the legacy dictionaries expected by the
API views.  These contracts make the boundaries between stages explicit while
the pipeline is migrated incrementally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple


BloomLevel = Literal[
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]
Difficulty = Literal["easy", "medium", "hard"]
ValidationStatus = Literal[
    "fully_validated",
    "tier1_only_policy",
    "critic_unavailable",
    "safe_fallback",
    "rejected",
]
TTSStatus = Literal["disabled", "pending", "ready", "failed"]
EvidenceType = Literal[
    "submission_chunk",
    "module_chunk",
    "kg_contradiction",
    "kg_alternative",
    "kg_dependency",
    "previous_answer",
    "presentation_segment",
]


@dataclass(frozen=True)
class EvidenceReference:
    """One stable, attributable source available to both question agents."""

    evidence_id: str
    evidence_type: EvidenceType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuestionEvidencePackage:
    """The single evidence boundary shared by Questioner and Critic."""

    references: Tuple[EvidenceReference, ...] = ()
    weak_grounding: bool = False

    @property
    def evidence_ids(self) -> Tuple[str, ...]:
        return tuple(reference.evidence_id for reference in self.references)

    def get(self, evidence_id: str) -> Optional[EvidenceReference]:
        return next(
            (
                reference
                for reference in self.references
                if reference.evidence_id == evidence_id
            ),
            None,
        )

    def of_type(self, *evidence_types: EvidenceType) -> Tuple[EvidenceReference, ...]:
        allowed = set(evidence_types)
        return tuple(
            reference
            for reference in self.references
            if reference.evidence_type in allowed
        )


@dataclass(frozen=True)
class VivaTopicRef:
    """Stable, typed view of a grouped viva topic."""

    name: str
    focus: str
    criterion_ids: Tuple[str, ...]
    suggested_questions: int

    @classmethod
    def from_mapping(cls, topic: Mapping[str, Any]) -> "VivaTopicRef":
        return cls(
            name=str(topic.get("topic_name") or "General"),
            focus=str(topic.get("topic_focus") or ""),
            criterion_ids=tuple(
                str(criterion_id)
                for criterion_id in topic.get("source_criteria_ids", [])
            ),
            suggested_questions=int(topic.get("suggested_questions", 2)),
        )

    def to_pipeline_dict(self) -> Dict[str, Any]:
        """Return the legacy shape consumed by existing pipeline code."""
        return {
            "topic_name": self.name,
            "topic_focus": self.focus,
            "source_criteria_ids": list(self.criterion_ids),
            "suggested_questions": self.suggested_questions,
        }


@dataclass(frozen=True)
class AnswerAssessment:
    """Output of retrieval, triage, answer analysis, and speech confidence."""

    retrieval: Dict[str, Any]
    transcript_recent: List[Dict[str, str]]
    triage: Dict[str, Any]
    analysis: Optional[Dict[str, Any]] = None
    soft_score: Optional[float] = None
    correctness: Optional[float] = None
    speech_confidence: Dict[str, Any] = field(default_factory=dict)
    clarification_required: bool = False
    is_restate: bool = False


@dataclass(frozen=True)
class FairnessCheckPlan:
    """Conditional fairness work that should run for an assessed answer."""

    check_consistency: bool
    check_charitable_interpretation: bool
    check_self_correction: bool
    consistency_score: float
    previous_answer: str = ""
    requested_checks: Tuple[str, ...] = ()
    routing_reasons: Tuple[str, ...] = ()
    max_llm_calls: int = 1


@dataclass(frozen=True)
class FairnessVerdicts:
    consistency: Optional[Dict[str, Any]] = None
    charitable: Optional[Dict[str, Any]] = None
    self_correction: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class FairnessAdjustedAssessment:
    """Authoritative score after all applicable fairness rescue checks."""

    analysis: Dict[str, Any]
    soft_score: float
    correctness: float


@dataclass(frozen=True)
class NextQuestionPlan:
    """Decision contract for the future dedicated planning stage."""

    topic: VivaTopicRef
    target_bloom: BloomLevel
    difficulty: Difficulty
    socratic_intent: str
    intent_prompt_hint: str
    rationale: str
    mastery_probability: float
    question_number_in_topic: int
    is_first_for_topic: bool


@dataclass(frozen=True)
class QuestionGroundingContext:
    """Retrieved and historical evidence accompanying a question plan."""

    retrieval: Dict[str, Any]
    module_chunks: Tuple[Dict[str, Any], ...]
    question_hints: Tuple[str, ...]
    recent_questions: Tuple[str, ...]
    weak_grounding: bool
    evidence_package: QuestionEvidencePackage


@dataclass(frozen=True)
class PlannedQuestion:
    """Complete handoff from planning to candidate generation."""

    plan: NextQuestionPlan
    grounding: QuestionGroundingContext
    strategy: Dict[str, Any]


@dataclass(frozen=True)
class QuestionCandidate:
    question_text: str
    blooms_level: BloomLevel
    difficulty: Difficulty
    raw_response: Dict[str, Any] = field(default_factory=dict)
    candidate_hash: str = ""
    socratic_intent: str = ""
    source_reference_ids: Tuple[str, ...] = ()
    schema_failures: Tuple[str, ...] = ()

    @property
    def target_bloom(self) -> BloomLevel:
        return self.blooms_level


@dataclass(frozen=True)
class ValidatedQuestion:
    question_text: str
    blooms_level: BloomLevel
    difficulty: Difficulty
    tier1_passed: bool
    tier1_failures: Tuple[str, ...] = ()
    critic_passed: Optional[bool] = None
    critic_critique: str = ""
    critic_scores: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    candidate_hash: str = ""
    socratic_intent: str = ""
    source_reference_ids: Tuple[str, ...] = ()
    schema_failures: Tuple[str, ...] = ()
    validation_status: ValidationStatus = "tier1_only_policy"
    validation_degraded: bool = False
    degradation_reason: str = ""
    fallback_used: bool = False
    critic_available: bool = True
    tts_status: TTSStatus = "disabled"
    tts_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def target_bloom(self) -> BloomLevel:
        return self.blooms_level

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Return the response shape used by existing views and persistence."""
        return {
            "question_text": self.question_text,
            "blooms_level": self.blooms_level,
            "difficulty": self.difficulty,
            "tier1_passed": self.tier1_passed,
            "tier1_failures": list(self.tier1_failures),
            "critic_ran": self.critic_passed is not None,
            "critic_passed": self.critic_passed,
            "critic_critique": self.critic_critique,
            "critic_scores": dict(self.critic_scores),
            "attempts": self.attempts,
            "candidate_hash": self.candidate_hash,
            "socratic_intent": self.socratic_intent,
            "source_reference_ids": list(self.source_reference_ids),
            "schema_failures": list(self.schema_failures),
            "validation_status": self.validation_status,
            "validation_degraded": self.validation_degraded,
            "degradation_reason": self.degradation_reason,
            "fallback_used": self.fallback_used,
            "critic_available": self.critic_available,
            "tts_status": self.tts_status,
            "tts_metadata": dict(self.tts_metadata),
        }


@dataclass(frozen=True)
class ClarificationOutcome:
    triage: Dict[str, Any]
    question_payload: Dict[str, Any]
    clarification_attempt: int


@dataclass(frozen=True)
class SessionCompletedOutcome:
    analysis: Dict[str, Any]
    soft_score: float
    speech_confidence: Dict[str, Any]
    termination_reason: str


@dataclass(frozen=True)
class NextQuestionOutcome:
    analysis: Dict[str, Any]
    soft_score: float
    speech_confidence: Dict[str, Any]
    strategy: Dict[str, Any]
    question_payload: Dict[str, Any]
