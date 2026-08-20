"""Build the evidence boundary shared by question generation and critique."""

import hashlib
from typing import Dict, Iterable, Optional

from viva_evaluator.services.pipeline.contracts import (
    EvidenceReference,
    QuestionEvidencePackage,
)


def build_question_evidence_package(
    *,
    retrieval: Dict,
    module_chunks: Iterable[Dict],
    previous_answer: str = "",
    weak_grounding: bool = False,
    session=None,
    session_id: Optional[str] = None,
) -> QuestionEvidencePackage:
    references = []
    seen = set()

    def add(reference: EvidenceReference) -> None:
        if reference.evidence_id and reference.evidence_id not in seen:
            seen.add(reference.evidence_id)
            references.append(reference)

    for chunk in retrieval.get("chunks") or []:
        add(_chunk_reference(chunk, "submission_chunk", "submission"))

    for chunk in module_chunks:
        add(_chunk_reference(chunk, "module_chunk", "module"))

    for edge in retrieval.get("contradicts_code_alerts") or []:
        evidence_id = edge.get("evidence_id") or stable_evidence_id(
            "kg:contradiction",
            edge.get("source"),
            edge.get("target"),
        )
        edge.setdefault("evidence_id", evidence_id)
        attrs = edge.get("attrs") or {}
        add(
            EvidenceReference(
                evidence_id=evidence_id,
                evidence_type="kg_contradiction",
                content=(
                    f"Code finding {edge.get('source', '')}: "
                    f"{attrs.get('finding_detail', '')}. Contradicts report "
                    f"claim {edge.get('target', '')}: "
                    f"{attrs.get('claim_excerpt', '')}"
                ).strip(),
                metadata=dict(edge),
            )
        )

    for edge in retrieval.get("alternative_edges") or []:
        evidence_id = edge.get("evidence_id") or stable_evidence_id(
            "kg:alternative",
            edge.get("base_tech"),
            edge.get("alternative"),
            edge.get("edge_type"),
        )
        edge.setdefault("evidence_id", evidence_id)
        add(
            EvidenceReference(
                evidence_id=evidence_id,
                evidence_type="kg_alternative",
                content=(
                    f"{edge.get('base_tech', '')} has supported alternative "
                    f"{edge.get('alternative', '')}. "
                    f"{edge.get('rationale', '')}"
                ).strip(),
                metadata=dict(edge),
            )
        )

    dependency_items = retrieval.get("dependency_evidence") or [
        {
            "dependency": dependency,
            "evidence_id": stable_evidence_id("kg:dependency", dependency),
        }
        for dependency in retrieval.get("depends_on_topics") or []
    ]
    for item in dependency_items:
        dependency = item.get("dependency", "")
        add(
            EvidenceReference(
                evidence_id=item.get("evidence_id")
                or stable_evidence_id("kg:dependency", dependency),
                evidence_type="kg_dependency",
                content=f"The student's code depends on {dependency}.",
                metadata={"dependency": str(dependency)},
            )
        )

    if previous_answer.strip():
        session_scope = str(
            getattr(session, "id", None) or session_id or "unscoped"
        )
        add(
            EvidenceReference(
                evidence_id=stable_evidence_id(
                    f"previous-answer:{session_scope}",
                    previous_answer.strip(),
                ),
                evidence_type="previous_answer",
                content=previous_answer.strip(),
                metadata={},
            )
        )

    for reference in _load_presentation_references(
        session=session,
        session_id=session_id,
    ):
        add(reference)

    return QuestionEvidencePackage(
        references=tuple(references),
        weak_grounding=weak_grounding,
    )


def ensure_question_evidence_package(questioner_input) -> QuestionEvidencePackage:
    """Return the caller's package or build it for a legacy input."""
    package = getattr(questioner_input, "evidence_package", None)
    if package is None:
        package = build_question_evidence_package(
            retrieval=questioner_input.kg_signals
            or {"chunks": questioner_input.retrieved_chunks},
            module_chunks=questioner_input.module_chunks,
            previous_answer=questioner_input.previous_answer or "",
            weak_grounding=questioner_input.weak_grounding,
            session_id=questioner_input.session_id,
        )
        questioner_input.evidence_package = package
    return package


def stable_evidence_id(namespace: str, *parts) -> str:
    normalized = "|".join(str(part or "").strip() for part in parts)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{namespace}:{digest}"


def _chunk_reference(chunk: Dict, evidence_type: str, namespace: str):
    content = str(chunk.get("text") or "").strip()
    evidence_id = chunk.get("evidence_id") or stable_evidence_id(
        f"{namespace}:chunk",
        chunk.get("chunk_idx"),
        chunk.get("source"),
        chunk.get("section"),
        content,
    )
    chunk.setdefault("evidence_id", evidence_id)
    metadata = {
        key: value
        for key, value in chunk.items()
        if key not in {"text", "embedding"}
    }
    return EvidenceReference(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        content=content,
        metadata=metadata,
    )


def _load_presentation_references(*, session=None, session_id=None):
    if session is None and not session_id:
        return ()
    try:
        from core.models import DemoCapturedSegment, EvaluationSession

        if session is None:
            session = EvaluationSession.objects.select_related("student").get(
                id=session_id
            )
        segments = DemoCapturedSegment.objects.filter(
            session=session,
            student=getattr(session, "student", None),
            is_processed=True,
        ).order_by("sequence_number", "timestamp")
        return tuple(
            EvidenceReference(
                evidence_id=f"presentation:{session.id}:segment:{segment.id}",
                evidence_type="presentation_segment",
                content=str(segment.processed_text or "").strip(),
                metadata={
                    "segment_type": segment.segment_type,
                    "sequence_number": segment.sequence_number,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                },
            )
            for segment in segments
            if str(segment.processed_text or "").strip()
        )
    except Exception:
        return ()
