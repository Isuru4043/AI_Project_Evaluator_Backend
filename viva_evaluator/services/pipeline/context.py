"""Read-only context helpers used by the viva turn pipeline."""

from typing import Any, Dict, List


WEAK_GROUNDING_THRESHOLD = 0.30

_RUBRIC_CACHE: Dict[str, List[Dict]] = {}


def grounding_is_weak(
    chunks: List[Dict],
    threshold: float = WEAK_GROUNDING_THRESHOLD,
) -> bool:
    """Return whether retrieved evidence is too weak for a specific question."""
    if not chunks:
        return True
    best = max((float(chunk.get("score", 0.0)) for chunk in chunks), default=0.0)
    return best < threshold


def load_rubric(project) -> List[Dict]:
    """Load all rubric criteria for a project in document order."""
    project_id = str(project.id)
    if project_id in _RUBRIC_CACHE:
        return _RUBRIC_CACHE[project_id]

    rubric: List[Dict] = []
    for category in project.rubric_categories.all().order_by("id"):
        for criterion in category.criteria.all().order_by("id"):
            hints = list(
                criterion.question_hints.values_list("hint_text", flat=True)
            )
            rubric.append(
                {
                    "id": str(criterion.id),
                    "name": criterion.criteria_name,
                    "description": criterion.description or "",
                    "max_score": float(criterion.max_score),
                    "category": category.category_name,
                    "questions_to_ask": int(criterion.questions_to_ask or 3),
                    "hints": hints,
                    "is_individual": criterion.is_individual,
                }
            )

    _RUBRIC_CACHE[project_id] = rubric
    return rubric


def clear_rubric_cache() -> None:
    """Clear the process-local rubric cache, primarily for tests and refreshes."""
    _RUBRIC_CACHE.clear()


def load_viva_topics(session) -> List[Dict]:
    """Return grouped viva topics, falling back to one topic per criterion."""
    if session.grouping_cache and session.grouping_cache.grouped_criteria:
        topics = session.grouping_cache.grouped_criteria.get("viva_topics", [])
        if topics:
            return topics

    return [
        {
            "topic_name": criterion["name"],
            "source_criteria_ids": [criterion["id"]],
            "suggested_questions": criterion["questions_to_ask"],
            "topic_focus": criterion["description"],
        }
        for criterion in load_rubric(session.project)
    ]


def resolve_answered_topic(question_obj, topics: List[Dict]) -> Dict:
    """Find the topic associated with a previously persisted question."""
    if getattr(question_obj, "viva_topic_name", None):
        for topic in topics:
            if topic["topic_name"] == question_obj.viva_topic_name:
                return topic

    criterion_ids = []
    try:
        extension = question_obj.extension
        if extension and extension.criteria_id:
            criterion_ids.append(str(extension.criteria_id))
    except Exception:
        pass

    return {
        "topic_name": "General",
        "source_criteria_ids": criterion_ids,
        "suggested_questions": 2,
        "topic_focus": "",
    }


def build_recent_transcript(session, limit: int = 5) -> List[Dict[str, Any]]:
    """Load the most recent Q/A pairs in chronological order."""
    from django.db.models import Prefetch

    from core.models import VivaAnswer

    questions = list(
        session.viva_questions.order_by("-question_order").prefetch_related(
            Prefetch(
                "answers",
                queryset=VivaAnswer.objects.order_by("-answered_at"),
            )
        )[:limit]
    )

    pairs = []
    for question in reversed(questions):
        answers = list(question.answers.all())
        last_answer = answers[0] if answers else None
        pairs.append(
            {
                "question_text": question.question_text,
                "question_order": question.question_order,
                "topic_name": question.viva_topic_name or "",
                "source_criteria_ids": list(
                    question.source_criteria_ids or []
                ),
                "answer_text": (
                    last_answer.transcribed_answer if last_answer else ""
                ),
            }
        )
    return pairs
