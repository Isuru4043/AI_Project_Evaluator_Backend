"""Next-question planning from the post-answer mastery state."""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple, cast

from viva_evaluator.services.pipeline.bloom import bloom_to_difficulty
from viva_evaluator.services.pipeline.context import grounding_is_weak
from viva_evaluator.services.pipeline.evidence import (
    build_question_evidence_package,
)
from viva_evaluator.services.pipeline.contracts import (
    BloomLevel,
    Difficulty,
    NextQuestionPlan,
    PlannedQuestion,
    QuestionGroundingContext,
    VivaTopicRef,
)
from viva_evaluator.services.pipeline.topic_selector import pick_next_topic


StageMarker = Optional[Callable[[str], None]]


def collect_topic_hints(topic: Dict, rubric: List[Dict]) -> Tuple[str, ...]:
    """Aggregate unique examiner hints across all criteria in a topic."""
    criterion_ids = {
        str(criterion_id)
        for criterion_id in topic.get("source_criteria_ids") or []
    }
    hints = []
    seen = set()
    for criterion in rubric:
        if str(criterion["id"]) not in criterion_ids:
            continue
        for hint in criterion.get("hints") or []:
            normalized = str(hint).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                hints.append(normalized)
    return tuple(hints)


def plan_next_question(
    *,
    session,
    submission,
    topics: List[Dict],
    rubric: List[Dict],
    unified_state,
    intent_history: List[str],
    adjusted_analysis: Dict,
    speech_confidence: Dict,
    answered_topic: Optional[Dict] = None,
    answered_retrieval: Optional[Dict] = None,
    student_answer: str = "",
    marker: StageMarker = None,
) -> PlannedQuestion:
    """Select, ground, and strategize the next question after state update."""
    from viva_evaluator.services.agents.strategist import (
        StrategistInput,
        select_strategy,
    )
    from viva_evaluator.services.rag.retrieval import (
        retrieve_hybrid_for_turn,
        retrieve_module_materials,
    )

    next_topic = pick_next_topic(topics, unified_state, session)
    if next_topic is None:
        raise ValueError(
            "No viva topic remains within its per-concept question cap."
        )

    criterion_ids = next_topic.get("source_criteria_ids") or []
    if not criterion_ids:
        raise ValueError("The selected viva topic has no source rubric criteria.")

    first_criterion_id = str(criterion_ids[0])
    ability = unified_state.bkt_states.get(first_criterion_id)
    coverage = unified_state.coverage.get(first_criterion_id)
    if ability is None or coverage is None:
        raise ValueError(
            f"No initialized mastery state for criterion {first_criterion_id}."
        )

    recent_questions = tuple(
        session.viva_questions.order_by("-question_order")
        .values_list("question_text", flat=True)[:5]
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        module_future = executor.submit(
            retrieve_module_materials,
            project_id=str(session.project.id),
            query=(
                next_topic["topic_name"]
                + " "
                + (next_topic.get("topic_focus") or "")
            ),
            top_k=2,
        )

        retrieval_future = None
        if (
            answered_topic is None
            or next_topic["topic_name"] != answered_topic["topic_name"]
        ):
            retrieval_future = executor.submit(
                retrieve_hybrid_for_turn,
                submission=submission,
                criterion_name=next_topic["topic_name"],
                criterion_description=next_topic.get("topic_focus") or "",
                last_answer=student_answer,
                top_k=3,
            )

        retrieval = (
            retrieval_future.result()
            if retrieval_future is not None
            else (answered_retrieval or {"chunks": []})
        )
        module_chunks = tuple(module_future.result())

    if marker:
        marker("E:planning-retrieval")

    strategy = select_strategy(
        StrategistInput(
            p_lt=ability.p_lt,
            analysis=adjusted_analysis,
            kg_signals=retrieval,
            intent_history=intent_history,
            speech_confidence=speech_confidence.get("flag"),
        )
    )
    if marker:
        marker("E:strategist")

    target_bloom = cast(BloomLevel, strategy["bloom_level"])
    difficulty = cast(Difficulty, bloom_to_difficulty(target_bloom))
    topic_ref = VivaTopicRef.from_mapping(next_topic)
    plan = NextQuestionPlan(
        topic=topic_ref,
        target_bloom=target_bloom,
        difficulty=difficulty,
        socratic_intent=strategy["socratic_intent"],
        intent_prompt_hint=strategy.get("intent_prompt_hint", ""),
        rationale=strategy.get("rationale", ""),
        mastery_probability=ability.p_lt,
        question_number_in_topic=coverage.turns + 1,
        is_first_for_topic=coverage.turns == 0,
    )
    weak_grounding = grounding_is_weak(retrieval.get("chunks") or [])
    evidence_package = build_question_evidence_package(
        retrieval=retrieval,
        module_chunks=module_chunks,
        previous_answer=student_answer,
        weak_grounding=weak_grounding,
        session=session,
    )
    grounding = QuestionGroundingContext(
        retrieval=retrieval,
        module_chunks=module_chunks,
        question_hints=collect_topic_hints(next_topic, rubric),
        recent_questions=recent_questions,
        weak_grounding=weak_grounding,
        evidence_package=evidence_package,
    )
    return PlannedQuestion(plan=plan, grounding=grounding, strategy=strategy)
