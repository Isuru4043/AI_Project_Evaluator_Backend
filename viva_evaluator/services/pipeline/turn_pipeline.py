"""
Turn pipeline — orchestrates one full viva turn end-to-end.

Replaces the legacy `viva_evaluator.services.session_manager` glue with a
staged state machine used by AnswerSubmitView.

FLOW (per turn):
    process_answer_and_pick_next(session_id, prev_question, student_answer)
        ↓ assess answer and apply fairness rescue
        ↓ update ability and coverage
        ↓ check termination from the updated state
        ↓ if continuing: plan from current mastery, then generate
        → returns a computation result for the persistence stage
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from viva_evaluator.services.pipeline.bloom import bloom_to_difficulty
from viva_evaluator.services.pipeline.context import (
    build_recent_transcript,
    grounding_is_weak,
    load_rubric,
    load_viva_topics,
    resolve_answered_topic,
)
from viva_evaluator.services.pipeline.stages import (
    apply_fairness_adjustments,
    assess_answer,
    plan_fairness_checks,
    record_scored_turn,
    resolve_fairness_futures,
    submit_fairness_checks,
    collect_topic_hints,
    plan_next_question,
    update_topic_ability,
)
from viva_evaluator.services.pipeline.state_bundle import load_state_bundle
from viva_evaluator.services.pipeline.topic_selector import pick_next_topic

logger = logging.getLogger(__name__)

# A1 Response Triage: maximum CONSECUTIVE clarification re-asks before the
# pipeline proceeds to score regardless (prevents a student stalling forever).
CLARIFICATION_CAP = 1

# Backward-compatible private names for older imports.
_grounding_is_weak = grounding_is_weak
_resolve_answered_topic = resolve_answered_topic
_build_recent_transcript = build_recent_transcript


# =============================================================================
# Public API
# =============================================================================

def process_answer_and_pick_next(
    session,
    submission,
    prev_question_obj,
    student_answer: str,
    speech_metrics: Optional[Dict] = None,
    speaker_id: str = "group",
    examiner_paused: bool = False,
) -> Dict:
    """
    Score the student's answer to prev_question_obj, update BKT, check
    termination, then either return a next-question payload or signal
    session_complete.

    Args:
        session:           EvaluationSession instance (already saved).
        submission:        ProjectSubmission tied to this session.
        prev_question_obj: VivaQuestion instance just answered.
        student_answer:    Raw text the student gave.
        speech_metrics:    Optional dict from frontend with pause intervals
                           etc. Used purely for the speech confidence flag —
                           never enters scoring math.

    Returns dict with shape:
        {
            'analysis': { ... 3D rubric output ... },
            'soft_score': float,
            'speech_confidence': dict,            # Week 6
            'session_complete': bool,
            'termination_reason': str | None,
            'next_question_payload': dict | None,   # set when not complete
        }
    """
    from viva_evaluator.services.pipeline.termination import should_terminate
    from viva_evaluator.services.agents import (
        generate_anchored_question, QuestionerInput,
    )

    import time as _t
    _turn_t0 = _t.time()
    _stage_marks = []

    def _mark(label):
        now = _t.time()
        step_elapsed = now - (_stage_marks[-1][1] if _stage_marks else _turn_t0)
        total_elapsed = now - _turn_t0
        _stage_marks.append((label, now, step_elapsed, total_elapsed))
        logger.info('[turn-timing] %-28s step: %5.2fs | cumul: %5.2fs', label, step_elapsed, total_elapsed)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    rubric = load_rubric(session.project)
    # Resolve which topic was being asked about
    viva_topics = load_viva_topics(session)
    answered_topic = resolve_answered_topic(prev_question_obj, viva_topics)
    state_bundle = load_state_bundle(
        session,
        rubric,
        answered_topic,
        speaker_id,
    )
    group_state = state_bundle.group_state
    student_state = state_bundle.student_state
    active_state = state_bundle.active_state
    unified_state = state_bundle.unified_state

    # ------------------------------------------------------------------
    # Step A — Hybrid retrieval for the answered criterion
    # ------------------------------------------------------------------
    _mark('setup')
    assessment = assess_answer(
        session=session,
        submission=submission,
        previous_question=prev_question_obj,
        student_answer=student_answer,
        answered_topic=answered_topic,
        speech_metrics=speech_metrics,
        clarification_allowed=(
            group_state.clarification_streak < CLARIFICATION_CAP
        ),
        marker=_mark,
    )
    retrieval = assessment.retrieval
    triage = assessment.triage

    if assessment.clarification_required:
        group_state.clarification_streak += 1
        prev_bloom = (
            getattr(prev_question_obj, 'blooms_level', 'Analyze') or 'Analyze'
        )

        if assessment.is_restate:
            question_data = {
                'question_text':  prev_question_obj.question_text,
                'blooms_level':   prev_bloom,
                'difficulty':     _bloom_to_difficulty(prev_bloom),
                'tier1_passed':   True,
                'tier1_failures': [],
                'critic_ran':     False,
                'critic_passed':  None,
                'critic_critique': '',
                'critic_scores':  {},
                'attempts':       0,
                'validation_status': 'tier1_only_policy',
                'validation_degraded': False,
                'degradation_reason': '',
                'fallback_used': False,
                'critic_available': True,
            }
        else:
            question_data = generate_anchored_question(QuestionerInput(
                criterion_name=answered_topic['topic_name'],
                criterion_description=answered_topic['topic_focus'],
                retrieved_chunks=retrieval['chunks'],
                kg_signals=retrieval,
                difficulty=_bloom_to_difficulty(prev_bloom),
                target_bloom=prev_bloom,
                socratic_intent='clarifying',
                intent_prompt_hint=(
                    'Rephrase the same underlying question more clearly.'
                ),
                question_hints=list(collect_topic_hints(answered_topic, rubric)),
                recent_questions=[],
                previous_question=prev_question_obj.question_text,
                previous_answer=student_answer,
                is_first_question=False,
                clarify_mode=True,
                clarify_reason=triage.get('rationale', ''),
                session_id=str(session.id),
            ))
            question_data['blooms_level'] = prev_bloom

        logger.info(
            '[turn] %s (streak=%d/%d) label=%s for topic=%s',
            'RESTATE' if assessment.is_restate else 'CLARIFICATION',
            group_state.clarification_streak, CLARIFICATION_CAP,
            triage['label'], answered_topic['topic_name'],
        )

        return {
            'clarification':       True,
            'triage':              triage,
            'session_complete':    False,
            'analysis':            None,
            'soft_score':          None,
            'speech_confidence':   {},
            'clarification_attempt': group_state.clarification_streak,
            'clarified_question_payload': {
                'question_data':   question_data,
                'topic':           answered_topic,
                'bloom_level':     prev_bloom,
                'difficulty':      _bloom_to_difficulty(prev_bloom),
            },
            '_state_bundle':       state_bundle,
        }

    # A real attempt resets the bounded clarification streak.
    group_state.clarification_streak = 0
    confidence = assessment.speech_confidence

    # ------------------------------------------------------------------
    # Step B.1-3 — finish all fairness adjustments before state changes.
    # ------------------------------------------------------------------
    fairness_plan = plan_fairness_checks(
        assessment,
        answered_topic=answered_topic,
        student_answer=student_answer,
    )
    with ThreadPoolExecutor(max_workers=3) as fairness_executor:
        fairness_futures = submit_fairness_checks(
            fairness_executor,
            plan=fairness_plan,
            assessment=assessment,
            previous_question=prev_question_obj,
            student_answer=student_answer,
            answered_topic=answered_topic,
        )
        fairness_verdicts = resolve_fairness_futures(fairness_futures)

    _mark('B:fairness(parallel)')
    adjusted_assessment = apply_fairness_adjustments(
        assessment,
        fairness_plan,
        fairness_verdicts,
        marker=_mark,
    )
    analysis = adjusted_assessment.analysis
    soft_score = adjusted_assessment.soft_score
    correctness = adjusted_assessment.correctness

    # Step C — update ability from the final, fairness-adjusted score.
    update_topic_ability(
        active_state=active_state,
        answered_topic=answered_topic,
        soft_score=soft_score,
        bloom_level=(
            getattr(prev_question_obj, 'blooms_level', 'Analyze') or 'Analyze'
        ),
    )

    # Record coverage and rebuild the authoritative group/individual view.
    record_scored_turn(
        active_state=active_state,
        group_state=group_state,
        answered_topic=answered_topic,
        correctness=correctness,
        soft_score=soft_score,
    )

    unified_state = state_bundle.rebuild_unified(rubric)

    # Step D — pure termination decision from explicit current counts.
    ai_question_count = session.viva_questions.filter(
        question_source='ai'
    ).count()
    decision = should_terminate(
        unified_state,
        rubric,
        ai_question_count=ai_question_count,
        hard_cap=session.max_total_questions,
    )
    _mark('D:ability+termination')
    if decision.should_end:
        _mark('G:computed')
        total_time = _t.time() - _turn_t0
        table_lines = ["\n" + "=" * 45]
        table_lines.append(f"{'PIPELINE STEP (TERMINATED)':<25} | {'STEP (s)':>10} | {'CUMULATIVE (s)':>15}")
        table_lines.append("-" * 55)
        for label, timestamp, step_elapsed, total_elapsed in _stage_marks:
            table_lines.append(f"{label:<25} | {step_elapsed:>9.2f}s | {total_elapsed:>14.2f}s")
        table_lines.append("-" * 55)
        table_lines.append(f"{'TOTAL (attempts=0)':<25} | {'':>10} | {total_time:>14.2f}s")
        table_lines.append("=" * 55 + "\n")
        logger.info("\n".join(table_lines))

        return {
            'analysis':              analysis,
            'soft_score':            soft_score,
            'speech_confidence':     confidence,
            'session_complete':      True,
            'termination_reason':    decision.reason,
            'next_question_payload': None,
            '_state_bundle':         state_bundle,
        }

    # A paused examiner still receives the scored answer, but no retrieval or
    # question-generation work is started until the session resumes.
    if examiner_paused:
        _mark('G:computed-paused')
        return {
            'analysis':              analysis,
            'soft_score':            soft_score,
            'speech_confidence':     confidence,
            'session_complete':      False,
            'termination_reason':    None,
            'paused_by_examiner':    True,
            'strategy':              {},
            'next_topic':            None,
            'next_question_payload': None,
            '_state_bundle':         state_bundle,
        }

    # Step E — plan from the post-answer mastery state and adjusted analysis.
    planned_question = plan_next_question(
        session=session,
        submission=submission,
        topics=viva_topics,
        rubric=rubric,
        unified_state=unified_state,
        intent_history=group_state.intent_history,
        adjusted_analysis=analysis,
        speech_confidence=confidence,
        answered_topic=answered_topic,
        answered_retrieval=retrieval,
        student_answer=student_answer,
        marker=_mark,
    )
    plan = planned_question.plan
    strategy = planned_question.strategy
    grounding = planned_question.grounding
    next_topic = plan.topic.to_pipeline_dict()

    group_state.intent_history.append(plan.socratic_intent)
    if len(group_state.intent_history) > 30:
        group_state.intent_history = group_state.intent_history[-30:]

    # Step F — generate with the exact planning decision and full context.
    question_data = generate_anchored_question(
        QuestionerInput(
            criterion_name=plan.topic.name,
            criterion_description=plan.topic.focus,
            retrieved_chunks=grounding.retrieval.get('chunks') or [],
            module_chunks=list(grounding.module_chunks),
            kg_signals=grounding.retrieval,
            difficulty=plan.difficulty,
            target_bloom=plan.target_bloom,
            socratic_intent=plan.socratic_intent,
            intent_prompt_hint=plan.intent_prompt_hint,
            question_hints=list(grounding.question_hints),
            recent_questions=list(grounding.recent_questions),
            previous_question=prev_question_obj.question_text,
            previous_answer=student_answer,
            is_first_question=plan.is_first_for_topic,
            question_number_in_criterion=plan.question_number_in_topic,
            weak_grounding=grounding.weak_grounding,
            session_id=str(session.id),
            evidence_package=grounding.evidence_package,
        )
    )
    _mark('F:questioner')

    # ------------------------------------------------------------------
    # Step G — Return computed state to the atomic persistence boundary.
    # ------------------------------------------------------------------
    _mark('G:computed')
    
    total_time = _t.time() - _turn_t0
    table_lines = ["\n" + "=" * 45]
    table_lines.append(f"{'PIPELINE STEP':<25} | {'STEP (s)':>10} | {'CUMULATIVE (s)':>15}")
    table_lines.append("-" * 55)
    for label, timestamp, step_elapsed, total_elapsed in _stage_marks:
        table_lines.append(f"{label:<25} | {step_elapsed:>9.2f}s | {total_elapsed:>14.2f}s")
    table_lines.append("-" * 55)
    table_lines.append(f"{'TOTAL (attempts=' + str(question_data.get('attempts', 1)) + ')':<25} | {'':>10} | {total_time:>14.2f}s")
    table_lines.append("=" * 55 + "\n")
    
    logger.info("\n".join(table_lines))

    return {
        'analysis':              analysis,
        'soft_score':            soft_score,
        'speech_confidence':     confidence,
        'session_complete':      False,
        'termination_reason':    None,
        'strategy':              strategy,
        'next_topic':            next_topic,
        'next_question_payload': {
            'question_data':        question_data,
            'topic':                next_topic,
            'bloom_level':          plan.target_bloom,
            'socratic_intent':      plan.socratic_intent,
            'difficulty':           plan.difficulty,
            'p_lt':                 plan.mastery_probability,
        },
        '_state_bundle':         state_bundle,
    }


# =============================================================================
# Internals
# =============================================================================

def _bloom_to_difficulty(bloom: str) -> str:
    """Backward-compatible alias for the shared Bloom helper."""
    return bloom_to_difficulty(bloom)
