"""
Turn pipeline — orchestrates one full viva turn end-to-end.

Replaces the legacy `viva_evaluator.services.session_manager` glue with a
clean state machine. Used by the AnswerSubmitView (and by SessionStartView
for the very first question, with student_answer='').

FLOW (per turn):
    process_answer_and_pick_next(session_id, prev_question, student_answer)
        ↓ load session, retrieve context, run Analyzer
        ↓ update BKT for the answered criterion
        ↓ check termination
        ↓ if not terminating: pick next criterion, run Strategist + Questioner
        ↓ persist state
        → returns dict with analysis + next_question (or session_complete)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# A1 Response Triage: maximum CONSECUTIVE clarification re-asks before the
# pipeline proceeds to score regardless (prevents a student stalling forever).
CLARIFICATION_CAP = 1

# B3 Weak-retrieval awareness: if the best retrieved chunk's cosine similarity
# is below this, the submission barely covers the criterion → ask a broader
# question instead of fabricating specifics.
WEAK_GROUNDING_THRESHOLD = 0.30

# A2 Charitable interpretation: only runs when correctness falls in this
# borderline band, and can only RAISE the score up to CHARITABLE_FLOOR.
CHARITABLE_BAND = (0.40, 0.60)
CHARITABLE_FLOOR = 0.65

# A3 Material-vs-superficial inconsistency: only review when consistency is
# flagged below this; if superficial, lift it to the neutral value (no penalty).
CONSISTENCY_REVIEW_THRESHOLD = 0.40
CONSISTENCY_NEUTRAL = 0.80

# A4 Self-correction: only run when there's a prior answer AND the current score
# left room to rescue; can only RAISE the score up to SELF_CORRECTION_FLOOR.
SELF_CORRECTION_TRIGGER_MAX = 0.70
SELF_CORRECTION_FLOOR = 0.65


def _grounding_is_weak(chunks: List[Dict], threshold: float = WEAK_GROUNDING_THRESHOLD) -> bool:
    """True if no retrieved chunk clears the similarity threshold (thin coverage)."""
    if not chunks:
        return True
    best = max((float(c.get('score', 0.0)) for c in chunks), default=0.0)
    return best < threshold


# =============================================================================
# Helpers — load rubric and resolve session submission
# =============================================================================

_RUBRIC_CACHE: Dict[str, List[Dict]] = {}


def load_rubric(project) -> List[Dict]:
    """Flat list of all rubric criteria for a project, in document order (cached in RAM)."""
    proj_id = str(project.id)
    if proj_id in _RUBRIC_CACHE:
        return _RUBRIC_CACHE[proj_id]

    out: List[Dict] = []
    for category in project.rubric_categories.all().order_by('id'):
        for crit in category.criteria.all().order_by('id'):
            hints = list(crit.question_hints.values_list('hint_text', flat=True))
            out.append({
                'id':                str(crit.id),
                'name':              crit.criteria_name,
                'description':       crit.description or '',
                'max_score':         float(crit.max_score),
                'category':          category.category_name,
                'questions_to_ask':  int(crit.questions_to_ask or 3),
                'hints':             hints,
            })
    _RUBRIC_CACHE[proj_id] = out
    return out


def load_viva_topics(session) -> List[Dict]:
    """Returns grouped viva topics if available, else falls back to raw criteria as topics."""
    if session.grouping_cache and session.grouping_cache.grouped_criteria:
        topics = session.grouping_cache.grouped_criteria.get('viva_topics', [])
        if topics:
            return topics
            
    # Fallback to 1-to-1 criteria mapping if no grouping exists
    rubric = load_rubric(session.project)
    fallback_topics = []
    for c in rubric:
        fallback_topics.append({
            'topic_name': c['name'],
            'source_criteria_ids': [c['id']],
            'suggested_questions': c['questions_to_ask'],
            'topic_focus': c['description'],
        })
    return fallback_topics


def pick_next_topic(topics: List[Dict], state, session) -> Optional[Dict]:
    """
    Round robin across topics based on topic budgets.
    Applies the 4 rules from the adaptive viva design.
    """
    from viva_evaluator.services.pipeline.termination import WEAK_MASTERY_THRESHOLD

    # We need a quick way to check if a topic has met its budget.
    # A topic meets its budget if ANY (or all?) of its criteria have enough correct turns.
    # Let's say a topic budget is met if its total questions asked >= suggested_questions.
    
    # We can calculate topic_turns by looking at state.coverage for its criteria.
    # Since one question applies to all criteria in a topic, the turns for the first criterion
    # accurately reflects the turns for the topic.
    
    # Rule 1: Find first topic that has not met its question budget yet
    for topic in topics:
        first_crit_id = str(topic['source_criteria_ids'][0])
        cov = state.coverage.get(first_crit_id)
        # Use turns (or correct_turns) against suggested_questions
        budget = topic.get('suggested_questions', 2)
        
        correct_turns = cov.correct_turns if cov else 0
        if correct_turns < budget:
            return topic

    # Rule 3: If all topics met their budget but total_turns < max_total_questions,
    # revisit topics where BKT mastery < 0.40 (WEAK_MASTERY_THRESHOLD)
    if state.total_turns < session.max_total_questions:
        weak_topics = []
        for topic in topics:
            for crit_id in topic['source_criteria_ids']:
                bkt = state.bkt_states.get(str(crit_id))
                if bkt and bkt.p_lt < WEAK_MASTERY_THRESHOLD:
                    weak_topics.append(topic)
                    break # Topic is weak if any criterion is weak
                    
        if weak_topics:
            # Rule 2: Among tied topics, prefer the one with LOWEST BKT mastery
            # For simplicity, just return the first weak topic found
            return weak_topics[0]

    # Rule 4: If all topics have mastery >= 0.40 (or total turns reached), terminate early
    return None


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
    from viva_evaluator.services.pipeline.session_state import (
        load_session_state, save_session_state,
    )
    from viva_evaluator.services.pipeline.termination import should_terminate
    from viva_evaluator.services.bkt.ability_engine import update_ability
    from viva_evaluator.services.agents.analyzer import (
        analyze_answer, AnalyzerInput,
    )
    from viva_evaluator.services.agents.strategist import (
        select_strategy, StrategistInput,
    )
    from viva_evaluator.services.agents import (
        generate_anchored_question, QuestionerInput,
    )
    from viva_evaluator.services.rag.retrieval import retrieve_hybrid_for_turn
    from viva_evaluator.services.confidence import analyze_speech_confidence

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
    state = load_session_state(session, speaker_id=speaker_id)

    # Initialize coverage entries for any criterion that isn't yet tracked
    for crit in rubric:
        state.get_or_init_coverage(str(crit['id']), questions_to_ask=int(crit['questions_to_ask']))
        state.get_or_init_bkt(str(crit['id']))

    # Resolve which topic was being asked about
    viva_topics = load_viva_topics(session)
    answered_topic = _resolve_answered_topic(prev_question_obj, viva_topics)

    # ------------------------------------------------------------------
    # Step A — Hybrid retrieval for the answered criterion
    # ------------------------------------------------------------------
    _mark('setup')
    retrieval = retrieve_hybrid_for_turn(
        submission=submission,
        criterion_name=answered_topic['topic_name'],
        criterion_description=answered_topic['topic_focus'],
        last_answer=student_answer,
        top_k=3,
    )
    _mark('A:retrieval')

    # ------------------------------------------------------------------
    # Step A.5 & Step B — Response Triage (A1) AND Analyzer (3D rubric)
    # Executed IN PARALLEL via ThreadPoolExecutor to minimize latency.
    # ------------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor
    from viva_evaluator.services.agents.response_triage import (
        triage_response, TriageInput, CLARIFY_LABELS, RESTATE_LABELS, LABEL_GARBLED,
    )

    triage_input = TriageInput(
        question_text=prev_question_obj.question_text,
        student_answer=student_answer,
        is_spoken=bool(speech_metrics),
    )

    transcript_recent = _build_recent_transcript(session)
    analyzer_input = AnalyzerInput(
        question_text=prev_question_obj.question_text,
        student_answer=student_answer,
        topic_name=answered_topic['topic_name'],
        topic_focus=answered_topic['topic_focus'],
        retrieved_chunks=retrieval['chunks'],
        contradicts_code_alerts=retrieval.get('contradicts_code_alerts') or [],
        transcript_recent=transcript_recent,
    )

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_triage = executor.submit(triage_response, triage_input)
        future_analyzer = executor.submit(analyze_answer, analyzer_input)

        triage = future_triage.result()
        _mark('A.5:triage(parallel)')

        gate_labels = CLARIFY_LABELS | RESTATE_LABELS
        if triage['label'] in gate_labels and state.clarification_streak < CLARIFICATION_CAP:
            state.clarification_streak += 1
            prev_bloom = getattr(prev_question_obj, 'blooms_level', 'Analyze') or 'Analyze'

            if triage['label'] in RESTATE_LABELS:
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
                }
            else:
                question_data = generate_anchored_question(QuestionerInput(
                    criterion_name=answered_topic['topic_name'],
                    criterion_description=answered_topic['topic_focus'],
                    retrieved_chunks=retrieval['chunks'],
                    kg_signals=retrieval,
                    difficulty=_bloom_to_difficulty(prev_bloom),
                    question_hints=[],
                    recent_questions=[],
                    previous_question=prev_question_obj.question_text,
                    previous_answer=student_answer,
                    is_first_question=False,
                    clarify_mode=True,
                    clarify_reason=triage.get('rationale', ''),
                    session_id=str(session.id),
                ))
                question_data['blooms_level'] = prev_bloom

            save_session_state(session, state)
            logger.info(
                '[turn] %s (streak=%d/%d) label=%s for topic=%s',
                'RESTATE' if triage['label'] in RESTATE_LABELS else 'CLARIFICATION',
                state.clarification_streak, CLARIFICATION_CAP,
                triage['label'], answered_topic['topic_name'],
            )

            return {
                'clarification':       True,
                'triage':              triage,
                'session_complete':    False,
                'analysis':            None,
                'soft_score':          None,
                'speech_confidence':   {},
                'clarification_attempt': state.clarification_streak,
                'clarified_question_payload': {
                    'question_data':   question_data,
                    'topic':           answered_topic,
                    'bloom_level':     prev_bloom,
                    'difficulty':      _bloom_to_difficulty(prev_bloom),
                },
            }

        # Not clarifying (real attempt, or clarification budget exhausted) →
        # reset the streak and proceed to normal scoring.
        state.clarification_streak = 0
        analysis = future_analyzer.result()
        _mark('B:analyzer(parallel)')

    soft_score = float(analysis.get('soft_score', 0.5))
    correctness = float((analysis.get('correctness') or {}).get('score', 0.5))

    # ------------------------------------------------------------------
    # Step B.5 — Confidence Analysis (Instant, needed for Strategist)
    # ------------------------------------------------------------------
    confidence = analyze_speech_confidence(
        answer_text=student_answer,
        speech_metrics=speech_metrics,
    )
    _mark('B.5:confidence')

    # ------------------------------------------------------------------
    # Step E — Pick next topic + run Strategist (instant <0.01s)
    # ------------------------------------------------------------------
    next_topic = pick_next_topic(viva_topics, state, session)
    if next_topic is None:
        next_topic = answered_topic

    # Check if retrieval is needed for the next topic
    if next_topic['topic_name'] != answered_topic['topic_name']:
        retrieval_next = retrieve_hybrid_for_turn(
            submission=submission,
            criterion_name=next_topic['topic_name'],
            criterion_description=next_topic['topic_focus'],
            last_answer=student_answer,
            top_k=3,
        )
    else:
        retrieval_next = retrieval

    first_crit_id = str(next_topic['source_criteria_ids'][0])

    strategy = select_strategy(StrategistInput(
        p_lt=state.bkt_states[first_crit_id].p_lt,
        analysis=analysis,
        kg_signals=retrieval_next,
        intent_history=state.intent_history,
        speech_confidence=confidence.get('flag'),
    ))
    _mark('E:strategist')

    state.intent_history.append(strategy['socratic_intent'])
    if len(state.intent_history) > 30:
        state.intent_history = state.intent_history[-30:]

    next_difficulty = _bloom_to_difficulty(strategy['bloom_level'])
    is_first_for_topic = (
        state.coverage[first_crit_id].turns == 0
    )

    recent_qs = list(
        session.viva_questions.order_by('-question_order')
        .values_list('question_text', flat=True)[:5]
    )

    # ------------------------------------------------------------------
    # Steps B.1-3 (Fairness Rescue) AND Step F (Questioner LLM)
    # Executed IN PARALLEL via ThreadPoolExecutor.
    # ------------------------------------------------------------------
    consistency_dim = analysis.get('consistency') or {}
    consistency_score = float(consistency_dim.get('score', 1.0))
    need_b1 = consistency_score < CONSISTENCY_REVIEW_THRESHOLD
    need_b2 = CHARITABLE_BAND[0] <= correctness <= CHARITABLE_BAND[1]
    need_b3 = soft_score < SELF_CORRECTION_TRIGGER_MAX

    previous_answer = ''
    if need_b3:
        for _pair in reversed(transcript_recent):
            if _pair.get('answer_text'):
                previous_answer = _pair['answer_text']
                break
        if not previous_answer:
            need_b3 = False

    futures = {}
    with ThreadPoolExecutor(max_workers=4) as main_executor:
        # Submit retrieval of module materials
        from viva_evaluator.services.rag.retrieval import retrieve_module_materials
        module_future = main_executor.submit(
            retrieve_module_materials,
            project_id=str(session.project.id),
            query=next_topic['topic_name'] + " " + (next_topic['topic_focus'] or ""),
            top_k=2
        )
        
        # Submit Questioner LLM task (needs module_future result first)
        def _run_questioner():
            mod_chunks = module_future.result()
            return generate_anchored_question(
                QuestionerInput(
                    criterion_name=next_topic['topic_name'],
                    criterion_description=next_topic['topic_focus'],
                    retrieved_chunks=retrieval_next['chunks'],
                    module_chunks=mod_chunks,
                    kg_signals=retrieval_next,
                    difficulty=next_difficulty,
                    question_hints=[], # Topics don't have explicit hints
                    recent_questions=recent_qs,
                    previous_question=prev_question_obj.question_text,
                    previous_answer=student_answer,
                    is_first_question=is_first_for_topic,
                    question_number_in_criterion=state.coverage[first_crit_id].turns + 1,
                    weak_grounding=_grounding_is_weak(retrieval_next['chunks']),
                    session_id=str(session.id),
                )
            )
            
        question_future = main_executor.submit(_run_questioner)

        # Submit Fairness Rescue tasks in parallel
        if need_b1:
            from viva_evaluator.services.agents.consistency_check import (
                classify_inconsistency, ConsistencyInput,
            )
            futures['b1'] = main_executor.submit(
                classify_inconsistency,
                ConsistencyInput(
                    question_text=prev_question_obj.question_text,
                    student_answer=student_answer,
                    transcript_recent=transcript_recent,
                    consistency_evidence=consistency_dim.get('evidence_quote', ''),
                )
            )

        if need_b2:
            from viva_evaluator.services.agents.charitable_check import (
                assess_understanding, CharitableInput,
            )
            futures['b2'] = main_executor.submit(
                assess_understanding,
                CharitableInput(
                    question_text=prev_question_obj.question_text,
                    student_answer=student_answer,
                    criterion_name=answered_topic['topic_name'],
                    criterion_description=answered_topic['topic_focus'],
                    retrieved_chunks=retrieval['chunks'],

                )
            )

        if need_b3:
            from viva_evaluator.services.agents.self_correction import (
                assess_self_correction, SelfCorrectionInput,
            )
            futures['b3'] = main_executor.submit(
                assess_self_correction,
                SelfCorrectionInput(
                    question_text=prev_question_obj.question_text,
                    current_answer=student_answer,
                    previous_answer=previous_answer,
                )
            )

        # Wait for Questioner & Fairness results in parallel
        question_data = question_future.result()
        for key in list(futures.keys()):
            futures[key] = futures[key].result()

    _mark('F:questioner+fairness(parallel)')

    # Apply B.1 (Consistency) update
    if need_b1 and 'b1' in futures:
        from viva_evaluator.services.agents.analyzer import recompute_soft_score
        verdict = futures['b1']
        if not verdict['material']:
            analysis['consistency']['score'] = max(consistency_score, CONSISTENCY_NEUTRAL)
            analysis['consistency_adjustment'] = {
                'applied':   True,
                'original':  round(consistency_score, 4),
                'rationale': verdict['rationale'],
            }
            soft_score = recompute_soft_score(analysis)
            analysis['soft_score'] = soft_score
        else:
            analysis['consistency_adjustment'] = {
                'applied': False, 'material': True, 'rationale': verdict['rationale'],
            }
        _mark('B.1:consistency')

    # ------------------------------------------------------------------
    # Step B.2 — Charitable interpretation (A2, FAIRNESS RESCUE).
    # If correctness is borderline, check whether the answer shows sound
    # understanding despite weak wording. This can only RAISE the score
    # (asymmetric) and only fires inside the borderline band.
    # ------------------------------------------------------------------
    if CHARITABLE_BAND[0] <= correctness <= CHARITABLE_BAND[1]:
        from viva_evaluator.services.agents.charitable_check import (
            assess_understanding, CharitableInput,
        )
        charitable = assess_understanding(CharitableInput(
            question_text=prev_question_obj.question_text,
            student_answer=student_answer,
            criterion_name=answered_topic['topic_name'],
            criterion_description=answered_topic['topic_focus'],
            retrieved_chunks=retrieval['chunks'],
        ))
        if charitable['understanding_sound'] and soft_score < CHARITABLE_FLOOR:
            original_soft = soft_score
            soft_score = CHARITABLE_FLOOR
            analysis['charitable'] = {
                'applied':       True,
                'original_soft': round(original_soft, 4),
                'adjusted_soft': CHARITABLE_FLOOR,
                'rationale':     charitable['rationale'],
            }
        else:
            analysis['charitable'] = {'applied': False,
                                      'rationale': charitable['rationale']}

    # Apply B.3 (Self-correction) update
    if need_b3 and 'b3' in futures:
        sc = futures['b3']
        if sc['is_correction'] and sc['improved'] and soft_score < SELF_CORRECTION_FLOOR:
            original_soft = soft_score
            soft_score = SELF_CORRECTION_FLOOR
            analysis['self_correction'] = {
                'applied':       True,
                'original_soft': round(original_soft, 4),
                'adjusted_soft': SELF_CORRECTION_FLOOR,
                'rationale':     sc['rationale'],
            }
            analysis['soft_score'] = soft_score
        else:
            analysis['self_correction'] = {'applied': False,
                                            'rationale': sc['rationale']}

    # ------------------------------------------------------------------
    # Step C, D — Ability Update & Termination Check
    # (Confidence was computed in Step B.5 before the Strategist)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step C — Bayesian ability update for all criteria in the answered topic.
    # Difficulty-aware: the answered question's Bloom level sets the item
    # difficulty, so a correct hard answer raises ability more than a
    # correct easy one (and a wrong easy answer costs more).
    # ------------------------------------------------------------------
    for crit_id in answered_topic['source_criteria_ids']:
        ability_state = state.get_or_init_bkt(str(crit_id))
        update_ability(
            ability_state,
            soft_score,
            bloom_level=getattr(prev_question_obj, 'blooms_level', 'Analyze') or 'Analyze',
        )

    # ------------------------------------------------------------------
    # Step D — Termination check (BEFORE strategist for the next turn)
    # ------------------------------------------------------------------
    # Coverage update happens through record_turn AFTER we've used the intent
    # for strategist input — but termination needs the *current* counts so
    # we pre-update coverage here for all criteria in the answered topic.
    for crit_id in answered_topic['source_criteria_ids']:
        cov = state.coverage[str(crit_id)]
        cov.turns += 1
        cov.sum_correctness += correctness
        if correctness >= 0.3:
            cov.correct_turns += 1
    state.total_turns += 1
    state.soft_score_history.append(round(soft_score, 4))

    decision = should_terminate(state, rubric, session=session)
    if decision.should_end:
        save_session_state(session, state, speaker_id=speaker_id)
        from core.models import EvaluationSession as ES
        session.status = ES.Status.COMPLETED
        session.save(update_fields=['status'])
        
        _mark('G:save')
        total_time = _t.time() - _turn_t0
        table_lines = ["\n" + "=" * 45]
        table_lines.append(f"{'PIPELINE STEP (TERMINATED)':<25} | {'STEP (s)':>10} | {'CUMULATIVE (s)':>15}")
        table_lines.append("-" * 55)
        for label, timestamp, step_elapsed, total_elapsed in _stage_marks:
            table_lines.append(f"{label:<25} | {step_elapsed:>9.2f}s | {total_elapsed:>14.2f}s")
        table_lines.append("-" * 55)
        table_lines.append(f"{'TOTAL (attempts=1)':<25} | {'':>10} | {total_time:>14.2f}s")
        table_lines.append("=" * 55 + "\n")
        logger.info("\n".join(table_lines))

        return {
            'analysis':              analysis,
            'soft_score':            soft_score,
            'speech_confidence':     confidence,
            'session_complete':      True,
            'termination_reason':    decision.reason,
            'next_question_payload': None,
        }

    # (Step E & F are now handled in parallel earlier in the pipeline)

    # ------------------------------------------------------------------
    # Step G — Persist state and print full latency timeline summary
    # ------------------------------------------------------------------
    save_session_state(session, state, speaker_id=speaker_id)
    _mark('G:save')
    
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
            'bloom_level':          strategy['bloom_level'],
            'socratic_intent':      strategy['socratic_intent'],
            'difficulty':           next_difficulty,
            'p_lt':                 state.bkt_states[str(next_topic['source_criteria_ids'][0])].p_lt,
        },
    }


# =============================================================================
# Internals
# =============================================================================

_BLOOM_TO_DIFFICULTY = {
    'Remember':   'easy',
    'Understand': 'easy',
    'Apply':      'medium',
    'Analyze':    'medium',
    'Evaluate':   'hard',
    'Create':     'hard',
}


def _bloom_to_difficulty(bloom: str) -> str:
    return _BLOOM_TO_DIFFICULTY.get(bloom, 'medium')


def _resolve_answered_topic(question_obj, topics):
    """Find which viva topic the answered question belonged to."""
    if hasattr(question_obj, 'viva_topic_name') and question_obj.viva_topic_name:
        for t in topics:
            if t['topic_name'] == question_obj.viva_topic_name:
                return t
    
    # Fallback for old sessions or missing data
    # Create a generic topic containing the single criterion if it exists
    crit_ids = []
    try:
        ext = question_obj.extension
        if ext and ext.criteria_id:
            crit_ids.append(str(ext.criteria_id))
    except Exception:
        pass
        
    return {
        'topic_name': 'General',
        'source_criteria_ids': crit_ids,
        'suggested_questions': 2,
        'topic_focus': '',
    }


def _build_recent_transcript(session, limit: int = 5):
    """
    Return the last N Q/A pairs in the order asked, for the Analyzer.

    Uses prefetch_related to fetch all answers in ONE extra query instead of
    one query per question (was N+1 round-trips to the remote DB).
    """
    from django.db.models import Prefetch
    from core.models import VivaAnswer

    questions = list(
        session.viva_questions
        .order_by('-question_order')
        .prefetch_related(
            Prefetch(
                'answers',
                queryset=VivaAnswer.objects.order_by('-answered_at'),
            )
        )[:limit]
    )

    pairs = []
    for q in reversed(questions):
        answers = list(q.answers.all())          # already prefetched, no new query
        last_answer = answers[0] if answers else None
        pairs.append({
            'question_text': q.question_text,
            'answer_text':   last_answer.transcribed_answer if last_answer else '',
        })
    return pairs
