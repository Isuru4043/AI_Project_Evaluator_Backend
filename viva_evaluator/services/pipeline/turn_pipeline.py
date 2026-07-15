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
CLARIFICATION_CAP = 2

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

def load_rubric(project) -> List[Dict]:
    """Flat list of all rubric criteria for a project, in document order."""
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
    return out


def pick_next_criterion(rubric: List[Dict], state) -> Optional[Dict]:
    """
    Walk criteria in rubric order, return the first that hasn't met its
    questions_to_ask quota yet (counting only "correct enough" turns,
    matching the termination logic).
    """
    for crit in rubric:
        cov = state.coverage.get(str(crit['id']))
        required = int(crit['questions_to_ask'])
        correct_turns = cov.correct_turns if cov else 0
        if correct_turns < required:
            return crit

        # Also revisit weak-mastery criteria up to MAX_TURNS
        from viva_evaluator.services.pipeline.termination import (
            MAX_TURNS_PER_CONCEPT, WEAK_MASTERY_THRESHOLD,
        )
        bkt = state.bkt_states.get(str(crit['id']))
        if (bkt
                and bkt.p_lt < WEAK_MASTERY_THRESHOLD
                and (cov.turns if cov else 0) < MAX_TURNS_PER_CONCEPT):
            return crit

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
        elapsed = now - (_stage_marks[-1][1] if _stage_marks else _turn_t0)
        _stage_marks.append((label, now, elapsed))
        logger.info('[turn-timing] %-22s %6.2fs', label, elapsed)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    rubric = load_rubric(session.project)
    state = load_session_state(session)

    # Initialize coverage entries for any criterion that isn't yet tracked
    for crit in rubric:
        state.get_or_init_coverage(str(crit['id']), questions_to_ask=int(crit['questions_to_ask']))
        state.get_or_init_bkt(str(crit['id']))

    # Resolve which criterion was being asked about
    answered_criterion = _resolve_answered_criterion(prev_question_obj, rubric)

    # ------------------------------------------------------------------
    # Step A — Hybrid retrieval for the answered criterion
    # ------------------------------------------------------------------
    _mark('setup')
    retrieval = retrieve_hybrid_for_turn(
        submission=submission,
        criterion_name=answered_criterion['name'],
        criterion_description=answered_criterion['description'],
        last_answer=student_answer,
        top_k=3,
    )
    _mark('A:retrieval')

    # ------------------------------------------------------------------
    # Step A.5 — Response Triage (A1, FAIRNESS GATE).
    # Decide whether the student actually engaged with the question or was
    # confused by it. If confused (and we still have clarification budget),
    # SUSPEND scoring entirely — no Analyzer, no ability update, no turn
    # consumed — and re-ask the same question in clearer words.
    # This can only ever HELP the student (asymmetric) and is bounded by
    # CLARIFICATION_CAP so it cannot be used to stall.
    # ------------------------------------------------------------------
    from viva_evaluator.services.agents.response_triage import (
        triage_response, TriageInput, CLARIFY_LABELS, RESTATE_LABELS, LABEL_GARBLED,
    )

    triage = triage_response(TriageInput(
        question_text=prev_question_obj.question_text,
        student_answer=student_answer,
        is_spoken=bool(speech_metrics),
    ))
    _mark('A.5:triage')

    gate_labels = CLARIFY_LABELS | RESTATE_LABELS
    if triage['label'] in gate_labels and state.clarification_streak < CLARIFICATION_CAP:
        state.clarification_streak += 1
        prev_bloom = getattr(prev_question_obj, 'blooms_level', 'Analyze') or 'Analyze'

        if triage['label'] in RESTATE_LABELS:
            # A5: transcription artifact — re-present the SAME question verbatim
            # and ask the student to restate. No rephrase, no LLM call needed.
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
            # A1: confusion — re-ask the same concept in clearer words.
            question_data = generate_anchored_question(QuestionerInput(
                criterion_name=answered_criterion['name'],
                criterion_description=answered_criterion['description'],
                retrieved_chunks=retrieval['chunks'],
                kg_signals=retrieval,
                difficulty=_bloom_to_difficulty(prev_bloom),
                question_hints=answered_criterion.get('hints', []),
                recent_questions=[],            # a rephrase SHOULD resemble the original
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
            '[turn] %s (streak=%d/%d) label=%s for criterion=%s',
            'RESTATE' if triage['label'] in RESTATE_LABELS else 'CLARIFICATION',
            state.clarification_streak, CLARIFICATION_CAP,
            triage['label'], answered_criterion['name'],
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
                'criterion':       answered_criterion,
                'bloom_level':     prev_bloom,
                'difficulty':      _bloom_to_difficulty(prev_bloom),
            },
        }

    # Not clarifying (real attempt, or clarification budget exhausted) →
    # reset the streak and proceed to normal scoring.
    state.clarification_streak = 0

    # ------------------------------------------------------------------
    # Step B — Analyzer (3D rubric)
    # ------------------------------------------------------------------
    transcript_recent = _build_recent_transcript(session)
    analysis = analyze_answer(AnalyzerInput(
        question_text=prev_question_obj.question_text,
        student_answer=student_answer,
        criterion_name=answered_criterion['name'],
        criterion_description=answered_criterion['description'],
        retrieved_chunks=retrieval['chunks'],
        contradicts_code_alerts=retrieval.get('contradicts_code_alerts') or [],
        transcript_recent=transcript_recent,
    ))
    _mark('B:analyzer(LLM)')

    soft_score = float(analysis.get('soft_score', 0.5))
    correctness = float((analysis.get('correctness') or {}).get('score', 0.5))

    # ------------------------------------------------------------------
    # Step B.1 — Material vs superficial inconsistency (A3, FAIRNESS).
    # If consistency was flagged low, decide whether it's a real
    # contradiction or just reworded phrasing. Superficial clashes get the
    # penalty neutralised (asymmetric: can only lift consistency, not lower).
    # ------------------------------------------------------------------
    consistency_dim = analysis.get('consistency') or {}
    consistency_score = float(consistency_dim.get('score', 1.0))
    if consistency_score < CONSISTENCY_REVIEW_THRESHOLD:
        from viva_evaluator.services.agents.consistency_check import (
            classify_inconsistency, ConsistencyInput,
        )
        from viva_evaluator.services.agents.analyzer import recompute_soft_score

        verdict = classify_inconsistency(ConsistencyInput(
            question_text=prev_question_obj.question_text,
            student_answer=student_answer,
            transcript_recent=transcript_recent,
            consistency_evidence=consistency_dim.get('evidence_quote', ''),
        ))
        if not verdict['material']:
            analysis['consistency']['score'] = max(consistency_score, CONSISTENCY_NEUTRAL)
            analysis['consistency_adjustment'] = {
                'applied':   True,
                'original':  round(consistency_score, 4),
                'rationale': verdict['rationale'],
            }
            soft_score = recompute_soft_score(analysis)
            analysis['soft_score'] = soft_score
            logger.info(
                '[turn] A3 consistency suppressed (superficial): %.2f -> %.2f',
                consistency_score, analysis['consistency']['score'],
            )
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
            criterion_name=answered_criterion['name'],
            criterion_description=answered_criterion['description'],
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
            logger.info(
                '[turn] CHARITABLE rescue: soft %.2f -> %.2f (%s)',
                original_soft, soft_score, charitable['rationale'][:80],
            )
        else:
            analysis['charitable'] = {'applied': False,
                                      'rationale': charitable['rationale']}
        _mark('B.2:charitable')

    # ------------------------------------------------------------------
    # Step B.3 — Self-correction crediting (A4, FAIRNESS RESCUE).
    # If the current answer corrects/improves the student's previous answer,
    # credit the recovery. Asymmetric: can only RAISE the score, and only when
    # the current score left room (below the trigger threshold).
    # ------------------------------------------------------------------
    if soft_score < SELF_CORRECTION_TRIGGER_MAX:
        previous_answer = ''
        for _pair in reversed(transcript_recent):
            if _pair.get('answer_text'):
                previous_answer = _pair['answer_text']
                break
        if previous_answer:
            from viva_evaluator.services.agents.self_correction import (
                assess_self_correction, SelfCorrectionInput,
            )
            sc = assess_self_correction(SelfCorrectionInput(
                question_text=prev_question_obj.question_text,
                current_answer=student_answer,
                previous_answer=previous_answer,
            ))
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
                logger.info(
                    '[turn] A4 self-correction credit: soft %.2f -> %.2f (%s)',
                    original_soft, soft_score, sc['rationale'][:80],
                )
            else:
                analysis['self_correction'] = {'applied': False,
                                                'rationale': sc['rationale']}
            _mark('B.3:self_correction')

    # ------------------------------------------------------------------
    # Step B.5 — Speech confidence (Week 6)
    # Strictly informational — does NOT affect BKT or rubric scoring.
    # ------------------------------------------------------------------
    confidence = analyze_speech_confidence(
        answer_text=student_answer,
        speech_metrics=speech_metrics,
    )
    _mark('B.5:confidence')

    # ------------------------------------------------------------------
    # Step C — Bayesian ability update for the answered criterion.
    # Difficulty-aware: the answered question's Bloom level sets the item
    # difficulty, so a correct hard answer raises ability more than a
    # correct easy one (and a wrong easy answer costs more).
    # ------------------------------------------------------------------
    ability_state = state.get_or_init_bkt(str(answered_criterion['id']))
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
    # we pre-update coverage here for the answered criterion only.
    answered_id = str(answered_criterion['id'])
    cov = state.coverage[answered_id]
    cov.turns += 1
    cov.sum_correctness += correctness
    if correctness >= 0.3:
        cov.correct_turns += 1
    state.total_turns += 1
    state.soft_score_history.append(round(soft_score, 4))

    decision = should_terminate(state, rubric)
    if decision.should_end:
        save_session_state(session, state)
        # Mark the EvaluationSession status
        from core.models import EvaluationSession as ES
        session.status = ES.Status.COMPLETED
        session.save(update_fields=['status'])

        return {
            'analysis':              analysis,
            'soft_score':            soft_score,
            'speech_confidence':     confidence,
            'session_complete':      True,
            'termination_reason':    decision.reason,
            'next_question_payload': None,
        }

    # ------------------------------------------------------------------
    # Step E — Pick next criterion + run Strategist
    # ------------------------------------------------------------------
    next_criterion = pick_next_criterion(rubric, state)
    if next_criterion is None:
        # All quotas met but termination didn't fire (e.g., min turns not met).
        # Fall back to the answered criterion to keep moving.
        next_criterion = answered_criterion

    strategy = select_strategy(StrategistInput(
        p_lt=state.bkt_states[str(next_criterion['id'])].p_lt,
        analysis=analysis,
        kg_signals=retrieval,                         # reuse retrieval
        intent_history=state.intent_history,
        speech_confidence=confidence.get('flag'),
    ))
    _mark('E:strategist')

    # Append intent to history for repetition prevention next turn
    state.intent_history.append(strategy['socratic_intent'])
    if len(state.intent_history) > 30:
        state.intent_history = state.intent_history[-30:]


    # ------------------------------------------------------------------
    # Step F — Generate next question via the Questioner
    # ------------------------------------------------------------------
    # Recompute retrieval for the *next* criterion if it differs from the
    # answered one. This keeps anchoring focused on the new topic.
    if str(next_criterion['id']) != str(answered_criterion['id']):
        retrieval = retrieve_hybrid_for_turn(
            submission=submission,
            criterion_name=next_criterion['name'],
            criterion_description=next_criterion['description'],
            last_answer=student_answer,
            top_k=3,
        )
    _mark('F:retrieval2')

    next_difficulty = _bloom_to_difficulty(strategy['bloom_level'])
    is_first_for_criterion = (
        state.coverage[str(next_criterion['id'])].turns == 0
    )

    recent_qs = list(
        session.viva_questions.order_by('-question_order')
        .values_list('question_text', flat=True)[:5]
    )

    question_data = generate_anchored_question(QuestionerInput(
        criterion_name=next_criterion['name'],
        criterion_description=next_criterion['description'],
        retrieved_chunks=retrieval['chunks'],
        kg_signals=retrieval,
        difficulty=next_difficulty,
        question_hints=next_criterion.get('hints', []),
        recent_questions=recent_qs,
        previous_question=prev_question_obj.question_text,
        previous_answer=student_answer,
        is_first_question=is_first_for_criterion,
        question_number_in_criterion=state.coverage[str(next_criterion['id'])].turns + 1,
        weak_grounding=_grounding_is_weak(retrieval['chunks']),
        session_id=str(session.id),
    ))
    _mark('F:questioner(LLM+critic)')

    # Override the bloom level the Questioner echoes back with the Strategist's choice
    question_data['blooms_level'] = strategy['bloom_level']

    # ------------------------------------------------------------------
    # Step G — Persist state and return
    # ------------------------------------------------------------------
    save_session_state(session, state)
    _mark('G:save')
    logger.info('[turn-timing] TOTAL %.2fs (attempts=%s)',
                _t.time() - _turn_t0, question_data.get('attempts'))

    return {
        'analysis':              analysis,
        'soft_score':            soft_score,
        'speech_confidence':     confidence,
        'session_complete':      False,
        'termination_reason':    None,
        'strategy':              strategy,
        'next_criterion':        next_criterion,
        'next_question_payload': {
            'question_data':        question_data,
            'criterion':            next_criterion,
            'bloom_level':          strategy['bloom_level'],
            'socratic_intent':      strategy['socratic_intent'],
            'difficulty':           next_difficulty,
            'p_lt':                 state.bkt_states[str(next_criterion['id'])].p_lt,
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


def _resolve_answered_criterion(question_obj, rubric):
    """Find which rubric criterion the answered question belonged to."""
    try:
        ext = question_obj.extension
        if ext and ext.criteria_id:
            crit_id = str(ext.criteria_id)
            for c in rubric:
                if c['id'] == crit_id:
                    return c
    except Exception:
        pass
    # Fallback to the first rubric criterion to avoid crashing
    return rubric[0] if rubric else {
        'id': 'unknown', 'name': 'General',
        'description': '', 'questions_to_ask': 3, 'hints': [],
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
