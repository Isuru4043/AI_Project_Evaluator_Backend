"""
Post-viva report generator — assembles the structured report dict consumed
by GET /api/viva/sessions/<id>/report/.

INPUTS (read from the live DB):
    - EvaluationSession (with bkt_state_json populated by the pipeline)
    - All VivaQuestion + VivaAnswer rows for the session
    - Per-question VivaQuestionExtension and per-answer VivaAnswerExtension
    - The submission's KG (for knowledge-audit section)

OUTPUT (one dict, the response body of /report/):
    {
        'session_id':            ...,
        'overall_score':         0..1 — weighted by rubric category weights,
        'final_grade_bracket':   'A' | 'B' | 'C' | 'D' | 'F',
        'per_criterion_means':   [...],
        'bkt_trajectories':      {criterion_id: [P(L_t) history]},
        'authorship_alerts':     [...],   # CONTRADICTS_CODE turns
        'weak_areas':            [...],   # turns with correctness < 0.4
        'knowledge_audit':       {tier1_used, tier2_used, tier3_used},
        'transcript':            [{question_text, answer_text, ...}],
        'charts': {
            'bkt_trajectory_png_base64': '...',
            'rubric_radar_png_base64':   '...',
        },
    }
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Public API
# =============================================================================

def generate_post_viva_report(session) -> Dict:
    """
    Build the full report dict for one EvaluationSession.
    Now returns a dictionary mapped by speaker_id.
    """
    from viva_evaluator.services.pipeline.session_state import load_session_state

    raw_state = getattr(session, 'bkt_state_json', None) or {}
    if 'bkt_states' in raw_state or 'total_turns' in raw_state:
        raw_state = {'group': raw_state}
    if not raw_state:
        raw_state = {'group': {}}

    rubric_meta = _load_rubric_meta(session.project)
    questions = list(session.viva_questions.all().order_by('question_order'))
    transcript = _build_transcript(questions)
    
    # We can share these across the group or filter them later. For now, keep shared.
    authorship_alerts = _extract_authorship_alerts(session)
    knowledge_audit = _build_knowledge_audit(session)

    reports = {}

    for speaker_id in raw_state.keys():
        state = load_session_state(session, speaker_id=speaker_id)
        
        # Filter answers for this specific student if possible to get their weak areas.
        # But for now, we'll just aggregate all questions they answered.
        # Let's get per-criterion data for this student's state.
        
        # Wait, _aggregate_per_criterion uses questions! If we don't filter questions by student, 
        # all students get the same mean score in the report.
        # Let's filter the questions to only those answered by the student.
        speaker_questions = []
        for q in questions:
            # check if any answer belongs to this speaker (or if speaker='group')
            if speaker_id == 'group':
                speaker_questions.append(q)
            else:
                has_answer = any(str(a.student_id) == speaker_id for a in q.answers.all())
                if has_answer:
                    speaker_questions.append(q)
                    
        per_criterion_data = _aggregate_per_criterion(speaker_questions, rubric_meta)
        weak_areas = _extract_weak_areas(speaker_questions, rubric_meta)
        
        bkt_traj = {
            crit_id: list(bkt.history)
            for crit_id, bkt in state.bkt_states.items()
        }

        from viva_evaluator.services.scoring_service import ScoringService
        from core.models import StudentProfile
        try:
            student = StudentProfile.objects.get(id=speaker_id) if speaker_id != 'group' else None
        except Exception:
            student = None
            
        scoring_result = ScoringService.aggregate_student_score(session, student)
        
        overall_score = scoring_result['percentage'] / 100.0 if scoring_result['grade'] != 'N/A' else 0.0
        final_grade = scoring_result['grade']

        charts = _render_charts(
            bkt_trajectories=bkt_traj,
            criterion_name_map={c['id']: c['name'] for c in rubric_meta},
            per_criterion_means=per_criterion_data,
        )

        try:
            # We must find the specific summary report for this student
            if speaker_id == 'group':
                summary = session.summary_report
            else:
                summary = session.summary_reports.get(student_id=speaker_id)
                
            scores_status    = summary.scores_status
            scores_approved_at = summary.scores_approved_at.isoformat() if summary.scores_approved_at else None
        except Exception:
            scores_status    = 'draft'
            scores_approved_at = None

        reports[speaker_id] = {
            'session_id':           str(session.id),
            'speaker_id':           speaker_id,
            'overall_score':        round(overall_score, 3),
            'final_grade_bracket':  final_grade,
            'per_criterion_means':  per_criterion_data,
            'bkt_trajectories':     bkt_traj,
            'authorship_alerts':    authorship_alerts,
            'weak_areas':           weak_areas,
            'knowledge_audit':      knowledge_audit,
            'transcript':           transcript, # Full group transcript
            'total_turns':          state.total_turns,
            'intent_history':       list(state.intent_history),
            'charts':               charts,
            'scores_status':        scores_status,
            'scores_approved_at':   scores_approved_at,
        }

    return reports


# =============================================================================
# Internals — rubric metadata
# =============================================================================

def _load_rubric_meta(project) -> List[Dict]:
    """
    Flatten the rubric with category weights so we can compute weighted
    overall scores. Each entry:
        {
            'id', 'name',
            'category_name', 'category_weight_pct' (0..100),
            'weight_in_category_pct' (0..100, may be None),
            'max_score',
        }
    """
    rows: List[Dict] = []
    for category in project.rubric_categories.all().order_by('id'):
        cat_weight = float(category.weight_percentage or 0)
        for crit in category.criteria.all().order_by('id'):
            rows.append({
                'id':                  str(crit.id),
                'name':                crit.criteria_name,
                'category_name':       category.category_name,
                'category_weight_pct': cat_weight,
                'weight_in_category_pct': (
                    float(crit.weight_in_category) if crit.weight_in_category is not None else None
                ),
                'max_score':           float(crit.max_score),
            })
    return rows


# =============================================================================
# Internals — aggregation of question-level data
# =============================================================================

def _aggregate_per_criterion(questions, rubric_meta: List[Dict]) -> List[Dict]:
    import json
    from viva_evaluator.services.scoring_service import ScoringService

    by_crit: Dict[str, Dict] = {}

    for q in questions:
        crit_ids = []
        if getattr(q, 'source_criteria_ids', None):
            crit_ids = [str(x) for x in q.source_criteria_ids]
        else:
            try:
                ext = getattr(q, 'extension', None)
                if ext and ext.criteria_id:
                    crit_ids.append(str(ext.criteria_id))
            except Exception:
                pass
                
        if not crit_ids:
            continue

        for crit_id in crit_ids:
            slot = by_crit.setdefault(crit_id, {
                'criterion_id':  crit_id,
                'samples':       [],
            })

        answer = q.answers.order_by('-answered_at').first()
        if not answer:
            continue

        effective_score = ScoringService.get_effective_score_for_answer(answer)
        if effective_score is None:
            continue
            
        soft = max(0.0, min(1.0, effective_score / 10.0))

        depth_est = max(0.0, min(1.0, soft * 0.95))
        consistency_est = max(0.0, min(1.0, 0.6 + soft * 0.4))
        
        try:
            ext = getattr(answer, 'extension', None)
            if ext and ext.detailed_ai_analysis:
                if isinstance(ext.detailed_ai_analysis, str):
                    analysis = json.loads(ext.detailed_ai_analysis)
                else:
                    analysis = ext.detailed_ai_analysis
                if 'rubric' in analysis:
                    r = analysis['rubric']
                    if 'depth' in r: depth_est = max(0.0, min(1.0, float(r['depth']) / 10.0))
                    if 'consistency' in r: consistency_est = max(0.0, min(1.0, float(r['consistency']) / 10.0))
        except Exception:
            pass

        for crit_id in crit_ids:
            slot = by_crit[crit_id]
            slot['samples'].append({
                'correctness':  soft,
                'depth':        depth_est,
                'consistency':  consistency_est,
            })

    out: List[Dict] = []
    for meta in rubric_meta:
        slot = by_crit.get(meta['id'], {'samples': []})
        n = len(slot['samples'])
        if n == 0:
            mean_c = mean_d = mean_con = 0.0
        else:
            mean_c   = sum(s['correctness'] for s in slot['samples']) / n
            mean_d   = sum(s['depth']       for s in slot['samples']) / n
            mean_con = sum(s['consistency'] for s in slot['samples']) / n

        out.append({
            'criterion_id':  meta['id'],
            'name':          meta['name'],
            'category_name': meta['category_name'],
            'samples':       n,
            'correctness':   round(mean_c, 3),
            'depth':         round(mean_d, 3),
            'consistency':   round(mean_con, 3),
            'max_score':     meta['max_score'],
        })
    return out


# Removed unused _compute_overall_score and _grade_bracket_for

# =============================================================================
# Internals — transcript + weak areas + authorship alerts
# =============================================================================

def _build_transcript(questions) -> List[Dict]:
    """Linear list of question/answer pairs in order asked."""
    out: List[Dict] = []
    for q in questions:
        try:
            ext = q.extension
            criterion_name = ext.criteria.criteria_name if ext.criteria else 'General'
            difficulty = ext.difficulty_level
        except Exception:
            criterion_name = 'General'
            difficulty = 'medium'

        answer = q.answers.order_by('-answered_at').first()
        ai_score = None
        reasoning = ''
        if answer:
            try:
                ai_score = float(answer.ai_answer_score) if answer.ai_answer_score is not None else None
            except (TypeError, ValueError):
                ai_score = None
            try:
                reasoning = (answer.extension.llm_reasoning or '').strip()
            except Exception:
                reasoning = ''

        # Collect examiner override if the examiner edited the score
        examiner_override_score = None
        examiner_override_note  = ''
        answer_id = None
        if answer:
            answer_id = str(answer.id)
            if answer.examiner_override_score is not None:
                examiner_override_score = float(answer.examiner_override_score)
            examiner_override_note = answer.examiner_override_note or ''

        # The effective score shown in the report: examiner override wins if set
        effective_score = examiner_override_score if examiner_override_score is not None else ai_score

        out.append({
            'answer_id':              answer_id,
            'question_order':         q.question_order,
            'question_text':          q.question_text,
            'blooms_level':           q.blooms_level,
            'criterion':              criterion_name,
            'difficulty':             difficulty,
            'answer_text':            answer.transcribed_answer if answer else '',
            'ai_answer_score':        ai_score,
            'examiner_override_score': examiner_override_score,
            'examiner_override_note': examiner_override_note,
            'effective_score':        effective_score,
            'reasoning':              reasoning,
        })
    return out


def _extract_weak_areas(questions, rubric_meta: List[Dict]) -> List[Dict]:
    """
    Every turn with correctness < 0.4 (i.e., ai_answer_score < 4.0).
    Examiners review these manually before finalizing scores.
    """
    out: List[Dict] = []
    for q in questions:
        answer = q.answers.order_by('-answered_at').first()
        if not answer:
            continue
        try:
            score = float(answer.ai_answer_score) if answer.ai_answer_score is not None else None
        except (TypeError, ValueError):
            score = None
        if score is None or score >= 4.0:
            continue

        try:
            ext = q.extension
            criterion_name = ext.criteria.criteria_name if ext.criteria else 'General'
        except Exception:
            criterion_name = 'General'

        try:
            reasoning = (answer.extension.llm_reasoning or '').strip()
        except Exception:
            reasoning = ''

        out.append({
            'question_order':  q.question_order,
            'question_text':   q.question_text,
            'criterion':       criterion_name,
            'student_answer':  answer.transcribed_answer or '',
            'ai_answer_score': round(score, 2),
            'reasoning':       reasoning,
        })
    return out


def _extract_authorship_alerts(session) -> List[Dict]:
    """
    Pull all CONTRADICTS_CODE edges from the submission's KG. For the FYP
    we surface every alert; in production this would filter to only those
    triggered during the session.
    """
    try:
        submission = session.submission or _resolve_submission(session)
    except Exception:
        submission = None
    if not submission:
        return []

    try:
        from viva_evaluator.services.knowledge_graph.kg_store import (
            retrieve_contradicts_code_edges,
        )
        edges = retrieve_contradicts_code_edges(submission)
    except Exception as exc:
        logger.warning('authorship alerts: %s', exc)
        return []

    return [
        {
            'code_finding':   e.get('source'),
            'report_claim':   e.get('target'),
            'severity':       e.get('attrs', {}).get('severity', 'medium'),
            'finding_detail': e.get('attrs', {}).get('finding_detail', ''),
            'claim_excerpt':  e.get('attrs', {}).get('claim_excerpt', ''),
        }
        for e in edges
    ]


def _resolve_submission(session):
    from core.models import ProjectSubmission

    if session.submission:
        return session.submission
    if session.group_id:
        return ProjectSubmission.objects.filter(
            project=session.project, group=session.group,
        ).first()
    if session.student_id:
        return ProjectSubmission.objects.filter(
            project=session.project, student=session.student,
        ).first()
    return None


# =============================================================================
# Internals — knowledge audit
# =============================================================================

def _build_knowledge_audit(session) -> Dict:
    """
    Group the KG edges by tier so the examiner sees:
      - T1: examiner-approved (no action needed)
      - T2: LLM drafts the examiner has not yet reviewed (one-click approve)
      - T3: web-sourced (verify if used in scoring decisions)

    The viva loop already filters out T4 from question generation, but we
    surface counts here for transparency.
    """
    submission = _resolve_submission(session)
    if not submission:
        return {'tier1_used': [], 'tier2_used': [], 'tier3_used': [], 't4_seen_count': 0}

    try:
        from viva_evaluator.services.knowledge_graph.kg_store import (
            load_kg_for_submission,
        )
        graph = load_kg_for_submission(submission)
    except Exception as exc:
        logger.warning('knowledge audit: %s', exc)
        return {'tier1_used': [], 'tier2_used': [], 'tier3_used': [], 't4_seen_count': 0}

    if graph is None:
        return {'tier1_used': [], 'tier2_used': [], 'tier3_used': [], 't4_seen_count': 0}

    tier1: List[Dict] = []
    tier2: List[Dict] = []
    tier3: List[Dict] = []
    t4_count = 0

    for u, v, data in graph.edges(data=True):
        tier = int(data.get('tier', 1))
        edge_type = data.get('edge_type', 'UNKNOWN')
        record = {
            'edge_type': edge_type,
            'source':    str(u),
            'target':    str(v),
            'trigger':   data.get('trigger', ''),
            'rationale': data.get('rationale', ''),
            'severity':  data.get('severity', ''),
        }
        if tier == 1:   tier1.append(record)
        elif tier == 2: tier2.append(record)
        elif tier == 3: tier3.append(record)
        else:           t4_count += 1

    return {
        'tier1_used':     tier1[:50],
        'tier2_used':     tier2[:50],
        'tier3_used':     tier3[:50],
        't4_seen_count':  t4_count,
        'total_edges':    graph.number_of_edges(),
        'total_nodes':    graph.number_of_nodes(),
    }


# =============================================================================
# Internals — chart rendering wrapper (handles missing matplotlib gracefully)
# =============================================================================

def _render_charts(
    bkt_trajectories: Dict[str, List[float]],
    criterion_name_map: Dict[str, str],
    per_criterion_means: List[Dict],
) -> Dict:
    charts = {
        'bkt_trajectory_png_base64': '',
        'rubric_radar_png_base64':   '',
    }
    try:
        from viva_evaluator.services.reporting.bkt_charts import render_bkt_trajectory_png
        charts['bkt_trajectory_png_base64'] = render_bkt_trajectory_png(
            bkt_trajectories, criterion_name_map,
        )
    except Exception as exc:
        logger.warning('BKT chart render failed: %s', exc)

    try:
        from viva_evaluator.services.reporting.rubric_radar import render_rubric_radar_png
        charts['rubric_radar_png_base64'] = render_rubric_radar_png(per_criterion_means)
    except Exception as exc:
        logger.warning('Radar chart render failed: %s', exc)

    return charts
