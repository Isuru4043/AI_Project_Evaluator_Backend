"""Per-participant score report lifecycle for individual and group vivas."""

from __future__ import annotations

from django.core.exceptions import ObjectDoesNotExist


def session_students(session):
    """Return the complete, stable student roster for a session."""
    from core.models import GroupMember

    if session.group_id:
        return [
            membership.student
            for membership in (
                GroupMember.objects
                .filter(group_id=session.group_id)
                .select_related('student__user')
                .order_by('joined_at', 'student_id')
            )
        ]
    return [session.student] if session.student_id else []


def ensure_participant_reports(session):
    """Guarantee one summary report for every enrolled participant."""
    from core.models import SessionSummaryReport

    reports = []
    for student in session_students(session):
        report, _ = SessionSummaryReport.objects.get_or_create(
            session=session,
            student=student,
        )
        reports.append(report)

    # A legacy/group aggregate remains supported, but it must never replace
    # the participant reports above.
    if not reports and not session.student_id:
        report, _ = SessionSummaryReport.objects.get_or_create(
            session=session,
            student=None,
        )
        reports.append(report)
    return reports


def refresh_draft_summary_reports(session):
    """Recalculate all unapproved participant reports from current ownership."""
    from core.models import SessionSummaryReport
    from viva_evaluator.services.scoring_service import ScoringService

    ensure_participant_reports(session)
    refreshed = 0
    for report in session.summary_reports.select_related('student').all():
        if report.scores_status == SessionSummaryReport.ScoresStatus.APPROVED:
            continue

        ai_result = ScoringService.aggregate_student_score(
            session,
            report.student,
            use_examiner_overrides=False,
        )
        effective = ScoringService.aggregate_student_score(session, report.student)
        report.total_ai_score = ai_result['percentage']
        report.total_final_score = effective['percentage']
        report.grade = effective['grade']
        report.save(update_fields=['total_ai_score', 'total_final_score', 'grade'])
        refreshed += 1
    return refreshed


def build_published_student_report(session, student, summary):
    """Build the stable, student-safe payload for one published result."""
    from viva_evaluator.services.reporting.post_viva_report import (
        _build_transcript,
    )
    from viva_evaluator.services.scoring_service import ScoringService

    scoring = ScoringService.aggregate_student_score(session, student)
    criteria_by_id = {
        str(criteria.id): criteria
        for category in session.project.rubric_categories.prefetch_related(
            'criteria'
        ).all()
        for criteria in category.criteria.all()
    }
    per_criterion_scores = {}
    for criteria_id, earned in scoring.get('per_criteria', {}).items():
        criteria = criteria_by_id.get(str(criteria_id))
        if criteria is None:
            continue
        maximum = float(criteria.max_score or 0)
        score_out_of_ten = (float(earned) / maximum * 10) if maximum else 0
        per_criterion_scores[criteria.criteria_name] = round(
            score_out_of_ten, 2
        )

    questions = list(
        session.viva_questions.select_related(
            'extension__criteria'
        ).prefetch_related(
            'answers__extension'
        ).order_by('question_order')
    )
    percentage = float(summary.total_final_score or 0)
    weight = float(session.viva_weight_percentage or 100)
    feedback = (
        summary.overall_feedback
        or 'The examiner has approved this final viva result.'
    )

    return {
        'session_id': str(session.id),
        'speaker_id': str(student.id),
        'overall_score': round(percentage / 100, 4),
        'viva_weight_percentage': weight,
        'scaled_score': round(percentage * weight / 100, 2),
        'per_criterion_scores': per_criterion_scores,
        'xai_report': {
            'overall_summary': feedback,
            'examiner_recommendation': feedback,
        },
        'transcript': _build_transcript(questions),
        'scores_status': summary.scores_status,
        'scores_approved_at': (
            summary.scores_approved_at.isoformat()
            if summary.scores_approved_at else None
        ),
    }


def unresolved_individual_answers(session):
    """Individual-rubric answers that cannot safely be assigned a mark yet."""
    from attribution.models import AttributionStatus
    from core.models import VivaAnswer

    answers = (
        VivaAnswer.objects
        .filter(
            question__session=session,
            question__extension__criteria__is_individual=True,
        )
        .select_related(
            'question',
            'question__extension__criteria',
            'student__user',
            'attribution',
        )
        .prefetch_related(
            'contributions__student',
            'contributions__unknown_speaker__resolved_student',
        )
        .order_by('question__question_order', 'answered_at')
    )

    unresolved = []
    for answer in answers:
        try:
            attribution = answer.attribution
        except ObjectDoesNotExist:
            attribution = None

        contributions = list(answer.contributions.all())
        has_known_contributor = any(
            contribution.effective_student_id is not None
            for contribution in contributions
        )
        has_unknown_contributor = any(
            contribution.effective_student_id is None
            for contribution in contributions
        )
        needs_review = (
            answer.student_id is None and not has_known_contributor
        ) or has_unknown_contributor
        if attribution is not None:
            # A collaborative answer with recognized participants does not
            # need one artificial winner. Unknown/disputed identity still
            # requires an examiner before scores can be published.
            needs_review = needs_review or (
                attribution.unknown_speaker_id is not None
                or attribution.status == AttributionStatus.DISPUTED
            )
        if needs_review:
            unresolved.append(answer)
    return unresolved
