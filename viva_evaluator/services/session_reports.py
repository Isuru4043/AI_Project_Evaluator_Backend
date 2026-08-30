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
        .order_by('question__question_order', 'answered_at')
    )

    unresolved = []
    for answer in answers:
        try:
            attribution = answer.attribution
        except ObjectDoesNotExist:
            attribution = None

        needs_review = answer.student_id is None
        if attribution is not None:
            needs_review = needs_review or attribution.needs_review or (
                attribution.status == AttributionStatus.DISPUTED
            )
        if needs_review:
            unresolved.append(answer)
    return unresolved
