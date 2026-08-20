from core.models import ProjectSubmission
from viva_evaluator.models import SubmissionIndexStatus


def resolve_submission(session):
    if session.submission_id:
        return session.submission
    if session.group_id:
        return ProjectSubmission.objects.filter(
            project=session.project, group_id=session.group_id,
        ).first()
    if session.student_id:
        return ProjectSubmission.objects.filter(
            project=session.project, student_id=session.student_id,
        ).first()
    return None


def submission_is_ready(session):
    submission = resolve_submission(session)
    if submission is None:
        return False
    return SubmissionIndexStatus.objects.filter(
        submission=submission,
        status=SubmissionIndexStatus.IndexStatus.READY,
    ).exists()
