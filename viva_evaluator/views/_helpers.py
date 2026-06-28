"""Shared view helpers (extracted from the original views.py)."""
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from urllib.request import urlopen

from core.models import ProjectSubmission
from viva_evaluator.models import SubmissionIndexStatus
from viva_evaluator.serializers import (
    SubmissionUploadSerializer,
    SubmissionIndexStatusSerializer,
)


def _resolve_session_submission(session):
    if session.submission:
        return session.submission

    if session.group_id:
        return ProjectSubmission.objects.filter(
            project=session.project,
            group=session.group,
        ).first()

    if session.student_id:
        return ProjectSubmission.objects.filter(
            project=session.project,
            student=session.student,
        ).first()

    return None


def _difficulty_signal_from_score(soft_score: float) -> str:
    """
    Legacy compatibility: VivaAnswerExtension expects 'lower'|'same'|'higher'.
    Map the new soft_score to that signal so the audit trail stays consistent.
    """
    if soft_score < 0.4:
        return 'lower'
    if soft_score < 0.7:
        return 'same'
    return 'higher'


def _get_or_create_index_status(submission):
    from core.utils.document_parser import extract_text_from_bytes

    index_status, _ = SubmissionIndexStatus.objects.get_or_create(
        submission=submission,
    )

    if index_status.status == SubmissionIndexStatus.IndexStatus.READY and index_status.extracted_text:
        return index_status

    if not submission.report_file_url:
        return index_status

    try:
        with urlopen(submission.report_file_url) as response:
            file_content = response.read()

        report_name = submission.report_file_url.split('?')[0].rsplit('/', 1)[-1] or 'submission-report.pdf'
        extracted_text = extract_text_from_bytes(file_content, report_name)

        index_status.extracted_text = extracted_text
        index_status.status = SubmissionIndexStatus.IndexStatus.READY
        index_status.processed_at = timezone.now()
        index_status.error_message = None
        index_status.save()
        return index_status
    except Exception as exc:
        index_status.status = SubmissionIndexStatus.IndexStatus.FAILED
        index_status.error_message = str(exc)
        index_status.processed_at = timezone.now()
        index_status.save()
        raise
