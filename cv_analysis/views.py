"""CV/behavioral analysis API.

Endpoints (prefixed with /api/ in root urls):
    POST /api/sessions/<session_id>/cv/recording/ — upload session recording
    POST /api/sessions/<session_id>/cv/analyze/   — (re)run analysis
    GET  /api/sessions/<session_id>/cv/summary/   — fetch the report

The recording upload is student-permitted (their browser holds the camera
during the viva); the report itself is examiner-only advisory
decision-support (HITL invariant) — students receive feedback through the
normal post-viva report instead.
"""

import logging
from urllib.parse import unquote, urlparse

from django.conf import settings
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import (
    EvaluationSession,
    ExaminerProfile,
    GroupMember,
    ProjectExaminer,
    SessionRecording,
)
from cv_analysis.models import CVSessionReport
from cv_analysis.services.runner import enqueue_cv_analysis
from projects.permissions import IsExaminer

logger = logging.getLogger(__name__)


def _err(msg, code=400):
    return Response({'success': False, 'message': msg}, status=code)


def _get_session_for_examiner(request, session_id):
    """Return (session, error_response). Examiner must be assigned to the
    session's project."""
    session = (
        EvaluationSession.objects
        .filter(id=session_id)
        .select_related('project')
        .first()
    )
    if not session:
        return None, _err('Session not found.', code=404)

    ep = ExaminerProfile.objects.filter(user=request.user).first()
    if not ep or not ProjectExaminer.objects.filter(
        project=session.project, examiner=ep,
    ).exists():
        return None, _err('You are not assigned to this project.', code=403)

    return session, None


def _can_upload_recording(user, session) -> bool:
    """The session's student, a member of the session's group, or an
    assigned examiner may upload the recording."""
    if session.student and session.student.user_id == user.id:
        return True
    if session.group_id and GroupMember.objects.filter(
        group_id=session.group_id, student__user=user,
    ).exists():
        return True
    ep = ExaminerProfile.objects.filter(user=user).first()
    if ep and ProjectExaminer.objects.filter(
        project=session.project, examiner=ep,
    ).exists():
        return True
    return False


class CVRecordingUploadView(APIView):
    """POST /api/sessions/<session_id>/cv/recording/

    The student's browser records the viva (it owns the camera via Agora)
    and uploads the file here at session end. Stores a SessionRecording and
    queues post-hoc CV analysis.
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    MAX_SIZE = 500 * 1024 * 1024  # 500 MB
    ALLOWED_TYPES = ('.webm', '.mp4')

    def post(self, request, session_id):
        session = (
            EvaluationSession.objects
            .filter(id=session_id)
            .select_related('project', 'student__user', 'group')
            .first()
        )
        if not session:
            return _err('Session not found.', code=404)
        if not _can_upload_recording(request.user, session):
            return _err('You are not part of this session.', code=403)

        video_file = request.FILES.get('video_file')
        if not video_file:
            return _err('video_file is required.')
        if not video_file.name.lower().endswith(self.ALLOWED_TYPES):
            return _err('Only .webm and .mp4 recordings are allowed.')
        if video_file.size > self.MAX_SIZE:
            return _err('File too large. Maximum recording size is 500MB.')

        from AI_Evaluator_Backend.azure_storage import upload_video_to_blob
        video_blob_url = upload_video_to_blob(
            video_file, str(session.project_id), str(session.id),
        )

        SessionRecording.objects.create(
            session=session,
            video_file_url=video_blob_url,
        )

        queued = enqueue_cv_analysis(session.id)
        return Response({
            'success': True,
            'message': 'Recording stored.'
                       + (' Analysis queued.' if queued else ''),
            'data': {'analysis_queued': queued},
        }, status=201)


class CVAnalyzeTriggerView(APIView):
    """POST /api/sessions/<session_id>/cv/analyze/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, session_id):
        session, error = _get_session_for_examiner(request, session_id)
        if error:
            return error

        if not getattr(settings, 'CV_ANALYSIS_ENABLED', False):
            return _err(
                'CV analysis is disabled on this deployment '
                '(CV_ANALYSIS_ENABLED is off).', code=503,
            )

        existing = CVSessionReport.objects.filter(session=session).first()
        if existing and existing.status == CVSessionReport.Status.PROCESSING:
            return Response({
                'success': True,
                'message': 'Analysis already in progress.',
                'data': {'status': existing.status},
            })

        enqueue_cv_analysis(session.id)
        return Response({
            'success': True,
            'message': 'CV analysis queued.',
            'data': {'status': CVSessionReport.Status.PROCESSING},
        }, status=202)


class CVSummaryView(APIView):
    """GET /api/sessions/<session_id>/cv/summary/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def get(self, request, session_id):
        session, error = _get_session_for_examiner(request, session_id)
        if error:
            return error

        report = CVSessionReport.objects.filter(session=session).first()
        if report is None:
            return _err('No CV analysis exists for this session yet.', code=404)

        # Fresh short-lived SAS URL so the examiner's player can stream the
        # recording (the stored blob URL is not publicly readable).
        playback_url = None
        if report.recording_url:
            try:
                from AI_Evaluator_Backend.azure_storage import generate_sas_url

                parsed = urlparse(report.recording_url)
                container, _, blob_path = (
                    unquote(parsed.path).lstrip('/').partition('/')
                )
                playback_url = generate_sas_url(container, blob_path)
            except Exception:
                logger.exception(
                    "SAS generation failed for session %s", session_id,
                )

        return Response({
            'success': True,
            'data': {
                'status': report.status,
                'artifact': report.artifact,
                'recording_url': report.recording_url,
                'playback_url': playback_url,
                'error_message': report.error_message,
                'updated_at': report.updated_at,
            },
        })
