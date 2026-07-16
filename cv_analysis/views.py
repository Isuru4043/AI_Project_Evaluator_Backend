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
from django.urls import reverse
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import AllowAny, IsAuthenticated
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
from cv_analysis.services.timeline import build_question_timeline
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

        from cv_analysis.services.storage import (
            save_recording_locally,
            storage_backend,
        )

        if storage_backend() == 'azure':
            from AI_Evaluator_Backend.azure_storage import upload_video_to_blob
            recording_ref = upload_video_to_blob(
                video_file, str(session.project_id), str(session.id),
            )
        else:
            recording_ref = save_recording_locally(video_file, session.id)

        SessionRecording.objects.create(
            session=session,
            video_file_url=recording_ref,
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

        # Resume a Modal job whose polling thread died with its gunicorn
        # worker. The examiner UI polls this endpoint, so a stuck PROCESSING
        # report heals itself without anyone re-triggering the analysis.
        if report.status == CVSessionReport.Status.PROCESSING and report.modal_call_id:
            from cv_analysis.services.runner import poll_modal_result
            try:
                if poll_modal_result(report):
                    report.refresh_from_db()
            except Exception:
                logger.exception(
                    "Resume poll failed for session %s", session_id,
                )

        # Short-lived playback URL the examiner's <video> can stream without
        # a bearer header: a signed local endpoint, or an Azure SAS URL.
        playback_url = self._playback_url(request, session_id, report.recording_url)

        return Response({
            'success': True,
            'data': {
                'status': report.status,
                'artifact': report.artifact,
                'recording_url': report.recording_url,
                'playback_url': playback_url,
                'question_timeline': build_question_timeline(session),
                'error_message': report.error_message,
                'updated_at': report.updated_at,
            },
        })

    @staticmethod
    def _playback_url(request, session_id, recording_ref):
        if not recording_ref:
            return None
        from cv_analysis.services.storage import (
            is_local_recording,
            make_playback_token,
        )
        try:
            if is_local_recording(recording_ref):
                token = make_playback_token(session_id)
                return request.build_absolute_uri(
                    reverse('cv_analysis:cv-recording-download',
                            args=[session_id]) + f'?token={token}'
                )
            from AI_Evaluator_Backend.azure_storage import generate_sas_url

            parsed = urlparse(recording_ref)
            container, _, blob_path = (
                unquote(parsed.path).lstrip('/').partition('/')
            )
            return generate_sas_url(container, blob_path)
        except Exception:
            logger.exception(
                "Playback URL generation failed for session %s", session_id,
            )
            return None


class CVRecordingDownloadView(APIView):
    """GET /api/sessions/<session_id>/cv/recording/download/?token=<token>

    Streams a locally stored recording with HTTP range support so the
    examiner's player can seek. Authenticated by a short-lived signed token
    (the <video> element cannot send a bearer header)."""
    permission_classes = [AllowAny]

    def get(self, request, session_id):
        from cv_analysis.services.storage import (
            check_playback_token,
            is_local_recording,
            serve_file_with_range,
        )
        from pathlib import Path

        token = request.query_params.get('token', '')
        if not check_playback_token(token, session_id):
            return _err('Invalid or expired playback token.', code=403)

        report = CVSessionReport.objects.filter(session_id=session_id).first()
        ref = report.recording_url if report else ''
        if not ref or not is_local_recording(ref):
            return _err('No local recording for this session.', code=404)

        path = Path(ref)
        if not path.exists():
            return _err('Recording file is missing on the server.', code=404)

        return serve_file_with_range(request, path)
