"""
Presentation demo capture views.

Endpoints for uploading audio chunks, slide screenshots, warming up
Modal containers, and checking queue drain status during the live
presentation phase.
"""

import logging
import threading

import requests as http_requests
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from django_q.tasks import async_task

from core.models import DemoCapturedSegment, EvaluationSession, StudentProfile

logger = logging.getLogger(__name__)


# ─── Helper ──────────────────────────────────────────────────────────────────

def _get_session_or_404(session_id):
    try:
        return EvaluationSession.objects.get(id=session_id)
    except EvaluationSession.DoesNotExist:
        return None


def _get_student_profile(user):
    """Return the StudentProfile for the authenticated user, or None."""
    try:
        return StudentProfile.objects.get(user=user)
    except StudentProfile.DoesNotExist:
        return None


# ─── Warm-up ─────────────────────────────────────────────────────────────────

class StartWarmupView(APIView):
    """Signal presentation readiness.

    With Gemini Multimodal, no GPU container warming is needed as Google's
    managed multimodal APIs are always-on.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        session = _get_session_or_404(session_id)
        if session is None:
            return Response(
                {'error': 'Session not found.'},
                status=status.HTTP_404_NOT_FOUND,
            )

        logger.info('Presentation demo session %s ready (Gemini Multimodal).', session_id)
        return Response({
            'status': 'warmup_triggered',
            'backend': 'gemini_multimodal',
        }, status=status.HTTP_200_OK)


# ─── Audio Upload ────────────────────────────────────────────────────────────

class DemoAudioUploadView(APIView):
    """Accept a 20-second audio chunk (WebM) from the presenter's browser.

    Saves the file to Azure Blob via the model's FileField, then enqueues
    a background task to transcribe it on Modal.
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, session_id):
        session = _get_session_or_404(session_id)
        if session is None:
            return Response(
                {'error': 'Session not found.'},
                status=status.HTTP_404_NOT_FOUND,
            )

        student = _get_student_profile(request.user)
        if student is None:
            return Response(
                {'error': 'Student profile not found.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        audio_file = request.FILES.get('audio')
        if not audio_file:
            return Response(
                {'error': 'No audio file provided.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        segment = DemoCapturedSegment.objects.create(
            session=session,
            student=student,
            segment_type=DemoCapturedSegment.SegmentType.AUDIO,
            sequence_number=int(request.data.get('sequence_number', 1)),
            start_time=float(request.data.get('start_time', 0.0)),
            end_time=float(request.data.get('end_time', 0.0)),
            file=audio_file,
        )

        # Enqueue only the ID — the worker reads bytes from Azure Blob
        async_task(
            'sessions_app.tasks.transcribe_audio_task',
            str(segment.id),
        )

        logger.info(
            'Queued audio segment %s (%.1fs–%.1fs) for session %s.',
            segment.id, segment.start_time, segment.end_time, session_id,
        )

        return Response(
            {'segment_id': str(segment.id), 'status': 'queued'},
            status=status.HTTP_201_CREATED,
        )


# ─── Screenshot Upload ───────────────────────────────────────────────────────

class DemoScreenshotUploadView(APIView):
    """Accept a JPEG slide screenshot when a slide change is detected.

    The frontend only sends this when the perceptual hash (pHash) of the
    current frame differs from the previous one — so bandwidth is only
    spent on genuine slide transitions.
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, session_id):
        session = _get_session_or_404(session_id)
        if session is None:
            return Response(
                {'error': 'Session not found.'},
                status=status.HTTP_404_NOT_FOUND,
            )

        student = _get_student_profile(request.user)
        if student is None:
            return Response(
                {'error': 'Student profile not found.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        image_file = request.FILES.get('image')
        if not image_file:
            return Response(
                {'error': 'No image file provided.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # For slides, start_time represents the relative timestamp of the
        # slide change (elapsed seconds since demo start).
        segment = DemoCapturedSegment.objects.create(
            session=session,
            student=student,
            segment_type=DemoCapturedSegment.SegmentType.SLIDE,
            sequence_number=int(request.data.get('sequence_number', 1)),
            start_time=float(request.data.get('timestamp', 0.0)),
            end_time=float(request.data.get('timestamp', 0.0)),
            file=image_file,
        )

        async_task(
            'sessions_app.tasks.analyze_slide_task',
            str(segment.id),
        )

        logger.info(
            'Queued slide segment %s (offset %.1fs) for session %s.',
            segment.id, segment.start_time, session_id,
        )

        return Response(
            {'segment_id': str(segment.id), 'status': 'queued'},
            status=status.HTTP_201_CREATED,
        )


# ─── Queue Status ────────────────────────────────────────────────────────────

class DemoQueueStatusView(APIView):
    """Check whether all demo segments for this session have been processed.

    The frontend polls this after "End Demo" is clicked to determine
    when it is safe to transition to the viva phase.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        session = _get_session_or_404(session_id)
        if session is None:
            return Response(
                {'error': 'Session not found.'},
                status=status.HTTP_404_NOT_FOUND,
            )

        segments = DemoCapturedSegment.objects.filter(session=session)
        total = segments.count()
        processed = segments.filter(is_processed=True).count()
        failed = segments.filter(is_processed=False).exclude(error_message='').count()

        drained = (total > 0 and processed + failed == total)

        return Response({
            'drained': drained,
            'total': total,
            'processed': processed,
            'failed': failed,
        })
