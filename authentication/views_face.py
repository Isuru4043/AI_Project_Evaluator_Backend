"""Student face enrollment.

A group viva is recorded as one composite video of everyone at once, so the
only way to tell the examiner WHO answered a question is to recognise faces
against a reference photo. Students enroll one here after signup.

    GET  /api/auth/me/face-photo/   -> {has_photo, photo_url}  (short SAS)
    POST /api/auth/me/face-photo/   -> stores/replaces the photo

Scope: a student may only read or write their OWN photo. This is biometric
reference data — it lives in a private blob container, is never returned to
anyone else, and is only ever handed to the CV engine as a short-lived SAS.
"""

import logging

from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import StudentProfile

logger = logging.getLogger(__name__)

MAX_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_TYPES = ('.jpg', '.jpeg', '.png')


def _err(message, code=status.HTTP_400_BAD_REQUEST):
    return Response({'success': False, 'message': message}, status=code)


class FacePhotoView(APIView):
    """GET/POST /api/auth/me/face-photo/ — the caller's own enrollment photo."""

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request):
        student = StudentProfile.objects.filter(user=request.user).first()
        if student is None:
            return _err('Only students have an enrollment photo.',
                        code=status.HTTP_403_FORBIDDEN)

        return Response({
            'success': True,
            'data': {
                'has_photo': bool(student.face_photo_url),
                'photo_url': _sas_or_none(student.face_photo_url),
            },
        })

    def post(self, request):
        student = StudentProfile.objects.filter(user=request.user).first()
        if student is None:
            return _err('Only students can enroll a face photo.',
                        code=status.HTTP_403_FORBIDDEN)

        photo = request.FILES.get('photo')
        if not photo:
            return _err('photo is required.')
        if not photo.name.lower().endswith(ALLOWED_TYPES):
            return _err('Only .jpg and .png photos are allowed.')
        if photo.size > MAX_SIZE:
            return _err('Photo too large. Maximum size is 5MB.')

        from AI_Evaluator_Backend.azure_storage import upload_face_photo_to_blob

        try:
            url = upload_face_photo_to_blob(photo, str(student.id))
        except Exception:
            logger.exception('Face photo upload failed for student %s', student.id)
            return _err('Could not store the photo. Please try again.',
                        code=status.HTTP_502_BAD_GATEWAY)

        student.face_photo_url = url
        student.save(update_fields=['face_photo_url'])

        return Response({
            'success': True,
            'message': 'Face photo saved.',
            'data': {'has_photo': True, 'photo_url': _sas_or_none(url)},
        }, status=status.HTTP_201_CREATED)


def _sas_or_none(blob_url):
    """Short-lived read URL so the student can preview their own photo."""
    if not blob_url:
        return None
    from urllib.parse import unquote, urlparse

    from AI_Evaluator_Backend.azure_storage import generate_sas_url

    try:
        parsed = urlparse(blob_url)
        container, _, blob_path = unquote(parsed.path).lstrip('/').partition('/')
        if not container or not blob_path:
            return None
        return generate_sas_url(container, blob_path, expiry_hours=1)
    except Exception:
        logger.exception('Could not sign face photo URL')
        return None
