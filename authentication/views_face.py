"""Private, guided face enrollment for group-viva speaker attribution."""

import logging
from io import BytesIO
from urllib.parse import unquote, urlparse

from django.db import transaction
from PIL import Image, UnidentifiedImageError
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import StudentProfile

logger = logging.getLogger(__name__)

MAX_SIZE = 5 * 1024 * 1024
MAX_TOTAL_SIZE = 20 * 1024 * 1024
MIN_SAMPLES = 3
MAX_SAMPLES = 5
MIN_DIMENSION = 320
ALLOWED_TYPES = ('.jpg', '.jpeg', '.png')


def _err(message, code=status.HTTP_400_BAD_REQUEST):
    return Response({'success': False, 'message': message}, status=code)


class FacePhotoView(APIView):
    """GET/POST the authenticated student's own enrollment samples."""

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request):
        student = StudentProfile.objects.filter(user=request.user).first()
        if student is None:
            return _err(
                'Only students have a face registration.',
                code=status.HTTP_403_FORBIDDEN,
            )
        return Response({
            'success': True,
            'data': _state(student.enrollment_face_photos()),
        })

    def post(self, request):
        student = StudentProfile.objects.filter(user=request.user).first()
        if student is None:
            return _err(
                'Only students can register face samples.',
                code=status.HTTP_403_FORBIDDEN,
            )

        photos = request.FILES.getlist('photos') or request.FILES.getlist('photo')
        if not photos:
            return _err('Send 3 to 5 face samples in the photos field.')
        if not MIN_SAMPLES <= len(photos) <= MAX_SAMPLES:
            return _err('Face registration requires 3 to 5 samples.')
        if sum(photo.size for photo in photos) > MAX_TOTAL_SIZE:
            return _err('Face samples are too large (20MB total maximum).')

        for photo in photos:
            error = _validate_photo(photo)
            if error:
                return _err(error)

        from AI_Evaluator_Backend.azure_storage import upload_face_photo_to_blob

        old_urls = student.enrollment_face_photos()
        urls = []
        try:
            for photo in photos:
                photo.seek(0)
                urls.append(upload_face_photo_to_blob(photo, str(student.id)))
        except Exception:
            logger.exception('Face enrollment upload failed for student %s', student.id)
            for url in urls:
                _delete_face_blob(url)
            return _err(
                'Could not store all face samples. Please try again.',
                code=status.HTTP_502_BAD_GATEWAY,
            )

        try:
            with transaction.atomic():
                locked = StudentProfile.objects.select_for_update().get(pk=student.pk)
                locked.face_photo_url = urls[0]
                locked.face_photo_urls = urls
                locked.save(update_fields=['face_photo_url', 'face_photo_urls'])
        except Exception:
            logger.exception('Could not commit face enrollment for student %s', student.id)
            for url in urls:
                _delete_face_blob(url)
            return _err(
                'Could not finish face registration. Please try again.',
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # New samples are durable before old ones are removed. Cleanup failure
        # is logged and must not undo a successful new registration.
        for old_url in old_urls:
            if old_url not in urls:
                _delete_face_blob(old_url)

        return Response({
            'success': True,
            'message': f'Face registration complete with {len(urls)} samples.',
            'data': _state(urls),
        }, status=status.HTTP_201_CREATED)


def _validate_photo(photo):
    if not photo.name.lower().endswith(ALLOWED_TYPES):
        return 'Only .jpg and .png face samples are allowed.'
    if photo.size > MAX_SIZE:
        return 'Each face sample must be under 5MB.'
    try:
        photo.seek(0)
        data = photo.read()
        image = Image.open(BytesIO(data))
        image.verify()
        image = Image.open(BytesIO(data))
        if min(image.size) < MIN_DIMENSION:
            return f'Each face sample must be at least {MIN_DIMENSION}px on both sides.'
    except (UnidentifiedImageError, OSError, ValueError):
        return 'One of the selected files is not a valid image.'
    finally:
        photo.seek(0)
    return None


def _state(urls):
    signed = [url for url in (_sas_or_none(item) for item in urls) if url]
    count = len(urls)
    return {
        'has_photo': count > 0,
        'photo_url': signed[0] if signed else None,
        'sample_urls': signed,
        'sample_count': count,
        'registration_status': 'complete' if count >= MIN_SAMPLES else (
            'needs_improvement' if count else 'required'
        ),
    }


def _delete_face_blob(blob_url):
    if not blob_url:
        return
    from AI_Evaluator_Backend.azure_storage import (
        AZURE_CONTAINER_FACES,
        delete_blob,
    )
    try:
        parsed = urlparse(blob_url)
        container, _, blob_path = unquote(parsed.path).lstrip('/').partition('/')
        if container != AZURE_CONTAINER_FACES or not blob_path:
            logger.warning('Refusing to delete unexpected face blob URL: %s', blob_url)
            return
        delete_blob(container, blob_path)
    except Exception:
        logger.exception('Could not delete replaced face enrollment blob')


def _sas_or_none(blob_url):
    """Short-lived read URL so the student can preview their own sample."""
    if not blob_url:
        return None
    from AI_Evaluator_Backend.azure_storage import generate_sas_url

    try:
        parsed = urlparse(blob_url)
        container, _, blob_path = unquote(parsed.path).lstrip('/').partition('/')
        if not container or not blob_path:
            return None
        return generate_sas_url(container, blob_path, expiry_hours=1)
    except Exception:
        logger.exception('Could not sign face sample URL')
        return None
