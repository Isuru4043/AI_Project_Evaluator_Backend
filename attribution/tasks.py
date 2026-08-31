"""Background preparation of private face-enrollment embeddings."""

import logging

from django.conf import settings

from attribution.models import FaceEnrollmentEmbeddingCache
from core.models import StudentProfile

logger = logging.getLogger(__name__)


def refresh_face_embedding_cache(student_id):
    """Build vectors once after enrollment instead of during every session."""
    import requests

    student = StudentProfile.objects.filter(id=student_id).first()
    if student is None:
        return
    refs = student.enrollment_face_photos()
    fingerprint = FaceEnrollmentEmbeddingCache.fingerprint(refs)
    cache, _ = FaceEnrollmentEmbeddingCache.objects.update_or_create(
        student=student,
        defaults={
            'photo_fingerprint': fingerprint,
            'embeddings': [],
            'engine_version': '',
            'status': FaceEnrollmentEmbeddingCache.Status.PENDING,
            'error_message': '',
        },
    )
    url = getattr(settings, 'MODAL_CV_ENROLL_URL', '')
    token = getattr(settings, 'MODAL_CV_TOKEN', '')
    if not url or not token:
        cache.status = FaceEnrollmentEmbeddingCache.Status.FAILED
        cache.error_message = 'MODAL_CV_ENROLL_URL is not configured.'
        cache.save(update_fields=['status', 'error_message', 'updated_at'])
        return

    from cv_analysis.services.runner import _sas_for

    try:
        response = requests.post(url, json={
            'token': token,
            'enrollment_photos': {str(student.id): [_sas_for(ref) for ref in refs]},
        }, timeout=120)
        response.raise_for_status()
        body = response.json()
        embeddings = body.get('embeddings', {}).get(str(student.id), [])
        cache.embeddings = embeddings
        cache.engine_version = body.get('engine_version', '')
        cache.status = (
            FaceEnrollmentEmbeddingCache.Status.READY
            if embeddings else FaceEnrollmentEmbeddingCache.Status.UNUSABLE
        )
        cache.error_message = '' if embeddings else 'No sample contained one clear face.'
    except Exception as exc:
        logger.exception('Could not precompute face embeddings for %s', student.id)
        cache.status = FaceEnrollmentEmbeddingCache.Status.FAILED
        cache.error_message = str(exc)[:1000]
    cache.save(update_fields=[
        'embeddings', 'engine_version', 'status', 'error_message', 'updated_at',
    ])
