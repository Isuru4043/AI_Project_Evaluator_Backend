"""Precompute face-recognition vectors so a kiosk scan is not the first one.

A seat-binding scan spends most of its time downloading every enrolled photo
and turning it into an ArcFace vector. Measured against the live engine with a
three-student roster that is roughly 24 seconds on a cold container. Those
vectors depend only on the photos, so they can be built once, ahead of time,
and reused by every later scan, which then finishes in a few seconds.

Run it before a demo or after students re-enrol:

    python manage.py warm_face_cache                 # every enrolled student
    python manage.py warm_face_cache --session <id>  # just this session's group
    python manage.py warm_face_cache --force         # rebuild current entries

The engine returns its version with the vectors, and only vectors from the
current engine are stored: an older deployment computes them differently and
mixing the two would silently degrade recognition rather than fail loudly.
"""

import base64
import io
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from attribution.models import FaceEnrollmentEmbeddingCache
from attribution.services.binding import (
    ENGINE_VERSION,
    cached_enrollment_embeddings,
    store_fresh_embeddings,
)
from core.models import EvaluationSession, GroupMember, StudentProfile


def _blank_frame_b64() -> str:
    """A plain frame carrying no faces.

    The engine needs a frame to look at, but here only the enrollment side of
    its answer is wanted, so an empty room is exactly right: it builds the
    gallery, finds nobody, and returns the vectors.
    """
    from PIL import Image

    buffer = io.BytesIO()
    Image.new('RGB', (640, 360), (110, 110, 110)).save(buffer, 'JPEG', quality=70)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


class Command(BaseCommand):
    help = 'Precompute enrollment face vectors so seat binding stays fast.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--session', help='Warm only the students in this session\'s group.',
        )
        parser.add_argument(
            '--student', action='append', default=[],
            help='Warm one student id (repeatable).',
        )
        parser.add_argument(
            '--force', action='store_true',
            help='Rebuild entries that are already current.',
        )

    def handle(self, *args, **options):
        import requests

        url = getattr(settings, 'MODAL_CV_BIND_URL', '')
        token = getattr(settings, 'MODAL_CV_TOKEN', '')
        if not url or not token:
            raise CommandError(
                'MODAL_CV_BIND_URL and MODAL_CV_TOKEN must be configured.'
            )

        students = self._select_students(options)
        if not students:
            self.stdout.write('No enrolled students matched.')
            return

        photos = {
            str(student.id): student.enrollment_face_photos()
            for student in students
        }
        photos = {sid: refs for sid, refs in photos.items() if refs}
        if not photos:
            self.stdout.write(self.style.WARNING(
                'Those students have no enrollment photos yet.'
            ))
            return

        already, _ = cached_enrollment_embeddings(photos)
        if not options['force']:
            skipped = sorted(already)
            photos = {sid: refs for sid, refs in photos.items() if sid not in already}
            if skipped:
                self.stdout.write(f'{len(skipped)} student(s) already current; skipping.')
        if not photos:
            self.stdout.write(self.style.SUCCESS('Every selected student is already warm.'))
            return

        from cv_analysis.services.runner import _sas_for

        frame = _blank_frame_b64()
        names = {str(s.id): (s.user.full_name or str(s.id)) for s in students}
        warmed = failed = 0
        # One student per request: a slow or unreadable photo then names the
        # student it belongs to instead of spoiling a whole batch.
        for student_id, refs in photos.items():
            label = names.get(student_id, student_id)
            started = time.time()
            try:
                response = requests.post(url, json={
                    'token': token,
                    'frame_b64': frame,
                    'frames_b64': [frame],
                    'enrollment_photos': {student_id: [_sas_for(ref) for ref in refs]},
                    'enrollment_embeddings': {},
                }, timeout=300)
                if response.status_code != 200:
                    raise RuntimeError(
                        f'HTTP {response.status_code}: {response.text[:200]}'
                    )
                body = response.json()
            except Exception as exc:
                failed += 1
                self.stderr.write(self.style.ERROR(f'  {label}: {exc}'))
                continue

            elapsed = time.time() - started
            version = body.get('engine_version')
            if version != ENGINE_VERSION:
                raise CommandError(
                    f'The deployed engine reports {version or "no version"}, but this '
                    f'code expects {ENGINE_VERSION}. Deploy the current '
                    'cv_analyze_modal.py before warming the cache, otherwise the '
                    'stored vectors would not match what binding computes.'
                )

            stored = store_fresh_embeddings(
                {student_id: refs},
                {
                    'engine_version': version,
                    'enrollment_embeddings': body.get('enrollment_embeddings') or {},
                },
            )
            if stored:
                warmed += 1
                count = len((body.get('enrollment_embeddings') or {}).get(student_id, []))
                self.stdout.write(
                    f'  {label}: {count} vector(s) from {len(refs)} photo(s) in {elapsed:.1f}s'
                )
            else:
                failed += 1
                unusable = body.get('unusable_enrollment') or []
                reason = (
                    'no photo showed exactly one clear face'
                    if student_id in {str(v) for v in unusable}
                    else 'the engine returned no vectors'
                )
                FaceEnrollmentEmbeddingCache.objects.update_or_create(
                    student_id=student_id,
                    defaults={
                        'photo_fingerprint': FaceEnrollmentEmbeddingCache.fingerprint(refs),
                        'embeddings': [],
                        'engine_version': ENGINE_VERSION,
                        'status': FaceEnrollmentEmbeddingCache.Status.UNUSABLE,
                        'error_message': reason,
                    },
                )
                self.stderr.write(self.style.WARNING(f'  {label}: {reason}'))

        summary = f'{warmed} student(s) warmed, {failed} unusable or failed.'
        self.stdout.write(
            self.style.SUCCESS(summary) if not failed else self.style.WARNING(summary)
        )

    def _select_students(self, options):
        if options['session']:
            session = EvaluationSession.objects.filter(id=options['session']).first()
            if session is None:
                raise CommandError(f'No session {options["session"]}.')
            if not session.group_id:
                raise CommandError('That session has no group, so it has no roster.')
            ids = GroupMember.objects.filter(
                group_id=session.group_id,
            ).values_list('student_id', flat=True)
            return list(
                StudentProfile.objects.filter(id__in=list(ids)).select_related('user')
            )
        if options['student']:
            return list(
                StudentProfile.objects
                .filter(id__in=options['student'])
                .select_related('user')
            )
        return [
            student
            for student in StudentProfile.objects.select_related('user')
            if student.enrollment_face_photos()
        ]
