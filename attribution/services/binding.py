"""Seat binding — which face in the physical room is which student.

A physical group viva has one camera and everyone in frame at once, so before
lip activity can name a speaker, each face position must be tied to a roster
member. That is done here: one still frame, MediaPipe for detection, ArcFace
against each student's enrollment photo.

The split matters for cost. Recognition is expensive but changes only when
someone moves seats; lip activity is cheap but changes many times a second.
So this runs rarely and server-side, and the kiosk does the per-frame work.

Backends mirror cv_analysis.services.runner: `modal` in the cloud, `local`
when the CV engine's virtualenv is available on the web tier.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

from django.conf import settings
from django.utils import timezone

from core.models import GroupMember
from attribution.models import BindingMethod, SpeakerBinding

logger = logging.getLogger(__name__)


def enrollment_photo_refs(session) -> dict[str, str]:
    """student_id -> blob URL of their enrollment photo (only those who have one)."""
    photos: dict[str, str] = {}
    if session.group_id:
        members = (
            GroupMember.objects
            .filter(group_id=session.group_id)
            .select_related('student')
        )
        for member in members:
            ref = getattr(member.student, 'face_photo_url', '') or ''
            if ref:
                photos[str(member.student_id)] = ref
    elif session.student_id:
        ref = getattr(session.student, 'face_photo_url', '') or ''
        if ref:
            photos[str(session.student_id)] = ref
    return photos


def missing_enrollments(session) -> list[str]:
    """Roster members with no enrollment photo — they can never be recognised."""
    from .engine import roster_ids

    return sorted(roster_ids(session) - set(enrollment_photo_refs(session)))


def bind_from_frame(session, frame_bytes: bytes) -> dict:
    """Detect faces in one frame and bind each to a student.

    Returns {'bindings': [...], 'unmatched': int, 'missing_enrollment': [...]}.
    Faces matching no enrolled student are bound with student=None rather than
    to the nearest guess — that is the extra-person case the integrity flags
    exist for.
    """
    photos = enrollment_photo_refs(session)
    if not photos:
        return {
            'bindings': [],
            'unmatched': 0,
            'missing_enrollment': missing_enrollments(session),
            'error': 'No enrollment photos for this session.',
        }

    backend = getattr(settings, 'ATTRIBUTION_BINDING_BACKEND', 'modal').lower()
    if backend == 'local':
        matches = _bind_local(frame_bytes, photos)
    else:
        matches = _bind_modal(frame_bytes, photos)

    # Supersede the previous pass rather than deleting it, so evidence already
    # recorded keeps the mapping that was true when it was captured.
    now = timezone.now()
    SpeakerBinding.objects.filter(
        session=session, superseded_at__isnull=True,
    ).update(superseded_at=now)

    created = []
    unmatched = 0
    for match in matches:
        sid = match.get('student_id')
        if not sid:
            unmatched += 1
        binding = SpeakerBinding.objects.create(
            session=session,
            student_id=sid or None,
            track_ref=str(match.get('track_ref', '') or ''),
            bbox=match.get('bbox'),
            method=BindingMethod.ARCFACE,
            confidence=float(match.get('confidence', 0.0) or 0.0),
        )
        created.append({
            'binding_id': str(binding.id),
            'student_id': sid,
            'bbox': binding.bbox,
            'confidence': binding.confidence,
        })

    return {
        'bindings': created,
        'unmatched': unmatched,
        'missing_enrollment': missing_enrollments(session),
    }


def bind_by_seating(session, order: list[str]) -> dict:
    """Fallback: bind students left-to-right in the order given.

    Valid only for a fixed single-camera view where nobody swaps seats. Used
    when no enrollment photos exist, or recognition is unavailable.
    """
    from .engine import roster_ids

    valid = roster_ids(session)
    now = timezone.now()
    SpeakerBinding.objects.filter(
        session=session, superseded_at__isnull=True,
    ).update(superseded_at=now)

    created = []
    for index, sid in enumerate(order):
        if str(sid) not in valid:
            continue
        binding = SpeakerBinding.objects.create(
            session=session,
            student_id=sid,
            track_ref=str(index),
            method=BindingMethod.SEATING,
            confidence=0.5,
        )
        created.append({'binding_id': str(binding.id), 'student_id': str(sid),
                        'seat': index})
    return {'bindings': created, 'unmatched': 0,
            'missing_enrollment': missing_enrollments(session)}


def current_bindings(session) -> list[dict]:
    rows = (
        SpeakerBinding.objects
        .filter(session=session, superseded_at__isnull=True)
        .select_related('student__user')
    )
    return [
        {
            'binding_id': str(r.id),
            'student_id': str(r.student_id) if r.student_id else None,
            'track_ref': r.track_ref,
            'bbox': r.bbox,
            'method': r.method,
            'confidence': r.confidence,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _bind_modal(frame_bytes: bytes, photos: dict[str, str]) -> list[dict]:
    """Ask the Modal CV app to match faces in this frame against the gallery."""
    import requests

    url = getattr(settings, 'MODAL_CV_BIND_URL', '')
    token = getattr(settings, 'MODAL_CV_TOKEN', '')
    if not url or not token:
        raise RuntimeError(
            "Face binding endpoint not configured. Set MODAL_CV_BIND_URL and "
            "MODAL_CV_TOKEN, or use ATTRIBUTION_BINDING_BACKEND=local."
        )

    from cv_analysis.services.runner import _sas_for

    payload = {
        'token': token,
        'frame_b64': base64.b64encode(frame_bytes).decode('ascii'),
        'enrollment_photos': {
            sid: _sas_for(ref) for sid, ref in photos.items()
        },
    }
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"Face binding failed ({response.status_code}): {response.text[:300]}"
        )
    return response.json().get('matches', [])


def _bind_local(frame_bytes: bytes, photos: dict[str, str]) -> list[dict]:
    """Run the CV engine in-process. Dev only — needs the engine's deps."""
    import tempfile
    from pathlib import Path

    import numpy as np

    import cv2  # noqa: F401  (engine dependency; imported for decode)
    from exam_cv.faces.identity import (  # type: ignore
        ArcFaceEmbedder,
        build_gallery_from_photos,
    )
    from exam_cv.faces.mesh import MeshPipeline  # type: ignore

    from cv_analysis.services.runner import _download_blob
    from cv_analysis.services.storage import is_local_recording

    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Could not decode the binding frame.")

    decoded: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix='attr_bind_') as tmp:
        for sid, ref in photos.items():
            dest = Path(tmp) / f"{sid}.jpg"
            try:
                if is_local_recording(ref):
                    dest.write_bytes(Path(ref).read_bytes())
                else:
                    _download_blob(ref, dest)
                image = cv2.imread(str(dest))
                if image is not None and image.size:
                    decoded[sid] = image
            except Exception:
                logger.exception("Could not load enrollment photo for %s", sid)

        if not decoded:
            return []

        embedder = ArcFaceEmbedder()
        enroll_mesh = MeshPipeline(max_faces=2)
        try:
            gallery, _skipped = build_gallery_from_photos(
                decoded, enroll_mesh, embedder,
            )
        finally:
            enroll_mesh.close()

        mesh = MeshPipeline(max_faces=max(5, len(photos) + 1))
        try:
            observations = mesh.process_frame(frame)
            matches = []
            for obs in observations:
                crop = mesh.crop(frame, obs)
                if crop.size == 0:
                    continue
                embedding = embedder.embed(crop)
                sid = gallery.match(embedding)
                matches.append({
                    'student_id': sid,
                    'track_ref': str(obs.track_id),
                    'bbox': list(obs.bbox),
                    'confidence': 1.0 if sid else 0.0,
                })
            return matches
        finally:
            mesh.close()
