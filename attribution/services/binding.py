"""Face binding — which person in the physical room is which student.

A physical group viva has one camera and everyone in frame at once, so before
lip activity can name a speaker, each face position must be tied to a roster
member. A short burst is sampled here, using MediaPipe for detection and
ArcFace against each student's enrollment photo. Repeated observations make
binding tolerant of different seating distances, momentary head turns and
blinks without weakening the identity-match threshold.

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
from attribution.models import (
    BindingMethod,
    FaceEnrollmentEmbeddingCache,
    SpeakerBinding,
)

logger = logging.getLogger(__name__)

ENGINE_VERSION = 'face-binding-v3-aligned'
COMPAT_ENGINE_VERSION = 'face-binding-v2-legacy-adapter'
DEFAULT_MATCH_THRESHOLD = 0.42
DEFAULT_MATCH_MARGIN = 0.05


def enrollment_photo_refs(session) -> dict[str, list[str]]:
    """student_id -> all guided enrollment sample blob URLs."""
    photos: dict[str, list[str]] = {}
    if session.group_id:
        members = (
            GroupMember.objects
            .filter(group_id=session.group_id)
            .select_related('student')
        )
        for member in members:
            refs = member.student.enrollment_face_photos()
            if refs:
                photos[str(member.student_id)] = refs
    elif session.student_id:
        refs = session.student.enrollment_face_photos()
        if refs:
            photos[str(session.student_id)] = refs
    return photos


def missing_enrollments(session) -> list[str]:
    """Roster members with no enrollment photo — they can never be recognised."""
    from .engine import roster_ids

    return sorted(roster_ids(session) - set(enrollment_photo_refs(session)))


def cached_enrollment_embeddings(photos: dict[str, list[str]]) -> tuple[dict, set[str]]:
    """Return only current, successfully precomputed enrollment vectors."""
    caches = FaceEnrollmentEmbeddingCache.objects.filter(student_id__in=photos)
    vectors = {}
    unusable = set()
    for cache in caches:
        if cache.photo_fingerprint != cache.fingerprint(photos[str(cache.student_id)]):
            continue
        if cache.engine_version != ENGINE_VERSION:
            continue
        if cache.status == FaceEnrollmentEmbeddingCache.Status.READY and cache.embeddings:
            vectors[str(cache.student_id)] = cache.embeddings
        elif cache.status == FaceEnrollmentEmbeddingCache.Status.UNUSABLE:
            unusable.add(str(cache.student_id))
    return vectors, unusable


def store_fresh_embeddings(
    photos: dict[str, list[str]], backend_result: dict, already_cached=(),
) -> int:
    """Keep vectors the engine just built from photos, so the next scan is fast.

    Only output from the current engine version is stored: a stale deployment
    (compat adapter) embeds unaligned crops that must never be compared with
    aligned ones. Failures are never cached, so a transient photo download
    error cannot hide a student until they re-enrol.
    """
    fresh = backend_result.get('enrollment_embeddings') or {}
    if backend_result.get('engine_version') != ENGINE_VERSION or not isinstance(fresh, dict):
        return 0
    stored = 0
    for student_id, vectors in fresh.items():
        sid = str(student_id)
        if sid in already_cached or sid not in photos or not vectors:
            continue
        try:
            FaceEnrollmentEmbeddingCache.objects.update_or_create(
                student_id=sid,
                defaults={
                    'photo_fingerprint': FaceEnrollmentEmbeddingCache.fingerprint(photos[sid]),
                    'embeddings': vectors,
                    'engine_version': ENGINE_VERSION,
                    'status': FaceEnrollmentEmbeddingCache.Status.READY,
                    'error_message': '',
                },
            )
            stored += 1
        except Exception:
            logger.exception('Could not cache enrollment vectors for %s', sid)
    return stored


def _aggregate_frame_matches(
    frame_matches: list[list[dict]],
    min_votes: int = 1,
) -> list[dict]:
    """Combine per-frame matches into one conservative identity result.

    A student needs the configured number of valid frame votes. Repeated
    votes raise temporal support while the ArcFace score remains a separate
    signal; sampling never turns a weak nearest-neighbour guess into an ID.
    """
    if not frame_matches:
        return []

    buckets: dict[str, dict] = {}
    max_detected = 0
    best_unknowns: list[dict] = []
    for frame_index, matches in enumerate(frame_matches):
        max_detected = max(max_detected, len(matches))
        unknowns = [match for match in matches if not match.get('student_id')]

        # One identity gets at most one vote per frame. If two crops both
        # claim the same student, keep the stronger match and leave the other
        # face unassigned rather than duplicating a person.
        strongest: dict[str, dict] = {}
        for match in matches:
            sid = str(match.get('student_id') or '')
            if not sid:
                continue
            current = strongest.get(sid)
            if current is None or float(match.get('confidence', 0) or 0) > float(
                current.get('confidence', 0) or 0
            ):
                if current is not None:
                    unknowns.append({**current, 'student_id': None})
                strongest[sid] = match
            else:
                unknowns.append({**match, 'student_id': None})

        if len(unknowns) > len(best_unknowns):
            best_unknowns = unknowns

        for sid, match in strongest.items():
            bucket = buckets.setdefault(sid, {
                'votes': 0,
                'confidence_total': 0.0,
                'latest': match,
                'latest_frame': -1,
            })
            bucket['votes'] += 1
            bucket['confidence_total'] += max(
                0.0, min(1.0, float(match.get('confidence', 0) or 0))
            )
            if frame_index >= bucket['latest_frame']:
                bucket['latest'] = match
                bucket['latest_frame'] = frame_index

    total_frames = len(frame_matches)
    aggregated: list[dict] = []
    rejected_known: list[dict] = []
    for sid, bucket in sorted(
        buckets.items(), key=lambda item: item[1]['votes'], reverse=True,
    ):
        votes = int(bucket['votes'])
        if votes < min_votes:
            rejected_known.append({**bucket['latest'], 'student_id': None})
            continue
        identity_confidence = bucket['confidence_total'] / votes
        latest = bucket['latest']
        aggregated.append({
            'student_id': sid,
            'track_ref': str(latest.get('track_ref', '') or ''),
            'bbox': latest.get('bbox'),
            # Keep this as the actual ArcFace identity score. Temporal support
            # is reported separately as votes/frames_processed so the two
            # signals are not mixed into a misleading percentage.
            'confidence': round(identity_confidence, 4),
            'identity_confidence': round(identity_confidence, 4),
            'identity_margin': latest.get('identity_margin'),
            'votes': votes,
            'frames_processed': total_frames,
        })

    unknown_candidates = [*best_unknowns, *rejected_known]
    unknown_count = max(0, max_detected - len(aggregated))
    for index, match in enumerate(unknown_candidates[:unknown_count]):
        aggregated.append({
            'student_id': None,
            'track_ref': str(match.get('track_ref', f'unknown-{index}') or ''),
            'bbox': match.get('bbox'),
            'confidence': 0.0,
            'identity_confidence': float(match.get('confidence', 0) or 0),
            'identity_margin': match.get('identity_margin'),
            'votes': 0,
            'frames_processed': total_frames,
        })
    return aggregated


def _roster_review(session, bindings: list[dict], unusable: set[str]) -> list[dict]:
    """Build the explicit, complete expected-member checklist for the kiosk."""
    if session.group_id:
        members = GroupMember.objects.filter(group_id=session.group_id).select_related(
            'student__user',
        )
        roster = [
            (str(member.student_id), member.student.user.full_name,
             member.student.registration_number)
            for member in members
        ]
    else:
        roster = [(
            str(session.student_id), session.student.user.full_name,
            session.student.registration_number,
        )]

    recognised = {
        str(item['student_id']): item
        for item in bindings
        if item.get('student_id')
    }
    return [
        {
            'student_id': student_id,
            'full_name': full_name,
            'registration_number': registration_number,
            'status': (
                'no_usable_enrollment'
                if student_id in unusable
                else 'verified' if student_id in recognised
                else 'not_detected'
            ),
            'confidence': recognised.get(student_id, {}).get('identity_confidence'),
            'identity_margin': recognised.get(student_id, {}).get('identity_margin'),
            'votes': recognised.get(student_id, {}).get('votes', 0),
            'bbox': recognised.get(student_id, {}).get('bbox'),
        }
        for student_id, full_name, registration_number in roster
    ]


def _empty_review(session, unusable: set[str], error: str) -> dict:
    roster = _roster_review(session, [], unusable)
    return {
        'complete': False,
        'bindings': [],
        'roster': roster,
        'unknown_faces': [],
        'unmatched': 0,
        'missing_enrollment': sorted(unusable),
        'unusable_enrollment': sorted(unusable),
        'frames_processed': 0,
        'required_frames': 5,
        'engine_version': ENGINE_VERSION,
        'error': error,
    }
def bind_from_frames(session, frame_bytes_list: list[bytes]) -> dict:
    """Detect faces across a short camera burst and bind them to students.

    Returns {'bindings': [...], 'unmatched': int, 'missing_enrollment': [...]}.
    Faces matching no enrolled student are bound with student=None rather than
    to the nearest guess — that is the extra-person case the integrity flags
    exist for.
    """
    photos = enrollment_photo_refs(session)
    if not photos:
        missing = set(missing_enrollments(session))
        return _empty_review(
            session, missing, 'No usable enrollment samples exist for this session.',
        )

    frames = [frame for frame in frame_bytes_list if frame]
    if not frames:
        raise ValueError('At least one camera frame is required.')

    backend = getattr(settings, 'ATTRIBUTION_BINDING_BACKEND', 'modal').lower()
    cached_vectors, cached_unusable = cached_enrollment_embeddings(photos)
    if backend == 'local':
        backend_result = _bind_local_frames(frames, photos)
    else:
        backend_result = _bind_modal_frames(frames, photos, cached_vectors)
    if isinstance(backend_result, list):
        backend_result = {'matches': backend_result}
    matches = backend_result.get('matches', [])
    recognised_ids = {
        str(match['student_id']) for match in matches if match.get('student_id')
    }
    backend_result['unusable_enrollment'] = sorted((
        set(backend_result.get('unusable_enrollment', [])) | cached_unusable
    ) - recognised_ids)
    store_fresh_embeddings(photos, backend_result, already_cached=set(cached_vectors))
    if not matches:
        unusable = set(missing_enrollments(session)) | {
            str(value) for value in backend_result.get('unusable_enrollment', [])
        }
        return _empty_review(
            session,
            unusable,
            'No faces were identified. Keep every participant visible and retry.',
        )
    frames_processed = max(
        (int(match.get('frames_processed', 0) or 0) for match in matches),
        default=int(backend_result.get('frames_processed', len(frames)) or len(frames)),
    ) or len(frames)

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
            'identity_confidence': match.get('identity_confidence'),
            'identity_margin': match.get('identity_margin'),
            'votes': match.get('votes', 1),
            'frames_processed': match.get('frames_processed', frames_processed),
        })

    unusable = set(missing_enrollments(session)) | {
        str(value) for value in backend_result.get('unusable_enrollment', [])
    }
    roster = _roster_review(session, created, unusable)
    unknown_faces = [item for item in created if not item.get('student_id')]
    required_frames = min(5, len(frames))
    engine_version = backend_result.get('engine_version', 'legacy')
    complete = (
        bool(roster)
        and all(item['status'] == 'verified' for item in roster)
        and not unknown_faces
        and frames_processed >= required_frames
        and engine_version in {ENGINE_VERSION, COMPAT_ENGINE_VERSION}
    )
    result = {
        'complete': complete,
        'bindings': created,
        'roster': roster,
        'unknown_faces': unknown_faces,
        'unmatched': unmatched,
        'missing_enrollment': sorted(unusable),
        'unusable_enrollment': sorted(unusable),
        'frames_processed': frames_processed,
        'required_frames': required_frames,
        'engine_version': engine_version,
    }
    if engine_version not in {ENGINE_VERSION, COMPAT_ENGINE_VERSION}:
        result['error'] = (
            f'Recognition service {engine_version} is stale; expected '
            f'{ENGINE_VERSION}. Deploy the current Modal engine or use the '
            'audited examiner override.'
        )
    return result


def bind_from_frame(session, frame_bytes: bytes) -> dict:
    """Backward-compatible one-frame wrapper."""
    return bind_from_frames(session, [frame_bytes])


def match_test_frames(
    frame_bytes_list: list[bytes],
    photos: dict[str, list[str]],
) -> list[dict]:
    """Match a diagnostic burst without creating session bindings."""
    frames = [frame for frame in frame_bytes_list if frame]
    if not frames:
        return []
    backend = getattr(settings, 'ATTRIBUTION_BINDING_BACKEND', 'modal').lower()
    result = (
        _bind_local_frames(frames, photos)
        if backend == 'local'
        else _bind_modal_frames(frames, photos)
    )
    return result.get('matches', []) if isinstance(result, dict) else result


def match_test_frame(frame_bytes: bytes, photos: dict[str, list[str]]) -> list[dict]:
    """Backward-compatible one-frame diagnostic wrapper."""
    return match_test_frames([frame_bytes], photos)


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
    unusable = set(missing_enrollments(session))
    roster = _roster_review(session, created, unusable)
    return {
        'complete': all(item['status'] == 'verified' for item in roster),
        'bindings': created,
        'roster': roster,
        'unknown_faces': [],
        'unmatched': 0,
        'missing_enrollment': sorted(unusable),
        'unusable_enrollment': sorted(unusable),
        'frames_processed': 0,
        'required_frames': 0,
        'engine_version': 'manual-seating',
    }


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


def _bind_modal_frames(
    frame_bytes_list: list[bytes],
    photos: dict[str, list[str]],
    cached_embeddings: Optional[dict] = None,
) -> dict:
    """Ask Modal to match a camera burst against one enrollment gallery."""
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
        # Keep frame_b64 for compatibility with an older deployed endpoint;
        # the updated Modal app uses frames_b64 and loads ArcFace only once.
        'frame_b64': base64.b64encode(frame_bytes_list[0]).decode('ascii'),
        'frames_b64': [
            base64.b64encode(frame).decode('ascii')
            for frame in frame_bytes_list
        ],
        'enrollment_photos': {
            sid: [_sas_for(ref) for ref in refs]
            for sid, refs in photos.items()
        },
        'enrollment_embeddings': cached_embeddings or {},
        'match_threshold': float(getattr(
            settings, 'ATTRIBUTION_IDENTITY_MIN_CONFIDENCE', DEFAULT_MATCH_THRESHOLD,
        )),
        'match_margin': float(getattr(
            settings, 'ATTRIBUTION_IDENTITY_MIN_MARGIN', DEFAULT_MATCH_MARGIN,
        )),
    }
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"Face binding failed ({response.status_code}): {response.text[:300]}"
        )
    body = response.json()
    if isinstance(body.get('frame_matches'), list):
        reported_version = body.get('engine_version') or COMPAT_ENGINE_VERSION
        return {
            'matches': _aggregate_frame_matches(
                body['frame_matches'],
                min_votes=min(
                    int(getattr(settings, 'ATTRIBUTION_IDENTITY_MIN_VOTES', 2)),
                    max(1, len(body['frame_matches'])),
                ),
            ),
            'frames_processed': body.get('frames_processed', len(body['frame_matches'])),
            'unusable_enrollment': body.get('unusable_enrollment', []),
            'enrollment_embeddings': body.get('enrollment_embeddings') or {},
            # The immediately preceding deployment already understood a full
            # frame burst but did not expose a version endpoint. Treat that
            # exact response shape as the bounded compatibility adapter.
            'engine_version': reported_version,
        }
    # A deployed pre-v2 Modal function ignores ``frames_b64`` and processes
    # only ``frame_b64``. Adapt it by sending each remaining frame as its own
    # real request, then run the same temporal vote/one-identity-per-frame
    # aggregation in Django. This preserves working recognition during a
    # rolling deployment instead of silently treating five frames as one.
    legacy_frame_matches = [body.get('matches', [])]
    unusable = set(body.get('unusable_enrollment', []))
    for encoded in payload['frames_b64'][1:]:
        legacy_payload = {
            **payload,
            'frame_b64': encoded,
            'frames_b64': [encoded],
        }
        legacy_response = requests.post(url, json=legacy_payload, timeout=120)
        if legacy_response.status_code != 200:
            raise RuntimeError(
                'Legacy face binding failed while processing the five-frame '
                f'burst ({legacy_response.status_code}): '
                f'{legacy_response.text[:300]}'
            )
        legacy_body = legacy_response.json()
        legacy_frame_matches.append(legacy_body.get('matches', []))
        unusable.update(legacy_body.get('unusable_enrollment', []))

    return {
        'matches': _aggregate_frame_matches(
            legacy_frame_matches,
            min_votes=min(
                int(getattr(settings, 'ATTRIBUTION_IDENTITY_MIN_VOTES', 2)),
                max(1, len(legacy_frame_matches)),
            ),
        ),
        'frames_processed': len(legacy_frame_matches),
        'unusable_enrollment': sorted(unusable),
        'engine_version': COMPAT_ENGINE_VERSION,
    }


def _bind_modal(frame_bytes: bytes, photos: dict[str, list[str]]) -> dict:
    """Backward-compatible one-frame Modal wrapper."""
    return _bind_modal_frames([frame_bytes], photos)


def _bind_local_frames(
    frame_bytes_list: list[bytes],
    photos: dict[str, list[str]],
) -> dict:
    """Run one gallery against a camera burst in-process (development)."""
    import tempfile
    from pathlib import Path

    import numpy as np

    import cv2  # noqa: F401  (engine dependency; imported for decode)
    from exam_cv.faces.identity import (  # type: ignore
        ArcFaceEmbedder,
        assign_embeddings_one_to_one,
        face_chip,
        build_gallery_from_photos,
    )
    from exam_cv.faces.mesh import (  # type: ignore
        MeshPipeline,
        detect_faces_multiscale,
    )

    from cv_analysis.services.runner import _download_blob
    from cv_analysis.services.storage import is_local_recording

    frames = [
        frame
        for frame in (
            cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            for raw in frame_bytes_list
        )
        if frame is not None
    ]
    if not frames:
        raise RuntimeError("Could not decode any binding frame.")

    decoded: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix='attr_bind_') as tmp:
        for sid, refs in photos.items():
            for index, ref in enumerate(refs):
                dest = Path(tmp) / f"{sid}__{index}.jpg"
                try:
                    if is_local_recording(ref):
                        dest.write_bytes(Path(ref).read_bytes())
                    else:
                        _download_blob(ref, dest)
                    image = cv2.imread(str(dest))
                    if image is not None and image.size:
                        decoded.setdefault(sid, []).append(image)
                except Exception:
                    logger.exception(
                        "Could not load enrollment sample %s for %s", index, sid,
                    )

        if not decoded:
            return {
                'matches': [],
                'frames_processed': len(frames),
                'unusable_enrollment': sorted(photos),
                'engine_version': ENGINE_VERSION,
            }

        embedder = ArcFaceEmbedder()
        enroll_mesh = MeshPipeline(max_faces=2)
        try:
            gallery, skipped = build_gallery_from_photos(
                decoded, enroll_mesh, embedder,
            )
        finally:
            enroll_mesh.close()

        mesh = MeshPipeline(
            max_faces=max(5, len(photos) + 1),
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        try:
            frame_matches = []
            for frame in frames:
                matches = []
                observations = []
                embeddings = []
                for obs in detect_faces_multiscale(
                    mesh, frame, expected_faces=len(gallery.enrolled_ids()),
                ):
                    crop = face_chip(mesh, frame, obs)
                    if crop.size == 0:
                        continue
                    observations.append(obs)
                    embeddings.append(embedder.embed(crop))
                assigned = assign_embeddings_one_to_one(
                    embeddings,
                    gallery,
                    threshold=float(getattr(
                        settings, 'ATTRIBUTION_IDENTITY_MIN_CONFIDENCE',
                        DEFAULT_MATCH_THRESHOLD,
                    )),
                    min_margin=float(getattr(
                        settings, 'ATTRIBUTION_IDENTITY_MIN_MARGIN',
                        DEFAULT_MATCH_MARGIN,
                    )),
                )
                for obs, identity in zip(observations, assigned):
                    matches.append({
                        'student_id': identity['student_id'],
                        'track_ref': str(obs.track_id),
                        'bbox': list(obs.bbox),
                        'confidence': identity['confidence'],
                        'identity_margin': identity['identity_margin'],
                    })
                frame_matches.append(matches)
            return {
                'matches': _aggregate_frame_matches(
                    frame_matches,
                    min_votes=min(
                        int(getattr(settings, 'ATTRIBUTION_IDENTITY_MIN_VOTES', 2)),
                        max(1, len(frame_matches)),
                    ),
                ),
                'frames_processed': len(frame_matches),
                'unusable_enrollment': sorted(skipped),
                'engine_version': ENGINE_VERSION,
            }
        finally:
            mesh.close()


def _bind_local(frame_bytes: bytes, photos: dict[str, list[str]]) -> dict:
    """Backward-compatible one-frame local wrapper."""
    return _bind_local_frames([frame_bytes], photos)
