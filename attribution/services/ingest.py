"""Evidence ingest — the providers' write path.

Every provider normalizes to SpeakerEvidence here. Providers are the only
mode-aware code in the component: everything downstream of this module is
shared between physical and virtual sessions.

Sources:
    agora_volume  virtual, live      browser volume-indicator events
    agora_stt     virtual, live      STT WebVTT <v UID> voice tags
    live_cv       physical, live     kiosk/sidecar lip-motion x VAD
    posthoc_cv    both, after        exam_cv AttributionEvent timeline
    submitter     virtual, live      whoever pressed submit (weak prior)
"""

from __future__ import annotations

import logging
from datetime import timedelta

from django.utils import timezone

from core.models import GroupMember, SessionRecording
from attribution.models import EvidenceSource, SpeakerEvidence

logger = logging.getLogger(__name__)

# Spans shorter than this are tracking flicker, not speech.
MIN_SPAN_MS = 200
# Guard against a client posting an absurd span that swamps every window.
MAX_SPAN_MS = 120_000


def uid_to_student_map(session) -> dict[str, str]:
    """Agora numeric UID -> student_id for this session's roster.

    Mirrors agora_service.token_builder._uid_from_user_id, which derives the
    UID deterministically from the user id, so no extra state is needed.
    """
    from agora_service.token_builder import _uid_from_user_id

    mapping: dict[str, str] = {}
    if session.group_id:
        members = (
            GroupMember.objects
            .filter(group_id=session.group_id)
            .select_related('student')
        )
        for member in members:
            mapping[str(_uid_from_user_id(member.student.user_id))] = str(member.student_id)
    elif session.student_id:
        mapping[str(_uid_from_user_id(session.student.user_id))] = str(session.student_id)
    return mapping


def _valid_student_ids(session) -> set[str]:
    from .engine import roster_ids
    return roster_ids(session)


def record_spans(session, spans, source, default_confidence=1.0) -> int:
    """Persist normalized spans as evidence. Returns how many were stored.

    `spans` is an iterable of dicts with `t_start`/`t_end` (aware datetimes),
    optional `student_id`, `track_ref`, `confidence` and `meta`. Rows that fail
    validation are skipped individually — one malformed span must not lose the
    whole batch.

    A span carrying a `track_ref` but no student is a person the CV could see
    and follow but not name. It is attached to a stable UnknownSpeaker instead
    of being discarded, so their turns accumulate somewhere an examiner can
    later hand to the right student.
    """
    from .engine import get_or_create_unknown_speaker

    valid = _valid_student_ids(session)
    stored = 0

    for span in spans:
        try:
            t_start, t_end = span['t_start'], span['t_end']
            if t_end <= t_start:
                continue
            span_ms = (t_end - t_start).total_seconds() * 1000
            if span_ms < MIN_SPAN_MS or span_ms > MAX_SPAN_MS:
                continue

            student_id = span.get('student_id')
            student_id = str(student_id) if student_id else None
            if student_id and student_id not in valid:
                # Unknown to this roster: keep the span, drop the name. The
                # window is recorded as contested rather than mis-attributed.
                student_id = None

            unknown = None
            track_ref = span.get('track_ref')
            if student_id is None and track_ref:
                unknown = get_or_create_unknown_speaker(session, track_ref)
                _touch_seen(unknown, t_start, t_end)

            _, created = SpeakerEvidence.objects.get_or_create(
                session=session,
                source=source,
                student_id=student_id,
                unknown_speaker=unknown,
                t_start=t_start,
                t_end=t_end,
                defaults={
                    'confidence': float(span.get('confidence', default_confidence)),
                    'meta': span.get('meta', {}),
                },
            )
            if created:
                stored += 1
        except Exception:
            logger.exception("Skipping malformed evidence span: %r", span)

    return stored


def _touch_seen(unknown, t_start, t_end) -> None:
    """Widen an unknown speaker's observed time range."""
    fields = []
    if unknown.first_seen is None or t_start < unknown.first_seen:
        unknown.first_seen = t_start
        fields.append('first_seen')
    if unknown.last_seen is None or t_end > unknown.last_seen:
        unknown.last_seen = t_end
        fields.append('last_seen')
    if fields:
        unknown.save(update_fields=fields)


# ---------------------------------------------------------------------------
# Virtual providers
# ---------------------------------------------------------------------------


def ingest_agora_volume(session, events) -> int:
    """Browser volume-indicator batches.

    Each event: {uid, t_start (ISO), t_end (ISO), level 0..100}. Level maps to
    confidence so a barely-audible blip counts for less than clear speech.
    """
    uid_map = uid_to_student_map(session)
    spans = []
    for e in events:
        uid = str(e.get('uid', ''))
        level = float(e.get('level', 0) or 0)
        spans.append({
            't_start': _parse_dt(e.get('t_start')),
            't_end': _parse_dt(e.get('t_end')),
            'student_id': uid_map.get(uid),
            'confidence': max(0.0, min(1.0, level / 100.0)) if level else 0.6,
            'meta': {'uid': uid, 'level': level},
        })
    spans = [s for s in spans if s['t_start'] and s['t_end']]
    return record_spans(session, spans, EvidenceSource.AGORA_VOLUME)


def ingest_stt_turns(session, turns, recording_t0=None) -> int:
    """Speaker turns parsed from the Agora STT WebVTT captions.

    `turns` come from agora_service.transcript_parser.parse_vtt_to_speaker_turns
    and carry offsets into the caption clock, so they need the same origin the
    question timeline uses (Agora's sliceStartTime).
    """
    t0 = recording_t0 or _recording_t0(session)
    if t0 is None:
        logger.info(
            "Session %s: no recording origin yet, deferring STT evidence.",
            session.id,
        )
        return 0

    uid_map = uid_to_student_map(session)
    spans = []
    for turn in turns:
        spans.append({
            't_start': t0 + timedelta(milliseconds=turn['t_start_ms']),
            't_end': t0 + timedelta(milliseconds=turn['t_end_ms']),
            'student_id': uid_map.get(str(turn.get('speaker', ''))),
            'confidence': 0.95,
            'meta': {'speaker': turn.get('speaker'), 'text': turn.get('text', '')[:500]},
        })
    return record_spans(session, spans, EvidenceSource.AGORA_STT)


def ingest_submitter(session, student_id, question, answered_at=None) -> int:
    """The authenticated user who pressed submit — a weak prior, not proof."""
    from .engine import answer_window

    if not student_id:
        return 0
    start, end = answer_window(question, answered_at)
    return record_spans(
        session,
        [{
            't_start': start,
            't_end': end,
            'student_id': str(student_id),
            'confidence': 1.0,
            'meta': {'question_id': str(question.id)},
        }],
        EvidenceSource.SUBMITTER,
    )


# ---------------------------------------------------------------------------
# Physical / CV providers
# ---------------------------------------------------------------------------


def ingest_live_cv(session, turns) -> int:
    """Live lip-motion x VAD turns from the kiosk or the exam-station sidecar.

    Each turn: {student_id, t_start (ISO), t_end (ISO), confidence}. The
    sidecar already speaks this shape — it is exam_cv's AttributionEvent with
    wall-clock timestamps instead of session offsets.
    """
    spans = []
    for t in turns:
        spans.append({
            't_start': _parse_dt(t.get('t_start')),
            't_end': _parse_dt(t.get('t_end')),
            'student_id': t.get('student_id'),
            # Present when the engine tracked a face it could not name.
            'track_ref': t.get('track_ref'),
            'confidence': float(t.get('confidence', 0.5) or 0.5),
            'meta': {k: v for k, v in t.items() if k not in ('t_start', 't_end')},
        })
    spans = [s for s in spans if s['t_start'] and s['t_end']]
    return record_spans(session, spans, EvidenceSource.LIVE_CV)


def ingest_posthoc_artifact(session, artifact) -> int:
    """Convert an exam_cv SessionSummary artifact into evidence.

    The artifact's timeline is a list of AttributionEvent with `t_start_ms` /
    `t_end_ms` offsets into the recording; `student_id` is either a roster id
    or the UNCERTAIN_SPEAKER sentinel. The sentinel is mapped to None so it
    contributes nothing to the vote, which is exactly what it means.
    """
    if not artifact:
        return 0

    t0 = _recording_t0(session)
    if t0 is None:
        logger.warning(
            "Session %s: cannot ingest post-hoc CV without a recording origin.",
            session.id,
        )
        return 0

    timeline = artifact.get('timeline') or []
    spans = []
    for event in timeline:
        sid = event.get('student_id')
        if sid == 'uncertain':
            sid = None
        # The engine names a tracked-but-unrecognised face
        # "unknown_track:<id>". Convert it to a track reference so it lands on
        # a stable UnknownSpeaker holding that person's marks.
        track_ref = None
        if sid and str(sid).startswith('unknown_track:'):
            track_ref = str(sid).split(':', 1)[1]
            sid = None
        spans.append({
            't_start': t0 + timedelta(milliseconds=int(event.get('t_start_ms', 0))),
            't_end': t0 + timedelta(milliseconds=int(event.get('t_end_ms', 0))),
            'student_id': sid,
            'track_ref': track_ref,
            'confidence': float(event.get('confidence', 0.5) or 0.5),
            'meta': {'from': 'cv_artifact'},
        })
    return record_spans(session, spans, EvidenceSource.POSTHOC_CV)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _recording_t0(session):
    """Wall-clock instant that maps to video position 00:00:00.

    Agora's sliceStartTime, already persisted by agora_service.cloud_recording
    and used by cv_analysis.services.timeline for the question timeline. Falls
    back to the session's actual start when no cloud recording exists (the
    physical kiosk path records locally).
    """
    recording = (
        SessionRecording.objects
        .filter(session=session)
        .exclude(recording_started_at__isnull=True)
        .order_by('-recorded_at')
        .first()
    )
    if recording is not None:
        return recording.recording_started_at
    return session.actual_start


def _parse_dt(value):
    """ISO-8601 string (or datetime) -> aware datetime, else None."""
    if value is None:
        return None
    if hasattr(value, 'tzinfo'):
        return value if timezone.is_aware(value) else timezone.make_aware(value)
    try:
        from django.utils.dateparse import parse_datetime

        parsed = parse_datetime(str(value))
        if parsed is None:
            return None
        return parsed if timezone.is_aware(parsed) else timezone.make_aware(parsed)
    except Exception:
        return None
