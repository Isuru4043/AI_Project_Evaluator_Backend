"""Baseline capture and the continuous arousal timeline.

Two jobs:

  close_baseline()   turn a calm window into that student's resting numbers
  build_timeline()   a rolling arousal series for the examiner to scrub,
                     aligned to the recording so points map to video time

The timeline is derived on read rather than stored. Windowing and thresholds
are the parameters most likely to need retuning after a pilot, and recomputing
from the raw intervals costs milliseconds - baking conclusions into rows would
mean reprocessing every session to change a number.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

from django.utils import timezone

from physiology.models import BaselineWindow, PhysioSample
from .metrics import arousal, compute

logger = logging.getLogger(__name__)

# RMSSD stays meaningful down to roughly 30 s; shorter and it is noise.
WINDOW_S = 30
# How often a point is emitted. Overlapping windows give a readable curve
# without pretending to more time resolution than 30 s of beats can support.
STEP_S = 5


def clock_origin(session):
    """Wall-clock instant that maps to video position 00:00:00.

    Prefers the recording's own origin so physiological points and CV flags
    land on exactly the same timeline; falls back through the physical run and
    finally the session start.
    """
    from core.models import SessionRecording

    recording = (
        SessionRecording.objects
        .filter(session=session)
        .exclude(recording_started_at__isnull=True)
        .order_by('-recorded_at')
        .first()
    )
    if recording is not None:
        return recording.recording_started_at

    run = getattr(session, 'physical_run', None)
    if run is not None and run.recording_started_at:
        return run.recording_started_at

    return session.actual_start


def intervals_between(session, student, start, end) -> list[float]:
    """Every inter-beat interval recorded in [start, end).

    Samples without sensor contact are skipped outright: the clip was off the
    finger, so whatever the sensor produced is not that student's pulse.
    """
    samples = PhysioSample.objects.filter(
        session=session, student=student, t__gte=start, t__lt=end, contact=True,
    ).order_by('t')

    ibis: list[float] = []
    for sample in samples:
        ibis.extend(float(v) for v in (sample.ibi_ms or []))
    return ibis


def close_baseline(window: BaselineWindow) -> BaselineWindow:
    """End a calm window and derive the student's resting values from it."""
    if window.ended_at is None:
        window.ended_at = timezone.now()

    ibis = intervals_between(
        window.session, window.student, window.started_at, window.ended_at,
    )
    m = compute(ibis)

    window.hr_mean = m.mean_hr
    window.rmssd = m.rmssd
    window.sdnn = m.sdnn
    window.beat_count = m.beat_count
    window.quality = m.quality
    window.computed_at = timezone.now()
    window.save()

    if not window.is_usable:
        logger.warning(
            'Baseline for student %s unusable: %d clean beats, quality %.2f',
            window.student_id, m.beat_count, m.quality,
        )
    return window


def active_baseline(session, student) -> Optional[BaselineWindow]:
    """The most recent usable baseline for this student, or None."""
    for window in BaselineWindow.objects.filter(
        session=session, student=student, ended_at__isnull=False,
    ).order_by('-started_at'):
        if window.is_usable:
            return window
    return None


def build_timeline(session, student, window_s=WINDOW_S, step_s=STEP_S) -> dict:
    """Rolling arousal series for one student.

    Every point reports `usable`. A point that could not be measured - clip
    off the finger, too few clean beats - is emitted as usable=False rather
    than omitted, so a gap in the data reads as a gap and never as calm.
    """
    baseline = active_baseline(session, student)
    samples = PhysioSample.objects.filter(
        session=session, student=student,
    ).order_by('t')

    first = samples.first()
    last = samples.last()
    if first is None or last is None:
        return {
            'points': [],
            'baseline': _baseline_payload(baseline),
            'window_s': window_s,
            'step_s': step_s,
            'has_data': False,
        }

    origin = clock_origin(session) or first.t
    # If sampling began before the recording did, measuring from the recording
    # would clamp every early point to 00:00:00 and silently hide the offset.
    # Fall back to the first sample so the curve keeps real relative time.
    if first.t < origin:
        origin = first.t
    points = []
    cursor = first.t
    end_of_data = last.t
    step = timedelta(seconds=step_s)
    width = timedelta(seconds=window_s)

    while cursor <= end_of_data:
        w_end = cursor + width
        ibis = intervals_between(session, student, cursor, w_end)
        m = compute(ibis)
        a = arousal(
            m,
            baseline.hr_mean if baseline else None,
            baseline.hr_sd if baseline else None,
            baseline.rmssd if baseline else None,
        )
        offset_ms = int((cursor - origin).total_seconds() * 1000)
        points.append({
            't': cursor.isoformat(),
            'offset_ms': max(offset_ms, 0),
            'video_timecode': _timecode(offset_ms),
            'hr': round(m.mean_hr, 1) if m.mean_hr else None,
            'rmssd': round(m.rmssd, 1) if m.rmssd else None,
            'beats': m.beat_count,
            'quality': m.quality,
            'hr_z': a.hr_z,
            'hr_delta': a.hr_delta,
            'rmssd_ratio': a.rmssd_ratio,
            'elevated': a.elevated,
            'usable': a.usable,
            'reason': a.reason,
        })
        cursor += step

    usable = [p for p in points if p['usable']]
    return {
        'points': points,
        'baseline': _baseline_payload(baseline),
        'window_s': window_s,
        'step_s': step_s,
        'has_data': True,
        'coverage': round(len(usable) / len(points), 3) if points else 0.0,
        'elevated_count': sum(1 for p in usable if p['elevated']),
    }


def _baseline_payload(window: Optional[BaselineWindow]) -> Optional[dict]:
    if window is None:
        return None
    return {
        'hr_mean': round(window.hr_mean, 1) if window.hr_mean else None,
        'hr_sd': round(window.hr_sd, 2) if window.hr_sd else None,
        'rmssd': round(window.rmssd, 1) if window.rmssd else None,
        'beat_count': window.beat_count,
        'quality': window.quality,
        'usable': window.is_usable,
        'started_at': window.started_at,
        'ended_at': window.ended_at,
    }


def _timecode(ms: int) -> str:
    total_s = max(ms, 0) // 1000
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'
