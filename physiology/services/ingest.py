"""Sample ingest from the exam-station BLE sidecar.

The sidecar batches notifications rather than posting each one: the band
notifies about once a second, and a request per beat would be pure overhead.
Batches are idempotent by (session, student, instant) so a retry after a
network blip cannot double-count beats into the variability maths.
"""

from __future__ import annotations

import logging

from django.utils import timezone
from django.utils.dateparse import parse_datetime

from physiology.models import PhysioDevice, PhysioSample

logger = logging.getLogger(__name__)

# A notification carrying more beats than this is malformed or a replay of a
# long backlog; either way it would distort a 30 s window.
MAX_IBIS_PER_SAMPLE = 60


def bound_device(session, device_id=None) -> PhysioDevice | None:
    """The device currently worn in this session."""
    query = PhysioDevice.objects.filter(session=session, unbound_at__isnull=True)
    if device_id:
        query = query.filter(device_id=device_id)
    return query.order_by('-bound_at').first()


def bind_device(session, device_id: str, student) -> PhysioDevice:
    """Assign the band to whoever is wearing it.

    Any previous binding is closed rather than deleted, so samples already
    captured keep the attribution that was true when they were recorded.

    The release covers OTHER SESSIONS holding the same device_id, not just
    this one. A band is a physical object on one wrist: leaving a stale claim
    behind on a previous session made "which session owns this band" genuinely
    ambiguous, and the relay then fed whichever claim happened to be newest -
    somebody else's viva.
    """
    now = timezone.now()
    PhysioDevice.objects.filter(
        session=session, unbound_at__isnull=True,
    ).update(unbound_at=now)
    PhysioDevice.objects.filter(
        device_id=device_id, unbound_at__isnull=True,
    ).exclude(session=session).update(unbound_at=now)

    return PhysioDevice.objects.create(
        session=session, device_id=device_id, student=student,
    )


def ingest_samples(session, samples, device_id=None) -> dict:
    """Store a batch. Returns counts; never raises on one bad row.

    Requires a bound device: an unattributed pulse is not evidence about
    anybody, so it is refused rather than stored against a guess.
    """
    device = bound_device(session, device_id)
    if device is None:
        return {
            'stored': 0,
            'skipped': len(list(samples)),
            'error': 'No device is bound to a student for this session.',
        }

    stored = 0
    skipped = 0

    for row in samples:
        try:
            t = row.get('t')
            t = parse_datetime(t) if isinstance(t, str) else t
            if t is None:
                skipped += 1
                continue
            if timezone.is_naive(t):
                t = timezone.make_aware(t)

            ibis = [float(v) for v in (row.get('ibi_ms') or [])][:MAX_IBIS_PER_SAMPLE]
            bpm = row.get('bpm')

            _, created = PhysioSample.objects.get_or_create(
                session=session,
                student=device.student,
                t=t,
                defaults={
                    'device': device,
                    'bpm': int(bpm) if bpm is not None else None,
                    'ibi_ms': ibis,
                    'contact': bool(row.get('contact', True)),
                },
            )
            stored += 1 if created else 0
        except Exception:
            logger.exception('Skipping malformed physio sample: %r', row)
            skipped += 1

    battery = _latest_battery(samples)
    if battery is not None and battery != device.battery_pct:
        device.battery_pct = battery
        device.save(update_fields=['battery_pct'])

    return {'stored': stored, 'skipped': skipped, 'student_id': str(device.student_id)}


def _latest_battery(samples):
    for row in reversed(list(samples)):
        value = row.get('battery_pct')
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def signal_status(session, student) -> dict:
    """Is the band actually producing usable signal right now?

    The panel needs this to be autonomous. Counting down a calm period while
    the clip is still being fitted burns the window and yields a baseline of
    nothing, so capture waits until beats are genuinely arriving with sensor
    contact rather than starting on a click.

    `live` means: a sample landed in the last few seconds, it reported
    contact, and it carried beats.
    """
    from django.utils import timezone
    from datetime import timedelta

    recent_from = timezone.now() - timedelta(seconds=LIVE_WINDOW_S)
    recent = list(
        PhysioSample.objects
        .filter(session=session, student=student, t__gte=recent_from)
        .order_by('-t')[:10]
    )

    beats = sum(len(s.ibi_ms or []) for s in recent)
    latest = recent[0] if recent else None
    return {
        'live': bool(recent) and bool(latest and latest.contact) and beats > 0,
        'contact': bool(latest.contact) if latest else False,
        'recent_samples': len(recent),
        'recent_beats': beats,
        'last_bpm': latest.bpm if latest else None,
    }


# How far back "right now" reaches. The band notifies about once a second, so
# a few seconds is enough to tell live signal from a stale trickle.
LIVE_WINDOW_S = 8
