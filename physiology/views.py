"""Physiological signals API (physical exam stations only).

    POST /api/sessions/<id>/physio/device/           bind the band to a student
    POST /api/sessions/<id>/physio/samples/          sidecar sample batches
    POST /api/sessions/<id>/physio/baseline/start/   open the calm window
    POST /api/sessions/<id>/physio/baseline/stop/    close it, derive resting values
    GET  /api/sessions/<id>/physio/timeline/         examiner: arousal over time

Ingest accepts the exam-station token (a headless sidecar has no browser
session) and the kiosk lease. Reading is examiner-only: this is health-adjacent
data about a named student, and it is not shown to their peers.
"""

import logging

from django.utils import timezone
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from attribution.authentication import (
    ExamStationAuthentication,
    is_station_principal,
)
from authentication.authentication import CookieJWTAuthentication
from core.models import (
    EvaluationSession,
    ExaminerProfile,
    ProjectExaminer,
    StudentProfile,
)
from physical_evaluation.authentication import PhysicalKioskAuthentication
from physiology.models import BaselineWindow
from physiology.services import analysis, ingest

logger = logging.getLogger(__name__)


def _err(message, code=400):
    return Response({'success': False, 'message': message}, status=code)


def _ok(data, message='', code=200):
    body = {'success': True, 'data': data}
    if message:
        body['message'] = message
    return Response(body, status=code)


def _get_session(session_id):
    return (
        EvaluationSession.objects
        .filter(id=session_id)
        .select_related('project', 'student__user', 'group')
        .first()
    )


def _is_examiner_for(user, session) -> bool:
    ep = ExaminerProfile.objects.filter(user=user).first()
    return bool(ep) and ProjectExaminer.objects.filter(
        project=session.project, examiner=ep,
    ).exists()


def _is_station(user, session) -> bool:
    """The kiosk, a station sidecar, or the assigned examiner."""
    if is_station_principal(user):
        return True
    return _is_examiner_for(user, session)


def _roster(session):
    if session.group_id:
        return list(
            StudentProfile.objects
            .filter(group_memberships__group_id=session.group_id)
            .select_related('user')
        )
    return [session.student] if session.student_id else []


def _name(profile):
    if profile is None:
        return None
    full = (profile.user.full_name or '').strip()
    if full and full.lower() != 'none':
        return full
    return profile.user.email


def _baseline_state(session, device) -> str:
    """Where the calm capture has got to, so the panel can drive itself.

    none      nothing attempted yet
    capturing a window is open and collecting
    ready     a usable resting baseline exists
    unusable  the last attempt closed with too little clean signal
    """
    if device is None:
        return 'none'

    windows = BaselineWindow.objects.filter(
        session=session, student=device.student,
    ).order_by('-started_at')

    if windows.filter(ended_at__isnull=True).exists():
        return 'capturing'
    for window in windows:
        if window.is_usable:
            return 'ready'
    return 'unusable' if windows.exists() else 'none'


class PhysioDeviceView(APIView):
    """GET/POST /api/sessions/<session_id>/physio/device/

    Names who is wearing the band. Everything downstream depends on it: an
    unattributed pulse is not evidence about anybody, so ingest refuses rather
    than storing it against a guess.
    """

    authentication_classes = [
        ExamStationAuthentication,
        PhysicalKioskAuthentication,
        CookieJWTAuthentication,
    ]
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_station(request.user, session):
            return _err('Not permitted for this session.', code=403)

        device = ingest.bound_device(session)
        signal = (
            ingest.signal_status(session, device.student) if device
            else {'live': False, 'contact': False, 'recent_samples': 0,
                  'recent_beats': 0, 'last_bpm': None}
        )
        return _ok({
            'device_id': device.device_id if device else None,
            'student_id': str(device.student_id) if device else None,
            'student_name': _name(device.student) if device else None,
            'battery_pct': device.battery_pct if device else None,
            'signal': signal,
            'baseline_state': _baseline_state(session, device),
            'roster': [
                {'student_id': str(s.id), 'name': _name(s)}
                for s in _roster(session)
            ],
        })

    def post(self, request, session_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_station(request.user, session):
            return _err('Not permitted for this session.', code=403)

        device_id = (request.data.get('device_id') or '').strip()
        if not device_id:
            return _err('device_id is required.')

        roster = _roster(session)
        student_id = request.data.get('student_id')
        if not student_id:
            # An individual session has exactly one candidate, so there is
            # nothing for a human to choose.
            if len(roster) == 1:
                student = roster[0]
            else:
                return _err(
                    'student_id is required: more than one student could be '
                    'wearing the band.'
                )
        else:
            student = next(
                (s for s in roster if str(s.id) == str(student_id)), None,
            )
            if student is None:
                return _err('That student is not part of this session.')

        device = ingest.bind_device(session, device_id, student)
        return _ok({
            'device_id': device.device_id,
            'student_id': str(device.student_id),
            'student_name': _name(student),
        }, f'Band assigned to {_name(student)}.', code=201)


class PhysioSampleView(APIView):
    """POST /api/sessions/<session_id>/physio/samples/

    Body: {"device_id": "...", "samples": [
              {"t": iso, "bpm": 72, "ibi_ms": [830, 845], "contact": true}
           ]}

    Batched because the band notifies about once a second and a request per
    beat would be pure overhead.
    """

    authentication_classes = [
        ExamStationAuthentication,
        PhysicalKioskAuthentication,
        CookieJWTAuthentication,
    ]
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_station(request.user, session):
            return _err('Not permitted for this session.', code=403)

        samples = request.data.get('samples') or []
        if not isinstance(samples, list):
            return _err('samples must be a list.')
        if len(samples) > 1000:
            return _err('Too many samples in one batch (max 1000).')
        if not samples:
            return _ok({'stored': 0})

        result = ingest.ingest_samples(
            session, samples, request.data.get('device_id'),
        )
        if result.get('error'):
            return _err(result['error'], code=409)
        return _ok(result)


class PhysioBaselineView(APIView):
    """POST /api/sessions/<session_id>/physio/baseline/<action>/

    'start' opens the calm window, 'stop' closes it and derives the student's
    resting rate and variability. Everything the examiner later sees is
    expressed relative to those numbers, which is why a failed baseline is
    reported loudly rather than silently defaulted.
    """

    authentication_classes = [
        ExamStationAuthentication,
        PhysicalKioskAuthentication,
        CookieJWTAuthentication,
    ]
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id, action):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_station(request.user, session):
            return _err('Not permitted for this session.', code=403)

        device = ingest.bound_device(session)
        if device is None:
            return _err('Bind the band to a student first.', code=409)

        if action == 'start':
            BaselineWindow.objects.filter(
                session=session, student=device.student, ended_at__isnull=True,
            ).delete()
            window = BaselineWindow.objects.create(
                session=session,
                student=device.student,
                started_at=timezone.now(),
            )
            return _ok({
                'baseline_id': str(window.id),
                'started_at': window.started_at,
            }, 'Calm period started. Ask the student to sit still and quiet.',
                code=201)

        if action == 'stop':
            window = BaselineWindow.objects.filter(
                session=session, student=device.student, ended_at__isnull=True,
            ).order_by('-started_at').first()
            if window is None:
                return _err('No calm period is running.', code=409)

            analysis.close_baseline(window)
            payload = {
                'baseline_id': str(window.id),
                'hr_mean': window.hr_mean,
                'rmssd': window.rmssd,
                'beat_count': window.beat_count,
                'quality': window.quality,
                'usable': window.is_usable,
            }
            if not window.is_usable:
                return _ok(
                    payload,
                    'Calm period recorded, but too few clean beats to use it. '
                    'Check the finger clip and run it again - without a '
                    'baseline no arousal reading can be produced.',
                )
            return _ok(payload, 'Baseline captured.')

        return _err("action must be 'start' or 'stop'.", code=404)


class PhysioTimelineView(APIView):
    """GET /api/sessions/<session_id>/physio/timeline/

    The examiner's view: a rolling arousal series aligned to the recording,
    and an explicit list of who was NOT measured.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_examiner_for(request.user, session):
            return _err('You are not assigned to this project.', code=403)

        device = ingest.bound_device(session)
        measured = device.student if device else None

        # Named explicitly so the UI cannot render an unmeasured student as
        # though they had been measured and found calm.
        unmeasured = [
            {'student_id': str(s.id), 'name': _name(s)}
            for s in _roster(session)
            if measured is None or s.id != measured.id
        ]

        payload = {
            'measured_student_id': str(measured.id) if measured else None,
            'measured_student_name': _name(measured),
            'unmeasured_students': unmeasured,
            'device_id': device.device_id if device else None,
            'battery_pct': device.battery_pct if device else None,
            'timeline': None,
        }
        if measured is not None:
            payload['timeline'] = analysis.build_timeline(session, measured)

        return _ok(payload)


class StationActiveSessionView(APIView):
    """GET /api/physio/station/active/

    Which physical session, if any, is running right now.

    This exists so the band relay can be a SERVICE rather than a command
    somebody types. A sidecar started with a session id in its URL has to be
    launched by hand once per viva, which in practice means it gets launched
    late or not at all - and a calm baseline can only be captured while it is
    already streaming. Asking the platform "who am I feeding?" lets one
    process start at boot and follow sessions as they come and go.

    Station-token only: it reveals which room is busy, which is not a
    student's business and not a browser's.

    Single station per deployment is assumed. With several exam rooms sharing
    one backend this would need scoping to the kiosk that opened the session.
    """

    authentication_classes = [
        ExamStationAuthentication,
        PhysicalKioskAuthentication,
    ]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if not is_station_principal(request.user):
            return _err('Exam stations only.', code=403)

        from physical_evaluation.models import PhysicalEvaluationRun

        run = (
            PhysicalEvaluationRun.objects
            .filter(status__in=[
                PhysicalEvaluationRun.Status.DEMO_IN_PROGRESS,
                PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS,
            ])
            .select_related('session__project')
            .order_by('-created_at')
            .first()
        )
        if run is None:
            return _ok({'session_id': None})

        session = run.session
        device = ingest.bound_device(session)
        return _ok({
            'session_id': str(session.id),
            'project': session.project.project_name,
            'phase': run.status,
            # The relay posts regardless, but this tells the log whether the
            # samples will be accepted or bounced for want of a wearer.
            'device_bound': device is not None,
            'student_name': _name(device.student) if device else None,
        })
