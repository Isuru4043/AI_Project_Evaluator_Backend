"""Bench harness for the physiological pipeline.

Lets the backend and the hardware be proved SEPARATELY. Debugging a band, a
BLE link, an auth token and an analysis chain all at once is how an afternoon
disappears; each half here can be shown correct on its own.

    python manage.py test_physio setup       # session + binding, prints the
                                             # sidecar command to paste
    python manage.py test_physio simulate    # synthetic beats through the real
                                             # services - no hardware needed
    python manage.py test_physio timeline    # the arousal curve, as text
    python manage.py test_physio reset       # clear this session's physio data

`simulate` deliberately calls the same ingest and analysis services the
sidecar does, so a pass means the chain works and anything left is the radio
or the sensor.
"""

import random
from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

MARKER = '[physio-test]'


class Command(BaseCommand):
    help = 'Exercise the physiological pipeline with or without hardware.'

    def add_arguments(self, parser):
        parser.add_argument(
            'action',
            choices=['setup', 'simulate', 'timeline', 'reset', 'sidecar'],
        )
        parser.add_argument('--session-id', default=None)
        parser.add_argument('--group', action='store_true',
                            help='Build a 2-student group session instead.')
        parser.add_argument('--device', default='VivaSense-HR-TEST')
        parser.add_argument('--backend', default='http://localhost:8000',
                            help='Base URL the sidecar should post to.')
        parser.add_argument('--calm-s', type=int, default=45)
        parser.add_argument('--viva-s', type=int, default=180)

    # -- output helpers ----------------------------------------------------

    def _head(self, text):
        self.stdout.write(self.style.MIGRATE_HEADING(f'\n{text}'))

    def _ok(self, text):
        self.stdout.write(self.style.SUCCESS(f'  OK  {text}'))

    def _info(self, text):
        self.stdout.write(f'      {text}')

    def _warn(self, text):
        self.stdout.write(self.style.WARNING(f'  !   {text}'))

    # -- entry -------------------------------------------------------------

    def handle(self, *args, **opts):
        self.opts = opts
        session = self._session(opts)
        getattr(self, f'_{opts["action"]}')(session)

    def _session(self, opts):
        from core.models import (
            EvaluationSession, GroupMember, Project, StudentGroup,
            StudentProfile, User,
        )

        if opts['session_id']:
            session = EvaluationSession.objects.filter(
                id=opts['session_id'],
            ).first()
            if session is None:
                raise CommandError(f'No session {opts["session_id"]}')
            return session

        is_group = opts['group']
        label = 'group' if is_group else 'individual'
        project, _ = Project.objects.get_or_create(
            project_name=f'{MARKER} {label}',
            defaults={
                'description': 'Reusable fixture for physiology testing.',
                'is_group_project': is_group,
                'status': Project.Status.ACTIVE,
                'evaluation_mode': Project.EvaluationMode.PHYSICAL,
            },
        )

        roster = []
        for i in range(2 if is_group else 1):
            user, _ = User.objects.get_or_create(
                email=f'physio-test-{i + 1}@example.invalid',
                defaults={
                    'full_name': f'Physio Test Student {i + 1}',
                    'role': User.Role.STUDENT,
                    'is_active': True,
                },
            )
            student, _ = StudentProfile.objects.get_or_create(
                user=user,
                defaults={'registration_number': f'PHYSIO{i + 1:03d}'},
            )
            roster.append(student)

        group = None
        if is_group:
            group, _ = StudentGroup.objects.get_or_create(
                project=project, group_name=f'{MARKER} group',
            )
            for student in roster:
                GroupMember.objects.get_or_create(group=group, student=student)

        session = (
            EvaluationSession.objects
            .filter(project=project)
            .order_by('-scheduled_start')
            .first()
        )
        if session is None:
            now = timezone.now()
            session = EvaluationSession.objects.create(
                project=project,
                student=None if is_group else roster[0],
                group=group,
                scheduled_start=now,
                scheduled_end=now + timedelta(hours=1),
                actual_start=now,
            )
        return session

    # -- actions -----------------------------------------------------------

    def _sidecar(self, session):
        """Print the ready-to-paste sidecar command for the LIVE session.

        The sidecar is a separate process and has to be told which session it
        is feeding, but hunting that UUID out of the database mid-viva is
        exactly the friction that gets it started late - and a baseline can
        only be captured while it is already streaming.
        """
        from physical_evaluation.models import PhysicalEvaluationRun
        from physiology.services import ingest

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
            self._warn('No physical session is running. Start one in the kiosk first.')
            return

        target = run.session
        device = ingest.bound_device(target)
        token = getattr(settings, 'EXAM_STATION_TOKEN', '')
        base = self.opts['backend'].rstrip('/')

        self._head('Live physical session')
        self._ok(f'{target.project.project_name}  [{run.status}]')
        self._info(f'session {target.id}')
        self._info(
            f'band    {device.device_id} -> {device.student.user.full_name}'
            if device else
            'band    NOT BOUND - pick the wearer in the kiosk panel first'
        )

        if not token:
            self._warn('EXAM_STATION_TOKEN is empty; the backend will reject this.')

        self._head('Run this now, and keep it running for the whole session')
        cmd = ' '.join([
            r'venv\Scripts\python.exe -m physiology.station_sidecar',
            f'--backend {base}/api/sessions/{target.id}/physio',
            f'--token {token or "<EXAM_STATION_TOKEN>"}',
            '--device VivaSense-HR',
        ])
        self.stdout.write('')
        self.stdout.write('  ' + cmd)
        self.stdout.write('')
        self._info('The calm baseline records itself once beats start arriving,')
        self._info('so start this BEFORE the demo phase ends.')

    def _setup(self, session):
        from physiology.services import ingest

        self._head('Session')
        self._ok(f'id {session.id}')
        roster = self._roster(session)
        for student in roster:
            self._info(f'{student.user.full_name}  ({student.id})')

        self._head('Device binding')
        device = ingest.bind_device(session, self.opts['device'], roster[0])
        self._ok(f'{device.device_id} -> {roster[0].user.full_name}')
        if len(roster) > 1:
            self._warn(
                f'{len(roster) - 1} other student(s) have no band and will '
                'report NO DATA - never "calm".'
            )

        token = getattr(settings, 'EXAM_STATION_TOKEN', '')
        self._head('Run the sidecar')
        if not token:
            self._warn(
                'EXAM_STATION_TOKEN is empty, so the backend will reject the '
                'sidecar. Set it in .env and restart the server.'
            )
        base = self.opts['backend'].rstrip('/')
        self.stdout.write(
            f'\n  python -m physiology.station_sidecar \\\n'
            f'      --backend {base}/api/sessions/{session.id}/physio \\\n'
            f'      --token   {token or "<EXAM_STATION_TOKEN>"} \\\n'
            f'      --device  VivaSense-HR\n'
        )

        self._head('Then, from the station')
        self.stdout.write(
            f'  1. start calm period:\n'
            f'     curl -X POST {base}/api/sessions/{session.id}/physio/baseline/start/ '
            f'-H "X-Station-Token: {token or "<TOKEN>"}"\n\n'
            f'  2. sit still for {self.opts["calm_s"]}s, then:\n'
            f'     curl -X POST {base}/api/sessions/{session.id}/physio/baseline/stop/ '
            f'-H "X-Station-Token: {token or "<TOKEN>"}"\n\n'
            f'  3. run the viva, then read the curve:\n'
            f'     python manage.py test_physio timeline --session-id {session.id}\n'
        )

    def _simulate(self, session):
        """Push synthetic beats through the real services.

        Two stretches: a calm baseline, then a viva containing one genuinely
        aroused passage - rate up AND variability down together, which is the
        only pattern the analyser treats as elevated.
        """
        from physiology.models import BaselineWindow
        from physiology.services import analysis, ingest

        random.seed(11)
        roster = self._roster(session)
        student = roster[0]
        device = ingest.bound_device(session)
        if device is None:
            device = ingest.bind_device(session, self.opts['device'], student)
            self._info(f'bound {device.device_id} -> {student.user.full_name}')
        student = device.student

        calm_s = self.opts['calm_s']
        viva_s = self.opts['viva_s']
        start = timezone.now() - timedelta(seconds=calm_s + viva_s + 5)

        # Anchor the session clock to the simulated data. The fixture session
        # is reused across days, so without this the samples sit hours after
        # actual_start and every timecode reads like 17:54:46 instead of an
        # offset into the recording. A real session starts when its samples do.
        session.actual_start = start
        session.save(update_fields=['actual_start'])

        self._head('Baseline (calm)')
        window = BaselineWindow.objects.create(
            session=session, student=student, started_at=start,
        )
        n = self._push(session, device, start, calm_s, hr=70, swing=45)
        self._ok(f'{n} sample(s), ~{calm_s}s at ~70 bpm with healthy variability')

        window.ended_at = start + timedelta(seconds=calm_s)
        analysis.close_baseline(window)
        if not window.is_usable:
            self._warn(f'baseline unusable: {window.beat_count} beats, '
                       f'quality {window.quality}')
            return
        self._ok(f'baseline: HR {window.hr_mean:.1f}, RMSSD {window.rmssd:.1f} ms, '
                 f'{window.beat_count} beats')

        self._head('Viva')
        viva_start = start + timedelta(seconds=calm_s + 5)
        third = viva_s // 3
        # calm answering
        self._push(session, device, viva_start, third, hr=72, swing=42)
        # a hard question: rate up, variability collapsing
        self._push(session, device, viva_start + timedelta(seconds=third),
                   third, hr=92, swing=6)
        # recovery
        self._push(session, device, viva_start + timedelta(seconds=2 * third),
                   third, hr=75, swing=38)
        self._ok(f'{viva_s}s simulated: calm -> aroused -> recovering')

        self._timeline(session)

    def _push(self, session, device, start, seconds, hr, swing):
        """Generate one second's worth of beats at a time and ingest them."""
        from physiology.services import ingest

        mean_ibi = 60000.0 / hr
        samples = []
        t = start
        carry = 0.0
        while t < start + timedelta(seconds=seconds):
            ibis = []
            carry += 1000.0
            while carry >= mean_ibi:
                ibi = mean_ibi + random.gauss(0, swing / 2.0)
                ibis.append(round(max(334.0, min(1499.0, ibi)), 1))
                carry -= mean_ibi
            samples.append({
                't': t.isoformat(),
                'bpm': int(round(hr)),
                'ibi_ms': ibis,
                'contact': True,
            })
            t += timedelta(seconds=1)
        result = ingest.ingest_samples(session, samples, device.device_id)
        return result.get('stored', 0)

    def _timeline(self, session):
        from physiology.services import analysis, ingest

        device = ingest.bound_device(session)
        if device is None:
            self._warn('no band bound to this session')
            return

        data = analysis.build_timeline(session, device.student)
        self._head(f'Arousal timeline - {device.student.user.full_name}')

        baseline = data.get('baseline')
        if not baseline or not baseline['usable']:
            self._warn('no usable baseline: no arousal reading can be produced')
        else:
            self._info(f"baseline HR {baseline['hr_mean']}, "
                       f"RMSSD {baseline['rmssd']} ms")

        points = data.get('points') or []
        if not points:
            self._warn('no samples recorded yet')
            return

        self.stdout.write('')
        self.stdout.write(
            '      time        HR   dHR   RMSSD   RMSSD%   state')
        for p in points:
            if not p['usable']:
                self.stdout.write(
                    f"      {p['video_timecode']}    --     --      --       --   "
                    f"{self.style.WARNING('no reading')}  ({p['reason']})"
                )
                continue
            ratio = f"{p['rmssd_ratio'] * 100:5.0f}%" if p['rmssd_ratio'] else '   --'
            state = (self.style.ERROR('ELEVATED') if p['elevated']
                     else self.style.SUCCESS('normal  '))
            self.stdout.write(
                f"      {p['video_timecode']}  {p['hr']:5.1f}  "
                f"{p['hr_delta']:+5.1f}  {p['rmssd']:6.1f}  {ratio}   {state}"
            )

        self.stdout.write('')
        self._ok(f"{data['elevated_count']} elevated window(s) of "
                 f"{len(points)}, coverage {data['coverage']:.0%}")
        self._info('ELEVATED = rate up AND variability down together.')
        self._info('Rate alone is not flagged: talking raises heart rate.')

    def _reset(self, session):
        from physiology.models import BaselineWindow, PhysioDevice, PhysioSample

        samples = PhysioSample.objects.filter(session=session).delete()[0]
        windows = BaselineWindow.objects.filter(session=session).delete()[0]
        devices = PhysioDevice.objects.filter(session=session).delete()[0]
        self._head('Reset')
        self._ok(f'{samples} sample(s), {windows} baseline(s), {devices} device(s)')

    def _roster(self, session):
        from core.models import StudentProfile

        if session.group_id:
            return list(
                StudentProfile.objects
                .filter(group_memberships__group_id=session.group_id)
                .select_related('user')
            )
        return [session.student]
