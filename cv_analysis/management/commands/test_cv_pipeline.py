"""End-to-end test of the CV pipeline against one video clip.

The `test_clip.py` script in exam-station-cv exercises the ENGINE. This
exercises the PLATFORM: storage, the session manifest, the analysis backend,
the summary artifact, speaker evidence, and attribution write-back. Use it to
answer "does the whole chain work", not "does the CV see my face".

    python manage.py test_cv_pipeline --video me.mp4
    python manage.py test_cv_pipeline --video me.mp4 --photo me.jpg --group

It reuses ONE test session per mode, keyed on a marker in the project name, so
running it repeatedly does not litter the database with sessions. Pass
--fresh to force a new one.

Stages, each reported as it completes:

    1. fixtures    reusable test project / student(s) / session
    2. storage     upload the clip (Azure blob, or local file)
    3. recording   SessionRecording row with a clock origin
    4. questions   a synthetic Q&A spanning the clip, so attribution has
                   something to attribute (skip with --no-qa)
    5. analysis    run the configured backend, store CVSessionReport
    6. attribution ingest the artifact as evidence and reconcile

Backends: CV_ANALYSIS_BACKEND=subprocess runs the engine locally and needs
only CV_ANALYSIS_PYTHON. The `modal` default needs MODAL_CV_* configured and
a deployed Modal app.
"""

from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

MARKER = '[cv-pipeline-test]'


class Command(BaseCommand):
    help = 'Run one video clip through the full CV + attribution pipeline.'

    def add_arguments(self, parser):
        parser.add_argument('--video', required=True, type=str)
        parser.add_argument(
            '--photo', action='append', default=[],
            help='Enrolment face photo (repeatable, one per student). '
                 'Required to exercise face recognition.',
        )
        parser.add_argument(
            '--group', action='store_true',
            help='Build a GROUP session. Individual mode never loads the '
                 'recognition model, so face matching is only tested here.',
        )
        parser.add_argument(
            '--students', type=int, default=1,
            help='Roster size for --group (default 1).',
        )
        parser.add_argument(
            '--no-qa', action='store_true',
            help='Skip the synthetic question/answer, testing CV only.',
        )
        parser.add_argument(
            '--fresh', action='store_true',
            help='Create a new session instead of reusing the test one.',
        )
        parser.add_argument(
            '--backend', type=str, default=None,
            help="Override CV_ANALYSIS_BACKEND for this run "
                 "('subprocess' or 'modal').",
        )

    # -- reporting helpers -------------------------------------------------

    def _stage(self, n, title):
        self.stdout.write(self.style.MIGRATE_HEADING(f'\n[{n}] {title}'))

    def _ok(self, msg):
        self.stdout.write(self.style.SUCCESS(f'    OK  {msg}'))

    def _info(self, msg):
        self.stdout.write(f'        {msg}')

    def _warn(self, msg):
        self.stdout.write(self.style.WARNING(f'    !   {msg}'))

    # -- main --------------------------------------------------------------

    def handle(self, *args, **opts):
        video = Path(opts['video']).expanduser().resolve()
        if not video.exists():
            raise CommandError(f'video not found: {video}')

        if opts['backend']:
            settings.CV_ANALYSIS_BACKEND = opts['backend'].lower()
        # Run inline: the point is to watch it happen, not to queue it.
        settings.CV_ANALYSIS_ENABLED = True
        settings.CV_ANALYSIS_ASYNC = False

        backend = getattr(settings, 'CV_ANALYSIS_BACKEND', 'modal')
        if backend == 'modal' and not getattr(settings, 'MODAL_CV_SUBMIT_URL', ''):
            raise CommandError(
                'CV_ANALYSIS_BACKEND is "modal" but MODAL_CV_SUBMIT_URL is not '
                'set. Deploy cv_analyze_modal.py and configure MODAL_CV_*, or '
                'rerun with --backend subprocess to analyse locally.'
            )

        is_group = opts['group'] or opts['students'] > 1 or len(opts['photo']) > 1
        self.stdout.write(
            f'clip    : {video.name}\n'
            f'mode    : {"group" if is_group else "individual"}\n'
            f'backend : {backend}\n'
            f'photos  : {len(opts["photo"])}'
        )
        if is_group and not opts['photo']:
            self._warn(
                'No --photo given: faces will resolve as UNKNOWN (seating '
                'fallback is off). Recognition is not being tested.'
            )
        if not is_group and opts['photo']:
            self._warn(
                'Individual mode never loads ArcFace, so --photo is ignored. '
                'Add --group to test recognition.'
            )

        session, roster = self._fixtures(is_group, opts['students'], opts['fresh'])
        recording_ref, started_at = self._store(session, video)
        self._recording(session, recording_ref, started_at, video)
        if not opts['no_qa']:
            self._questions(session, started_at, video)
        self._attach_photos(roster, opts['photo'])
        report = self._analyse(session)
        self._attribution(session, report)

    # -- stages ------------------------------------------------------------

    def _fixtures(self, is_group, n_students, fresh):
        from core.models import (
            EvaluationSession, GroupMember, Project, StudentGroup,
            StudentProfile, User,
        )

        self._stage(1, 'Fixtures')
        label = 'group' if is_group else 'individual'
        project_name = f'{MARKER} {label}'

        project, created = Project.objects.get_or_create(
            project_name=project_name,
            defaults={
                'description': 'Reusable fixture for CV pipeline testing.',
                'is_group_project': is_group,
                'status': Project.Status.ACTIVE,
                'evaluation_mode': Project.EvaluationMode.PHYSICAL,
            },
        )
        self._ok(f'project {"created" if created else "reused"}: {project_name}')

        roster = []
        for i in range(n_students if is_group else 1):
            email = f'cv-test-student-{i + 1}@example.invalid'
            user, _ = User.objects.get_or_create(
                email=email,
                defaults={
                    'full_name': f'CV Test Student {i + 1}',
                    'role': User.Role.STUDENT,
                    'is_active': True,
                },
            )
            student, _ = StudentProfile.objects.get_or_create(
                user=user,
                defaults={'registration_number': f'CVTEST{i + 1:03d}'},
            )
            roster.append(student)
        self._ok(f'roster: {", ".join(s.user.full_name for s in roster)}')

        group = None
        if is_group:
            group, _ = StudentGroup.objects.get_or_create(
                project=project, group_name=f'{MARKER} group',
            )
            for student in roster:
                GroupMember.objects.get_or_create(group=group, student=student)

        now = timezone.now()
        session = None
        if not fresh:
            session = (
                EvaluationSession.objects
                .filter(project=project)
                .order_by('-scheduled_start')
                .first()
            )
        if session is None:
            session = EvaluationSession.objects.create(
                project=project,
                student=None if is_group else roster[0],
                group=group,
                scheduled_start=now,
                scheduled_end=now + timedelta(hours=1),
                actual_start=now,
            )
            self._ok(f'session created: {session.id}')
        else:
            # Reusing a session means clearing what the last run left behind,
            # or evidence would accumulate across runs and skew attribution.
            self._reset(session)
            session.actual_start = now
            session.save(update_fields=['actual_start'])
            self._ok(f'session reused (previous run cleared): {session.id}')

        return session, roster

    def _reset(self, session):
        from attribution.models import (
            AnswerAttribution, AnswerContribution, SpeakerBinding,
            SpeakerEvidence, UnknownSpeaker,
        )
        from core.models import SessionRecording, VivaQuestion
        from cv_analysis.models import CVSessionReport

        AnswerContribution.objects.filter(attribution__session=session).delete()
        AnswerAttribution.objects.filter(session=session).delete()
        SpeakerEvidence.objects.filter(session=session).delete()
        UnknownSpeaker.objects.filter(session=session).delete()
        SpeakerBinding.objects.filter(session=session).delete()
        VivaQuestion.objects.filter(session=session).delete()
        SessionRecording.objects.filter(session=session).delete()
        CVSessionReport.objects.filter(session=session).delete()

    def _store(self, session, video):
        from cv_analysis.services.storage import save_recording_locally

        self._stage(2, 'Storage')
        started_at = timezone.now()

        with open(video, 'rb') as fh:
            upload = SimpleUploadedFile(
                video.name, fh.read(),
                content_type='video/mp4' if video.suffix == '.mp4' else 'video/webm',
            )

        try:
            from AI_Evaluator_Backend.azure_storage import upload_video_to_blob

            ref = upload_video_to_blob(
                upload, str(session.project_id), str(session.id),
            )
            self._ok('uploaded to Azure blob')
            self._info(ref)
        except Exception as e:
            self._warn(f'Azure upload failed ({str(e)[:120]}) - storing locally')
            upload.seek(0)
            ref = save_recording_locally(upload, session.id)
            self._ok(f'stored locally: {ref}')

        return ref, started_at

    def _recording(self, session, ref, started_at, video):
        from core.models import SessionRecording

        self._stage(3, 'Recording row')
        SessionRecording.objects.create(
            session=session,
            video_file_url=ref,
            recording_started_at=started_at,
        )
        # recording_started_at is the origin every timestamp is measured from:
        # without it the question timeline is dropped and post-hoc CV evidence
        # cannot be placed on the session clock at all.
        self._ok(f'SessionRecording created, clock origin {started_at.isoformat()}')

    def _questions(self, session, started_at, video):
        from core.models import VivaAnswer, VivaQuestion

        self._stage(4, 'Synthetic Q&A')
        question = VivaQuestion.objects.create(
            session=session,
            question_text='Describe the architecture of your system.',
            question_order=1,
            question_source=VivaQuestion.QuestionSource.AI,
        )
        # generated_at/answered_at are auto_now_add, so bend them to straddle
        # the clip - the answer window is what evidence is matched against.
        VivaQuestion.objects.filter(pk=question.pk).update(generated_at=started_at)
        answer = VivaAnswer.objects.create(
            question=question,
            student=None,
            transcribed_answer='(spoken during the test clip)',
            ai_answer_score=8.0,
        )
        VivaAnswer.objects.filter(pk=answer.pk).update(
            answered_at=started_at + timedelta(minutes=5),
        )
        self._ok('one question + one unattributed answer spanning the clip')

    def _attach_photos(self, roster, photos):
        if not photos:
            return
        from AI_Evaluator_Backend.azure_storage import upload_face_photo_to_blob

        self._stage('4b', 'Enrolment photos')
        for student, photo_path in zip(roster, photos):
            path = Path(photo_path).expanduser().resolve()
            if not path.exists():
                self._warn(f'photo not found, skipping: {path}')
                continue
            with open(path, 'rb') as fh:
                upload = SimpleUploadedFile(
                    path.name, fh.read(), content_type='image/jpeg',
                )
            try:
                url = upload_face_photo_to_blob(upload, str(student.id))
                student.face_photo_url = url
                student.save(update_fields=['face_photo_url'])
                self._ok(f'{student.user.full_name} enrolled')
            except Exception as e:
                self._warn(f'enrolment upload failed: {str(e)[:120]}')

    def _analyse(self, session):
        from cv_analysis.models import CVSessionReport
        from cv_analysis.services.runner import run_cv_analysis

        self._stage(5, 'Analysis')
        self._info('running inline; a 20s clip takes roughly a minute...')
        report = run_cv_analysis(session.id)

        if report.status != CVSessionReport.Status.COMPLETED:
            self._warn(f'status={report.status}')
            self.stdout.write(self.style.ERROR(
                f'    {report.error_message[:600]}'
            ))
            return report

        artifact = report.artifact or {}
        self._ok('analysis completed, artifact stored on CVSessionReport')
        self._info(f"schema {artifact.get('schema_version')} "
                   f"mode={artifact.get('mode')}")
        for s in artifact.get('per_student', []):
            self._info(
                f"{s.get('display_name')}: {s.get('turn_count')} turns, "
                f"{(s.get('speaking_time_ms') or 0) / 1000:.1f}s speaking, "
                f"attention={s.get('attention_pct')}, "
                f"{s.get('off_screen_glance_count')} look-aways, "
                f"{len(s.get('integrity_flags') or [])} flags"
            )
        for flag in artifact.get('session_flags', []):
            self._info(f"session flag [{flag.get('video_timecode')}] "
                       f"{flag.get('kind')}")
        self._info(f"timeline entries: {len(artifact.get('timeline') or [])}")
        return report

    def _attribution(self, session, report):
        from attribution.models import (
            AnswerAttribution, AnswerContribution, SpeakerEvidence,
            UnknownSpeaker,
        )

        self._stage(6, 'Attribution')
        if not report.artifact:
            self._warn('no artifact, nothing to attribute')
            return

        # run_cv_analysis already calls feed_attribution on success; report
        # what actually landed rather than doing it twice.
        evidence = SpeakerEvidence.objects.filter(session=session)
        self._ok(f'{evidence.count()} evidence rows')
        for source, in evidence.values_list('source').distinct():
            self._info(f'source {source}: {evidence.filter(source=source).count()}')

        unknown = UnknownSpeaker.objects.filter(session=session)
        if unknown.exists():
            self._ok(f'{unknown.count()} unknown speaker(s) holding marks')
            for u in unknown:
                held = AnswerContribution.objects.filter(unknown_speaker=u).count()
                self._info(f'{u.label}: {held} contribution(s) - '
                           f'resolve via the unknown-speakers endpoint')

        attributions = (
            AnswerAttribution.objects
            .filter(session=session)
            .select_related('student__user', 'unknown_speaker')
        )
        if not attributions.exists():
            # Distinguish the two very different reasons for an empty result,
            # because "no answers" and "no evidence" call for opposite fixes.
            from core.models import VivaAnswer

            answers = VivaAnswer.objects.filter(question__session=session).count()
            if not answers:
                self._info('no answers in this session (--no-qa)')
            elif not evidence.exists():
                self._warn(
                    f'{answers} answer(s), but the CV found no speaking turns, '
                    'so there was nothing to attribute.'
                )
                self._info(
                    'A turn needs a detected face AND voice activity AND lip '
                    'motion. Check the "Faces seen" line above: no face means '
                    'the clip never gave the recogniser anything to work with.'
                )
            else:
                self._warn(
                    f'{answers} answer(s) and {evidence.count()} evidence row(s), '
                    'but no attribution was recorded - the evidence may not '
                    'overlap the answer window.'
                )
            return

        self._ok(f'{attributions.count()} answer attribution(s)')
        for a in attributions:
            if a.student_id:
                who = a.student.user.full_name
            elif a.unknown_speaker_id:
                who = a.unknown_speaker.label
            else:
                who = 'uncertain -> group'
            self._info(
                f'answer {str(a.answer_id)[:8]}: {who} '
                f'[{a.outcome}] share={a.share} margin={a.margin}'
            )
            for c in a.contributions.all():
                label = (
                    c.student.user.full_name if c.student_id
                    else (c.unknown_speaker.label if c.unknown_speaker_id else '?')
                )
                self._info(f'    {label}: {c.share:.0%}'
                           f'{" (dominant)" if c.is_dominant else ""}')

        self.stdout.write(self.style.SUCCESS('\nPipeline run complete.'))
