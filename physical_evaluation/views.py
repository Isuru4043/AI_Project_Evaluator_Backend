import logging
import secrets
from datetime import timedelta

from django.conf import settings
from django.db import close_old_connections, transaction
from django.utils import timezone
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import (
    EvaluationSession,
    Project,
    ProjectExaminer,
    SessionRecording,
)
from physical_evaluation.authentication import (
    PhysicalKioskAuthentication,
    PhysicalRecordingAuthentication,
    make_recording_upload_token,
)
from physical_evaluation.models import (
    PhysicalEvaluationRun,
    PhysicalKioskAccess,
    PhysicalProjectConfig,
    PhysicalRecordingUpload,
)
from physical_evaluation.serializers import (
    IdentityOverrideSerializer,
    PanelPinSerializer,
    PhysicalProjectConfigSerializer,
    PhysicalRunSerializer,
    PhysicalSessionSerializer,
    PhysicalSettingsUpdateSerializer,
    PhysicalRecordingUploadSerializer,
)
from physical_evaluation.services import submission_is_ready
from projects.permissions import IsExaminer, IsProjectLead
from authentication.cookies import clear_auth_cookies

logger = logging.getLogger(__name__)


def _ok(message, data=None, code=200):
    return Response({'success': True, 'message': message, 'data': data}, status=code)


def _err(message, errors=None, code=400):
    return Response(
        {'success': False, 'message': message, 'errors': errors or {}},
        status=code,
    )


def _assigned_examiner(user, project):
    try:
        examiner = user.examiner_profile
    except Exception:
        return None
    if not ProjectExaminer.objects.filter(project=project, examiner=examiner).exists():
        return None
    return examiner


def _kiosk_access(request):
    return request.auth if isinstance(request.auth, PhysicalKioskAccess) else None


def _recording_upload_access(request):
    return request.auth if isinstance(request.auth, PhysicalRecordingUpload) else None


def _run_for_access(access, session_id):
    return PhysicalEvaluationRun.objects.select_related(
        'session__project', 'session__student__user', 'session__group',
    ).filter(kiosk_access=access, session_id=session_id).first()


def _run_for_recording_request(request, session_id):
    upload = _recording_upload_access(request)
    if upload is not None:
        return upload.run if str(upload.run.session_id) == str(session_id) else None
    access = _kiosk_access(request)
    return _run_for_access(access, session_id) if access is not None else None


def _attempt_recording_finalize(upload_id):
    """Commit a physical recording once every expected Azure block exists."""
    with transaction.atomic():
        # Lock only the upload row. Including the nullable run.recording join in
        # this FOR UPDATE query is rejected by PostgreSQL.
        upload = PhysicalRecordingUpload.objects.select_for_update().get(id=upload_id)
        if upload.status == PhysicalRecordingUpload.Status.READY:
            return True, None
        if upload.status == PhysicalRecordingUpload.Status.FINALIZING:
            return False, None
        if not upload.finalization_requested or upload.expected_chunks is None:
            return False, None

        uploaded = set(upload.uploaded_chunk_indices or [])
        expected = set(range(upload.expected_chunks))
        if not expected.issubset(uploaded):
            return False, None

        upload.status = PhysicalRecordingUpload.Status.FINALIZING
        upload.error_message = ''
        upload.save(update_fields=['status', 'error_message', 'updated_at'])

    try:
        from AI_Evaluator_Backend.azure_storage import commit_physical_video_blocks

        video_url = commit_physical_video_blocks(
            upload.blob_path,
            upload.expected_chunks,
            upload.mime_type,
        )
    except Exception as exc:
        logger.exception('Could not finalize physical recording upload %s', upload_id)
        with transaction.atomic():
            failed_upload = PhysicalRecordingUpload.objects.select_for_update().get(
                id=upload_id,
            )
            failed_run = PhysicalEvaluationRun.objects.select_for_update().get(
                id=failed_upload.run_id,
            )
            failed_upload.status = PhysicalRecordingUpload.Status.FAILED
            failed_upload.error_message = str(exc)[:1000]
            failed_upload.save(update_fields=['status', 'error_message', 'updated_at'])
            failed_run.status = PhysicalEvaluationRun.Status.RECORDING_FAILED
            failed_run.save(update_fields=['status', 'updated_at'])
        return False, str(exc)

    now = timezone.now()
    with transaction.atomic():
        ready_upload = PhysicalRecordingUpload.objects.select_for_update().get(
            id=upload_id,
        )
        run = PhysicalEvaluationRun.objects.select_for_update().select_related(
            'session',
        ).get(id=ready_upload.run_id)
        if run.recording_id:
            recording = SessionRecording.objects.select_for_update().get(
                id=run.recording_id,
            )
            recording.video_file_url = video_url
            recording.duration_seconds = ready_upload.duration_seconds
            recording.recording_started_at = run.recording_started_at
            recording.save(update_fields=[
                'video_file_url', 'duration_seconds', 'recording_started_at',
            ])
        else:
            recording = SessionRecording.objects.create(
                session=run.session,
                video_file_url=video_url,
                duration_seconds=ready_upload.duration_seconds,
                recording_started_at=run.recording_started_at,
            )
            run.recording = recording

        run.status = PhysicalEvaluationRun.Status.COMPLETED
        if run.completed_at is None:
            run.completed_at = now
        run.save(update_fields=['recording', 'status', 'completed_at', 'updated_at'])

        ready_upload.status = PhysicalRecordingUpload.Status.READY
        ready_upload.error_message = ''
        ready_upload.finalized_at = now
        ready_upload.save(update_fields=[
            'status', 'error_message', 'finalized_at', 'updated_at',
        ])

    try:
        from cv_analysis.services.runner import enqueue_cv_analysis

        enqueue_cv_analysis(run.session_id)
    except Exception:
        logger.exception('Could not enqueue CV analysis for physical session %s', run.session_id)

    return True, None


class PhysicalProjectSettingsView(APIView):
    """Read or update the venue and hashed kiosk PIN for a physical project."""

    permission_classes = [IsAuthenticated, IsProjectLead]

    def get(self, request, project_id):
        config = PhysicalProjectConfig.objects.select_related('project').filter(
            project_id=project_id,
        ).first()
        if config is None:
            return _err('Physical project configuration not found.', code=404)
        return _ok('Physical project configuration retrieved.', PhysicalProjectConfigSerializer(config).data)

    def put(self, request, project_id):
        project = Project.objects.filter(id=project_id).first()
        if project is None:
            return _err('Project not found.', code=404)
        if project.evaluation_mode != Project.EvaluationMode.PHYSICAL:
            return _err('This endpoint is only available for physical projects.')

        serializer = PhysicalSettingsUpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return _err('Validation failed.', serializer.errors)

        config = PhysicalProjectConfig.objects.filter(project=project).first()
        if config is None:
            config = PhysicalProjectConfig(
                project=project,
                created_by=request.user.examiner_profile,
                location=serializer.validated_data.get('location', ''),
            )

        if 'location' in serializer.validated_data:
            config.location = serializer.validated_data['location'].strip()
        if 'panel_pin' in serializer.validated_data:
            config.set_panel_pin(serializer.validated_data['panel_pin'])
        if not config.location:
            return _err('A physical location is required.')
        config.save()

        return _ok('Physical project configuration updated.', PhysicalProjectConfigSerializer(config).data)


class KioskOpenView(APIView):
    """Unlock a physical panel and issue a limited, revocable kiosk token."""

    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, project_id):
        serializer = PanelPinSerializer(data=request.data)
        if not serializer.is_valid():
            return _err('Validation failed.', serializer.errors)

        project = Project.objects.filter(id=project_id).first()
        if project is None:
            return _err('Project not found.', code=404)
        examiner = _assigned_examiner(request.user, project)
        if examiner is None:
            return _err('You are not assigned to this project.', code=403)
        if project.evaluation_mode != Project.EvaluationMode.PHYSICAL:
            return _err('Only physical projects can open a physical session panel.')

        config = PhysicalProjectConfig.objects.filter(project=project).first()
        if config is None:
            return _err('Physical project configuration not found.', code=404)
        if not config.check_panel_pin(serializer.validated_data['pin']):
            return _err('Invalid panel PIN/password.', code=403)

        active_run = PhysicalEvaluationRun.objects.filter(
            kiosk_access__config=config,
            status__in=[
                PhysicalEvaluationRun.Status.DEMO_IN_PROGRESS,
                PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS,
            ],
        ).first()
        if active_run:
            return _err(
                'An evaluation is still active in the current panel.',
                code=409,
            )

        raw_token = secrets.token_urlsafe(32)
        now = timezone.now()
        lifetime = timedelta(
            hours=max(1, settings.PHYSICAL_KIOSK_TOKEN_LIFETIME_HOURS),
        )
        with transaction.atomic():
            config = PhysicalProjectConfig.objects.select_for_update().get(pk=config.pk)
            PhysicalKioskAccess.objects.filter(
                config=config, closed_at__isnull=True,
            ).update(closed_at=now)
            access = PhysicalKioskAccess.objects.create(
                config=config,
                opened_by=examiner,
                token_digest=PhysicalKioskAccess.digest_token(raw_token),
                expires_at=now + lifetime,
            )

        response = _ok('Physical session panel opened.', {
            'kiosk_token': raw_token,
            'token_header': PhysicalKioskAuthentication.header_name,
            'expires_at': access.expires_at,
            'project_id': str(project.id),
            'project_name': project.project_name,
            'location': config.location,
            'examiner_session_cleared': True,
        })
        # The browser now holds only the limited kiosk capability. This stops
        # students using the examiner's HttpOnly login cookie to call unrelated
        # dashboard APIs while the full-screen panel is open.
        return clear_auth_cookies(response)


class KioskCloseView(APIView):
    """Lock the kiosk panel. The project PIN is required again."""

    authentication_classes = [PhysicalKioskAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = PanelPinSerializer(data=request.data)
        if not serializer.is_valid():
            return _err('Validation failed.', serializer.errors)
        access = _kiosk_access(request)
        if not access.config.check_panel_pin(serializer.validated_data['pin']):
            return _err('Invalid panel PIN/password.', code=403)
        if access.runs.filter(
            status__in=[
                PhysicalEvaluationRun.Status.DEMO_IN_PROGRESS,
                PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS,
            ],
        ).exists():
            return _err(
                'Complete the active evaluation before closing the panel.',
                code=409,
            )
        access.close()
        return _ok('Physical session panel closed.')


class KioskSessionListView(APIView):
    """List today's non-completed sessions for the kiosk's physical project."""

    authentication_classes = [PhysicalKioskAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        access = _kiosk_access(request)
        today = timezone.localdate()
        sessions = EvaluationSession.objects.filter(
            project=access.config.project,
            scheduled_start__date=today,
            status__in=[EvaluationSession.Status.SCHEDULED, EvaluationSession.Status.IN_PROGRESS],
        ).select_related(
            'project', 'project__physical_config', 'student__user', 'group',
            'submission__index_status',
        ).order_by('scheduled_start')
        return _ok('Physical sessions retrieved.', {
            'date': today,
            'project_id': str(access.config.project_id),
            'project_name': access.config.project.project_name,
            'location': access.config.location,
            'sessions': PhysicalSessionSerializer(sessions, many=True).data,
        })


class KioskActiveRunView(APIView):
    """Return the current physical run so a refreshed kiosk can resume safely."""

    authentication_classes = [PhysicalKioskAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        access = _kiosk_access(request)
        run = access.runs.filter(
            status__in=[
                PhysicalEvaluationRun.Status.DEMO_IN_PROGRESS,
                PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS,
            ],
        ).select_related(
            'session__project__physical_config', 'session__student__user', 'session__group',
        ).order_by('-updated_at').first()
        if run is None:
            return _ok('No physical evaluation is active.', None)
        data = PhysicalRunSerializer(run).data
        data['viva'] = {
            'start_url': '/api/viva/sessions/start/',
            'answer_url': f'/api/viva/sessions/{run.session_id}/answer/',
            'current_question_url': f'/api/viva/sessions/{run.session_id}/current/',
            'status_url': f'/api/viva/sessions/{run.session_id}/status/',
        }
        return _ok('Active physical evaluation retrieved.', data)


class KioskSessionStartView(APIView):
    """Activate a selected physical session with room-camera access."""

    authentication_classes = [PhysicalKioskAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        access = _kiosk_access(request)
        session = EvaluationSession.objects.select_related(
            'project__physical_config', 'student__user', 'group', 'submission',
        ).filter(id=session_id, project=access.config.project).first()
        if session is None:
            return _err('Physical session not found for this panel.', code=404)
        if timezone.localtime(session.scheduled_start).date() != timezone.localdate():
            return _err('Only sessions scheduled for today can be started.')

        existing = PhysicalEvaluationRun.objects.filter(session=session).first()
        if existing:
            if existing.kiosk_access_id == access.id and existing.is_active:
                return _ok('Physical evaluation resumed.', PhysicalRunSerializer(existing).data)
            return _err('This physical evaluation has already been started.', code=409)
        if session.status != EvaluationSession.Status.SCHEDULED:
            return _err('Only a scheduled session can be started.', code=409)
        if not submission_is_ready(session):
            return _err('The project submission is not ready for the viva yet.', code=409)
        if not session.project.rubric_categories.filter(criteria__isnull=False).exists():
            return _err('The project has no rubric criteria configured.', code=409)

        now = timezone.now()
        run_status = (
            PhysicalEvaluationRun.Status.DEMO_IN_PROGRESS
            if session.demo_enabled
            else PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS
        )
        with transaction.atomic():
            # Serialize double-clicks/concurrent starts for one kiosk lease.
            PhysicalKioskAccess.objects.select_for_update().get(pk=access.pk)
            another_run = PhysicalEvaluationRun.objects.filter(
                kiosk_access__config=access.config,
                status__in=[
                    PhysicalEvaluationRun.Status.DEMO_IN_PROGRESS,
                    PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS,
                ],
            ).exists()
            if another_run:
                return _err('Another physical evaluation is already active on this panel.', code=409)
            locked_session = EvaluationSession.objects.select_for_update().get(id=session.id)
            if locked_session.status != EvaluationSession.Status.SCHEDULED:
                return _err('Only a scheduled session can be started.', code=409)
            locked_session.status = EvaluationSession.Status.IN_PROGRESS
            locked_session.actual_start = now
            locked_session.demo_completed_at = None if session.demo_enabled else now
            # Physical sessions never start Agora/STT recording. The kiosk
            # records the room locally and uploads protected chunks through
            # the dedicated physical-recording endpoints instead.
            locked_session.agora_channel_name = ''
            locked_session.agora_stt_task_id = ''
            locked_session.agora_recording_resource_id = ''
            locked_session.agora_recording_sid = ''
            locked_session.save(update_fields=[
                'status', 'actual_start', 'demo_completed_at',
                'agora_channel_name', 'agora_stt_task_id',
                'agora_recording_resource_id', 'agora_recording_sid',
            ])
            run = PhysicalEvaluationRun.objects.create(
                session=locked_session,
                kiosk_access=access,
                status=run_status,
                # The room preview is active during identity/sensor setup, but
                # protected recording begins only at the demo/viva boundary.
                recording_started_at=None,
                viva_started_at=now if not session.demo_enabled else None,
                identity_status=(
                    PhysicalEvaluationRun.IdentityStatus.PENDING
                    if session.group_id
                    else PhysicalEvaluationRun.IdentityStatus.NOT_REQUIRED
                ),
            )

        data = PhysicalRunSerializer(run).data
        data['next_action'] = 'start_demo' if session.demo_enabled else 'start_viva'
        data['viva_start_url'] = '/api/viva/sessions/start/'
        return _ok('Physical evaluation started. Keep the room camera active.', data)


class KioskRecordingStartView(APIView):
    """Stamp the exact browser capture origin at the demo/viva boundary."""

    authentication_classes = [PhysicalKioskAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        from django.utils.dateparse import parse_datetime

        access = _kiosk_access(request)
        run = _run_for_access(access, session_id)
        if run is None:
            return _err('Active physical evaluation not found.', code=404)
        if run.status not in {
            PhysicalEvaluationRun.Status.DEMO_IN_PROGRESS,
            PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS,
        }:
            return _err('This physical evaluation is not accepting a recording.', code=409)
        if not run.identity_authorized:
            return _err(
                'At least one present group member must be verified, with no unknown faces, before recording starts.',
                code=409,
            )

        now = timezone.now()
        raw_started_at = str(request.data.get('started_at') or '').strip()
        browser_started_at = parse_datetime(raw_started_at) if raw_started_at else None
        if browser_started_at is not None and timezone.is_naive(browser_started_at):
            browser_started_at = timezone.make_aware(browser_started_at)
        # The kiosk is trusted, but reject stale/tampered origins. Normal HTTP
        # latency is well inside this one-minute envelope.
        if browser_started_at is None or abs((now - browser_started_at).total_seconds()) > 60:
            browser_started_at = now

        with transaction.atomic():
            locked = PhysicalEvaluationRun.objects.select_for_update().get(id=run.id)
            if locked.recording_started_at is None:
                locked.recording_started_at = browser_started_at
                locked.save(update_fields=['recording_started_at', 'updated_at'])

        locked.refresh_from_db()
        return _ok('Protected room recording started.', PhysicalRunSerializer(locked).data)


class KioskDemoCompleteView(APIView):
    """End the camera-assisted demo and transition the same run to AI viva."""

    authentication_classes = [PhysicalKioskAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        access = _kiosk_access(request)
        run = _run_for_access(access, session_id)
        if run is None:
            return _err('Active physical evaluation not found.', code=404)
        if run.status != PhysicalEvaluationRun.Status.DEMO_IN_PROGRESS:
            return _err('This physical evaluation is not in the demo phase.', code=409)
        if not run.identity_authorized:
            return _err(
                'At least one present group member must be verified with no '
                'unknown faces, or an examiner must authorize the identity-review override.',
                code=409,
            )

        now = timezone.now()
        with transaction.atomic():
            run.status = PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS
            run.viva_started_at = now
            run.save(update_fields=['status', 'viva_started_at', 'updated_at'])
            run.session.demo_completed_at = now
            run.session.save(update_fields=['demo_completed_at'])

        return _ok('Demo completed. Start the shared AI viva now.', {
            'run': PhysicalRunSerializer(run).data,
            'viva_start_url': '/api/viva/sessions/start/',
            'viva_start_payload': {'session_id': str(run.session_id)},
        })


class KioskIdentityOverrideView(APIView):
    """Allow an assigned examiner at the kiosk to acknowledge an incomplete scan."""

    authentication_classes = [PhysicalKioskAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        serializer = IdentityOverrideSerializer(data=request.data)
        if not serializer.is_valid():
            return _err('Validation failed.', serializer.errors)
        access = _kiosk_access(request)
        run = _run_for_access(access, session_id)
        if run is None:
            return _err('Active physical evaluation not found.', code=404)
        if not run.session.group_id:
            return _err('Identity override is only required for group sessions.')
        if not access.config.check_panel_pin(serializer.validated_data['pin']):
            return _err('Invalid examiner PIN/password.', code=403)

        now = timezone.now()
        run.identity_status = PhysicalEvaluationRun.IdentityStatus.OVERRIDDEN
        run.identity_override_at = now
        run.identity_override_by = access.opened_by
        run.identity_override_reason = serializer.validated_data['reason'].strip()
        run.save(update_fields=[
            'identity_status', 'identity_override_at', 'identity_override_by',
            'identity_override_reason', 'updated_at',
        ])
        logger.warning(
            'Identity review overridden for physical session %s by examiner %s',
            session_id,
            access.opened_by_id,
        )
        return _ok(
            'Examiner override recorded. Heart-rate wearer setup remains a separate step.',
            PhysicalRunSerializer(run).data,
        )


class KioskRecordingChunkUploadView(APIView):
    """Stage one small recording block without holding the full video in RAM."""

    authentication_classes = [PhysicalRecordingAuthentication, PhysicalKioskAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    MAX_CHUNK_SIZE = 15 * 1024 * 1024
    MAX_CHUNKS = 20000

    def post(self, request, session_id, chunk_index):
        run = _run_for_recording_request(request, session_id)
        if run is None:
            return _err('Physical evaluation not found.', code=404)
        if run.status not in [
            PhysicalEvaluationRun.Status.DEMO_IN_PROGRESS,
            PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS,
            PhysicalEvaluationRun.Status.RECORDING_UPLOADING,
            PhysicalEvaluationRun.Status.RECORDING_FAILED,
        ]:
            return _err('This recording no longer accepts chunks.', code=409)
        if run.recording_started_at is None:
            return _err('Protected recording has not started yet.', code=409)
        if chunk_index < 0 or chunk_index >= self.MAX_CHUNKS:
            return _err('Invalid recording chunk index.')
        authorized_upload = _recording_upload_access(request)
        if (
            authorized_upload is not None
            and authorized_upload.expected_chunks is not None
            and chunk_index >= authorized_upload.expected_chunks
        ):
            return _err('Chunk index exceeds the authorized recording size.', code=403)

        chunk = request.FILES.get('chunk')
        if chunk is None:
            return _err('chunk is required.')
        if chunk.size <= 0:
            return _err('The recording chunk is empty.')
        if chunk.size > self.MAX_CHUNK_SIZE:
            return _err('Recording chunk is too large. Maximum chunk size is 15MB.')

        extension = str(request.data.get('extension', 'webm')).lower()
        if extension not in {'webm', 'mp4'}:
            return _err('Only WebM and MP4 recording chunks are supported.')
        mime_type = str(request.data.get('mime_type') or chunk.content_type or 'video/webm')[:100]

        from AI_Evaluator_Backend.azure_storage import (
            physical_video_blob_path,
            stage_physical_video_block,
        )

        with transaction.atomic():
            upload, _ = PhysicalRecordingUpload.objects.select_for_update().get_or_create(
                run=run,
            )
            if not upload.blob_path:
                upload.blob_path = physical_video_blob_path(
                    run.session.project_id,
                    run.session_id,
                    extension,
                )
                upload.mime_type = mime_type
                upload.save(update_fields=['blob_path', 'mime_type', 'updated_at'])

        try:
            stage_physical_video_block(chunk, upload.blob_path, chunk_index)
        except Exception as exc:
            logger.exception(
                'Physical recording chunk %s failed for session %s',
                chunk_index,
                session_id,
            )
            upload.status = PhysicalRecordingUpload.Status.FAILED
            upload.error_message = str(exc)[:1000]
            upload.save(update_fields=['status', 'error_message', 'updated_at'])
            return _err('Could not upload this recording chunk. It can be retried.', code=503)

        with transaction.atomic():
            upload = PhysicalRecordingUpload.objects.select_for_update().select_related('run').get(
                id=upload.id,
            )
            indices = set(upload.uploaded_chunk_indices or [])
            indices.add(chunk_index)
            upload.uploaded_chunk_indices = sorted(indices)
            upload.status = (
                PhysicalRecordingUpload.Status.UPLOADING
                if upload.finalization_requested
                else PhysicalRecordingUpload.Status.CAPTURING
            )
            upload.error_message = ''
            upload.save(update_fields=[
                'uploaded_chunk_indices', 'status', 'error_message', 'updated_at',
            ])
            if upload.run.status == PhysicalEvaluationRun.Status.RECORDING_FAILED:
                upload.run.status = PhysicalEvaluationRun.Status.RECORDING_UPLOADING
                upload.run.save(update_fields=['status', 'updated_at'])

        finalize_error = None
        if upload.finalization_requested:
            _, finalize_error = _attempt_recording_finalize(upload.id)
        upload.refresh_from_db()
        if finalize_error:
            return _err('The chunks were uploaded, but video finalization failed. Retry is available.', code=503)
        return _ok(
            'Recording chunk uploaded.',
            PhysicalRecordingUploadSerializer(upload).data,
            code=201,
        )


class KioskRecordingFinalizeView(APIView):
    """Release the kiosk immediately and finalize when all chunks arrive."""

    authentication_classes = [PhysicalRecordingAuthentication, PhysicalKioskAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        run = _run_for_recording_request(request, session_id)
        if run is None:
            return _err('Physical evaluation not found.', code=404)
        if run.session.status != EvaluationSession.Status.COMPLETED:
            return _err('The shared viva evaluator has not completed this session yet.', code=409)
        if run.recording_started_at is None:
            return _err('No protected recording was started for this session.', code=409)
        if run.status not in [
            PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS,
            PhysicalEvaluationRun.Status.RECORDING_UPLOADING,
            PhysicalEvaluationRun.Status.RECORDING_FAILED,
            PhysicalEvaluationRun.Status.COMPLETED,
        ]:
            return _err('The physical evaluation is not ready to finish.', code=409)

        try:
            expected_chunks = int(request.data.get('total_chunks'))
            duration_seconds = max(0, int(request.data.get('duration_seconds', 0)))
        except (TypeError, ValueError):
            return _err('total_chunks and duration_seconds must be integers.')
        if expected_chunks < 1 or expected_chunks > KioskRecordingChunkUploadView.MAX_CHUNKS:
            return _err('total_chunks is outside the allowed range.')

        authorized_upload = _recording_upload_access(request)
        if authorized_upload is not None and (
            authorized_upload.expected_chunks != expected_chunks
            or authorized_upload.duration_seconds != duration_seconds
        ):
            return _err('Recording metadata does not match the authorized upload.', code=403)

        extension = str(request.data.get('extension', 'webm')).lower()
        if extension not in {'webm', 'mp4'}:
            return _err('Only WebM and MP4 recordings are supported.')
        mime_type = str(request.data.get('mime_type') or 'video/webm')[:100]

        from AI_Evaluator_Backend.azure_storage import physical_video_blob_path

        with transaction.atomic():
            locked_run = PhysicalEvaluationRun.objects.select_for_update().get(id=run.id)
            upload, _ = PhysicalRecordingUpload.objects.select_for_update().get_or_create(
                run=locked_run,
            )
            if not upload.blob_path:
                upload.blob_path = physical_video_blob_path(
                    locked_run.session.project_id,
                    locked_run.session_id,
                    extension,
                )
            upload.mime_type = mime_type
            upload.expected_chunks = expected_chunks
            upload.duration_seconds = duration_seconds
            upload.finalization_requested = True
            if upload.status != PhysicalRecordingUpload.Status.READY:
                upload.status = PhysicalRecordingUpload.Status.UPLOADING
            upload.error_message = ''
            upload.save(update_fields=[
                'blob_path', 'mime_type', 'expected_chunks', 'duration_seconds',
                'finalization_requested', 'status', 'error_message', 'updated_at',
            ])

            if locked_run.status != PhysicalEvaluationRun.Status.COMPLETED:
                locked_run.status = PhysicalEvaluationRun.Status.RECORDING_UPLOADING
                locked_run.completed_at = locked_run.completed_at or timezone.now()
                locked_run.save(update_fields=['status', 'completed_at', 'updated_at'])

        if request.data.get('defer_commit') is True:
            upload.refresh_from_db()
            data = PhysicalRecordingUploadSerializer(upload).data
            data['upload_token'] = make_recording_upload_token(upload)
            return _ok(
                'Evaluation completed. Recording continues uploading in the background.',
                data,
                code=202,
            )

        finalized, finalize_error = _attempt_recording_finalize(upload.id)
        upload.refresh_from_db()
        if finalize_error:
            return _err('Evaluation finished, but recording finalization must be retried.', code=503)

        data = PhysicalRecordingUploadSerializer(upload).data
        data['upload_token'] = make_recording_upload_token(upload)
        return _ok(
            'Evaluation completed. Recording is ready.' if finalized
            else 'Evaluation completed. Recording continues uploading in the background.',
            data,
            code=200 if finalized else 202,
        )


class KioskRecordingStatusView(APIView):
    authentication_classes = [PhysicalRecordingAuthentication, PhysicalKioskAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        run = _run_for_recording_request(request, session_id)
        if run is None:
            return _err('Physical evaluation not found.', code=404)
        upload = PhysicalRecordingUpload.objects.filter(run=run).first()
        if upload is None:
            return _err('Recording upload not found.', code=404)
        return _ok('Recording upload status retrieved.', PhysicalRecordingUploadSerializer(upload).data)


class KioskSessionFinishView(APIView):
    """Finish a physical run without creating or uploading a video recording."""

    authentication_classes = [PhysicalKioskAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        access = _kiosk_access(request)

        with transaction.atomic():
            # Lock only concrete rows. EvaluationSession.student/group are
            # nullable, so joining them in this FOR UPDATE query is rejected
            # by PostgreSQL ("FOR UPDATE cannot be applied to the nullable
            # side of an outer join"). Related display data is loaded after
            # the transaction for serialization.
            run = PhysicalEvaluationRun.objects.select_for_update().filter(
                kiosk_access=access,
                session_id=session_id,
            ).first()
            if run is None:
                return _err('Physical evaluation not found.', code=404)
            if run.status == PhysicalEvaluationRun.Status.COMPLETED:
                completed_run = _run_for_access(access, session_id)
                return _ok(
                    'Physical evaluation was already completed.',
                    PhysicalRunSerializer(completed_run).data,
                )
            session = EvaluationSession.objects.select_for_update().get(
                id=run.session_id,
            )
            if session.status != EvaluationSession.Status.COMPLETED:
                return _err(
                    'The shared viva evaluator has not completed this session yet.',
                    code=409,
                )
            if run.status not in [
                PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS,
                PhysicalEvaluationRun.Status.RECORDING_UPLOADING,
                PhysicalEvaluationRun.Status.RECORDING_FAILED,
            ]:
                return _err('The physical evaluation is not ready to finish.', code=409)

            # A run started by the retired browser-upload flow may still have a
            # tracking row. Removing metadata here prevents a stale upload from
            # appearing as pending; no video or answer data is deleted.
            PhysicalRecordingUpload.objects.filter(run=run).delete()
            run.status = PhysicalEvaluationRun.Status.COMPLETED
            run.completed_at = run.completed_at or timezone.now()
            run.save(update_fields=['status', 'completed_at', 'updated_at'])

        run = _run_for_access(access, session_id)
        return _ok(
            'Physical evaluation completed without a video recording.',
            PhysicalRunSerializer(run).data,
        )


class KioskSessionCompleteView(APIView):
    """Upload the full local recording after the shared evaluator completes."""

    authentication_classes = [PhysicalKioskAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    MAX_VIDEO_SIZE = 500 * 1024 * 1024
    MAX_AUDIO_SIZE = 100 * 1024 * 1024
    ALLOWED_VIDEO_TYPES = ('.mp4', '.webm')
    ALLOWED_AUDIO_TYPES = ('.mp3', '.wav', '.webm')

    def post(self, request, session_id):
        access = _kiosk_access(request)
        run = _run_for_access(access, session_id)
        if run is None:
            return _err('Physical evaluation not found.', code=404)
        if run.status == PhysicalEvaluationRun.Status.COMPLETED:
            return _ok('Physical evaluation was already completed.', PhysicalRunSerializer(run).data)
        if run.status != PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS:
            return _err('The physical evaluation is not in the viva phase.', code=409)
        if run.session.status != EvaluationSession.Status.COMPLETED:
            return _err('The shared viva evaluator has not completed this session yet.', code=409)
        if run.recording_started_at is None:
            return _err('No protected recording was started for this session.', code=409)

        video_file = request.FILES.get('video_file')
        audio_file = request.FILES.get('audio_file')
        if video_file is None:
            return _err('video_file is required for a physical evaluation recording.')
        if not video_file.name.lower().endswith(self.ALLOWED_VIDEO_TYPES):
            return _err('Only .mp4 and .webm video files are allowed.')
        if video_file.size > self.MAX_VIDEO_SIZE:
            return _err('File too large. Maximum video size is 500MB.')
        if audio_file:
            if not audio_file.name.lower().endswith(self.ALLOWED_AUDIO_TYPES):
                return _err('Only .mp3, .wav, and .webm audio files are allowed.')
            if audio_file.size > self.MAX_AUDIO_SIZE:
                return _err('File too large. Maximum audio size is 100MB.')

        from AI_Evaluator_Backend.azure_storage import (
            upload_audio_to_blob,
            upload_video_to_blob,
        )

        video_url = upload_video_to_blob(
            video_file, str(run.session.project_id), str(run.session_id),
        )
        audio_url = None
        if audio_file:
            audio_url = upload_audio_to_blob(
                audio_file, str(run.session.project_id), str(run.session_id),
            )

        close_old_connections()
        now = timezone.now()
        duration_seconds = max(0, int((now - run.recording_started_at).total_seconds()))
        with transaction.atomic():
            recording = SessionRecording.objects.create(
                session=run.session,
                video_file_url=video_url,
                audio_file_url=audio_url,
                duration_seconds=duration_seconds,
                recording_started_at=run.recording_started_at,
            )
            run.recording = recording
            run.status = PhysicalEvaluationRun.Status.COMPLETED
            run.completed_at = now
            run.save(update_fields=['recording', 'status', 'completed_at', 'updated_at'])

        try:
            from cv_analysis.services.runner import enqueue_cv_analysis

            enqueue_cv_analysis(run.session_id)
        except Exception:
            logger.exception('Could not enqueue CV analysis for physical session %s', run.session_id)

        return _ok('Physical evaluation and recording completed.', {
            'run': PhysicalRunSerializer(run).data,
            'recording_id': str(recording.id),
            'video_file_url': video_url,
            'audio_file_url': audio_url,
            'duration_seconds': duration_seconds,
        })
