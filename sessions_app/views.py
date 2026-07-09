"""
Views for Session Panel, Demo Flow, Viva Questions,
End-Viva media upload, and Student Session Status.
"""

from datetime import date

from django.db import transaction
from django.utils import timezone
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import (
    EvaluationSession, ExaminerProfile, GroupMember, Project,
    ProjectExaminer, SessionRecording, StudentProfile, VivaQuestion,
)
from projects.permissions import IsExaminer, IsStudent
from sessions_app.serializers import (
    EvaluationSessionDetailSerializer, SessionPanelEntrySerializer,
    StudentSessionStatusSerializer, VivaQuestionCreateSerializer,
    VivaQuestionSerializer, VivaQuestionUpdateSerializer,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _err(msg, errors=None, code=400):
    return Response(
        {'success': False, 'message': msg, 'errors': errors or {}},
        status=code,
    )

def _ok(msg, data=None, code=200):
    return Response(
        {'success': True, 'message': msg, 'data': data},
        status=code,
    )

def _500(e):
    return Response(
        {'success': False, 'message': f'An unexpected error occurred: {str(e)}', 'errors': {}},
        status=500,
    )

def _get_examiner_profile(user):
    try:
        return user.examiner_profile
    except ExaminerProfile.DoesNotExist:
        return None

def _is_assigned(examiner_profile, project):
    return ProjectExaminer.objects.filter(
        project=project, examiner=examiner_profile,
    ).exists()


def _start_recording_and_stt(session):
    """Kick off the Agora STT bot + cloud recording (non-blocking, optional).

    Called when a session first becomes active (start-demo / start-viva) so the
    whole session — demo included — is captured server-side.
    """
    import threading
    from agora_service.stt_manager import start_stt, is_enabled as stt_enabled
    if stt_enabled():
        threading.Thread(target=start_stt, args=(session,), daemon=True).start()

    from agora_service.cloud_recording import (
        start_recording, is_enabled as rec_enabled,
    )
    if rec_enabled():
        threading.Thread(target=start_recording, args=(session,), daemon=True).start()


def _activate_session(session, skip_demo):
    """Flip a scheduled session to in_progress and start capture.

    ``skip_demo=True`` also stamps ``demo_completed_at`` so the session lands
    straight in the viva phase (used when the session has no demo). Group
    sessions transition every sibling row together and share one Agora channel.
    """
    now = timezone.now()
    updates = {
        'status': 'in_progress',
        'actual_start': now,
        'agora_channel_name': str(session.id),
    }
    if skip_demo:
        updates['demo_completed_at'] = now

    with transaction.atomic():
        if session.group:
            EvaluationSession.objects.filter(
                project=session.project, group=session.group,
            ).update(**updates)
        else:
            for key, value in updates.items():
                setattr(session, key, value)
            session.save()

    session.refresh_from_db()
    _start_recording_and_stt(session)
    return session


def _student_session_or_error(request, session_id):
    """Resolve (session, student_profile) ensuring the caller participates.

    Returns (session, sp, None) on success or (None, None, error_response).
    """
    try:
        sp = request.user.student_profile
    except StudentProfile.DoesNotExist:
        return None, None, _err('Student profile not found.', code=404)

    session = EvaluationSession.objects.filter(
        id=session_id,
    ).select_related('project', 'group').first()
    if not session:
        return None, None, _err('Session not found.', code=404)

    is_participant = (
        (session.student_id == sp.id) or
        (session.group_id and GroupMember.objects.filter(
            group=session.group, student=sp,
        ).exists())
    )
    if not is_participant:
        return None, None, _err('You are not a participant of this session.', code=403)

    return session, sp, None


# =============================================================================
# PART 3 — SESSION PANEL (EXAMINER)
# =============================================================================

class SessionPanelOpenView(APIView):
    """POST /api/projects/<project_id>/session-panel/open/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, project):
                return _err('You are not assigned to this project.', code=403)

            today = date.today()
            sessions = EvaluationSession.objects.filter(
                project=project,
                scheduled_start__date=today,
            ).select_related('student__user', 'group').order_by('scheduled_start')

            # Deduplicate for group projects (show one entry per group)
            if project.is_group_project:
                seen_groups = set()
                unique_sessions = []
                for s in sessions:
                    key = str(s.group_id) if s.group_id else str(s.id)
                    if key not in seen_groups:
                        seen_groups.add(key)
                        unique_sessions.append(s)
                sessions_list = unique_sessions
            else:
                sessions_list = list(sessions)

            data = SessionPanelEntrySerializer(sessions_list, many=True).data

            total = len(sessions_list)
            completed = sum(1 for s in sessions_list if s.status == 'completed')
            in_progress = sum(1 for s in sessions_list if s.status == 'in_progress')
            remaining = total - completed - in_progress

            return _ok('Session panel opened.', {
                'sessions': data,
                'panel_summary': {
                    'total_sessions': total,
                    'completed': completed,
                    'in_progress': in_progress,
                    'remaining': remaining,
                },
            })
        except Exception as e:
            return _500(e)


class ActiveSessionView(APIView):
    """GET /api/projects/<project_id>/session-panel/active/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def get(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, project):
                return _err('You are not assigned to this project.', code=403)

            session = EvaluationSession.objects.filter(
                project=project, status='in_progress',
            ).select_related('student__user', 'group').first()

            if not session:
                return _ok('No session currently in progress.', None)

            return _ok('Active session retrieved.', EvaluationSessionDetailSerializer(session).data)
        except Exception as e:
            return _500(e)


# =============================================================================
# PART 4 — DEMO SESSION FLOW
# =============================================================================

class StartDemoView(APIView):
    """POST /api/sessions/<session_id>/start-demo/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, session_id):
        try:
            session = EvaluationSession.objects.filter(
                id=session_id,
            ).select_related('project', 'student__user', 'group').first()
            if not session:
                return _err('Session not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, session.project):
                return _err('You are not assigned to this project.', code=403)

            if session.status == 'in_progress':
                return _err('This session is already in progress.')
            if session.status == 'completed':
                return _err('This session has already been completed.')

            # Check no other session is in progress for this project
            if EvaluationSession.objects.filter(
                project=session.project, status='in_progress',
            ).exists():
                return _err(
                    'Another session is currently in progress. '
                    'Please complete it first.'
                )

            now = timezone.now()
            with transaction.atomic():
                if session.group:
                    EvaluationSession.objects.filter(
                        project=session.project, group=session.group,
                    ).update(
                        status='in_progress',
                        actual_start=now,
                        agora_channel_name=str(session.id),
                    )
                else:
                    session.status = 'in_progress'
                    session.actual_start = now
                    session.agora_channel_name = str(session.id)
                    session.save()

            session.refresh_from_db()

            # Start Agora STT bot + cloud recording (non-blocking, optional)
            import threading
            from agora_service.stt_manager import start_stt, is_enabled as stt_enabled
            if stt_enabled():
                threading.Thread(
                    target=start_stt, args=(session,), daemon=True,
                ).start()

            from agora_service.cloud_recording import (
                start_recording, is_enabled as rec_enabled,
            )
            if rec_enabled():
                threading.Thread(
                    target=start_recording, args=(session,), daemon=True,
                ).start()

            return _ok('Demo started successfully.', EvaluationSessionDetailSerializer(session).data)
        except Exception as e:
            return _500(e)


class CompleteDemoView(APIView):
    """POST /api/sessions/<session_id>/complete-demo/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, session_id):
        try:
            session = EvaluationSession.objects.filter(
                id=session_id,
            ).select_related('project', 'group').first()
            if not session:
                return _err('Session not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, session.project):
                return _err('You are not assigned to this project.', code=403)

            if session.status != 'in_progress':
                return _err('No demo is currently in progress for this session.')

            now = timezone.now()
            with transaction.atomic():
                if session.group:
                    EvaluationSession.objects.filter(
                        project=session.project, group=session.group,
                    ).update(demo_completed_at=now)
                else:
                    session.demo_completed_at = now
                    session.save()

            session.refresh_from_db()
            return _ok('Demo completed. Viva session will begin now.', EvaluationSessionDetailSerializer(session).data)
        except Exception as e:
            return _500(e)


class StudentEndDemoView(APIView):
    """POST /api/sessions/<session_id>/end-demo/

    Lets the presenting student end the demo phase from their own screen (the
    "End Demo & Start Viva" button). Sets demo_completed_at for the session (and
    for all sibling group sessions), which every participant's UI polls to move
    on to the AI viva. The examiner's Complete Demo remains available too.
    """
    permission_classes = [IsAuthenticated, IsStudent]

    def post(self, request, session_id):
        try:
            session, sp, error = _student_session_or_error(request, session_id)
            if error:
                return error

            if session.phase != 'demo_in_progress':
                return _err('No demo is currently in progress for this session.')

            now = timezone.now()
            with transaction.atomic():
                if session.group:
                    EvaluationSession.objects.filter(
                        project=session.project, group=session.group,
                    ).update(demo_completed_at=now)
                else:
                    session.demo_completed_at = now
                    session.save(update_fields=['demo_completed_at'])

            session.refresh_from_db()
            return _ok('Demo ended. Your viva will begin now.', EvaluationSessionDetailSerializer(session).data)
        except Exception as e:
            return _500(e)


class StudentStartDemoView(APIView):
    """POST /api/sessions/<session_id>/student/start-demo/

    A participating student starts the demo/presentation phase from the live
    viva room ("Start Demo" button). Moves the session scheduled → in_progress
    (demo phase) and starts the server-side recording. Only valid when the
    examiner enabled a demo for this session.
    """
    permission_classes = [IsAuthenticated, IsStudent]

    def post(self, request, session_id):
        try:
            session, sp, error = _student_session_or_error(request, session_id)
            if error:
                return error

            if not session.demo_enabled:
                return _err('This session has no demo phase. Use Start Viva instead.')
            if session.phase != 'scheduled':
                return _err('This session has already started.')

            session = _activate_session(session, skip_demo=False)
            return _ok('Demo started. You can present your work now.', EvaluationSessionDetailSerializer(session).data)
        except Exception as e:
            return _500(e)


class StudentStartVivaView(APIView):
    """POST /api/sessions/<session_id>/student/start-viva/

    A participating student starts the viva directly (no demo phase) from the
    live viva room ("Start Viva" button). Moves the session scheduled →
    in_progress (viva phase) and starts the server-side recording.
    """
    permission_classes = [IsAuthenticated, IsStudent]

    def post(self, request, session_id):
        try:
            session, sp, error = _student_session_or_error(request, session_id)
            if error:
                return error

            if session.phase != 'scheduled':
                return _err('This session has already started.')

            session = _activate_session(session, skip_demo=True)
            return _ok('Viva started.', EvaluationSessionDetailSerializer(session).data)
        except Exception as e:
            return _500(e)


class EndVivaView(APIView):
    """POST /api/sessions/<session_id>/end-viva/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB
    MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100 MB
    ALLOWED_VIDEO_TYPES = ('.mp4', '.webm')
    ALLOWED_AUDIO_TYPES = ('.mp3', '.wav', '.webm')

    def post(self, request, session_id):
        try:
            session = EvaluationSession.objects.filter(
                id=session_id,
            ).select_related('project', 'group').first()
            if not session:
                return _err('Session not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, session.project):
                return _err('You are not assigned to this project.', code=403)

            if session.status != 'in_progress':
                return _err('No viva is currently in progress for this session.')

            video_file = request.FILES.get('video_file')
            audio_file = request.FILES.get('audio_file')

            video_blob_url = None
            audio_blob_url = None

            # Validate and upload video
            if video_file:
                if not video_file.name.lower().endswith(self.ALLOWED_VIDEO_TYPES):
                    return _err('Only .mp4 and .webm video files are allowed.')
                if video_file.size > self.MAX_VIDEO_SIZE:
                    return _err('File too large. Maximum video size is 500MB.')

                from AI_Evaluator_Backend.azure_storage import upload_video_to_blob
                video_blob_url = upload_video_to_blob(
                    video_file, str(session.project_id), str(session.id),
                )

            # Validate and upload audio
            if audio_file:
                if not audio_file.name.lower().endswith(self.ALLOWED_AUDIO_TYPES):
                    return _err('Only .mp3, .wav, and .webm audio files are allowed.')
                if audio_file.size > self.MAX_AUDIO_SIZE:
                    return _err('File too large. Maximum audio size is 100MB.')

                from AI_Evaluator_Backend.azure_storage import upload_audio_to_blob
                audio_blob_url = upload_audio_to_blob(
                    audio_file, str(session.project_id), str(session.id),
                )

            # Calculate duration
            duration_seconds = None
            if session.actual_start:
                duration_seconds = int((timezone.now() - session.actual_start).total_seconds())

            # Stop Agora STT bot if running (non-blocking)
            from agora_service.stt_manager import stop_stt, is_enabled as stt_enabled
            if stt_enabled() and session.agora_stt_task_id:
                import threading
                threading.Thread(
                    target=stop_stt, args=(session,), daemon=True,
                ).start()

            # Stop Agora Cloud Recording (synchronous — we need the resulting
            # Azure blob URL now). This is the authoritative full-channel
            # recording; prefer it over any client upload when present.
            from agora_service.cloud_recording import (
                stop_recording, is_enabled as rec_enabled,
            )
            if rec_enabled() and session.agora_recording_sid:
                cloud_url = stop_recording(session)
                if cloud_url:
                    video_blob_url = cloud_url

            now = timezone.now()
            with transaction.atomic():
                # Create session recording
                recording = SessionRecording.objects.create(
                    session=session,
                    video_file_url=video_blob_url,
                    audio_file_url=audio_blob_url,
                    duration_seconds=duration_seconds,
                )

                # Update session status
                if session.group:
                    EvaluationSession.objects.filter(
                        project=session.project, group=session.group,
                    ).update(status='completed')
                else:
                    session.status = 'completed'
                    session.save()

            # Queue post-hoc CV/behavioral analysis of the recording
            # (no-op when CV_ANALYSIS_ENABLED is off).
            if video_blob_url:
                from cv_analysis.services.runner import enqueue_cv_analysis
                enqueue_cv_analysis(session.id)

            # Generate SAS URLs for response
            video_sas = None
            audio_sas = None
            if video_blob_url:
                try:
                    from AI_Evaluator_Backend.azure_storage import generate_sas_url, AZURE_CONTAINER_VIDEOS
                    blob_path = f"{session.project_id}/{session.id}/{video_file.name}"
                    video_sas = generate_sas_url(AZURE_CONTAINER_VIDEOS, blob_path)
                except Exception:
                    video_sas = video_blob_url

            if audio_blob_url:
                try:
                    from AI_Evaluator_Backend.azure_storage import generate_sas_url, AZURE_CONTAINER_AUDIOS
                    blob_path = f"{session.project_id}/{session.id}/{audio_file.name}"
                    audio_sas = generate_sas_url(AZURE_CONTAINER_AUDIOS, blob_path)
                except Exception:
                    audio_sas = audio_blob_url

            return _ok('Viva session completed successfully.', {
                'session_id': str(session.id),
                'video_url': video_sas,
                'audio_url': audio_sas,
                'duration_seconds': duration_seconds,
                'status': 'completed',
            })
        except Exception as e:
            return _500(e)


# =============================================================================
# PART 4.3 — EXAMINER VIVA QUESTIONS
# =============================================================================

class ExaminerVivaQuestionCreateView(APIView):
    """POST /api/projects/<project_id>/viva/questions/create/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, project):
                return _err('You are not assigned to this project.', code=403)

            ser = VivaQuestionCreateSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            question = VivaQuestion.objects.create(
                project=project,
                session=None,
                question_text=ser.validated_data['question_text'],
                blooms_level=ser.validated_data.get('blooms_level'),
                question_order=ser.validated_data['question_order'],
                question_source='examiner',
            )

            return _ok('Viva question created.', VivaQuestionSerializer(question).data, 201)
        except Exception as e:
            return _500(e)


class ExaminerVivaQuestionListView(APIView):
    """GET /api/projects/<project_id>/viva/questions/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def get(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, project):
                return _err('You are not assigned to this project.', code=403)

            questions = VivaQuestion.objects.filter(
                project=project, question_source='examiner',
            ).order_by('question_order')

            return _ok('Viva questions retrieved.', VivaQuestionSerializer(questions, many=True).data)
        except Exception as e:
            return _500(e)


class ExaminerVivaQuestionUpdateView(APIView):
    """PUT /api/projects/<project_id>/viva/questions/<question_id>/update/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def put(self, request, project_id, question_id):
        try:
            question = VivaQuestion.objects.filter(
                id=question_id, project_id=project_id, question_source='examiner',
            ).first()
            if not question:
                return _err('Examiner question not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, question.project):
                return _err('You are not assigned to this project.', code=403)

            ser = VivaQuestionUpdateSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            for field in ('question_text', 'blooms_level', 'question_order'):
                if field in ser.validated_data:
                    setattr(question, field, ser.validated_data[field])
            question.save()

            return _ok('Viva question updated.', VivaQuestionSerializer(question).data)
        except Exception as e:
            return _500(e)


class ExaminerVivaQuestionDeleteView(APIView):
    """DELETE /api/projects/<project_id>/viva/questions/<question_id>/delete/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def delete(self, request, project_id, question_id):
        try:
            question = VivaQuestion.objects.filter(
                id=question_id, project_id=project_id,
            ).first()
            if not question:
                return _err('Question not found.', code=404)

            if question.question_source != 'examiner':
                return _err('Only examiner-created questions can be deleted.')

            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, question.project):
                return _err('You are not assigned to this project.', code=403)

            question.delete()
            return _ok('Viva question deleted.')
        except Exception as e:
            return _500(e)


# =============================================================================
# PART 5 — STUDENT SESSION STATUS
# =============================================================================

class StudentSessionStatusView(APIView):
    """GET /api/sessions/my-status/?status=upcoming|ongoing|completed"""
    permission_classes = [IsAuthenticated, IsStudent]

    def get(self, request):
        try:
            try:
                sp = request.user.student_profile
            except StudentProfile.DoesNotExist:
                return _err('Student profile not found.', code=404)

            from django.db.models import Q
            sessions_qs = EvaluationSession.objects.filter(
                Q(student=sp) | Q(group__members__student=sp)
            ).distinct().select_related('project', 'group').order_by('scheduled_start')

            sessions = list(sessions_qs)
            # NOTE: status is now driven ONLY by explicit actions (the student's
            # Start Demo / Start Viva buttons and End Viva), never by the clock.
            # We do not auto-advance sessions here — a scheduled session stays
            # scheduled until the student starts it, and an in-progress viva is
            # never silently marked completed just because its slot elapsed.

            status_filter = request.query_params.get('status')
            status_map = {
                'upcoming': 'scheduled',
                'scheduled': 'scheduled',
                'ongoing': 'in_progress',
                'in_progress': 'in_progress',
                'completed': 'completed',
            }

            if status_filter:
                mapped_status = status_map.get(status_filter)
                if not mapped_status:
                    return _err(
                        'Invalid status filter. Use upcoming, ongoing, or completed.',
                        code=400,
                    )
                sessions = [s for s in sessions if s.status == mapped_status]

            data = StudentSessionStatusSerializer(sessions, many=True).data
            return _ok('Session status retrieved.', data)
        except Exception as e:
            return _500(e)
