import logging
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import FormParser, MultiPartParser
from django.http import HttpResponse
from django.utils import timezone

logger = logging.getLogger(__name__)

from viva_evaluator.models import SubmissionIndexStatus

from viva_evaluator.views._helpers import (
    _resolve_session_submission,
    _get_or_create_index_status,
)
from authentication.authentication import CookieJWTAuthentication
from physical_evaluation.authentication import PhysicalKioskAuthentication
from physical_evaluation.models import PhysicalEvaluationRun, PhysicalKioskAccess
from viva_evaluator.permissions import (
    EXAMINER,
    CanParticipateInVivaSession,
    IsAssignedProjectExaminer,
    IsAssignedSessionExaminer,
    VivaSessionPermission,
    session_roles_for_user,
)
from viva_evaluator.services.pipeline.exceptions import (
    QuestionGenerationUnavailableError,
)


class SessionStartView(APIView):
    """
    POST /api/viva/sessions/start/

    Starts a viva session for a submission.
    Generates and returns the first question for the first rubric criterion.

    Request body:
    {
        "session_id": "uuid-of-evaluation-session"
    }
    """
    authentication_classes = [PhysicalKioskAuthentication, CookieJWTAuthentication]
    permission_classes = [IsAuthenticated, VivaSessionPermission]

    def post(self, request):
        session_id = request.data.get('session_id')

        if not session_id:
            return Response(
                {"error": "session_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            from core.models import EvaluationSession
            from viva_evaluator.services.pipeline.orchestrator import (
                VivaPipeline,
                VivaPipelineInputError,
            )

            session = EvaluationSession.objects.get(id=session_id)
            if isinstance(request.auth, PhysicalKioskAccess) and session.group_id:
                physical_run = PhysicalEvaluationRun.objects.filter(
                    session=session,
                    kiosk_access=request.auth,
                ).first()
                if physical_run is None or not physical_run.identity_authorized:
                    return Response(
                        {
                            "error": "Verify at least one present group member with no unknown faces, or use an examiner PIN override before starting the viva.",
                            "code": "identity_review_required",
                        },
                        status=status.HTTP_409_CONFLICT,
                    )
            submission = _resolve_session_submission(session)

            if not submission:
                return Response(
                    {"error": "No processed submission found for this session."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Make sure submission is ready
            try:
                index_status = _get_or_create_index_status(submission)
                if index_status.status != SubmissionIndexStatus.IndexStatus.READY:
                    return Response(
                        {"error": "Submission is not ready yet. Please wait for processing to complete."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
            except Exception:
                return Response(
                    {"error": "No processed submission found for this session."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if session.status == EvaluationSession.Status.COMPLETED:
                return Response(
                    {"error": "This session is already complete."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                payload = VivaPipeline().start_session(
                    session=session,
                    submission=submission,
                )
            except VivaPipelineInputError as exc:
                return Response(
                    {"error": str(exc)},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            return Response(payload, status=status.HTTP_200_OK)

        except EvaluationSession.DoesNotExist:
            return Response(
                {"error": "Session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        except QuestionGenerationUnavailableError as exc:
            return Response(
                {"error": str(exc), "code": exc.code},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as e:
            from viva_evaluator.services.llm_service import LLMQuotaError
            if isinstance(e, LLMQuotaError):
                return Response(
                    {
                        "error": "The AI service is busy right now (quota limit reached). "
                                 "Please try again in a moment.",
                        "code": "ai_quota_exceeded",
                        "retry_after_seconds": getattr(e, 'retry_after_seconds', None),
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AnswerSubmitView(APIView):
    """
    POST /api/viva/sessions/<session_id>/answer/

    Student submits their answer to the current question.
    The answer is evaluated, saved, and the next question is generated.

    Request body:
    {
        "question_id": "uuid",
        "answer_text": "student's answer here"
    }
    """
    authentication_classes = [PhysicalKioskAuthentication, CookieJWTAuthentication]
    permission_classes = [IsAuthenticated, CanParticipateInVivaSession]

    def post(self, request, session_id):
        claim = None
        submitted_at = timezone.now()
        question_id = request.data.get('question_id')
        answer_text = request.data.get('answer_text', '').strip()
        speech_metrics = request.data.get('speech_metrics')   # Week 6: optional
        # Distinguish a speaker the CALLER named from one the attribution engine
        # infers further down. Only the former is an authorisation question;
        # conflating them let an inferred name block the person who was actually
        # typing from submitting their own answer.
        named_speaker_id = request.data.get('speaker_id')
        speaker_id = named_speaker_id or 'group'

        if not question_id or not answer_text:
            return Response(
                {"error": "question_id and answer_text are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if (
            speaker_id != 'group'
            and not isinstance(request.auth, PhysicalKioskAccess)
        ):
            return Response(
                {"error": "Only the physical kiosk may select a speaker explicitly."},
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            from core.models import (
                EvaluationSession, VivaQuestion,
                GroupMember, StudentProfile,
            )
            from viva_evaluator.services.pipeline.orchestrator import VivaPipeline

            session = EvaluationSession.objects.get(id=session_id)
            question = VivaQuestion.objects.get(id=question_id, session=session)

            submission = _resolve_session_submission(session)
            student_profile = None
            if speaker_id != 'group':
                try:
                    student_profile = StudentProfile.objects.get(id=speaker_id)
                except (StudentProfile.DoesNotExist, ValueError):
                    return Response(
                        {"error": "Invalid speaker_id."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                belongs_to_session = (
                    session.student_id == student_profile.id
                    or (
                        session.group_id
                        and GroupMember.objects.filter(
                            group_id=session.group_id,
                            student=student_profile,
                        ).exists()
                    )
                )
                if not belongs_to_session:
                    return Response(
                        {"error": "speaker_id is not a participant of this session."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                from attribution.services.engine import resolve_speaker_id

                # An explicit kiosk selection is a manual decision, so persist
                # it as evidence before later reconciliation runs.
                resolve_speaker_id(
                    session,
                    question,
                    speaker_id,
                    answered_at=submitted_at,
                )
            elif session.student:
                # Individual session, use the session's student
                student_profile = session.student
            else:
                # Group session and the client named nobody. Ask the speaker
                # attribution engine who was talking during this answer's
                # window (Agora per-UID audio, live CV lip motion, examiner
                # override). It answers 'group' unless it is confident, so an
                # ambiguous window still scores to the group rather than being
                # guessed onto a student.
                from attribution.services.engine import resolve_speaker_id

                resolved = resolve_speaker_id(
                    session,
                    question,
                    speaker_id,
                    answered_at=submitted_at,
                )
                if resolved != 'group':
                    student_profile = StudentProfile.objects.filter(
                        id=resolved,
                    ).first()
                    if student_profile is not None:
                        speaker_id = resolved

            caller_student = None
            if not isinstance(request.auth, PhysicalKioskAccess):
                caller_student = getattr(request.user, 'student_profile', None)
                # Guard the caller's own CLAIM, never the engine's inference.
                # In a remote group viva the attribution engine names whichever
                # member it heard speaking, which is frequently not the member
                # holding the keyboard — checking that name here rejected a
                # student for submitting their own answer.
                claimed_another_participant = (
                    named_speaker_id not in (None, '', 'group')
                    and (
                        caller_student is None
                        or student_profile is None
                        or caller_student.id != student_profile.id
                    )
                )
                if claimed_another_participant:
                    return Response(
                        {"error": "You cannot submit an answer for another participant."},
                        status=status.HTTP_403_FORBIDDEN,
                    )

            if not submission:
                return Response(
                    {"error": "No submission found for this session."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            from viva_evaluator.services.answer_idempotency import (
                IdempotencyConflict, acquire_claim, complete_claim,
                has_client_key, request_fingerprint, resolve_idempotency_key,
                speaker_key,
            )
            logical_speaker = (
                f"student:{caller_student.id}"
                if caller_student is not None
                else (
                    f"student:{student_profile.id}"
                    if request.data.get('speaker_id', 'group') != 'group'
                    and student_profile is not None
                    else speaker_key('group')
                )
            )
            try:
                idempotency_key = resolve_idempotency_key(
                    request, question_id=question.id, speaker_id=logical_speaker,
                )
            except ValueError as exc:
                return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
            try:
                claim_result = acquire_claim(
                    session=session,
                    question=question,
                    speaker=logical_speaker,
                    idempotency_key=idempotency_key,
                    request_hash=request_fingerprint(
                        answer_text=answer_text,
                        speech_metrics=speech_metrics,
                        speaker_id=logical_speaker,
                    ),
                    # Only a key the caller chose carries a promise about the
                    # payload. A key we derived from question and speaker must
                    # not turn a re-submitted answer into a dead end.
                    strict_hash=has_client_key(request),
                )
            except IdempotencyConflict as exc:
                return Response(
                    {"error": str(exc), "code": "idempotency_conflict"},
                    status=status.HTTP_409_CONFLICT,
                )
            claim = claim_result.claim
            if claim_result.action == "replay":
                response = Response(claim.response_payload, status=claim.response_status)
                response["Idempotency-Replayed"] = "true"
                return response
            if claim_result.action == "in_progress":
                response = Response(
                    {"error": "This answer is already being processed.", "code": "answer_processing"},
                    status=status.HTTP_409_CONFLICT,
                )
                response["Retry-After"] = "1"
                return response

            payload = VivaPipeline().submit_answer(
                session=session,
                submission=submission,
                question=question,
                answer_text=answer_text,
                speech_metrics=speech_metrics,
                speaker_id=speaker_id,
                student_profile=student_profile,
                submitter_student_profile=caller_student,
                answered_at=submitted_at,
                deduplication_key=logical_speaker,
            )
            from sessions_app.services.recording_finalizer import (
                finalize_completed_online_session,
            )
            payload = finalize_completed_online_session(session, payload)
            complete_claim(claim, payload, status.HTTP_200_OK)
            return Response(payload, status=status.HTTP_200_OK)

        except EvaluationSession.DoesNotExist:
            return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
        except VivaQuestion.DoesNotExist:
            return Response({"error": "Question not found."}, status=status.HTTP_404_NOT_FOUND)
        except QuestionGenerationUnavailableError as exc:
            return Response(
                {"error": str(exc), "code": exc.code},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as e:
            if claim is not None:
                from viva_evaluator.services.answer_idempotency import fail_claim
                fail_claim(claim)
            from viva_evaluator.services.llm_service import LLMQuotaError
            if isinstance(e, LLMQuotaError):
                return Response(
                    {
                        "error": "The AI service is busy right now (quota limit reached). "
                                 "Please try again in a moment.",
                        "code": "ai_quota_exceeded",
                        "retry_after_seconds": getattr(e, 'retry_after_seconds', None),
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SessionReportView(APIView):
    """
    GET /api/viva/sessions/<session_id>/report/

    Returns the structured post-viva report. Allowed even before the
    session is COMPLETED so examiners can review mid-session if needed —
    the report just reflects whatever turns have happened so far.
    """
    permission_classes = [IsAuthenticated, VivaSessionPermission]

    def get(self, request, session_id):
        try:
            from core.models import EvaluationSession, SessionSummaryReport
            from viva_evaluator.services.reporting import generate_post_viva_report

            session = EvaluationSession.objects.select_related(
                'student', 'group'
            ).get(id=session_id)

            roles = session_roles_for_user(request.user, session)
            is_examiner = EXAMINER in roles
            participant_student = None
            if not is_examiner:
                participant_student = request.user.student_profile
                summary = session.summary_reports.filter(
                    student=participant_student,
                    scores_status=SessionSummaryReport.ScoresStatus.APPROVED,
                    is_published=True,
                ).first()
                if summary is None:
                    return Response(
                        {
                            "message": "Final results are waiting for examiner approval.",
                            "scores_status": "draft",
                            "session_status": session.status,
                        },
                        status=status.HTTP_202_ACCEPTED,
                    )

                from viva_evaluator.services.session_reports import (
                    build_published_student_report,
                )

                student_key = str(participant_student.id)
                student_report = build_published_student_report(
                    session, participant_student, summary
                )
                return Response(
                    {
                        "reports": {student_key: student_report},
                        "session_status": session.status,
                        "data": student_report,
                    },
                    status=status.HTTP_200_OK,
                )

            reports = generate_post_viva_report(session)

            return Response({
                "reports": reports,
                "session_status": session.status,
                "data": next(iter(reports.values())) if reports else None,
            }, status=status.HTTP_200_OK)

        except EvaluationSession.DoesNotExist:
            return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class EvaluationSessionCreateView(APIView):
    """
    POST /api/viva/sessions/create/

    Examiner creates a viva session linking project, student, submission.
    """
    permission_classes = [IsAuthenticated, IsAssignedProjectExaminer]

    def post(self, request):
        from viva_evaluator.serializers import (
            EvaluationSessionCreateSerializer,
            EvaluationSessionDetailSerializer,
        )
        serializer = EvaluationSessionCreateSerializer(data=request.data)
        if serializer.is_valid():
            session = serializer.save()
            return Response(
                EvaluationSessionDetailSerializer(session).data,
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SessionListView(APIView):
    """
    GET /api/viva/projects/<project_id>/sessions/

    Returns all sessions for a project.
    """
    permission_classes = [IsAuthenticated, IsAssignedProjectExaminer]

    def get(self, request, project_id):
        from core.models import EvaluationSession, Project
        from viva_evaluator.serializers import EvaluationSessionDetailSerializer
        try:
            project = Project.objects.get(id=project_id)
        except Project.DoesNotExist:
            return Response(
                {"error": "Project not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        sessions = EvaluationSession.objects.filter(
            project=project
        ).order_by('scheduled_start')
        serializer = EvaluationSessionDetailSerializer(sessions, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class SessionStatusView(APIView):
    """
    GET /api/viva/sessions/<session_id>/status/

    Returns current status of a session.
    Frontend polls this to know if session is scheduled, in progress, or complete.
    """
    authentication_classes = [PhysicalKioskAuthentication, CookieJWTAuthentication]
    permission_classes = [IsAuthenticated, VivaSessionPermission]

    def get(self, request, session_id):
        from core.models import EvaluationSession
        try:
            session = EvaluationSession.objects.get(id=session_id)
        except EvaluationSession.DoesNotExist:
            return Response(
                {"error": "Session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Count questions and answers so far
        questions = session.viva_questions.all()
        total_questions = questions.count()
        total_answers = sum(q.answers.count() for q in questions)

        return Response(
            {
                "session_id": str(session.id),
                "status": session.status,
                "phase": session.phase,
                "demo_enabled": session.demo_enabled,
                "demo_completed_at": session.demo_completed_at,
                "scheduled_start": session.scheduled_start,
                "scheduled_end": session.scheduled_end,
                "actual_start": session.actual_start,
                "total_questions_asked": total_questions,
                "total_answers_submitted": total_answers,
            },
            status=status.HTTP_200_OK,
        )


class CurrentQuestionView(APIView):
    """
    GET /api/viva/sessions/<session_id>/current/

    Returns the latest AI-generated question for the session (read-only, no
    generation). In group mode every member's viva UI polls this so that when
    one teammate answers and the AI advances, the others' screens catch up.

    Examiner-interjected questions are delivered through the separate
    live-questions endpoints, so they are excluded here.
    """
    authentication_classes = [PhysicalKioskAuthentication, CookieJWTAuthentication]
    permission_classes = [IsAuthenticated, VivaSessionPermission]

    def get(self, request, session_id):
        from core.models import EvaluationSession, VivaQuestion
        from viva_evaluator.services.pipeline.presenter import (
            persisted_validation_metadata,
        )

        session = EvaluationSession.objects.filter(id=session_id).first()
        if not session:
            return Response({"error": "Session not found."},
                            status=status.HTTP_404_NOT_FOUND)

        latest_q = (
            session.viva_questions
            .exclude(question_source=VivaQuestion.QuestionSource.EXAMINER)
            .order_by('question_order')
            .last()
        )
        if latest_q is None:
            return Response({"question": None, "session_complete": False},
                            status=status.HTTP_200_OK)

        try:
            ext = latest_q.extension
        except Exception:
            ext = None
        return Response(
            {
                "question": {
                    "question_id": str(latest_q.id),
                    "question_text": latest_q.question_text,
                    "blooms_level": latest_q.blooms_level,
                    "difficulty": ext.difficulty_level if ext else "medium",
                    "criterion": (
                        ext.criteria.criteria_name if ext and ext.criteria else ""
                    ),
                    "question_number": latest_q.question_order,
                    **persisted_validation_metadata(latest_q),
                },
                "session_complete": session.status == 'completed',
            },
            status=status.HTTP_200_OK,
        )


class QuestionAudioView(APIView):
    """Return a signed URL for direct Azure streaming, or 202 while pending."""

    authentication_classes = [PhysicalKioskAuthentication, CookieJWTAuthentication]
    permission_classes = [IsAuthenticated, VivaSessionPermission]

    def get(self, request, session_id, question_id):
        from core.models import VivaQuestion
        from viva_evaluator.services.tts import (
            get_tts_audio,
            get_tts_signed_url,
            get_tts_status,
        )

        logger.info("[QuestionAudioView] Incoming audio request for question_id=%s, session_id=%s", question_id, session_id)

        question = (
            VivaQuestion.objects.select_related("extension")
            .filter(id=question_id, session_id=session_id)
            .first()
        )
        if question is None:
            logger.warning("[QuestionAudioView] Question %s not found in session %s", question_id, session_id)
            return Response(
                {"error": "Question not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            audit = dict(question.extension.generation_audit or {})
        except Exception:
            audit = {}
        tts = dict(audit.get("tts") or {})
        cache_key = str(tts.get("cache_key") or "")
        if (
            not cache_key
            or tts.get("candidate_hash") != audit.get("candidate_hash")
        ):
            logger.warning("[QuestionAudioView] TTS cache_key missing or candidate_hash mismatch for question %s", question_id)
            return Response(
                {"tts_status": "unavailable"},
                status=status.HTTP_404_NOT_FOUND,
            )

        current = get_tts_status(cache_key)
        current_status = current.get("status", "pending")
        logger.info("[QuestionAudioView] Status for key=%s is '%s'", cache_key[:12], current_status)

        if current_status == "ready":
            # Try signed URL first (browser streams directly from Azure)
            signed = get_tts_signed_url(cache_key)
            if signed is not None:
                logger.info("[QuestionAudioView] Returning 200 JSON with signed Azure URL for question %s", question_id)
                return Response(
                    {
                        "tts_status": "ready",
                        "audio_url": signed["audio_url"],
                        "cache_hit": signed["cache_hit"],
                    }
                )

            # Fallback: proxy the bytes through Django
            logger.info("[QuestionAudioView] SAS URL failed; falling back to binary proxy through Django")
            try:
                audio = get_tts_audio(cache_key)
            except Exception:
                audio = None
            if audio is not None:
                audio_bytes, mime_type, audio_status = audio
                response = HttpResponse(audio_bytes, content_type=mime_type)
                response["Cache-Control"] = "private, max-age=3600"
                response["Content-Length"] = str(len(audio_bytes))
                response["X-TTS-Cache-Hit"] = str(
                    audio_status.get("cache_hit") is True
                ).lower()
                return response

        if current_status == "pending":
            logger.info("[QuestionAudioView] Audio still generating. Returning 202 Accepted (Retry-After: 0.2)")
            response = Response(
                {"tts_status": "pending"},
                status=status.HTTP_202_ACCEPTED,
            )
            response["Retry-After"] = "0.2"
            return response
        logger.warning("[QuestionAudioView] Returning 503 for tts_status='%s'", current_status)
        return Response(
            {"tts_status": current_status},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class AnswerTranscriptionView(APIView):
    """
    POST /api/viva/sessions/<session_id>/transcribe/

    Transcribes one recorded utterance with ElevenLabs Scribe and returns the
    text.  The browser records short segments and posts them here so the API
    key stays server-side; a disabled or failing provider returns a status the
    client uses to fall back to browser dictation or typing.
    """

    authentication_classes = [PhysicalKioskAuthentication, CookieJWTAuthentication]
    permission_classes = [IsAuthenticated, VivaSessionPermission]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, session_id):
        from viva_evaluator.services.stt import (
            is_enabled,
            max_audio_bytes,
            transcribe_answer_audio,
        )

        if not is_enabled():
            return Response(
                {"stt_status": "disabled", "text": ""},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        upload = request.FILES.get("audio") or request.FILES.get("file")
        if upload is None:
            return Response(
                {"error": "An 'audio' file is required.", "stt_status": "failed"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if upload.size and upload.size > max_audio_bytes():
            return Response(
                {"error": "Audio clip is too large.", "stt_status": "failed"},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        result = transcribe_answer_audio(
            upload.read(),
            content_type=getattr(upload, "content_type", "") or "",
            filename=getattr(upload, "name", "") or "",
        )

        if result.status == "disabled":
            return Response(
                {"stt_status": "disabled", "text": ""},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        if result.status == "failed":
            logger.warning(
                "[AnswerTranscriptionView] Transcription failed for session %s (%s)",
                session_id, result.error,
            )
            return Response(
                {"stt_status": "failed", "text": "", "error": result.error[:160]},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        # "empty" is a normal outcome for a clip that held only silence.
        return Response({
            "stt_status": result.status,
            "text": result.text,
            "language_code": result.language_code,
            "language_probability": result.language_probability,
            "latency_ms": result.latency_ms,
        })


class SessionDetailedReportView(APIView):
    """
    GET /api/evaluator/sessions/<session_id>/detailed-report/
    
    Returns the comprehensive timeline of the AI viva session for the examiner.
    Includes every question asked, the student's transcribed answer, and the
    granular AI reasoning (gaps identified, strategies, soft scores) stored
    in the detailed_ai_analysis JSON field.
    """
    permission_classes = [IsAuthenticated, IsAssignedSessionExaminer]

    def get(self, request, session_id):
        from core.models import EvaluationSession, VivaQuestion
        import traceback as tb_module

        try:
            try:
                session = EvaluationSession.objects.select_related(
                    'project', 'student__user', 'group'
                ).get(id=session_id)
            except EvaluationSession.DoesNotExist:
                return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)

            # Get session info
            session_data = {
                "session_id": str(session.id),
                "project_name": session.project.project_name,
                "status": session.status,
                "scheduled_start": session.scheduled_start,
                "scheduled_end": session.scheduled_end,
                "location_room": session.location_room,
                "student_name": session.student.user.full_name if session.student and getattr(session.student, 'user', None) else None,
                "group_name": session.group.group_name if session.group else None,
                "viva_weight_percentage": session.viva_weight_percentage,
            }

            # Get summary report if exists
            # Get summary reports for all students in the session
            reports_data = {}
            try:
                # If session is completed, ensure reports exist so examiners can approve scores
                from core.models import EvaluationSession as ES
                if session.status == ES.Status.COMPLETED:
                    from viva_evaluator.services.session_reports import (
                        refresh_draft_summary_reports,
                    )
                    refresh_draft_summary_reports(session)

                summaries = session.summary_reports.all()
                for summary in summaries:
                    # Group Overview is rendered separately by the frontend.
                    # Its score header must always represent a real student.
                    if session.group_id and summary.student_id is None:
                        continue
                    # In a group session with individual tracking, the student might be linked
                    # If not, it defaults to 'group'
                    student_key = str(summary.student.id) if summary.student else 'group'
                    student_name = summary.student.user.full_name if summary.student and getattr(summary.student, 'user', None) else 'Group'
                    final_score_percentage = float(summary.total_final_score or 0)
                    viva_weight_percentage = session.viva_weight_percentage
                    reports_data[student_key] = {
                        "student_name": student_name,
                        "total_ai_score": summary.total_ai_score,
                        "total_final_score": summary.total_final_score,
                        "viva_weight_percentage": viva_weight_percentage,
                        "scaled_score": round(
                            final_score_percentage * viva_weight_percentage / 100,
                            2,
                        ),
                        "grade": summary.grade,
                        "overall_feedback": summary.overall_feedback,
                        "emotional_summary": summary.emotional_summary,
                        "integrity_flags_summary": summary.integrity_flags_summary,
                        "scores_status": summary.scores_status,
                        "scores_approved_at": summary.scores_approved_at.isoformat() if summary.scores_approved_at else None,
                    }
            except Exception:
                pass

            # Fetch all questions and their answers in chronological order
            questions_qs = (
                session.viva_questions.all()
                .select_related('extension__criteria')
                .prefetch_related(
                    'answers__student__user',
                    'answers__extension',
                    'answers__attribution',
                    'answers__contributions__student__user',
                    'answers__contributions__unknown_speaker__resolved_student__user',
                )
                .order_by('question_order')
            )

            def serialize_answer(answer):
                try:
                    ans_ext = answer.extension
                except Exception:
                    ans_ext = None

                contributions = []
                for contribution in answer.contributions.all():
                    effective_student = contribution.student
                    unknown_speaker = contribution.unknown_speaker
                    if (
                        effective_student is None
                        and unknown_speaker is not None
                        and unknown_speaker.resolved_student_id
                    ):
                        effective_student = unknown_speaker.resolved_student

                    if effective_student and getattr(effective_student, 'user', None):
                        contributor_name = effective_student.user.full_name
                    elif unknown_speaker is not None:
                        contributor_name = unknown_speaker.label
                    else:
                        contributor_name = 'Unidentified speaker'

                    contributions.append({
                        "student_id": (
                            str(effective_student.id) if effective_student else None
                        ),
                        "student_name": contributor_name,
                        "unknown_speaker_id": (
                            str(unknown_speaker.id) if unknown_speaker else None
                        ),
                        "share": float(contribution.share),
                        "is_dominant": contribution.is_dominant,
                    })

                try:
                    attribution = answer.attribution
                except Exception:
                    attribution = None

                student = getattr(answer, 'student', None)
                student_user = getattr(student, 'user', None) if student else None
                return {
                    "answer_id": str(answer.id),
                    "student_id": str(student.id) if student else None,
                    "transcribed_answer": answer.transcribed_answer,
                    "answered_at": str(answer.answered_at) if answer.answered_at else None,
                    "answered_by": student_user.full_name if student_user else None,
                    "llm_score": (
                        float(ans_ext.llm_score)
                        if ans_ext and ans_ext.llm_score is not None
                        else None
                    ),
                    "examiner_override_score": (
                        float(answer.examiner_override_score)
                        if answer.examiner_override_score is not None
                        else None
                    ),
                    "examiner_override_note": answer.examiner_override_note,
                    "llm_reasoning": ans_ext.llm_reasoning if ans_ext else None,
                    "detailed_ai_analysis": (
                        ans_ext.detailed_ai_analysis if ans_ext else None
                    ),
                    "contributions": contributions,
                    "attribution": ({
                        "outcome": attribution.outcome,
                        "status": attribution.status,
                        "share": float(attribution.share),
                        "margin": float(attribution.margin),
                        "needs_review": attribution.needs_review,
                    } if attribution else None),
                }

            timeline = []
            for q in questions_qs:
                try:
                    q_ext = q.extension
                except Exception:
                    q_ext = None

                # A group question can have one response per student or a
                # collaborative response containing multiple contributors.
                answers = sorted(
                    q.answers.all(),
                    key=lambda item: item.answered_at,
                )
                serialized_answers = [serialize_answer(answer) for answer in answers]

                criteria = q_ext.criteria if q_ext else None
                if criteria is None:
                    criterion_type = 'unclassified'
                else:
                    criterion_type = 'individual' if criteria.is_individual else 'group'

                timeline.append({
                    "question_id": str(q.id),
                    "question_text": q.question_text,
                    "question_source": q.question_source,
                    "question_order": q.question_order,
                    "blooms_level": q.blooms_level,
                    "difficulty": q_ext.difficulty_level if q_ext else None,
                    "criterion": criteria.criteria_name if criteria else getattr(q, 'viva_topic_name', None),
                    "criterion_type": criterion_type,
                    "is_individual_criterion": (
                        criteria.is_individual if criteria is not None else None
                    ),
                    "asked_at": str(getattr(q, 'asked_at', None) or q.generated_at),
                    "validation_status": (
                        q_ext.validation_status if q_ext else 'not_applicable'
                    ),
                    "validation_degraded": bool(
                        q_ext.validation_degraded if q_ext else False
                    ),
                    "fallback_used": bool(
                        q_ext.fallback_used if q_ext else False
                    ),
                    "generation_audit": (
                        q_ext.generation_audit if q_ext else {}
                    ),
                    # Keep `answer` for older clients while exposing the full
                    # group response history through `answers`.
                    "answer": serialized_answers[-1] if serialized_answers else None,
                    "answers": serialized_answers,
                })

            return Response({
                "session": session_data,
                "reports": reports_data,
                "report": next(iter(reports_data.values())) if reports_data else None,
                "timeline": timeline,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({
                "error": str(e),
                "traceback": tb_module.format_exc(),
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
