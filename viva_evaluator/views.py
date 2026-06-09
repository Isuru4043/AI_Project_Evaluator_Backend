from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from urllib.request import urlopen

from core.models import ProjectSubmission
from viva_evaluator.models import SubmissionIndexStatus
from viva_evaluator.serializers import (
    SubmissionUploadSerializer,
    SubmissionIndexStatusSerializer,
)


def _resolve_session_submission(session):
    if session.submission:
        return session.submission

    if session.group_id:
        return ProjectSubmission.objects.filter(
            project=session.project,
            group=session.group,
        ).first()

    if session.student_id:
        return ProjectSubmission.objects.filter(
            project=session.project,
            student=session.student,
        ).first()

    return None


def _difficulty_signal_from_score(soft_score: float) -> str:
    """
    Legacy compatibility: VivaAnswerExtension expects 'lower'|'same'|'higher'.
    Map the new soft_score to that signal so the audit trail stays consistent.
    """
    if soft_score < 0.4:
        return 'lower'
    if soft_score < 0.7:
        return 'same'
    return 'higher'


def _get_or_create_index_status(submission):
    from core.utils.document_parser import extract_text_from_bytes

    index_status, _ = SubmissionIndexStatus.objects.get_or_create(
        submission=submission,
    )

    if index_status.status == SubmissionIndexStatus.IndexStatus.READY and index_status.extracted_text:
        return index_status

    if not submission.report_file_url:
        return index_status

    try:
        with urlopen(submission.report_file_url) as response:
            file_content = response.read()

        report_name = submission.report_file_url.split('?')[0].rsplit('/', 1)[-1] or 'submission-report.pdf'
        extracted_text = extract_text_from_bytes(file_content, report_name)

        index_status.extracted_text = extracted_text
        index_status.status = SubmissionIndexStatus.IndexStatus.READY
        index_status.processed_at = timezone.now()
        index_status.error_message = None
        index_status.save()
        return index_status
    except Exception as exc:
        index_status.status = SubmissionIndexStatus.IndexStatus.FAILED
        index_status.error_message = str(exc)
        index_status.processed_at = timezone.now()
        index_status.save()
        raise


class SubmissionUploadView(APIView):
    """
    POST /api/viva/submissions/upload/

    Handles report upload for both students and examiners.
    - Student logged in: auto-attaches their profile, only needs project + file
    - Examiner logged in: requires student UUID in request body
    """
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.utils.document_parser import extract_text_from_bytes

        data = request.data.copy()
        user = request.user

        # Auto-detect student if logged in user is a student
        if user.role == 'student':
            try:
                student_profile = user.student_profile
                data['student'] = str(student_profile.id)
            except Exception:
                return Response(
                    {"error": "Student profile not found for this user."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            # Examiner must supply student UUID
            if not data.get('student'):
                return Response(
                    {"error": "student UUID is required when uploading as examiner."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        serializer = SubmissionUploadSerializer(data=data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            submission = serializer.save()
            index_status = submission.index_status

            index_status.status = SubmissionIndexStatus.IndexStatus.PROCESSING
            index_status.save()

            # Read file content directly so this works with cloud-backed storage.
            with index_status.report_file.open('rb') as f:
                file_content = f.read()
            extracted_text = extract_text_from_bytes(
                file_content, index_status.report_file.name
            )

            index_status.extracted_text = extracted_text

            # =========================================================
            # Build FAISS index from the extracted text (Week 2 RAG).
            # Uses section-aware chunking + multimodal image captioning.
            # On indexing failure we still mark the submission READY so
            # the legacy text-only flow keeps working — but log it loudly.
            # =========================================================
            try:
                from viva_evaluator.services.indexing import index_report
                from viva_evaluator.services.rag.vector_store import save_index_for_submission

                index_result = index_report(file_content, enable_image_captions=True)
                chunks = index_result['chunks']
                num_chunks, _ = save_index_for_submission(submission, chunks)
                indexed_chunks = num_chunks
                images_captioned = index_result['images_captioned']
            except Exception as idx_exc:
                import logging
                logging.getLogger(__name__).exception(
                    'FAISS indexing failed for submission=%s: %s', submission.id, idx_exc,
                )
                indexed_chunks = 0
                images_captioned = 0

            index_status.status = SubmissionIndexStatus.IndexStatus.READY
            index_status.processed_at = timezone.now()
            index_status.save()

            return Response(
                {
                    "message": "Report uploaded and processed successfully.",
                    "submission_id": str(submission.id),
                    "status": "ready",
                    "characters_extracted": len(extracted_text),
                    "chunks_indexed": indexed_chunks,
                    "images_captioned": images_captioned,
                },
                status=status.HTTP_201_CREATED,
            )

        except Exception as e:
            try:
                index_status.status = SubmissionIndexStatus.IndexStatus.FAILED
                index_status.error_message = str(e)
                index_status.save()
            except Exception:
                pass
            return Response(
                {"error": f"Upload failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SubmissionStatusView(APIView):
    """
    GET /api/viva/submissions/<submission_id>/status/

    Returns the current processing status of a submission.
    Frontend checks this before allowing a session to start.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, submission_id):
        try:
            index_status = SubmissionIndexStatus.objects.get(
                submission__id=submission_id
            )
        except SubmissionIndexStatus.DoesNotExist:
            return Response(
                {"error": "Submission not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = SubmissionIndexStatusSerializer(index_status)
        return Response(serializer.data, status=status.HTTP_200_OK)

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
    permission_classes = [IsAuthenticated]

    def post(self, request):
        session_id = request.data.get('session_id')

        if not session_id:
            return Response(
                {"error": "session_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            from core.models import EvaluationSession
            from viva_evaluator.services.session_manager import get_session_context
            from viva_evaluator.services.agents import (
                generate_anchored_question, QuestionerInput,
            )
            from viva_evaluator.services.rag.retrieval import retrieve_hybrid_for_turn
            from viva_evaluator.models import VivaQuestionExtension
            from core.models import VivaQuestion

            session = EvaluationSession.objects.get(id=session_id)
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

            # Get session context to find first criterion
            context = get_session_context(session_id)

            # If project has no rubric criteria configured, treat as misconfiguration
            if not context.get('all_criteria'):
                return Response(
                    {"error": "No rubric configured for this project. Please add rubric criteria before starting a viva session."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if context['is_complete']:
                return Response(
                    {"error": "This session is already complete."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            next_criterion = context['next_criterion']

            # =================================================================
            # Hybrid retrieval — pull both FAISS chunks AND KG signals
            # (CONTRADICTS_CODE alerts, DEPENDS_ON topics).
            # =================================================================
            retrieval = retrieve_hybrid_for_turn(
                submission=submission,
                criterion_name=next_criterion['name'],
                criterion_description=next_criterion['description'],
                last_answer='',
                top_k=3,
            )
            retrieved = retrieval['chunks']

            # =================================================================
            # Generate the anchored question via the new pipeline.
            # =================================================================
            question_data = generate_anchored_question(QuestionerInput(
                criterion_name=next_criterion['name'],
                criterion_description=next_criterion['description'],
                retrieved_chunks=retrieved,
                kg_signals=retrieval,
                difficulty='medium',
                question_hints=next_criterion.get('hints', []),
                recent_questions=[],
                is_first_question=True,
                question_number_in_criterion=1,
            ))

            # Save the question to DB
            question_order = context['total_questions_asked'] + 1
            question = VivaQuestion.objects.create(
                session=session,
                question_text=question_data['question_text'],
                blooms_level=question_data.get('blooms_level', 'Understand'),
                question_order=question_order,
                question_source='ai',
            )

            # Save extension with criterion and difficulty
            from core.models import RubricCriteria
            criterion_obj = RubricCriteria.objects.get(id=next_criterion['id'])
            VivaQuestionExtension.objects.create(
                question=question,
                criteria=criterion_obj,
                difficulty_level=question_data.get('difficulty', 'medium'),
            )

            # Mark session as in progress
            from core.models import EvaluationSession as ES
            session.status = ES.Status.IN_PROGRESS
            from django.utils import timezone
            session.actual_start = timezone.now()
            session.save()

            return Response(
                {
                    "message": "Session started.",
                    "session_id": session_id,
                    "question_id": str(question.id),
                    "question_text": question.question_text,
                    "blooms_level": question.blooms_level,
                    "difficulty": question_data.get('difficulty', 'medium'),
                    "criterion": next_criterion['name'],
                    "question_number": question_order,
                    "tier1_passed": question_data.get('tier1_passed', False),
                    "tier1_failures": question_data.get('tier1_failures', []),
                    "critic_passed": question_data.get('critic_passed', True),
                    "critic_critique": question_data.get('critic_critique', ''),
                    "critic_scores": question_data.get('critic_scores', {}),
                },
                status=status.HTTP_200_OK,
            )

        except EvaluationSession.DoesNotExist:
            return Response(
                {"error": "Session not found."},
                status=status.HTTP_404_NOT_FOUND,
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
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        question_id = request.data.get('question_id')
        answer_text = request.data.get('answer_text', '').strip()
        speech_metrics = request.data.get('speech_metrics')   # Week 6: optional

        if not question_id or not answer_text:
            return Response(
                {"error": "question_id and answer_text are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            from core.models import (
                EvaluationSession, VivaQuestion,
                VivaAnswer, RubricCriteria,
            )
            from viva_evaluator.models import (
                VivaAnswerExtension, VivaQuestionExtension,
            )
            from viva_evaluator.services.pipeline import (
                process_answer_and_pick_next,
            )

            session = EvaluationSession.objects.get(id=session_id)
            question = VivaQuestion.objects.get(id=question_id, session=session)

            submission = _resolve_session_submission(session)
            if not submission:
                return Response(
                    {"error": "No submission found for this session."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # =================================================================
            # WEEK 5 — single pipeline call replaces legacy answer_evaluator
            # + session_manager glue. Runs Analyzer → BKT update →
            # termination check → Strategist → Questioner.
            # =================================================================
            result = process_answer_and_pick_next(
                session=session,
                submission=submission,
                prev_question_obj=question,
                student_answer=answer_text,
                speech_metrics=speech_metrics,
            )

            analysis = result['analysis']
            soft_score = result['soft_score']
            confidence = result.get('speech_confidence') or {}

            # Persist the answer + extension (audit trail).
            answer = VivaAnswer.objects.create(
                question=question,
                student=session.student,
                transcribed_answer=answer_text,
                # ai_answer_score is on a 0-10 scale historically; map from soft_score
                ai_answer_score=round(soft_score * 10.0, 2),
            )
            VivaAnswerExtension.objects.create(
                answer=answer,
                llm_score=round(soft_score * 10.0, 2),
                llm_reasoning=analysis.get('reasoning', '') or '',
                next_difficulty_signal=_difficulty_signal_from_score(soft_score),
            )

            # Build a unified rubric scores block for the response.
            rubric_payload = {
                'correctness': analysis.get('correctness', {}),
                'depth':       analysis.get('depth', {}),
                'consistency': analysis.get('consistency', {}),
                'soft_score':  soft_score,
                'reasoning':   analysis.get('reasoning', ''),
                'gap_identified':       analysis.get('gap_identified', ''),
                'revealed_assumption':  analysis.get('revealed_assumption', ''),
                'contradicts_code_flag': analysis.get('contradicts_code_flag', False),
            }

            # ---- Session complete -------------------------------------------
            if result['session_complete']:
                return Response(
                    {
                        "answer_saved": True,
                        "session_complete": True,
                        "termination_reason": result.get('termination_reason'),
                        "rubric": rubric_payload,
                        "speech_confidence": confidence,
                        "message": "All termination conditions satisfied — session complete.",
                    },
                    status=status.HTTP_200_OK,
                )

            # ---- Save next question -----------------------------------------
            payload = result['next_question_payload']
            next_criterion = payload['criterion']
            qd = payload['question_data']

            total_asked = session.viva_questions.count()

            next_question = VivaQuestion.objects.create(
                session=session,
                question_text=qd['question_text'],
                blooms_level=qd.get('blooms_level', payload['bloom_level']),
                question_order=total_asked + 1,
                question_source='ai',
            )
            try:
                next_criterion_obj = RubricCriteria.objects.get(id=next_criterion['id'])
                VivaQuestionExtension.objects.create(
                    question=next_question,
                    criteria=next_criterion_obj,
                    difficulty_level=qd.get('difficulty', payload['difficulty']),
                )
            except RubricCriteria.DoesNotExist:
                pass

            return Response(
                {
                    "answer_saved": True,
                    "session_complete": False,
                    "rubric": rubric_payload,
                    "speech_confidence": confidence,
                    "strategy": {
                        "bloom_level":     payload['bloom_level'],
                        "socratic_intent": payload['socratic_intent'],
                        "p_lt":            round(payload['p_lt'], 3),
                        "rationale":       result['strategy'].get('rationale', ''),
                    },
                    "next_question": {
                        "question_id":     str(next_question.id),
                        "question_text":   next_question.question_text,
                        "blooms_level":    payload['bloom_level'],
                        "difficulty":      payload['difficulty'],
                        "criterion":       next_criterion['name'],
                        "question_number": total_asked + 1,
                        "tier1_passed":    qd.get('tier1_passed', False),
                        "tier1_failures":  qd.get('tier1_failures', []),
                        "critic_passed":   qd.get('critic_passed', True),
                        "critic_critique": qd.get('critic_critique', ''),
                        "critic_scores":   qd.get('critic_scores', {}),
                        "attempts":        qd.get('attempts', 1),
                    },
                },
                status=status.HTTP_200_OK,
            )

        except EvaluationSession.DoesNotExist:
            return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
        except VivaQuestion.DoesNotExist:
            return Response({"error": "Question not found."}, status=status.HTTP_404_NOT_FOUND)
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
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SessionReportView(APIView):
    """
    GET /api/viva/sessions/<session_id>/report/

    Returns the structured post-viva report. Allowed even before the
    session is COMPLETED so examiners can review mid-session if needed —
    the report just reflects whatever turns have happened so far.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        try:
            from core.models import EvaluationSession
            from viva_evaluator.services.reporting import generate_post_viva_report

            session = EvaluationSession.objects.get(id=session_id)
            report = generate_post_viva_report(session)
            report['session_status'] = session.status
            return Response(report, status=status.HTTP_200_OK)

        except EvaluationSession.DoesNotExist:
            return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ProjectCreateView(APIView):
    """
    POST /api/viva/projects/

    Examiner creates a project with full rubric in one call.
    Returns warnings if weights do not add up to 100%.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from viva_evaluator.serializers import (
            ProjectCreateSerializer, ProjectDetailSerializer
        )

        context = {'warnings': []}
        serializer = ProjectCreateSerializer(
            data=request.data,
            context=context,
        )

        if serializer.is_valid():
            project = serializer.save()
            response_data = ProjectDetailSerializer(project).data

            # Include warnings if any
            warnings = serializer.context.get('warnings', [])
            if warnings:
                response_data['warnings'] = warnings
            else:
                response_data['warnings'] = []

            return Response(response_data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ProjectDetailView(APIView):
    """
    GET /api/viva/projects/<project_id>/

    Returns full project details including rubric.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, project_id):
        from core.models import Project
        from viva_evaluator.serializers import ProjectDetailSerializer
        try:
            project = Project.objects.get(id=project_id)
            return Response(
                ProjectDetailSerializer(project).data,
                status=status.HTTP_200_OK,
            )
        except Project.DoesNotExist:
            return Response(
                {"error": "Project not found."},
                status=status.HTTP_404_NOT_FOUND,
            )


class RubricCategoryCreateView(APIView):
    """
    POST /api/viva/projects/<project_id>/categories/

    Add a rubric category to an existing project.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, project_id):
        from core.models import Project, RubricCategory
        from viva_evaluator.serializers import RubricCategorySerializer
        try:
            project = Project.objects.get(id=project_id)
        except Project.DoesNotExist:
            return Response(
                {"error": "Project not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        data = request.data.copy()
        serializer = RubricCategorySerializer(data=data)
        if serializer.is_valid():
            category = serializer.save(project=project)
            return Response(
                RubricCategorySerializer(category).data,
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RubricCriteriaCreateView(APIView):
    """
    POST /api/viva/categories/<category_id>/criteria/

    Add a criterion to an existing rubric category.
    Includes optional question hints and questions_to_ask count.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, category_id):
        from core.models import RubricCategory
        from viva_evaluator.serializers import RubricCriteriaSerializer
        try:
            category = RubricCategory.objects.get(id=category_id)
        except RubricCategory.DoesNotExist:
            return Response(
                {"error": "Category not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = RubricCriteriaSerializer(data=request.data)
        if serializer.is_valid():
            criteria = serializer.save(category=category)
            return Response(
                RubricCriteriaSerializer(criteria).data,
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class QuestionHintCreateView(APIView):
    """
    POST /api/viva/criteria/<criteria_id>/hints/

    Add question hints to an existing criterion.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, criteria_id):
        from core.models import RubricCriteria
        from viva_evaluator.serializers import CriteriaQuestionHintSerializer
        try:
            criteria = RubricCriteria.objects.get(id=criteria_id)
        except RubricCriteria.DoesNotExist:
            return Response(
                {"error": "Criteria not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = CriteriaQuestionHintSerializer(data=request.data)
        if serializer.is_valid():
            hint = serializer.save(criteria=criteria)
            return Response(
                CriteriaQuestionHintSerializer(hint).data,
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class RubricUploadPreviewView(APIView):
    """
    POST /api/viva/rubric/upload-preview/

    Examiner uploads a rubric PDF or DOCX.
    Gemini extracts the structure and returns a preview for the examiner to review.
    Nothing is saved to the database yet.

    Request: multipart/form-data with 'rubric_file' field.
    Response: structured rubric JSON for examiner to review and edit.
    """
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        rubric_file = request.FILES.get('rubric_file')

        if not rubric_file:
            return Response(
                {"error": "rubric_file is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        ext = rubric_file.name.split('.')[-1].lower()
        if ext not in ['pdf', 'docx']:
            return Response(
                {"error": "Only PDF and DOCX files are accepted."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            import os
            import tempfile
            from core.utils.document_parser import extract_text_from_file
            from viva_evaluator.services.rubric_extractor import extract_rubric_from_text

            # Save to a temp file so document_parser can read it
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f'.{ext}'
            ) as tmp:
                for chunk in rubric_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            # Extract text from the file
            rubric_text = extract_text_from_file(tmp_path)
            os.unlink(tmp_path)  # Clean up temp file

            if not rubric_text.strip():
                return Response(
                    {"error": "Could not extract text from the uploaded file."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Send to Gemini for structure extraction
            extracted = extract_rubric_from_text(rubric_text)

            if "error" in extracted:
                return Response(
                    {"error": extracted["error"]},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            return Response(
                {
                    "message": "Rubric extracted successfully. Review and confirm to save.",
                    "preview": extracted,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class RubricConfirmSaveView(APIView):
    """
    POST /api/viva/rubric/confirm-save/
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from viva_evaluator.serializers import (
            ProjectCreateSerializer, ProjectDetailSerializer
        )

        data = request.data.copy()

        if 'project_description' in data and 'description' not in data:
            data['description'] = data.pop('project_description')

        context = {'warnings': []}
        serializer = ProjectCreateSerializer(data=data, context=context)

        if serializer.is_valid():
            project = serializer.save()
            response_data = ProjectDetailSerializer(project).data

            warnings = serializer.context.get('warnings', [])
            if warnings:
                response_data['warnings'] = warnings
            else:
                response_data['warnings'] = []

            return Response(
                {
                    "message": "Rubric saved successfully.",
                    "project": response_data,
                },
                status=status.HTTP_201_CREATED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class EvaluationSessionCreateView(APIView):
    """
    POST /api/viva/sessions/create/

    Examiner creates a viva session linking project, student, submission.
    """
    permission_classes = [IsAuthenticated]

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


class ProjectListView(APIView):
    """
    GET  /api/viva/projects/ — Returns all projects
    POST /api/viva/projects/ — Creates a new project with rubric
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        from core.models import Project
        from viva_evaluator.serializers import ProjectDetailSerializer
        projects = Project.objects.all().order_by('-created_at')
        serializer = ProjectDetailSerializer(projects, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        from viva_evaluator.serializers import (
            ProjectCreateSerializer, ProjectDetailSerializer
        )
        context = {'warnings': []}
        serializer = ProjectCreateSerializer(
            data=request.data,
            context=context,
        )
        if serializer.is_valid():
            project = serializer.save()
            response_data = ProjectDetailSerializer(project).data
            warnings = serializer.context.get('warnings', [])
            if warnings:
                response_data['warnings'] = warnings
            else:
                response_data['warnings'] = []
            return Response(response_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class StudentListView(APIView):
    """
    GET /api/viva/students/

    Returns all students. Used by examiner when creating a session.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        from core.models import StudentProfile
        students = StudentProfile.objects.select_related('user').all()
        data = [
            {
                'id': str(s.id),
                'full_name': s.user.full_name,
                'email': s.user.email,
                'registration_number': s.registration_number,
                'degree_program': s.degree_program,
                'academic_year': s.academic_year,
                'batch': s.batch,
            }
            for s in students
        ]
        return Response(data, status=status.HTTP_200_OK)


class SessionListView(APIView):
    """
    GET /api/viva/projects/<project_id>/sessions/

    Returns all sessions for a project.
    """
    permission_classes = [IsAuthenticated]

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
    permission_classes = [IsAuthenticated]

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
                "scheduled_start": session.scheduled_start,
                "scheduled_end": session.scheduled_end,
                "actual_start": session.actual_start,
                "total_questions_asked": total_questions,
                "total_answers_submitted": total_answers,
            },
            status=status.HTTP_200_OK,
        )

class FinalScoreSubmitView(APIView):
    """
    POST /api/viva/sessions/<session_id>/final-scores/

    Examiner submits final approved scores after reviewing the XAI report.
    Saves to FinalScore and creates SessionSummaryReport.

    Request body:
    {
        "scores": [
            {
                "criteria_id": "uuid",
                "examiner_final_score": 8.5,
                "examiner_note": "Good understanding overall"
            }
        ],
        "overall_feedback": "Student demonstrated solid knowledge",
        "grade": "A"
    }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        from core.models import (
            EvaluationSession, RubricCriteria,
            FinalScore, SessionSummaryReport, ExaminerProfile
        )
        from django.utils import timezone

        try:
            session = EvaluationSession.objects.get(id=session_id)
        except EvaluationSession.DoesNotExist:
            return Response(
                {"error": "Session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Get examiner profile from logged in user
        try:
            examiner = request.user.examiner_profile
        except Exception:
            return Response(
                {"error": "Only examiners can submit final scores."},
                status=status.HTTP_403_FORBIDDEN,
            )

        scores_data = request.data.get('scores', [])
        overall_feedback = request.data.get('overall_feedback', '')
        grade = request.data.get('grade', '')

        if not scores_data:
            return Response(
                {"error": "scores list is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        saved_scores = []
        total_final_score = 0
        total_ai_score = 0

        for score_item in scores_data:
            criteria_id = score_item.get('criteria_id')
            examiner_final_score = score_item.get('examiner_final_score')
            examiner_note = score_item.get('examiner_note', '')

            if not criteria_id or examiner_final_score is None:
                continue

            try:
                criteria = RubricCriteria.objects.get(id=criteria_id)
            except RubricCriteria.DoesNotExist:
                continue

            # Get AI recommended score for this criterion
            ai_score = None
            try:
                ai_rec = session.ai_score_recommendations.filter(
                    criteria=criteria
                ).last()
                if ai_rec:
                    ai_score = float(ai_rec.ai_recommended_score)
            except Exception:
                pass

            # Save final score
            final_score, _ = FinalScore.objects.update_or_create(
                session=session,
                criteria=criteria,
                examiner=examiner,
                defaults={
                    'examiner_final_score': examiner_final_score,
                    'ai_recommended_score': ai_score,
                    'examiner_note': examiner_note,
                }
            )

            saved_scores.append({
                'criteria': criteria.criteria_name,
                'ai_recommended_score': ai_score,
                'examiner_final_score': float(examiner_final_score),
                'examiner_note': examiner_note,
            })

            total_final_score += float(examiner_final_score)
            if ai_score:
                total_ai_score += ai_score

        # Create or update session summary report
        summary, _ = SessionSummaryReport.objects.update_or_create(
            session=session,
            defaults={
                'total_ai_score': total_ai_score,
                'total_final_score': total_final_score,
                'grade': grade,
                'overall_feedback': overall_feedback,
                'finalized_by': examiner,
                'is_published': True,
                'published_at': timezone.now(),
            }
        )

        return Response(
            {
                "message": "Final scores submitted successfully.",
                "session_id": session_id,
                "grade": grade,
                "total_final_score": total_final_score,
                "total_ai_score": total_ai_score,
                "scores": saved_scores,
            },
            status=status.HTTP_201_CREATED,
        )

class RubricCategoryUpdateView(APIView):
    """
    PATCH /api/viva/categories/<category_id>/

    Examiner updates a rubric category's name, weight, or description.
    """
    permission_classes = [IsAuthenticated]

    def patch(self, request, category_id):
        from core.models import RubricCategory
        from viva_evaluator.serializers import RubricCategoryUpdateSerializer
        try:
            category = RubricCategory.objects.get(id=category_id)
        except RubricCategory.DoesNotExist:
            return Response(
                {"error": "Category not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = RubricCategoryUpdateSerializer(
            category, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RubricCriteriaUpdateView(APIView):
    """
    PATCH /api/viva/criteria/<criteria_id>/

    Examiner updates a rubric criterion's fields.
    """
    permission_classes = [IsAuthenticated]

    def patch(self, request, criteria_id):
        from core.models import RubricCriteria
        from viva_evaluator.serializers import RubricCriteriaUpdateSerializer
        try:
            criteria = RubricCriteria.objects.get(id=criteria_id)
        except RubricCriteria.DoesNotExist:
            return Response(
                {"error": "Criteria not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = RubricCriteriaUpdateSerializer(
            criteria, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class QuestionHintDeleteView(APIView):
    """
    DELETE /api/viva/hints/<hint_id>/

    Examiner removes a question hint from a criterion.
    """
    permission_classes = [IsAuthenticated]

    def delete(self, request, hint_id):
        from viva_evaluator.models import CriteriaQuestionHint
        try:
            hint = CriteriaQuestionHint.objects.get(id=hint_id)
        except CriteriaQuestionHint.DoesNotExist:
            return Response(
                {"error": "Hint not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        hint.delete()
        return Response(
            {"message": "Hint deleted successfully."},
            status=status.HTTP_200_OK,
        )



# =============================================================================
# WEEK 4 — Examiner-in-the-Loop Brief Review API
# =============================================================================

class BriefListView(APIView):
    """
    GET /api/viva/briefs/

    List domain briefs. Default returns pending (T2) drafts for the examiner
    to review. Filterable via query params:
        ?status=pending|active|archived
        ?technology=PostgreSQL
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        from viva_evaluator.models import ApprovedDomainBrief
        from viva_evaluator.serializers import ApprovedDomainBriefListSerializer

        qs = ApprovedDomainBrief.objects.all()

        status_filter = request.query_params.get('status', 'pending')
        if status_filter:
            qs = qs.filter(status=status_filter)

        tech = request.query_params.get('technology')
        if tech:
            qs = qs.filter(technology__iexact=tech)

        qs = qs.order_by('-drafted_at')[:100]
        serializer = ApprovedDomainBriefListSerializer(qs, many=True)
        return Response(
            {'count': len(serializer.data), 'results': serializer.data},
            status=status.HTTP_200_OK,
        )


class BriefDetailView(APIView):
    """
    GET /api/viva/briefs/<brief_id>/

    Full brief including the JSON body the examiner is reviewing.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, brief_id):
        from viva_evaluator.models import ApprovedDomainBrief
        from viva_evaluator.serializers import ApprovedDomainBriefDetailSerializer
        try:
            brief = ApprovedDomainBrief.objects.get(id=brief_id)
        except ApprovedDomainBrief.DoesNotExist:
            return Response(
                {"error": "Brief not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(
            ApprovedDomainBriefDetailSerializer(brief).data,
            status=status.HTTP_200_OK,
        )


class BriefApproveView(APIView):
    """
    POST /api/viva/briefs/<brief_id>/approve/

    Examiner approves a pending T2 draft → status=ACTIVE, tier=1.
    Optional body:
        {
            "scope": "examiner" | "department",
            "tech_version": "..."
        }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, brief_id):
        from viva_evaluator.models import ApprovedDomainBrief
        from viva_evaluator.serializers import ApprovedDomainBriefDetailSerializer
        from django.utils import timezone

        try:
            brief = ApprovedDomainBrief.objects.get(id=brief_id)
        except ApprovedDomainBrief.DoesNotExist:
            return Response(
                {"error": "Brief not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Examiner check
        try:
            examiner = request.user.examiner_profile
        except Exception:
            return Response(
                {"error": "Only examiners can approve briefs."},
                status=status.HTTP_403_FORBIDDEN,
            )

        scope = request.data.get('scope', ApprovedDomainBrief.Scope.EXAMINER)
        if scope not in dict(ApprovedDomainBrief.Scope.choices):
            return Response(
                {"error": f"Invalid scope: {scope}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        tech_version = request.data.get('tech_version')
        if tech_version is not None:
            brief.tech_version = tech_version

        brief.status = ApprovedDomainBrief.Status.ACTIVE
        brief.tier = 1
        brief.scope = scope
        brief.approved_by = examiner
        brief.approved_at = timezone.now()
        brief.last_verified_at = timezone.now()
        brief.save()

        return Response(
            ApprovedDomainBriefDetailSerializer(brief).data,
            status=status.HTTP_200_OK,
        )


class BriefEditView(APIView):
    """
    PATCH /api/viva/briefs/<brief_id>/edit/

    Examiner edits the brief body before approving. Status stays PENDING
    until they call /approve/ explicitly.
    """
    permission_classes = [IsAuthenticated]

    def patch(self, request, brief_id):
        from viva_evaluator.models import ApprovedDomainBrief
        from viva_evaluator.serializers import (
            ApprovedDomainBriefEditSerializer,
            ApprovedDomainBriefDetailSerializer,
        )
        try:
            brief = ApprovedDomainBrief.objects.get(id=brief_id)
        except ApprovedDomainBrief.DoesNotExist:
            return Response(
                {"error": "Brief not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            request.user.examiner_profile
        except Exception:
            return Response(
                {"error": "Only examiners can edit briefs."},
                status=status.HTTP_403_FORBIDDEN,
            )

        serializer = ApprovedDomainBriefEditSerializer(
            brief, data=request.data, partial=True,
        )
        if not serializer.is_valid():
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST,
            )
        serializer.save()
        return Response(
            ApprovedDomainBriefDetailSerializer(brief).data,
            status=status.HTTP_200_OK,
        )


class BriefRejectView(APIView):
    """
    POST /api/viva/briefs/<brief_id>/reject/

    Examiner rejects the draft → status=ARCHIVED. The system will not
    re-draft a brief for the same technology unless this is explicitly
    cleared (admin operation).
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, brief_id):
        from viva_evaluator.models import ApprovedDomainBrief
        try:
            brief = ApprovedDomainBrief.objects.get(id=brief_id)
        except ApprovedDomainBrief.DoesNotExist:
            return Response(
                {"error": "Brief not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            request.user.examiner_profile
        except Exception:
            return Response(
                {"error": "Only examiners can reject briefs."},
                status=status.HTTP_403_FORBIDDEN,
            )

        brief.status = ApprovedDomainBrief.Status.ARCHIVED
        brief.save(update_fields=['status'])
        return Response(
            {"message": "Brief archived.", "id": str(brief.id)},
            status=status.HTTP_200_OK,
        )



# =============================================================================
# WEEK 7 — Ablation experiment endpoint
# =============================================================================

class AblationRunView(APIView):
    """
    POST /api/viva/ablation/run/

    Generate the same question under multiple ablation conditions for the
    dissertation evaluation chapter. Examiners use this to compare the
    full system against ablated variants on identical inputs.

    Body:
        {
            "submission_id":         "<uuid>",
            "criterion_name":        "...",
            "criterion_description": "...",
            "last_answer":           "...",            (optional)
            "previous_question":     "...",            (optional)
            "difficulty":            "easy|medium|hard", (optional, default medium)
            "conditions": [                            (optional)
                {},
                {"disable_anchoring": true},
                {"disable_kg": true}
            ]
        }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.models import ProjectSubmission
        from viva_evaluator.services.ablation import run_ablation_set

        submission_id = request.data.get('submission_id')
        criterion_name = request.data.get('criterion_name')
        criterion_description = request.data.get('criterion_description', '')
        last_answer = request.data.get('last_answer', '') or ''
        previous_question = request.data.get('previous_question')
        difficulty = request.data.get('difficulty', 'medium')
        conditions = request.data.get('conditions')

        if not submission_id or not criterion_name:
            return Response(
                {"error": "submission_id and criterion_name are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            submission = ProjectSubmission.objects.get(id=submission_id)
        except ProjectSubmission.DoesNotExist:
            return Response(
                {"error": "Submission not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        runs = run_ablation_set(
            submission=submission,
            criterion_name=criterion_name,
            criterion_description=criterion_description,
            last_answer=last_answer,
            previous_question=previous_question,
            difficulty=difficulty,
            conditions=conditions,
        )

        return Response(
            {
                "submission_id": str(submission.id),
                "criterion_name": criterion_name,
                "runs": runs,
            },
            status=status.HTTP_200_OK,
        )
