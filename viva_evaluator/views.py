from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone

from core.models import ProjectSubmission
from viva_evaluator.models import SubmissionIndexStatus
from viva_evaluator.serializers import (
    SubmissionUploadSerializer,
    SubmissionIndexStatusSerializer,
)


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
        from core.utils.document_parser import extract_text_from_file

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

            file_path = index_status.report_file.path
            extracted_text = extract_text_from_file(file_path)

            index_status.extracted_text = extracted_text
            index_status.status = SubmissionIndexStatus.IndexStatus.READY
            index_status.processed_at = timezone.now()
            index_status.save()

            return Response(
                {
                    "message": "Report uploaded and processed successfully.",
                    "submission_id": str(submission.id),
                    "status": "ready",
                    "characters_extracted": len(extracted_text),
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
            from viva_evaluator.services.question_generator import generate_first_question
            from viva_evaluator.models import VivaQuestionExtension
            from core.models import VivaQuestion

            session = EvaluationSession.objects.get(id=session_id)

            # Make sure submission is ready
            try:
                index_status = session.submission.index_status
                if index_status.status != SubmissionIndexStatus.IndexStatus.READY:
                    return Response(
                        {"error": "Submission is not ready yet. Please wait for processing to complete."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                report_text = index_status.extracted_text or ""
            except Exception:
                return Response(
                    {"error": "No processed submission found for this session."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Get session context to find first criterion
            context = get_session_context(session_id)

            if context['is_complete']:
                return Response(
                    {"error": "This session is already complete."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            next_criterion = context['next_criterion']

            # Generate the first question
            question_data = generate_first_question(
                report_text=report_text,
                criteria_name=next_criterion['name'],
                criteria_description=next_criterion['description'],
                difficulty='medium',
                question_hints=next_criterion.get('hints', []),
            )

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
                },
                status=status.HTTP_200_OK,
            )

        except EvaluationSession.DoesNotExist:
            return Response(
                {"error": "Session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
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

        if not question_id or not answer_text:
            return Response(
                {"error": "question_id and answer_text are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            from core.models import (
                EvaluationSession, VivaQuestion,
                VivaAnswer, RubricCriteria
            )
            from viva_evaluator.models import VivaAnswerExtension, VivaQuestionExtension
            from viva_evaluator.services.answer_evaluator import evaluate_answer
            from viva_evaluator.services.session_manager import get_session_context
            from viva_evaluator.services.question_generator import (
                generate_first_question, generate_followup_question
            )

            session = EvaluationSession.objects.get(id=session_id)
            question = VivaQuestion.objects.get(id=question_id, session=session)

            # Get report text
            report_text = session.submission.index_status.extracted_text or ""

            # Get question extension for criterion info
            try:
                q_ext = question.extension
                criteria = q_ext.criteria
                criteria_name = criteria.criteria_name
                criteria_description = criteria.description or ""
                current_difficulty = q_ext.difficulty_level
            except Exception:
                criteria = None
                criteria_name = "General"
                criteria_description = ""
                current_difficulty = "medium"

            # Evaluate the answer
            evaluation = evaluate_answer(
                question_text=question.question_text,
                student_answer=answer_text,
                criteria_name=criteria_name,
                criteria_description=criteria_description,
                report_text=report_text,
                current_difficulty=current_difficulty,
            )

            # Get student profile
            student_profile = session.student

            # Save the answer
            answer = VivaAnswer.objects.create(
                question=question,
                student=student_profile,
                transcribed_answer=answer_text,
                ai_answer_score=evaluation.get('llm_score', 0),
            )

            # Save answer extension
            VivaAnswerExtension.objects.create(
                answer=answer,
                llm_score=evaluation.get('llm_score', 0),
                llm_reasoning=evaluation.get('llm_reasoning', ''),
                next_difficulty_signal=evaluation.get('next_difficulty', 'same'),
            )

            # Get updated session context
            context = get_session_context(session_id)

            # Check if session is complete
            if context['is_complete']:
                session.status = EvaluationSession.Status.COMPLETED
                session.save()

                return Response(
                    {
                        "answer_saved": True,
                        "score": evaluation.get('llm_score'),
                        "reasoning": evaluation.get('llm_reasoning'),
                        "strengths": evaluation.get('strengths', ''),
                        "gaps": evaluation.get('gaps', ''),
                        "session_complete": True,
                        "message": "All criteria have been covered. Session is complete.",
                    },
                    status=status.HTTP_200_OK,
                )

            # Generate next question
            next_criterion = context['next_criterion']
            next_difficulty = context['current_difficulty']
            question_count = context['current_criterion_question_count']
            total_asked = context['total_questions_asked']

            # Decide: follow-up on same criterion or first question of new criterion
            if question_count > 0:
                next_question_data = generate_followup_question(
                    report_text=report_text,
                    criteria_name=next_criterion['name'],
                    criteria_description=next_criterion['description'],
                    previous_question=question.question_text,
                    previous_answer=answer_text,
                    next_difficulty=next_difficulty,
                    question_number=question_count + 1,
                    question_hints=next_criterion.get('hints', [])
                )
            else:
                next_question_data = generate_first_question(
                    report_text=report_text,
                    criteria_name=next_criterion['name'],
                    criteria_description=next_criterion['description'],
                    difficulty=next_difficulty,
                    question_hints=next_criterion.get('hints', []),
                )

            # Save next question
            next_question = VivaQuestion.objects.create(
                session=session,
                question_text=next_question_data['question_text'],
                blooms_level=next_question_data.get('blooms_level', 'Understand'),
                question_order=total_asked + 1,
                question_source='ai',
            )

            # Save next question extension
            next_criterion_obj = RubricCriteria.objects.get(id=next_criterion['id'])
            VivaQuestionExtension.objects.create(
                question=next_question,
                criteria=next_criterion_obj,
                difficulty_level=next_question_data.get('difficulty', 'medium'),
            )

            return Response(
                {
                    "answer_saved": True,
                    "score": evaluation.get('llm_score'),
                    "reasoning": evaluation.get('llm_reasoning'),
                    "strengths": evaluation.get('strengths', ''),
                    "gaps": evaluation.get('gaps', ''),
                    "session_complete": False,
                    "next_question": {
                        "question_id": str(next_question.id),
                        "question_text": next_question.question_text,
                        "blooms_level": next_question.blooms_level,
                        "difficulty": next_question_data.get('difficulty', 'medium'),
                        "criterion": next_criterion['name'],
                        "question_number": total_asked + 1,
                    },
                },
                status=status.HTTP_200_OK,
            )

        except EvaluationSession.DoesNotExist:
            return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
        except VivaQuestion.DoesNotExist:
            return Response({"error": "Question not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SessionReportView(APIView):
    """
    GET /api/viva/sessions/<session_id>/report/

    Returns the full XAI report for a completed session.
    The examiner reviews this before making final grade decisions.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        try:
            from core.models import EvaluationSession
            from viva_evaluator.services.session_manager import generate_session_report

            session = EvaluationSession.objects.get(id=session_id)

            if session.status != EvaluationSession.Status.COMPLETED:
                return Response(
                    {"error": "Session is not completed yet."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            report = generate_session_report(str(session_id))

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