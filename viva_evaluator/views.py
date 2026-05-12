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

    Student uploads their report file here.
    Text is extracted immediately and stored in the DB.
    No separate indexing step needed.
    """
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = SubmissionUploadSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            submission = serializer.save()
            index_status = submission.index_status

            # Extract text immediately after upload
            from core.utils.document_parser import extract_text_from_bytes

            index_status.status = SubmissionIndexStatus.IndexStatus.PROCESSING
            index_status.save()

            # Read file content directly (works with both local and cloud storage)
            with index_status.report_file.open('rb') as f:
                file_content = f.read()
            extracted_text = extract_text_from_bytes(
                file_content, index_status.report_file.name
            )

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
            # If something went wrong during extraction, mark as failed
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
            )

            # Save the question to DB
            question_order = context['total_questions_asked'] + 1
            question = VivaQuestion.objects.create(
                session=session,
                question_text=question_data['question_text'],
                blooms_level=question_data.get('blooms_level', 'Understand'),
                question_order=question_order,
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
                )
            else:
                next_question_data = generate_first_question(
                    report_text=report_text,
                    criteria_name=next_criterion['name'],
                    criteria_description=next_criterion['description'],
                    difficulty=next_difficulty,
                )

            # Save next question
            next_question = VivaQuestion.objects.create(
                session=session,
                question_text=next_question_data['question_text'],
                blooms_level=next_question_data.get('blooms_level', 'Understand'),
                question_order=total_asked + 1,
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