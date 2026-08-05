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

from viva_evaluator.views._helpers import (
    _resolve_session_submission,
    _difficulty_signal_from_score,
    _get_or_create_index_status,
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


class PatchAnswerScoreView(APIView):
    """
    PATCH /api/viva/sessions/<session_id>/answers/<answer_id>/score/

    Allows an examiner to override the AI-generated score for a single
    VivaAnswer. The override is stored alongside the original AI score so
    both values remain visible in the report for transparency.

    Request body:
    {
        "override_score": 7.5,
        "override_note": "Student showed understanding but missed edge cases."
    }
    """
    permission_classes = [IsAuthenticated]

    def patch(self, request, session_id, answer_id):
        from core.models import EvaluationSession, VivaAnswer

        # Verify the session exists
        try:
            session = EvaluationSession.objects.get(id=session_id)
        except EvaluationSession.DoesNotExist:
            return Response(
                {"error": "Session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Examiner-only
        if not hasattr(request.user, 'examiner_profile'):
            return Response(
                {"error": "Only examiners can edit scores."},
                status=status.HTTP_403_FORBIDDEN,
            )

        # Fetch the answer and make sure it belongs to this session
        try:
            answer = VivaAnswer.objects.get(
                id=answer_id,
                question__session=session,
            )
        except VivaAnswer.DoesNotExist:
            return Response(
                {"error": "Answer not found in this session."},
                status=status.HTTP_404_NOT_FOUND,
            )

        override_score = request.data.get('override_score')
        override_note  = request.data.get('override_note', '')

        if override_score is None:
            return Response(
                {"error": "override_score is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate range
        try:
            override_score = float(override_score)
            if override_score < 0 or override_score > 10:
                raise ValueError
        except (ValueError, TypeError):
            return Response(
                {"error": "override_score must be a number between 0 and 10."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        answer.examiner_override_score = override_score
        answer.examiner_override_note  = override_note
        answer.save(update_fields=['examiner_override_score', 'examiner_override_note'])

        return Response(
            {
                "answer_id": str(answer.id),
                "ai_answer_score": float(answer.ai_answer_score) if answer.ai_answer_score is not None else None,
                "examiner_override_score": override_score,
                "examiner_override_note": override_note,
                "message": "Score updated successfully.",
            },
            status=status.HTTP_200_OK,
        )


class ApproveSessionScoresView(APIView):
    """
    POST /api/viva/sessions/<session_id>/approve-scores/

    Examiner approves all scores for a session, moving the report from
    DRAFT → APPROVED. This action is irreversible from the student's
    perspective (scores are now considered final).

    No request body required.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        from core.models import EvaluationSession, SessionSummaryReport
        from django.utils import timezone

        # Verify session
        try:
            session = EvaluationSession.objects.get(id=session_id)
        except EvaluationSession.DoesNotExist:
            return Response(
                {"error": "Session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Examiner-only
        if not hasattr(request.user, 'examiner_profile'):
            return Response(
                {"error": "Only examiners can approve scores."},
                status=status.HTTP_403_FORBIDDEN,
            )

        examiner = request.user.examiner_profile

        # Get or create the summary report
        report, _ = SessionSummaryReport.objects.get_or_create(
            session=session,
            defaults={
                'finalized_by': examiner,
            }
        )

        if report.scores_status == SessionSummaryReport.ScoresStatus.APPROVED:
            return Response(
                {
                    "message": "Scores are already approved.",
                    "scores_status": "approved",
                    "scores_approved_at": report.scores_approved_at,
                },
                status=status.HTTP_200_OK,
            )

        now = timezone.now()
        
        # Recalculate the final score based on overrides
        total_possible = 0
        total_earned = 0
        
        for q in session.viva_questions.all():
            answer = q.answers.order_by('-answered_at').first()
            if not answer:
                continue
            
            total_possible += 10
            if answer.examiner_override_score is not None:
                total_earned += float(answer.examiner_override_score)
            else:
                try:
                    if hasattr(answer, 'extension') and answer.extension.llm_score is not None:
                        total_earned += float(answer.extension.llm_score)
                except Exception:
                    pass
                    
        final_percentage = 0
        if total_possible > 0:
            final_percentage = round((total_earned / total_possible) * 100, 2)
            
        grade = "F"
        if final_percentage >= 75: grade = "A"
        elif final_percentage >= 65: grade = "B"
        elif final_percentage >= 50: grade = "C"
        elif final_percentage >= 35: grade = "S"

        report.total_final_score = final_percentage
        report.grade = grade
        report.scores_status     = SessionSummaryReport.ScoresStatus.APPROVED
        report.scores_approved_at = now
        report.is_published      = True
        report.published_at      = now
        report.finalized_by      = examiner
        report.save(update_fields=[
            'total_final_score', 'grade',
            'scores_status', 'scores_approved_at',
            'is_published', 'published_at', 'finalized_by',
        ])

        return Response(
            {
                "message": "Session scores have been approved and finalized.",
                "session_id": str(session_id),
                "scores_status": "approved",
                "scores_approved_at": now.isoformat(),
                "approved_by": examiner.user.full_name,
            },
            status=status.HTTP_200_OK,
        )

