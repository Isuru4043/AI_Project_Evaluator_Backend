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
