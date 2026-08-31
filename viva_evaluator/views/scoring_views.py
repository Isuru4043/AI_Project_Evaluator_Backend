import logging

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from urllib.request import urlopen

from core.models import ProjectSubmission
from viva_evaluator.permissions import IsAssignedSessionExaminer
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

logger = logging.getLogger(__name__)


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
    permission_classes = [IsAuthenticated, IsAssignedSessionExaminer]

    def post(self, request, session_id):
        from core.models import (
            EvaluationSession, RubricCriteria,
            FinalScore, SessionSummaryReport, ExaminerProfile, StudentProfile
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

        # Expecting a dictionary: { "speaker_id": { "scores": [...], "overall_feedback": "...", "grade": "..." } }
        # Or backward compatibility: { "scores": [...] } -> treat as "group"
        payload = request.data
        if 'scores' in payload:
            payload = {'group': payload}

        if not payload:
            return Response(
                {"error": "Payload is empty."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        all_saved_scores = {}

        for speaker_id, speaker_data in payload.items():
            scores_data = speaker_data.get('scores', [])
            overall_feedback = speaker_data.get('overall_feedback', '')
            grade = speaker_data.get('grade', '')

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

                # Get AI recommended score for this criterion using the ScoringService
                ai_score = None
                try:
                    # In Phase 2, AIScoreRecommendation is dead. We use the actual BKT/pipeline answers.
                    from viva_evaluator.services.scoring_service import ScoringService
                    student_obj = None
                    if speaker_id != 'group':
                        from core.models import StudentProfile
                        student_obj = StudentProfile.objects.filter(id=speaker_id).first()
                        
                    scoring_result = ScoringService.aggregate_student_score(session, student_obj)
                    if str(criteria.id) in scoring_result['per_criteria']:
                        # The scoring service returns out of max_score, so normalize back to 10
                        max_s = float(criteria.max_score)
                        earned = float(scoring_result['per_criteria'][str(criteria.id)])
                        ai_score = (earned / max_s) * 10.0 if max_s > 0 else 0
                except Exception:
                    pass

                student = None
                if speaker_id != 'group':
                    try:
                        student = StudentProfile.objects.get(id=speaker_id)
                    except (StudentProfile.DoesNotExist, ValueError):
                        pass

                # Save final score
                final_score, _ = FinalScore.objects.update_or_create(
                    session=session,
                    criteria=criteria,
                    examiner=examiner,
                    student=student,
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

            # Calculate actual final score and grade from backend source of truth
            from viva_evaluator.services.scoring_service import ScoringService
            scoring_result = ScoringService.aggregate_student_score(session, student)
            
            total_final_score = scoring_result['percentage']
            grade = scoring_result['grade']

            summary, _ = SessionSummaryReport.objects.update_or_create(
                session=session,
                student=student,
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
            
            all_saved_scores[speaker_id] = {
                "grade": grade,
                "total_final_score": total_final_score,
                "total_ai_score": total_ai_score,
                "scores": saved_scores,
            }

        return Response(
            {
                "message": "Final scores submitted successfully.",
                "session_id": session_id,
                "student_reports": all_saved_scores,
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
    permission_classes = [IsAuthenticated, IsAssignedSessionExaminer]

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
        from viva_evaluator.services.session_reports import (
            refresh_draft_summary_reports,
        )
        try:
            refresh_draft_summary_reports(session)
        except Exception:
            logger.exception(
                "Draft report refresh failed after score override for answer %s",
                answer.id,
            )

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
    permission_classes = [IsAuthenticated, IsAssignedSessionExaminer]

    def post(self, request, session_id):
        from core.models import EvaluationSession, SessionSummaryReport
        from django.db import transaction
        from django.utils import timezone
        from viva_evaluator.services.scoring_service import ScoringService
        from viva_evaluator.services.session_reports import (
            ensure_participant_reports,
            unresolved_individual_answers,
        )

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

        with transaction.atomic():
            # Lock the session so two approval clicks cannot publish a partial
            # mixture of old and newly reconciled participant scores.
            session = EvaluationSession.objects.select_for_update().get(id=session_id)
            unresolved = unresolved_individual_answers(session)
            if unresolved:
                return Response(
                    {
                        "error": (
                            "Resolve the answerer for every individual question "
                            "before approving scores."
                        ),
                        "code": "unresolved_individual_attribution",
                        "unresolved_count": len(unresolved),
                        "answers": [
                            {
                                "answer_id": str(answer.id),
                                "question_id": str(answer.question_id),
                                "question_order": answer.question.question_order,
                            }
                            for answer in unresolved
                        ],
                    },
                    status=status.HTTP_409_CONFLICT,
                )
            ensure_participant_reports(session)
            # Lock only SessionSummaryReport rows. ``student`` is nullable, so
            # select_related() makes PostgreSQL generate a LEFT OUTER JOIN;
            # PostgreSQL rejects FOR UPDATE on the nullable side of that join.
            # Loading each student lazily keeps the approval atomic without
            # attempting to lock StudentProfile rows.
            reports = session.summary_reports.select_for_update()
            now = timezone.now()

            for report in reports:
                if report.scores_status == SessionSummaryReport.ScoresStatus.APPROVED:
                    continue

                scoring_result = ScoringService.aggregate_student_score(
                    session, report.student,
                )
                ai_result = ScoringService.aggregate_student_score(
                    session,
                    report.student,
                    use_examiner_overrides=False,
                )
                report.total_ai_score = ai_result['percentage']
                report.total_final_score = scoring_result['percentage']
                report.grade = scoring_result['grade']
                report.scores_status = SessionSummaryReport.ScoresStatus.APPROVED
                report.scores_approved_at = now
                report.is_published = True
                report.published_at = now
                report.finalized_by = examiner
                report.save(update_fields=[
                    'total_ai_score', 'total_final_score', 'grade',
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

