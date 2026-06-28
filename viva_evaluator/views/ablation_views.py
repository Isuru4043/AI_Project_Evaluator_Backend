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
