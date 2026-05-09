from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import CodeSubmission, GroupMember, ProjectSubmission, User
from .serializers import (
    CodeSubmissionCreateSerializer,
    CodeSubmissionStatusSerializer,
    CodeSubmissionQuestionsSerializer,
    CodeSubmissionSonarSummarySerializer,
)
from .services.analysis_runner import enqueue_code_analysis
from .services.analysis_service import CodeAnalysisService


def _ensure_submission_access(user, project_submission):
    if user.role == User.Role.EXAMINER:
        return

    if project_submission.student and project_submission.student.user_id == user.id:
        return

    if project_submission.group:
        has_membership = GroupMember.objects.filter(
            group=project_submission.group,
            student__user=user,
        ).exists()
        if has_membership:
            return

    raise PermissionDenied("You do not have access to this submission.")


class CodeSubmissionCreateView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        serializer = CodeSubmissionCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        project_submission = get_object_or_404(
            ProjectSubmission,
            id=serializer.validated_data["project_submission_id"],
        )
        _ensure_submission_access(request.user, project_submission)

        code_submission = CodeSubmission.objects.create(
            project_submission=project_submission,
            source_type=serializer.validated_data["source_type"],
            github_url=serializer.validated_data.get("github_url"),
            zip_file=serializer.validated_data.get("zip_file"),
            build_command=serializer.validated_data.get("build_command"),
        )

        if getattr(settings, "CODE_ANALYSIS_ASYNC", True):
            enqueue_code_analysis(code_submission.id)
        else:
            CodeAnalysisService().analyze_submission(code_submission.id)

        return Response(
            {
                "success": True,
                "message": "Code submission received. Analysis started.",
                "data": {
                    "code_submission_id": str(code_submission.id),
                },
            },
            status=status.HTTP_202_ACCEPTED,
        )


class CodeSubmissionStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, code_submission_id):
        code_submission = get_object_or_404(CodeSubmission, id=code_submission_id)
        _ensure_submission_access(request.user, code_submission.project_submission)

        code_submission = CodeAnalysisService().refresh_submission(code_submission.id)

        serializer = CodeSubmissionStatusSerializer(code_submission)
        return Response(
            {
                "success": True,
                "message": "Status retrieved.",
                "data": serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class CodeSubmissionSonarSummaryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, code_submission_id):
        code_submission = get_object_or_404(CodeSubmission, id=code_submission_id)
        _ensure_submission_access(request.user, code_submission.project_submission)

        code_submission = CodeAnalysisService().refresh_submission(code_submission.id)

        serializer = CodeSubmissionSonarSummarySerializer(code_submission)
        return Response(
            serializer.data,
            status=status.HTTP_200_OK,
        )


class CodeSubmissionQuestionsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, code_submission_id):
        code_submission = get_object_or_404(CodeSubmission, id=code_submission_id)
        _ensure_submission_access(request.user, code_submission.project_submission)

        code_submission = CodeAnalysisService().refresh_submission(code_submission.id)

        serializer = CodeSubmissionQuestionsSerializer(code_submission)
        return Response(
            {
                "success": True,
                "message": "Questions retrieved.",
                "data": serializer.data,
            },
            status=status.HTTP_200_OK,
        )
