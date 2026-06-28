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
