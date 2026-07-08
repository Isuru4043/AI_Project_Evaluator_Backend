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
        if ext not in ['pdf', 'docx', 'md', 'markdown', 'txt']:
            return Response(
                {"error": "Only PDF, DOCX, MD and TXT files are accepted."},
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
