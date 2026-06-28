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
