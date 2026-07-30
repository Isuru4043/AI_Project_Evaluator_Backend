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


# =============================================================================
# 3. MODULE MATERIALS
# =============================================================================

class ModuleMaterialListView(APIView):
    """
    GET /api/viva/projects/<project_id>/module-materials/
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, project_id):
        from core.models import ModuleMaterial
        from viva_evaluator.serializers import ModuleMaterialSerializer
        
        materials = ModuleMaterial.objects.filter(project_id=project_id)
        return Response(
            ModuleMaterialSerializer(materials, many=True).data,
            status=status.HTTP_200_OK
        )


class ModuleMaterialUploadView(APIView):
    """
    POST /api/viva/projects/<project_id>/module-materials/upload/
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, project_id):
        from core.models import Project, ModuleMaterial
        from viva_evaluator.serializers import ModuleMaterialSerializer
        import uuid
        import os
        from django.core.files.storage import default_storage
        from django_q.tasks import async_task
        
        try:
            project = Project.objects.get(id=project_id)
        except Project.DoesNotExist:
            return Response({"error": "Project not found"}, status=status.HTTP_404_NOT_FOUND)

        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Validate extension
        ext = os.path.splitext(file_obj.name)[1].lower()
        if ext not in ['.pdf', '.pptx', '.docx']:
            return Response({"error": "Only PDF, PPTX, and DOCX files are supported"}, status=status.HTTP_400_BAD_REQUEST)

        # Save file
        filename = f"module_materials/{project_id}/{uuid.uuid4()}{ext}"
        saved_path = default_storage.save(filename, file_obj)
        file_url = default_storage.url(saved_path)

        # Create record
        material = ModuleMaterial.objects.create(
            project=project,
            file_url=file_url,
            original_filename=file_obj.name,
            processing_status=ModuleMaterial.ProcessingStatus.PENDING
        )

        # Trigger background task
        try:
            async_task('viva_evaluator.tasks.process_module_material_task', material.id)
        except Exception as e:
            # If Q cluster isn't running or something fails synchronously
            pass

        return Response(
            ModuleMaterialSerializer(material).data,
            status=status.HTTP_201_CREATED
        )

class ModuleMaterialDeleteView(APIView):
    """
    DELETE /api/viva/projects/<project_id>/module-materials/<material_id>/
    """
    permission_classes = [IsAuthenticated]

    def delete(self, request, project_id, material_id):
        from core.models import ModuleMaterial
        from django.core.files.storage import default_storage
        import urllib.parse
        
        try:
            material = ModuleMaterial.objects.get(id=material_id, project_id=project_id)
        except ModuleMaterial.DoesNotExist:
            return Response({"error": "Material not found"}, status=status.HTTP_404_NOT_FOUND)

        # 1. Delete original file from storage
        if material.file_url:
            parsed = urllib.parse.urlparse(material.file_url)
            path = parsed.path
            if 'module_materials/' in path:
                path = path[path.index('module_materials/'):]
            else:
                path = material.file_url.replace('/media/', '')
            
            if default_storage.exists(path):
                default_storage.delete(path)

        # 2. Delete vectors/chunks if they exist
        chunks_path = f"module_materials/{material.id}_chunks.json"
        faiss_path = f"module_materials/{material.id}_faiss.bin"
        
        if default_storage.exists(chunks_path):
            default_storage.delete(chunks_path)
        if default_storage.exists(faiss_path):
            default_storage.delete(faiss_path)

        # 3. Delete DB record
        material.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)
