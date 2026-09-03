from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from urllib.request import urlopen
import logging

from core.models import ProjectSubmission
from projects.permissions import IsExaminer
from viva_evaluator.permissions import IsAssignedProjectExaminer
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

MODULE_MATERIAL_MAX_BYTES = 50 * 1024 * 1024
MODULE_MATERIAL_EXTENSIONS = {'.pdf', '.pptx', '.docx'}


def _assign_creator_as_lead(project, user):
    from core.models import ProjectExaminer

    ProjectExaminer.objects.get_or_create(
        project=project,
        examiner=user.examiner_profile,
        defaults={'role_in_project': ProjectExaminer.RoleInProject.LEAD},
    )


class ProjectCreateView(APIView):
    """
    POST /api/viva/projects/

    Examiner creates a project with full rubric in one call.
    Returns warnings if weights do not add up to 100%.
    """
    permission_classes = [IsAuthenticated, IsExaminer]

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
            _assign_creator_as_lead(project, request.user)
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
    permission_classes = [IsAuthenticated, IsAssignedProjectExaminer]

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
    permission_classes = [IsAuthenticated, IsExaminer]

    def get(self, request):
        from core.models import Project
        from viva_evaluator.serializers import ProjectDetailSerializer
        projects = Project.objects.filter(
            project_examiners__examiner=request.user.examiner_profile,
        ).distinct().order_by('-created_at')
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
            _assign_creator_as_lead(project, request.user)
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
    permission_classes = [IsAuthenticated, IsExaminer]

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
    permission_classes = [IsAuthenticated, IsAssignedProjectExaminer]

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
    permission_classes = [IsAuthenticated, IsAssignedProjectExaminer]
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

        original_filename = os.path.basename(file_obj.name)
        ext = os.path.splitext(original_filename)[1].lower()
        if ext not in MODULE_MATERIAL_EXTENSIONS:
            return Response({"error": "Only PDF, PPTX, and DOCX files are supported"}, status=status.HTTP_400_BAD_REQUEST)
        if file_obj.size > MODULE_MATERIAL_MAX_BYTES:
            return Response(
                {"error": "The file is larger than the 50 MB upload limit."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        filename = f"module_materials/{project_id}/{uuid.uuid4()}{ext}"
        saved_path = None
        try:
            saved_path = default_storage.save(filename, file_obj)
            file_url = default_storage.url(saved_path)
        except Exception:
            logger.exception(
                "Failed to store module material project=%s file=%s",
                project_id,
                original_filename,
            )
            if saved_path:
                try:
                    default_storage.delete(saved_path)
                except Exception:
                    logger.exception(
                        "Failed to clean up incomplete module material %s",
                        saved_path,
                    )
            return Response(
                {
                    "error": "The module material could not be stored. Please try again.",
                    "code": "module_material_storage_failed",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        try:
            material = ModuleMaterial.objects.create(
                project=project,
                file_url=file_url,
                original_filename=original_filename[:255],
                processing_status=ModuleMaterial.ProcessingStatus.PENDING,
            )
        except Exception:
            logger.exception(
                "Failed to create module material record project=%s file=%s",
                project_id,
                original_filename,
            )
            try:
                default_storage.delete(saved_path)
            except Exception:
                logger.exception(
                    "Failed to clean up orphaned module material %s",
                    saved_path,
                )
            return Response(
                {
                    "error": "The upload was stored but could not be registered. Please try again.",
                    "code": "module_material_record_failed",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Trigger background task
        try:
            async_task('viva_evaluator.tasks.process_module_material_task', material.id)
        except Exception:
            # The upload is valid even when the worker is temporarily offline.
            # It remains pending and can be picked up/retried by a worker.
            logger.exception(
                "Failed to enqueue module material indexing material=%s",
                material.id,
            )

        return Response(
            ModuleMaterialSerializer(material).data,
            status=status.HTTP_201_CREATED
        )

class ModuleMaterialDeleteView(APIView):
    """
    DELETE /api/viva/projects/<project_id>/module-materials/<material_id>/
    """
    permission_classes = [IsAuthenticated, IsAssignedProjectExaminer]

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
