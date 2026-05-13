"""
Views for Project Management, Examiner Assignment,
Student Enrollment, and Submissions (Features 1-3).
"""

from django.db import transaction
from django.utils import timezone
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import (
    CodeSubmission, EvaluationSession, ExaminerProfile, GroupMember, Project,
    ProjectExaminer, ProjectSubmission, StudentGroup, StudentProfile, User,
)
from code_analysis.services.analysis_runner import enqueue_code_analysis
from projects.permissions import IsExaminer, IsExaminerOrStudent, IsProjectLead, IsStudent
from projects.serializers import (
    AddExaminerSerializer, AvailableProjectSerializer, MyEnrollmentSerializer,
    ProjectCreateSerializer, ProjectDetailSerializer, ProjectExaminerSerializer,
    ProjectSerializer, ProjectSubmissionSerializer, ProjectUpdateSerializer,
    RemoveExaminerSerializer, StudentEnrollSerializer, SubmitProjectSerializer,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_examiner_profile(user):
    try:
        return user.examiner_profile
    except ExaminerProfile.DoesNotExist:
        return None


def _get_student_profile(user):
    try:
        return user.student_profile
    except StudentProfile.DoesNotExist:
        return None


def _is_lead(examiner_profile, project):
    return ProjectExaminer.objects.filter(
        project=project, examiner=examiner_profile, role_in_project='lead',
    ).exists()


def _is_assigned(examiner_profile, project):
    return ProjectExaminer.objects.filter(
        project=project, examiner=examiner_profile,
    ).exists()


def _err(msg, errors=None, code=400):
    return Response(
        {'success': False, 'message': msg, 'errors': errors or {}},
        status=code,
    )


def _ok(msg, data=None, code=200):
    return Response(
        {'success': True, 'message': msg, 'data': data},
        status=code,
    )


def _500(e):
    return Response(
        {'success': False, 'message': f'An unexpected error occurred: {str(e)}', 'errors': {}},
        status=500,
    )


# =============================================================================
# FEATURE 1 — PROJECT MANAGEMENT
# =============================================================================

class ProjectCreateView(APIView):
    """POST /api/projects/create/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request):
        try:
            ser = ProjectCreateSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            ep = _get_examiner_profile(request.user)
            if not ep:
                return _err('Examiner profile not found.', code=404)

            with transaction.atomic():
                project = Project.objects.create(
                    project_name=ser.validated_data['project_name'],
                    description=ser.validated_data.get('description'),
                    is_group_project=ser.validated_data.get('is_group_project', False),
                    submission_deadline=ser.validated_data.get('submission_deadline'),
                    academic_year=ser.validated_data.get('academic_year'),
                    status='draft',
                )
                ProjectExaminer.objects.create(
                    project=project, examiner=ep, role_in_project='lead',
                )

            return _ok('Project created successfully.', ProjectSerializer(project).data, 201)
        except Exception as e:
            return _500(e)


class ProjectListView(APIView):
    """GET /api/projects/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def get(self, request):
        try:
            ep = _get_examiner_profile(request.user)
            if not ep:
                return _err('Examiner profile not found.', code=404)

            project_ids = ProjectExaminer.objects.filter(
                examiner=ep,
            ).values_list('project_id', flat=True)
            projects = Project.objects.filter(id__in=project_ids)
            return _ok('Projects retrieved.', ProjectSerializer(projects, many=True).data)
        except Exception as e:
            return _500(e)


class ProjectDetailView(APIView):
    """GET /api/projects/<project_id>/"""
    permission_classes = [IsAuthenticated, IsExaminerOrStudent]

    def get(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)
            return _ok('Project details retrieved.', ProjectDetailSerializer(project).data)
        except Exception as e:
            return _500(e)


class ProjectUpdateView(APIView):
    """PUT /api/projects/<project_id>/update/"""
    permission_classes = [IsAuthenticated, IsProjectLead]

    def put(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            ser = ProjectUpdateSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            data = ser.validated_data

            # Check is_group_project change
            if 'is_group_project' in data and data['is_group_project'] != project.is_group_project:
                has_students = project.submissions.filter(student__isnull=False).exists()
                has_groups = project.student_groups.exists()
                if has_students or has_groups:
                    return _err('Cannot change project type after students have enrolled.')

            for field in ('project_name', 'description', 'submission_deadline', 'academic_year', 'is_group_project'):
                if field in data:
                    setattr(project, field, data[field])
            project.save()

            return _ok('Project updated successfully.', ProjectSerializer(project).data)
        except Exception as e:
            return _500(e)


class ProjectActivateView(APIView):
    """PATCH /api/projects/<project_id>/activate/"""
    permission_classes = [IsAuthenticated, IsProjectLead]

    def patch(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)
            if project.status != 'draft':
                return _err(f'Project is already "{project.status}". Only draft projects can be activated.')
            project.status = 'active'
            project.save()
            return _ok('Project activated successfully.')
        except Exception as e:
            return _500(e)


# ── Examiner Assignment ──────────────────────────────────────────────────────

class AddExaminerView(APIView):
    """POST /api/projects/<project_id>/examiners/add/"""
    permission_classes = [IsAuthenticated, IsProjectLead]

    def post(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            ser = AddExaminerSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            target_user = User.objects.filter(
                id=ser.validated_data['examiner_user_id'], role='examiner',
            ).first()
            if not target_user:
                return _err('Examiner user not found.')

            target_ep = _get_examiner_profile(target_user)
            if not target_ep:
                return _err('Examiner profile not found for this user.')

            if ProjectExaminer.objects.filter(project=project, examiner=target_ep).exists():
                return _err('Examiner already assigned to this project.')

            ProjectExaminer.objects.create(
                project=project, examiner=target_ep,
                role_in_project=ser.validated_data.get('role_in_project', 'co-examiner'),
            )
            return _ok('Examiner added successfully.', code=201)
        except Exception as e:
            return _500(e)


class RemoveExaminerView(APIView):
    """DELETE /api/projects/<project_id>/examiners/remove/"""
    permission_classes = [IsAuthenticated, IsProjectLead]

    def delete(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            ser = RemoveExaminerSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            uid = ser.validated_data['examiner_user_id']
            if str(request.user.id) == str(uid):
                return _err('Lead examiner cannot remove themselves.')

            target_user = User.objects.filter(id=uid, role='examiner').first()
            if not target_user:
                return _err('Examiner user not found.')

            target_ep = _get_examiner_profile(target_user)
            pe = ProjectExaminer.objects.filter(project=project, examiner=target_ep).first()
            if not pe:
                return _err('Examiner is not assigned to this project.')

            pe.delete()
            return _ok('Examiner removed successfully.')
        except Exception as e:
            return _500(e)


class ListExaminersView(APIView):
    """GET /api/projects/<project_id>/examiners/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def get(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            qs = project.project_examiners.select_related('examiner__user').all()
            return _ok('Examiners retrieved.', ProjectExaminerSerializer(qs, many=True).data)
        except Exception as e:
            return _500(e)


# =============================================================================
# FEATURE 2 — STUDENT ENROLLMENT
# =============================================================================

class AvailableProjectsView(APIView):
    """GET /api/projects/available/"""
    permission_classes = [IsAuthenticated, IsStudent]

    def get(self, request):
        try:
            sp = _get_student_profile(request.user)
            if not sp:
                return _err('Student profile not found.', code=404)

            projects = Project.objects.filter(status='active')
            data = AvailableProjectSerializer(
                projects, many=True, context={'student_profile': sp},
            ).data
            return _ok('Available projects retrieved.', data)
        except Exception as e:
            return _500(e)


class EnrollInProjectView(APIView):
    """POST /api/projects/<project_id>/enroll/"""
    permission_classes = [IsAuthenticated, IsStudent]

    def post(self, request, project_id):
        try:
            sp = _get_student_profile(request.user)
            if not sp:
                return _err('Student profile not found.', code=404)

            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)
            if project.status != 'active':
                return _err('Project is not active.')
            if project.submission_deadline and project.submission_deadline < timezone.now():
                return _err('Enrollment deadline has passed.')

            ser = StudentEnrollSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            group_number = ser.validated_data.get('group_number')

            with transaction.atomic():
                if not project.is_group_project:
                    # Individual project
                    if ProjectSubmission.objects.filter(project=project, student=sp).exists():
                        return _err('You are already enrolled in this project.')
                    ProjectSubmission.objects.create(project=project, student=sp)
                else:
                    # Group project
                    if not group_number:
                        return _err('Group number is required for group projects.')

                    if GroupMember.objects.filter(group__project=project, student=sp).exists():
                        return _err('You are already enrolled in this project.')

                    group, _ = StudentGroup.objects.get_or_create(
                        project=project, group_name=group_number,
                    )
                    GroupMember.objects.create(group=group, student=sp)

                    # One submission record per group
                    ProjectSubmission.objects.get_or_create(
                        project=project, group=group,
                        defaults={'student': None},
                    )

            return _ok('Enrolled successfully.', code=201)
        except Exception as e:
            return _500(e)


class MyEnrollmentsView(APIView):
    """GET /api/projects/my-enrollments/"""
    permission_classes = [IsAuthenticated, IsStudent]

    def get(self, request):
        try:
            sp = _get_student_profile(request.user)
            if not sp:
                return _err('Student profile not found.', code=404)

            # Individual enrollments
            ind_ids = ProjectSubmission.objects.filter(
                student=sp,
            ).values_list('project_id', flat=True)
            # Group enrollments
            grp_ids = GroupMember.objects.filter(
                student=sp,
            ).values_list('group__project_id', flat=True)

            project_ids = set(list(ind_ids) + list(grp_ids))
            projects = Project.objects.filter(id__in=project_ids)
            data = MyEnrollmentSerializer(
                projects, many=True, context={'student_profile': sp},
            ).data
            return _ok('Enrolled projects retrieved.', data)
        except Exception as e:
            return _500(e)


# =============================================================================
# FEATURE 3 — SUBMISSIONS
# =============================================================================

class SubmitProjectView(APIView):
    """POST /api/projects/<project_id>/submit/"""
    permission_classes = [IsAuthenticated, IsStudent]
    parser_classes = [MultiPartParser, FormParser]

    # File validation constants
    MAX_REPORT_SIZE = 50 * 1024 * 1024  # 50 MB

    def post(self, request, project_id):
        try:
            sp = _get_student_profile(request.user)
            if not sp:
                return _err('Student profile not found.', code=404)

            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)
            if project.submission_deadline and project.submission_deadline < timezone.now():
                return _err('Submission deadline has passed.')

            if project.is_group_project:
                membership = GroupMember.objects.filter(
                    group__project=project, student=sp,
                ).select_related('group').first()
                if not membership:
                    return _err('You are not enrolled in this project.')
                submission = ProjectSubmission.objects.filter(
                    project=project, group=membership.group,
                ).first()
            else:
                submission = ProjectSubmission.objects.filter(
                    project=project, student=sp,
                ).first()

            if not submission:
                return _err('You are not enrolled in this project.')

            if submission.report_file_url or submission.github_repo_url:
                return _err('You have already submitted. Resubmission is not allowed.')

            submission_serializer = SubmitProjectSerializer(data=request.data)
            if not submission_serializer.is_valid():
                return _err('Validation failed.', submission_serializer.errors)

            # Handle file upload
            report_file = submission_serializer.validated_data['report_file']
            github_repo_url = submission_serializer.validated_data.get('github_repo_url', '')

            report_blob_url = None
            from core.utils.document_parser import extract_text_from_bytes
            # Validate file size
            if report_file.size > self.MAX_REPORT_SIZE:
                return _err('File too large. Maximum report size is 50MB.')

            from AI_Evaluator_Backend.azure_storage import upload_report_to_blob
            if project.is_group_project:
                report_blob_url = upload_report_to_blob(
                    report_file, str(project.id), group_id=str(membership.group.id),
                )
            else:
                report_blob_url = upload_report_to_blob(
                    report_file, str(project.id), student_id=str(sp.id),
                )

            code_submission = None

            report_file.seek(0)
            report_bytes = report_file.read()
            extracted_text = extract_text_from_bytes(report_bytes, report_file.name)

            with transaction.atomic():
                submission.report_file_url = report_blob_url
                submission.github_repo_url = github_repo_url or None
                submission.save()

                from viva_evaluator.models import SubmissionIndexStatus
                index_status, _ = SubmissionIndexStatus.objects.get_or_create(
                    submission=submission,
                    defaults={'report_file': report_file},
                )
                index_status.report_file = report_file
                index_status.status = SubmissionIndexStatus.IndexStatus.READY
                index_status.extracted_text = extracted_text
                index_status.processed_at = timezone.now()
                index_status.error_message = None
                index_status.save()

                if github_repo_url:
                    code_submission = CodeSubmission.objects.create(
                        project_submission=submission,
                        source_type=CodeSubmission.SourceType.GITHUB,
                        github_url=github_repo_url,
                    )

                    transaction.on_commit(
                        lambda code_submission_id=code_submission.id: enqueue_code_analysis(
                            code_submission_id,
                        )
                    )

            response_data = ProjectSubmissionSerializer(submission).data
            if code_submission:
                response_data['code_submission_id'] = str(code_submission.id)
                response_data['code_analysis_status'] = code_submission.analysis_status

            return _ok('Submission successful.', response_data)
        except Exception as e:
            return _500(e)


class SubmissionDetailView(APIView):
    """GET /api/projects/<project_id>/submission/"""
    permission_classes = [IsAuthenticated, IsExaminerOrStudent]

    def get(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            if request.user.role == 'student':
                sp = _get_student_profile(request.user)
                if project.is_group_project:
                    membership = GroupMember.objects.filter(
                        group__project=project, student=sp,
                    ).select_related('group').first()
                    if not membership:
                        return _err('You are not enrolled in this project.')
                    subs = ProjectSubmission.objects.filter(
                        project=project, group=membership.group,
                    )
                else:
                    subs = ProjectSubmission.objects.filter(project=project, student=sp)
                data = ProjectSubmissionSerializer(subs, many=True).data
            else:
                subs = ProjectSubmission.objects.filter(
                    project=project,
                ).select_related('student__user', 'group')
                data = ProjectSubmissionSerializer(subs, many=True).data

            return _ok('Submissions retrieved.', data)
        except Exception as e:
            return _500(e)
