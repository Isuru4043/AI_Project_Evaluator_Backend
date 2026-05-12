"""
Serializers for the Projects app.

Covers: project CRUD, examiner assignment, student enrollment,
submissions, rubric management, and session scheduling.
"""

from datetime import datetime, timedelta

from django.utils import timezone
from rest_framework import serializers

from core.models import (
    EvaluationSession,
    ExaminerProfile,
    GroupMember,
    Project,
    ProjectExaminer,
    ProjectSubmission,
    RubricCategory,
    RubricCriteria,
    StudentGroup,
    StudentProfile,
    User,
)


# =============================================================================
# PROJECT SERIALIZERS
# =============================================================================

class ProjectCreateSerializer(serializers.Serializer):
    """Validates input for creating a new project."""

    project_name = serializers.CharField(max_length=255)
    description = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    is_group_project = serializers.BooleanField(default=False)
    submission_deadline = serializers.DateTimeField(required=False, allow_null=True, default=None)
    academic_year = serializers.CharField(max_length=50, required=False, allow_blank=True, allow_null=True, default=None)


class ProjectUpdateSerializer(serializers.Serializer):
    """Validates input for updating an existing project."""

    project_name = serializers.CharField(max_length=255, required=False)
    description = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    is_group_project = serializers.BooleanField(required=False)
    submission_deadline = serializers.DateTimeField(required=False, allow_null=True)
    academic_year = serializers.CharField(max_length=50, required=False, allow_blank=True, allow_null=True)


class ProjectExaminerSerializer(serializers.ModelSerializer):
    """Serializes a ProjectExaminer record with examiner details."""

    examiner_user_id = serializers.UUIDField(source='examiner.user.id', read_only=True)
    full_name = serializers.CharField(source='examiner.user.full_name', read_only=True)
    email = serializers.CharField(source='examiner.user.email', read_only=True)
    employee_id = serializers.CharField(source='examiner.employee_id', read_only=True)
    department = serializers.CharField(source='examiner.department', read_only=True)

    class Meta:
        model = ProjectExaminer
        fields = [
            'examiner_user_id', 'full_name', 'email',
            'employee_id', 'department', 'role_in_project', 'assigned_at',
        ]


class ProjectSerializer(serializers.ModelSerializer):
    """Full project serializer for list / detail responses."""

    examiners = serializers.SerializerMethodField()
    enrolled_students_count = serializers.SerializerMethodField()
    sessions_count = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = [
            'id', 'project_name', 'description', 'is_group_project',
            'submission_deadline', 'status', 'academic_year', 'created_at',
            'examiners', 'enrolled_students_count', 'sessions_count',
        ]

    def get_examiners(self, obj):
        qs = obj.project_examiners.select_related('examiner__user').all()
        return ProjectExaminerSerializer(qs, many=True).data

    def get_enrolled_students_count(self, obj):
        if obj.is_group_project:
            return GroupMember.objects.filter(
                group__project=obj,
            ).count()
        return obj.submissions.filter(student__isnull=False).count()

    def get_sessions_count(self, obj):
        return obj.evaluation_sessions.count()


class ProjectDetailSerializer(serializers.ModelSerializer):
    """Detailed project serializer including rubrics."""

    examiners = serializers.SerializerMethodField()
    rubrics = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = [
            'id', 'project_name', 'description', 'is_group_project',
            'submission_deadline', 'status', 'academic_year', 'created_at',
            'examiners', 'rubrics',
        ]

    def get_examiners(self, obj):
        qs = obj.project_examiners.select_related('examiner__user').all()
        return ProjectExaminerSerializer(qs, many=True).data

    def get_rubrics(self, obj):
        qs = obj.rubric_categories.prefetch_related('criteria').all()
        return RubricCategorySerializer(qs, many=True).data


# =============================================================================
# EXAMINER ASSIGNMENT SERIALIZERS
# =============================================================================

class AddExaminerSerializer(serializers.Serializer):
    """Validates input for adding an examiner to a project."""

    examiner_user_id = serializers.UUIDField()
    role_in_project = serializers.ChoiceField(
        choices=['co-examiner'],
        default='co-examiner',
    )


class RemoveExaminerSerializer(serializers.Serializer):
    """Validates input for removing an examiner from a project."""

    examiner_user_id = serializers.UUIDField()


# =============================================================================
# STUDENT ENROLLMENT SERIALIZERS
# =============================================================================

class StudentEnrollSerializer(serializers.Serializer):
    """Validates input for student enrollment in a project."""

    group_number = serializers.CharField(
        max_length=255, required=False, allow_null=True, allow_blank=True, default=None,
    )


class AvailableProjectSerializer(serializers.ModelSerializer):
    """Serializes active projects for the student available projects list."""

    enrolled = serializers.SerializerMethodField()
    lead_examiner_name = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = [
            'id', 'project_name', 'description', 'is_group_project',
            'submission_deadline', 'lead_examiner_name', 'enrolled',
        ]

    def get_enrolled(self, obj):
        student_profile = self.context.get('student_profile')
        if not student_profile:
            return False
        if obj.is_group_project:
            return GroupMember.objects.filter(
                group__project=obj,
                student=student_profile,
            ).exists()
        return obj.submissions.filter(student=student_profile).exists()

    def get_lead_examiner_name(self, obj):
        lead = obj.project_examiners.filter(role_in_project='lead').select_related('examiner__user').first()
        if lead:
            return lead.examiner.user.full_name
        return None


class MyEnrollmentSerializer(serializers.ModelSerializer):
    """Serializes projects from the student's enrolled perspective."""

    submission_status = serializers.SerializerMethodField()
    session_details = serializers.SerializerMethodField()
    group_info = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = [
            'id', 'project_name', 'description', 'is_group_project',
            'submission_deadline', 'status', 'academic_year',
            'submission_status', 'session_details', 'group_info',
        ]

    def get_submission_status(self, obj):
        student_profile = self.context.get('student_profile')
        if obj.is_group_project:
            membership = GroupMember.objects.filter(
                group__project=obj, student=student_profile,
            ).select_related('group').first()
            if membership:
                sub = ProjectSubmission.objects.filter(
                    project=obj, group=membership.group,
                ).first()
                if sub and (sub.report_file_url or sub.github_repo_url):
                    return 'submitted'
        else:
            sub = ProjectSubmission.objects.filter(
                project=obj, student=student_profile,
            ).first()
            if sub and (sub.report_file_url or sub.github_repo_url):
                return 'submitted'
        return 'not_submitted'

    def get_session_details(self, obj):
        student_profile = self.context.get('student_profile')
        session = EvaluationSession.objects.filter(
            project=obj, student=student_profile,
        ).first()
        if session:
            return {
                'session_id': str(session.id),
                'scheduled_start': session.scheduled_start,
                'scheduled_end': session.scheduled_end,
                'location_room': session.location_room,
                'status': session.status,
            }
        return None

    def get_group_info(self, obj):
        if not obj.is_group_project:
            return None
        student_profile = self.context.get('student_profile')
        membership = GroupMember.objects.filter(
            group__project=obj, student=student_profile,
        ).select_related('group').first()
        if membership:
            members = GroupMember.objects.filter(
                group=membership.group,
            ).select_related('student__user').values_list(
                'student__user__full_name', flat=True,
            )
            return {
                'group_id': str(membership.group.id),
                'group_name': membership.group.group_name,
                'members': list(members),
            }
        return None


# =============================================================================
# SUBMISSION SERIALIZERS
# =============================================================================

class SubmitProjectSerializer(serializers.Serializer):
    """Validates input for submitting project files."""

    report_file = serializers.FileField(required=True)
    github_repo_url = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)

    def validate_report_file(self, value):
        if not value.name.lower().endswith('.pdf'):
            raise serializers.ValidationError('Only PDF files are allowed for reports.')
        return value


class ProjectSubmissionSerializer(serializers.ModelSerializer):
    """Serializes a project submission record."""

    student_name = serializers.CharField(source='student.user.full_name', read_only=True, default=None)
    student_reg_no = serializers.CharField(source='student.registration_number', read_only=True, default=None)
    group_name = serializers.CharField(source='group.group_name', read_only=True, default=None)
    latest_code_submission_id = serializers.SerializerMethodField()
    latest_code_analysis_status = serializers.SerializerMethodField()

    class Meta:
        model = ProjectSubmission
        fields = [
            'id', 'project', 'student_name', 'student_reg_no',
            'group_name', 'report_file_url', 'github_repo_url', 'submitted_at',
            'latest_code_submission_id', 'latest_code_analysis_status',
        ]

    def get_latest_code_submission_id(self, obj):
        latest_code_submission = obj.code_submissions.order_by('-uploaded_at').first()
        if not latest_code_submission:
            return None
        return str(latest_code_submission.id)

    def get_latest_code_analysis_status(self, obj):
        latest_code_submission = obj.code_submissions.order_by('-uploaded_at').first()
        if not latest_code_submission:
            return None
        return latest_code_submission.analysis_status


# =============================================================================
# RUBRIC SERIALIZERS
# =============================================================================

class RubricCriteriaSerializer(serializers.ModelSerializer):
    """Serializes a single rubric criteria."""

    criteria_id = serializers.UUIDField(source='id', read_only=True)

    class Meta:
        model = RubricCriteria
        fields = [
            'criteria_id', 'criteria_name', 'max_score',
            'weight_in_category', 'description',
        ]


class RubricCategorySerializer(serializers.ModelSerializer):
    """Serializes a rubric category with nested criteria."""

    category_id = serializers.UUIDField(source='id', read_only=True)
    criteria = RubricCriteriaSerializer(many=True, read_only=True)

    class Meta:
        model = RubricCategory
        fields = [
            'category_id', 'category_name', 'weight_percentage',
            'description', 'criteria',
        ]


class RubricCategoryCreateSerializer(serializers.Serializer):
    """Validates input for creating a rubric category."""

    category_name = serializers.CharField(max_length=255)
    weight_percentage = serializers.DecimalField(max_digits=5, decimal_places=2)
    description = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)


class RubricCategoryUpdateSerializer(serializers.Serializer):
    """Validates input for updating a rubric category."""

    category_name = serializers.CharField(max_length=255, required=False)
    weight_percentage = serializers.DecimalField(max_digits=5, decimal_places=2, required=False)
    description = serializers.CharField(required=False, allow_blank=True, allow_null=True)


class RubricCriteriaCreateSerializer(serializers.Serializer):
    """Validates input for creating a rubric criteria."""

    criteria_name = serializers.CharField(max_length=255)
    max_score = serializers.DecimalField(max_digits=6, decimal_places=2)
    weight_in_category = serializers.DecimalField(
        max_digits=5, decimal_places=2, required=False, allow_null=True, default=None,
    )
    description = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)


class RubricCriteriaUpdateSerializer(serializers.Serializer):
    """Validates input for updating a rubric criteria."""

    criteria_name = serializers.CharField(max_length=255, required=False)
    max_score = serializers.DecimalField(max_digits=6, decimal_places=2, required=False)
    weight_in_category = serializers.DecimalField(
        max_digits=5, decimal_places=2, required=False, allow_null=True,
    )
    description = serializers.CharField(required=False, allow_blank=True, allow_null=True)


# =============================================================================
# SESSION SCHEDULING SERIALIZERS
# =============================================================================

class ManualSessionEntrySerializer(serializers.Serializer):
    """A single entry in the manual schedule array."""

    student_id = serializers.UUIDField(required=False, allow_null=True, default=None)
    group_id = serializers.UUIDField(required=False, allow_null=True, default=None)
    scheduled_start = serializers.DateTimeField()
    scheduled_end = serializers.DateTimeField()
    location_room = serializers.CharField(max_length=255, required=False, allow_blank=True, default='')


class ManualScheduleSerializer(serializers.Serializer):
    """Validates input for manual session scheduling."""

    sessions = ManualSessionEntrySerializer(many=True)


class DateRangeEntrySerializer(serializers.Serializer):
    """A single date range for auto scheduling."""

    date = serializers.DateField()
    start_time = serializers.TimeField()
    end_time = serializers.TimeField()

    def validate(self, attrs):
        if attrs['start_time'] >= attrs['end_time']:
            raise serializers.ValidationError(
                'start_time must be before end_time.'
            )
        return attrs


class AutoScheduleSerializer(serializers.Serializer):
    """Validates input for automatic session scheduling."""

    date_ranges = DateRangeEntrySerializer(many=True)
    duration_per_slot_minutes = serializers.IntegerField(min_value=1)
    location_room = serializers.CharField(max_length=255, required=False, allow_blank=True, default='')


# =============================================================================
# EVALUATION SESSION SERIALIZER
# =============================================================================

class EvaluationSessionSerializer(serializers.ModelSerializer):
    """Serializes an evaluation session with student / group info."""

    student_name = serializers.CharField(source='student.user.full_name', read_only=True, default=None)
    student_reg_no = serializers.CharField(source='student.registration_number', read_only=True, default=None)
    group_name = serializers.CharField(source='group.group_name', read_only=True, default=None)

    class Meta:
        model = EvaluationSession
        fields = [
            'id', 'project', 'student_name', 'student_reg_no',
            'group_name', 'scheduled_start', 'scheduled_end',
            'actual_start', 'location_room', 'status',
        ]


class SessionUpdateSerializer(serializers.Serializer):
    """Validates input for updating a session."""

    scheduled_start = serializers.DateTimeField(required=False)
    scheduled_end = serializers.DateTimeField(required=False)
    location_room = serializers.CharField(max_length=255, required=False, allow_blank=True)
