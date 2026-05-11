"""
Models for the AI Project Evaluator Backend.

IMPORTANT: Add the following to your settings.py:
    AUTH_USER_MODEL = 'core.User'

Also add 'core' to INSTALLED_APPS before running migrations.
"""

import uuid

from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models


# =============================================================================
# Custom User Manager (required when username field is removed)
# =============================================================================

class CustomUserManager(BaseUserManager):
    """
    Custom manager for User model where email is the unique identifier
    instead of username.
    """

    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field is required.')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self.create_user(email, password, **extra_fields)


# =============================================================================
# 1. USERS — Custom User Model
# =============================================================================

class User(AbstractUser):
    """Custom user model using email as the primary login identifier."""

    class Role(models.TextChoices):
        EXAMINER = 'examiner', 'Examiner'
        STUDENT = 'student', 'Student'

    # Override the default `id` with a UUID primary key
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Remove the default `username` field — email is used instead
    username = None
    first_name = None
    last_name = None

    full_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    role = models.CharField(max_length=20, choices=Role.choices)
    profile_picture_url = models.TextField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['full_name', 'role']

    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.full_name} ({self.email})"


# =============================================================================
# 2. EXAMINER PROFILES
# =============================================================================

class ExaminerProfile(models.Model):
    """Extended profile for users with the examiner role."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='examiner_profile',
    )
    employee_id = models.CharField(max_length=100, unique=True, null=True, blank=True)
    department = models.CharField(max_length=255, null=True, blank=True)
    designation = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        verbose_name = 'Examiner Profile'
        verbose_name_plural = 'Examiner Profiles'

    def __str__(self):
        return f"Examiner: {self.user.full_name}"


# =============================================================================
# 3. STUDENT PROFILES
# =============================================================================

class StudentProfile(models.Model):
    """Extended profile for users with the student role."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='student_profile',
    )
    registration_number = models.CharField(max_length=100, unique=True)
    degree_program = models.CharField(max_length=255, null=True, blank=True)
    academic_year = models.IntegerField(null=True, blank=True)
    batch = models.CharField(max_length=100, null=True, blank=True)

    class Meta:
        verbose_name = 'Student Profile'
        verbose_name_plural = 'Student Profiles'

    def __str__(self):
        return f"Student: {self.user.full_name} ({self.registration_number})"


# =============================================================================
# 4. PROJECTS
# =============================================================================

class Project(models.Model):
    """A project that students submit and examiners evaluate."""

    class Status(models.TextChoices):
        DRAFT = 'draft', 'Draft'
        ACTIVE = 'active', 'Active'
        COMPLETED = 'completed', 'Completed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project_name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    is_group_project = models.BooleanField(default=False)
    submission_deadline = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.DRAFT,
    )
    academic_year = models.CharField(max_length=50, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Project'
        verbose_name_plural = 'Projects'
        ordering = ['-created_at']

    def __str__(self):
        return self.project_name


# =============================================================================
# 5. PROJECT EXAMINERS — Many-to-Many bridge
# =============================================================================

class ProjectExaminer(models.Model):
    """Links examiners to projects with a specific role."""

    class RoleInProject(models.TextChoices):
        LEAD = 'lead', 'Lead'
        CO_EXAMINER = 'co-examiner', 'Co-Examiner'

    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='project_examiners',
    )
    examiner = models.ForeignKey(
        ExaminerProfile,
        on_delete=models.CASCADE,
        related_name='project_assignments',
    )
    role_in_project = models.CharField(
        max_length=20,
        choices=RoleInProject.choices,
    )
    assigned_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Project Examiner'
        verbose_name_plural = 'Project Examiners'
        constraints = [
            models.UniqueConstraint(
                fields=['project', 'examiner'],
                name='unique_project_examiner',
            ),
        ]

    def __str__(self):
        return f"{self.examiner} — {self.project} ({self.role_in_project})"


# =============================================================================
# 6. STUDENT GROUPS
# =============================================================================

class StudentGroup(models.Model):
    """A group of students working together on a project."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='student_groups',
    )
    group_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Student Group'
        verbose_name_plural = 'Student Groups'

    def __str__(self):
        return f"{self.group_name} ({self.project.project_name})"


# =============================================================================
# 7. GROUP MEMBERS — Many-to-Many bridge
# =============================================================================

class GroupMember(models.Model):
    """Links students to their respective groups."""

    group = models.ForeignKey(
        StudentGroup,
        on_delete=models.CASCADE,
        related_name='members',
    )
    student = models.ForeignKey(
        StudentProfile,
        on_delete=models.CASCADE,
        related_name='group_memberships',
    )
    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Group Member'
        verbose_name_plural = 'Group Members'
        constraints = [
            models.UniqueConstraint(
                fields=['group', 'student'],
                name='unique_group_student',
            ),
        ]

    def __str__(self):
        return f"{self.student} in {self.group}"


# =============================================================================
# 8. RUBRIC CATEGORIES
# =============================================================================

class RubricCategory(models.Model):
    """Top-level scoring category within a project rubric."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='rubric_categories',
    )
    category_name = models.CharField(max_length=255)
    weight_percentage = models.DecimalField(max_digits=5, decimal_places=2)
    description = models.TextField(null=True, blank=True)

    class Meta:
        verbose_name = 'Rubric Category'
        verbose_name_plural = 'Rubric Categories'

    def __str__(self):
        return f"{self.category_name} ({self.weight_percentage}%)"


# =============================================================================
# 9. RUBRIC CRITERIA
# =============================================================================

class RubricCriteria(models.Model):
    """Individual scoring criterion within a rubric category."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    category = models.ForeignKey(
        RubricCategory,
        on_delete=models.CASCADE,
        related_name='criteria',
    )
    criteria_name = models.CharField(max_length=255)
    max_score = models.DecimalField(max_digits=6, decimal_places=2)
    weight_in_category = models.DecimalField(
        max_digits=5, decimal_places=2, null=True, blank=True,
    )
    description = models.TextField(null=True, blank=True)

    class Meta:
        verbose_name = 'Rubric Criteria'
        verbose_name_plural = 'Rubric Criteria'

    def __str__(self):
        return f"{self.criteria_name} (max {self.max_score})"


# =============================================================================
# 10. PROJECT SUBMISSIONS
# =============================================================================

class ProjectSubmission(models.Model):
    """A submission made by a student or group for a project."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='submissions',
    )
    student = models.ForeignKey(
        StudentProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='submissions',
    )
    group = models.ForeignKey(
        StudentGroup,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='submissions',
    )
    report_file_url = models.TextField(null=True, blank=True)
    github_repo_url = models.TextField(null=True, blank=True)
    submitted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Project Submission'
        verbose_name_plural = 'Project Submissions'
        ordering = ['-submitted_at']

    def __str__(self):
        submitter = self.group or self.student or 'Unknown'
        return f"Submission for {self.project} by {submitter}"


# =============================================================================
# 11. EVALUATION SESSIONS
# =============================================================================

class EvaluationSession(models.Model):
    """A scheduled evaluation/viva session for a project."""

    class Status(models.TextChoices):
        SCHEDULED = 'scheduled', 'Scheduled'
        IN_PROGRESS = 'in_progress', 'In Progress'
        COMPLETED = 'completed', 'Completed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='evaluation_sessions',
    )
    student = models.ForeignKey(
        StudentProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='evaluation_sessions',
    )
    group = models.ForeignKey(
        StudentGroup,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='evaluation_sessions',
    )
    submission = models.ForeignKey(
        ProjectSubmission,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='evaluation_sessions',
    )
    scheduled_start = models.DateTimeField()
    scheduled_end = models.DateTimeField()
    actual_start = models.DateTimeField(null=True, blank=True)
    demo_completed_at = models.DateTimeField(null=True, blank=True)
    location_room = models.CharField(max_length=255, null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.SCHEDULED,
    )

    class Meta:
        verbose_name = 'Evaluation Session'
        verbose_name_plural = 'Evaluation Sessions'
        ordering = ['scheduled_start']

    def __str__(self):
        target = self.group or self.student or 'Unassigned'
        return f"Session: {self.project} — {target} ({self.status})"


# =============================================================================
# 12. SESSION RECORDINGS
# =============================================================================

class SessionRecording(models.Model):
    """Audio/video recording of an evaluation session."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='recordings',
    )
    video_file_url = models.TextField(null=True, blank=True)
    audio_file_url = models.TextField(null=True, blank=True)
    duration_seconds = models.IntegerField(null=True, blank=True)
    recorded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Session Recording'
        verbose_name_plural = 'Session Recordings'

    def __str__(self):
        return f"Recording for {self.session}"


# =============================================================================
# 13. HEARTRATE RAW DATA
# =============================================================================

class HeartrateRawData(models.Model):
    """Raw heart-rate telemetry captured during an evaluation session."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='heartrate_data',
    )
    timestamp = models.DateTimeField()
    bpm = models.IntegerField()
    hrv_value = models.DecimalField(
        max_digits=6, decimal_places=3, null=True, blank=True,
    )
    is_baseline = models.BooleanField(default=False)

    class Meta:
        verbose_name = 'Heart Rate Raw Data'
        verbose_name_plural = 'Heart Rate Raw Data'
        ordering = ['timestamp']

    def __str__(self):
        return f"HR {self.bpm} bpm @ {self.timestamp}"


# =============================================================================
# 14. CV ANALYSIS RESULTS
# =============================================================================

class CVAnalysisResult(models.Model):
    """Computer-vision analysis output for a point in time during a session."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='cv_analysis_results',
    )
    timestamp = models.DateTimeField()
    gaze_direction = models.CharField(max_length=100, null=True, blank=True)
    head_pose = models.CharField(max_length=100, null=True, blank=True)
    hand_gesture = models.CharField(max_length=100, null=True, blank=True)
    facial_expression = models.CharField(max_length=100, null=True, blank=True)
    engagement_score = models.DecimalField(
        max_digits=5, decimal_places=2, null=True, blank=True,
    )
    cheating_flag = models.BooleanField(default=False)
    cheating_reason = models.TextField(null=True, blank=True)

    class Meta:
        verbose_name = 'CV Analysis Result'
        verbose_name_plural = 'CV Analysis Results'
        ordering = ['timestamp']

    def __str__(self):
        flag = "⚠ FLAGGED" if self.cheating_flag else "OK"
        return f"CV Analysis @ {self.timestamp} [{flag}]"


# =============================================================================
# 15. VIVA QUESTIONS
# =============================================================================

class VivaQuestion(models.Model):
    """A question generated or assigned for a viva session."""

    class BloomsLevel(models.TextChoices):
        REMEMBER = 'Remember', 'Remember'
        UNDERSTAND = 'Understand', 'Understand'
        APPLY = 'Apply', 'Apply'
        ANALYZE = 'Analyze', 'Analyze'
        EVALUATE = 'Evaluate', 'Evaluate'
        CREATE = 'Create', 'Create'

    class QuestionSource(models.TextChoices):
        AI = 'ai', 'AI'
        EXAMINER = 'examiner', 'Examiner'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='viva_questions',
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='examiner_questions',
    )
    question_text = models.TextField()
    blooms_level = models.CharField(
        max_length=20,
        choices=BloomsLevel.choices,
        null=True,
        blank=True,
    )
    question_source = models.CharField(
        max_length=20,
        choices=QuestionSource.choices,
        default=QuestionSource.AI,
    )
    question_order = models.IntegerField()
    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Viva Question'
        verbose_name_plural = 'Viva Questions'
        ordering = ['question_order']

    def __str__(self):
        return f"Q{self.question_order}: {self.question_text[:60]}"


# =============================================================================
# 16. VIVA ANSWERS
# =============================================================================

class VivaAnswer(models.Model):
    """A student's answer to a viva question, with analysis metrics."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    question = models.ForeignKey(
        VivaQuestion,
        on_delete=models.CASCADE,
        related_name='answers',
    )
    student = models.ForeignKey(
        StudentProfile,
        on_delete=models.CASCADE,
        related_name='viva_answers',
    )
    transcribed_answer = models.TextField(null=True, blank=True)
    audio_clip_url = models.TextField(null=True, blank=True)
    response_delay_ms = models.IntegerField(null=True, blank=True)
    pause_frequency = models.IntegerField(null=True, blank=True)
    speech_clarity_score = models.DecimalField(
        max_digits=5, decimal_places=2, null=True, blank=True,
    )
    ai_answer_score = models.DecimalField(
        max_digits=5, decimal_places=2, null=True, blank=True,
    )
    answered_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Viva Answer'
        verbose_name_plural = 'Viva Answers'

    def __str__(self):
        return f"Answer by {self.student} to {self.question}"


# =============================================================================
# 17. AI SCORE RECOMMENDATIONS
# =============================================================================

class AIScoreRecommendation(models.Model):
    """AI-generated score suggestion for a specific rubric criteria in a session."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='ai_score_recommendations',
    )
    criteria = models.ForeignKey(
        RubricCriteria,
        on_delete=models.CASCADE,
        related_name='ai_score_recommendations',
    )
    ai_recommended_score = models.DecimalField(max_digits=6, decimal_places=2)
    xai_explanation = models.TextField(null=True, blank=True)
    shap_values = models.JSONField(null=True, blank=True)
    confidence_level = models.DecimalField(
        max_digits=5, decimal_places=2, null=True, blank=True,
    )
    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'AI Score Recommendation'
        verbose_name_plural = 'AI Score Recommendations'

    def __str__(self):
        return f"AI Score: {self.ai_recommended_score} for {self.criteria}"


# =============================================================================
# 18. FINAL SCORES
# =============================================================================

class FinalScore(models.Model):
    """Examiner-confirmed final score for a rubric criteria in a session."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='final_scores',
    )
    examiner = models.ForeignKey(
        ExaminerProfile,
        on_delete=models.CASCADE,
        related_name='final_scores',
    )
    criteria = models.ForeignKey(
        RubricCriteria,
        on_delete=models.CASCADE,
        related_name='final_scores',
    )
    ai_recommended_score = models.DecimalField(
        max_digits=6, decimal_places=2, null=True, blank=True,
    )
    examiner_final_score = models.DecimalField(max_digits=6, decimal_places=2)
    examiner_note = models.TextField(null=True, blank=True)
    approved_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Final Score'
        verbose_name_plural = 'Final Scores'

    def __str__(self):
        return (
            f"Final Score: {self.examiner_final_score} "
            f"for {self.criteria} by {self.examiner}"
        )


# =============================================================================
# 19. SESSION SUMMARY REPORTS
# =============================================================================

class SessionSummaryReport(models.Model):
    """Aggregated summary report for an evaluation session."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.OneToOneField(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='summary_report',
    )
    total_ai_score = models.DecimalField(
        max_digits=6, decimal_places=2, null=True, blank=True,
    )
    total_final_score = models.DecimalField(
        max_digits=6, decimal_places=2, null=True, blank=True,
    )
    grade = models.CharField(max_length=10, null=True, blank=True)
    overall_feedback = models.TextField(null=True, blank=True)
    emotional_summary = models.TextField(null=True, blank=True)
    integrity_flags_summary = models.TextField(null=True, blank=True)
    is_published = models.BooleanField(default=False)
    published_at = models.DateTimeField(null=True, blank=True)
    finalized_by = models.ForeignKey(
        ExaminerProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='finalized_reports',
    )

    class Meta:
        verbose_name = 'Session Summary Report'
        verbose_name_plural = 'Session Summary Reports'

    def __str__(self):
        return f"Report for {self.session} — Grade: {self.grade or 'N/A'}"


# =============================================================================
# 20. EXAMINER ACTION LOGS
# =============================================================================

class ExaminerActionLog(models.Model):
    """Audit log of examiner actions within the system."""

    class ActionType(models.TextChoices):
        SCORE_MODIFIED = 'score_modified', 'Score Modified'
        QUESTION_VALIDATED = 'question_validated', 'Question Validated'
        GRADE_APPROVED = 'grade_approved', 'Grade Approved'
        BULK_APPROVED = 'bulk_approved', 'Bulk Approved'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    examiner = models.ForeignKey(
        ExaminerProfile,
        on_delete=models.CASCADE,
        related_name='action_logs',
    )
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='examiner_action_logs',
    )
    action_type = models.CharField(
        max_length=30,
        choices=ActionType.choices,
    )
    action_detail = models.JSONField(null=True, blank=True)
    performed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Examiner Action Log'
        verbose_name_plural = 'Examiner Action Logs'
        ordering = ['-performed_at']

    def __str__(self):
        return f"{self.examiner} — {self.action_type} @ {self.performed_at}"


# =============================================================================
# 21. CODE SUBMISSIONS (SOURCE UPLOADS)
# =============================================================================

class CodeSubmission(models.Model):
    """Source code submitted for automated analysis."""

    class SourceType(models.TextChoices):
        GITHUB = 'github', 'GitHub Repository'
        ZIP = 'zip', 'ZIP Upload'

    class AnalysisStatus(models.TextChoices):
        PENDING = 'pending', 'Pending'
        FETCHING = 'fetching', 'Fetching'
        SCANNING = 'scanning', 'Scanning'
        SUMMARIZING = 'summarizing', 'Summarizing'
        QUESTIONING = 'questioning', 'Generating Questions'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'

    class QualityStatus(models.TextChoices):
        UNKNOWN = 'unknown', 'Unknown'
        PASSED = 'passed', 'Passed'
        FAILED = 'failed', 'Failed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project_submission = models.ForeignKey(
        ProjectSubmission,
        on_delete=models.CASCADE,
        related_name='code_submissions',
    )
    source_type = models.CharField(max_length=20, choices=SourceType.choices)
    github_url = models.TextField(null=True, blank=True)
    zip_file = models.FileField(upload_to='code_uploads/', null=True, blank=True)

    language_detected = models.CharField(max_length=100, null=True, blank=True)
    build_system_detected = models.CharField(max_length=100, null=True, blank=True)
    build_command = models.TextField(null=True, blank=True)

    sonar_project_key = models.CharField(max_length=200, null=True, blank=True)
    sonar_task_id = models.CharField(max_length=200, null=True, blank=True)
    sonar_summary = models.JSONField(null=True, blank=True)
    sonar_report_url = models.TextField(null=True, blank=True)

    quality_status = models.CharField(
        max_length=20,
        choices=QualityStatus.choices,
        default=QualityStatus.UNKNOWN,
    )
    quality_reason = models.TextField(null=True, blank=True)

    code_summary = models.TextField(null=True, blank=True)
    analysis_status = models.CharField(
        max_length=20,
        choices=AnalysisStatus.choices,
        default=AnalysisStatus.PENDING,
    )
    analysis_error = models.TextField(null=True, blank=True)

    uploaded_at = models.DateTimeField(auto_now_add=True)
    analyzed_at = models.DateTimeField(null=True, blank=True)
    questions_generated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = 'Code Submission'
        verbose_name_plural = 'Code Submissions'
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"Code Submission {self.id} ({self.source_type})"


# =============================================================================
# 22. GENERATED VIVA QUESTIONS (FROM CODE ANALYSIS)
# =============================================================================

class GeneratedVivaQuestion(models.Model):
    """Viva questions generated from code analysis."""

    class SourceType(models.TextChoices):
        CODE = 'code', 'Code Analysis'
        SONAR = 'sonar', 'Sonar Issues'
        COMBINED = 'combined', 'Combined'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    code_submission = models.ForeignKey(
        CodeSubmission,
        on_delete=models.CASCADE,
        related_name='generated_questions',
    )
    evaluation_session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='generated_code_questions',
    )
    question_text = models.TextField()
    blooms_level = models.CharField(
        max_length=20,
        choices=VivaQuestion.BloomsLevel.choices,
        null=True,
        blank=True,
    )
    source_type = models.CharField(max_length=20, choices=SourceType.choices)
    code_reference = models.TextField(null=True, blank=True)
    sonar_issue_reference = models.JSONField(null=True, blank=True)
    reasoning = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Generated Viva Question'
        verbose_name_plural = 'Generated Viva Questions'
        ordering = ['-created_at']

    def __str__(self):
        return f"Generated Question {self.id}"
