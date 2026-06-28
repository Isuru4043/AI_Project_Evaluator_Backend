import uuid
from django.db import models
from core.models import ProjectSubmission, VivaQuestion, VivaAnswer, RubricCriteria


class SubmissionIndexStatus(models.Model):

    class IndexStatus(models.TextChoices):
        PENDING    = 'pending',    'Pending'
        PROCESSING = 'processing', 'Processing'
        READY      = 'ready',      'Ready'
        FAILED     = 'failed',     'Failed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    submission = models.OneToOneField(
        ProjectSubmission,
        on_delete=models.CASCADE,
        related_name='index_status',
    )

    report_file = models.FileField(
        upload_to='submissions/reports/',
        null=True,
        blank=True,
    )

    # Full extracted text stored directly — fed to LLM during session
    extracted_text = models.TextField(null=True, blank=True)

    status = models.CharField(
        max_length=20,
        choices=IndexStatus.choices,
        default=IndexStatus.PENDING,
    )

    error_message = models.TextField(null=True, blank=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # =========================================================================
    # RAG / FAISS persistence (Week 1 — Phase 1).
    # The FAISS index is serialized to bytes and stored directly in PostgreSQL.
    # The chunks JSON keeps text + metadata parallel to the vectors so we can
    # return rich results from a search (text, source, section, score).
    # =========================================================================
    faiss_index_blob = models.BinaryField(null=True, blank=True)
    faiss_chunks_json = models.JSONField(null=True, blank=True)

    # =========================================================================
    # Knowledge Graph persistence (Week 3 — Phase 1.3).
    # Stored as NetworkX node_link_data() JSON. Holds DEPENDS_ON,
    # CONTRADICTS_CODE, ALTERNATIVE_TO, etc. with per-edge confidence tier.
    # =========================================================================
    kg_graph_json = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = 'Submission Index Status'
        verbose_name_plural = 'Submission Index Statuses'

    def __str__(self):
        return f"Submission [{self.status}] — {self.submission_id}"


class VivaQuestionExtension(models.Model):

    class DifficultyLevel(models.TextChoices):
        EASY   = 'easy',   'Easy'
        MEDIUM = 'medium', 'Medium'
        HARD   = 'hard',   'Hard'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    question = models.OneToOneField(
        VivaQuestion,
        on_delete=models.CASCADE,
        related_name='extension',
    )

    criteria = models.ForeignKey(
        RubricCriteria,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='viva_questions',
    )

    difficulty_level = models.CharField(
        max_length=10,
        choices=DifficultyLevel.choices,
        default=DifficultyLevel.MEDIUM,
    )

    class Meta:
        verbose_name = 'Viva Question Extension'
        verbose_name_plural = 'Viva Question Extensions'

    def __str__(self):
        return f"Q{self.question.question_order} [{self.difficulty_level}]"


class VivaAnswerExtension(models.Model):

    class NextDifficulty(models.TextChoices):
        LOWER  = 'lower',  'Lower'
        SAME   = 'same',   'Same'
        HIGHER = 'higher', 'Higher'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    answer = models.OneToOneField(
        VivaAnswer,
        on_delete=models.CASCADE,
        related_name='extension',
    )

    # Score given by LLM (0-10)
    llm_score = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True,
    )

    # LLM's reasoning for the score — feeds into XAI report
    llm_reasoning = models.TextField(null=True, blank=True)

    # Signal for what difficulty the next question should be
    next_difficulty_signal = models.CharField(
        max_length=10,
        choices=NextDifficulty.choices,
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name = 'Viva Answer Extension'
        verbose_name_plural = 'Viva Answer Extensions'

    def __str__(self):
        return f"Answer extension — LLM score: {self.llm_score}"


class CriteriaQuestionHint(models.Model):
    """
    Optional sample questions provided by the examiner for a rubric criterion.
    Used as guidelines for the AI question generator.
    If none are provided the AI generates freely from the rubric description.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    criteria = models.ForeignKey(
        RubricCriteria,
        on_delete=models.CASCADE,
        related_name='question_hints',
    )
    hint_text = models.TextField()
    order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Criteria Question Hint'
        verbose_name_plural = 'Criteria Question Hints'
        ordering = ['order']

    def __str__(self):
        return f"Hint for {self.criteria.criteria_name}: {self.hint_text[:60]}"



# =============================================================================
# WEEK 4 — Examiner-in-the-Loop Knowledge Accumulation
# =============================================================================

class ApprovedDomainBrief(models.Model):
    """
    A structured brief about a single technology, used by the Knowledge Graph
    to drive alternative-probing questions during a viva.

    Lifecycle:
        - LLM auto-drafts → status=PENDING, tier=2 (T2: LLM draft, unreviewed)
        - Examiner approves → status=ACTIVE, tier=1 (T1: examiner-verified)
        - Examiner rejects → status=ARCHIVED (never used)
        - Older briefs become STALE after 18 months → flagged for re-review

    Scope:
        - EXAMINER: brief is private to the examiner who approved it
        - DEPARTMENT: brief is shared across all examiners in the department
                      (requires a second examiner's approval to promote)
    """

    class Status(models.TextChoices):
        PENDING    = 'pending',    'Pending Review'
        ACTIVE     = 'active',     'Active'
        ARCHIVED   = 'archived',   'Archived'

    class Scope(models.TextChoices):
        EXAMINER   = 'examiner',   'Examiner Scope'
        DEPARTMENT = 'department', 'Department Scope'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Tech identity
    technology = models.CharField(max_length=200)             # 'PostgreSQL', 'FAISS'
    tech_version = models.CharField(max_length=50, null=True, blank=True)

    # The structured brief content (alternatives, best practices, mistakes)
    brief_json = models.JSONField()

    # Workflow / governance
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    scope = models.CharField(
        max_length=20,
        choices=Scope.choices,
        default=Scope.EXAMINER,
    )
    tier = models.IntegerField(default=2)   # 1 = examiner-approved, 2 = unreviewed draft

    # Audit / lineage
    drafted_for_submission = models.ForeignKey(
        ProjectSubmission,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='drafted_briefs',
        help_text='Submission whose tech extraction triggered this draft.',
    )
    drafted_at = models.DateTimeField(auto_now_add=True)
    approved_by = models.ForeignKey(
        'core.ExaminerProfile',
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='approved_briefs',
    )
    approved_at = models.DateTimeField(null=True, blank=True)
    last_verified_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = 'Approved Domain Brief'
        verbose_name_plural = 'Approved Domain Briefs'
        ordering = ['-drafted_at']
        indexes = [
            models.Index(fields=['technology', 'status']),
            models.Index(fields=['status', 'scope']),
        ]

    def __str__(self):
        return (
            f"{self.technology}"
            f"{' v' + self.tech_version if self.tech_version else ''} "
            f"[{self.status}, T{self.tier}, {self.scope}]"
        )
