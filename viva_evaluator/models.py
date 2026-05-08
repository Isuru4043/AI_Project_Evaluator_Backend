import uuid
from django.db import models
from core.models import ProjectSubmission, VivaQuestion, VivaAnswer, RubricCriteria


# =============================================================================
# 1. SUBMISSION INDEX STATUS
# Tracks whether a submission's report has been parsed and indexed into FAISS.
# A viva session cannot start until index_status is 'indexed'.
# =============================================================================

class SubmissionIndexStatus(models.Model):

    class IndexStatus(models.TextChoices):
        PENDING   = 'pending',   'Pending'
        INDEXING  = 'indexing',  'Indexing'
        INDEXED   = 'indexed',   'Indexed'
        FAILED    = 'failed',    'Failed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    submission = models.OneToOneField(
        ProjectSubmission,
        on_delete=models.CASCADE,
        related_name='index_status',
    )

    # Local path to the uploaded report file (PDF or DOCX)
    report_file = models.FileField(
        upload_to='submissions/reports/',
        null=True,
        blank=True,
    )

    # Local path where the FAISS index is saved after indexing
    faiss_index_path = models.CharField(max_length=512, null=True, blank=True)

    # Plain text extracted from the report — stored for quick access during session
    extracted_text = models.TextField(null=True, blank=True)

    status = models.CharField(
        max_length=20,
        choices=IndexStatus.choices,
        default=IndexStatus.PENDING,
    )

    error_message = models.TextField(null=True, blank=True)
    indexed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Submission Index Status'
        verbose_name_plural = 'Submission Index Statuses'

    def __str__(self):
        return f"Index [{self.status}] for submission {self.submission_id}"


# =============================================================================
# 2. VIVA QUESTION EXTENSION
# Adds rubric criterion linkage and difficulty tracking to the core VivaQuestion.
# One extension per question — OneToOne relationship.
# =============================================================================

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

    # Which rubric criterion this question is targeting
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

    # The report chunks that were retrieved from FAISS to generate this question
    # Stored as JSON list of strings — useful for XAI and debugging
    source_chunks = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = 'Viva Question Extension'
        verbose_name_plural = 'Viva Question Extensions'

    def __str__(self):
        return f"Extension for Q{self.question.question_order} [{self.difficulty_level}]"


# =============================================================================
# 3. VIVA ANSWER EXTENSION
# Adds semantic scoring details to the core VivaAnswer.
# One extension per answer — OneToOne relationship.
# =============================================================================

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

    # Raw cosine similarity between answer embedding and ideal context embedding
    # Value between 0.0 and 1.0
    semantic_similarity_score = models.DecimalField(
        max_digits=5,
        decimal_places=4,
        null=True,
        blank=True,
    )

    # Score given by the LLM based on reasoning (0-10)
    llm_score = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True,
    )

    # The LLM's reasoning for the score — this feeds into XAI explanation
    llm_reasoning = models.TextField(null=True, blank=True)

    # What the adaptive engine decided the next question difficulty should be
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
        return f"Answer extension — similarity: {self.semantic_similarity_score}"