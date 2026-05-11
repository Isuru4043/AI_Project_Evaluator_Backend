import uuid
from django.db import models
from core.models import ProjectSubmission, VivaQuestion, VivaAnswer, RubricCriteria


class SubmissionIndexStatus(models.Model):

    class IndexStatus(models.TextChoices):
        PENDING   = 'pending',   'Pending'
        PROCESSING = 'processing', 'Processing'
        READY     = 'ready',     'Ready'
        FAILED    = 'failed',    'Failed'

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