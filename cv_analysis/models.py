"""CV/behavioral session report — seam 3 of the exam-station-cv module.

Stores the versioned summary artifact produced by post-hoc analysis of the
session recording (attribution timeline, per-student behavioral summary,
integrity flags with video timecodes).

INVARIANT: everything in the artifact is advisory decision-support for the
examiner. Nothing here feeds score computation; integrity flags are
timecoded evidence pointers into the recording, never verdicts.
"""

import uuid

from django.db import models

from core.models import EvaluationSession


class CVSessionReport(models.Model):
    """One behavioral analysis report per evaluation session."""

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.OneToOneField(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='cv_report',
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    # SessionSummary JSON from exam-station-cv (schema_version inside).
    artifact = models.JSONField(null=True, blank=True)
    # Blob URL of the recording that was analyzed.
    recording_url = models.TextField(blank=True, default='')
    error_message = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'CV Session Report'
        verbose_name_plural = 'CV Session Reports'
        ordering = ['-created_at']

    def __str__(self):
        return f"CV report for {self.session_id} [{self.status}]"
