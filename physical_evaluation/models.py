import hashlib
import uuid

from django.contrib.auth.hashers import check_password, make_password
from django.db import models
from django.utils import timezone

from core.models import EvaluationSession, ExaminerProfile, Project, SessionRecording


class PhysicalProjectConfig(models.Model):
    """Physical venue and secret used to lock/unlock a project's kiosk panel."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.OneToOneField(
        Project,
        on_delete=models.CASCADE,
        related_name='physical_config',
    )
    location = models.CharField(max_length=255)
    panel_pin_hash = models.CharField(max_length=128)
    created_by = models.ForeignKey(
        ExaminerProfile,
        on_delete=models.PROTECT,
        related_name='physical_project_configs',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def set_panel_pin(self, raw_pin):
        self.panel_pin_hash = make_password(raw_pin)

    def check_panel_pin(self, raw_pin):
        return bool(raw_pin) and check_password(raw_pin, self.panel_pin_hash)

    def __str__(self):
        return f'{self.project.project_name} — {self.location}'


class PhysicalKioskAccess(models.Model):
    """A revocable, limited-scope lease for one opened kiosk panel."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    config = models.ForeignKey(
        PhysicalProjectConfig,
        on_delete=models.CASCADE,
        related_name='kiosk_accesses',
    )
    opened_by = models.ForeignKey(
        ExaminerProfile,
        on_delete=models.PROTECT,
        related_name='opened_physical_kiosks',
    )
    token_digest = models.CharField(max_length=64, unique=True, db_index=True)
    expires_at = models.DateTimeField()
    closed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity_at = models.DateTimeField(auto_now_add=True)

    @staticmethod
    def digest_token(raw_token):
        return hashlib.sha256(raw_token.encode('utf-8')).hexdigest()

    @property
    def is_active(self):
        return self.closed_at is None and self.expires_at > timezone.now()

    def close(self):
        if self.closed_at is None:
            self.closed_at = timezone.now()
            self.save(update_fields=['closed_at'])

    def __str__(self):
        return f'Kiosk {self.id} for {self.config.project}'


class PhysicalEvaluationRun(models.Model):
    """Physical-only capture state around the shared EvaluationSession."""

    class Status(models.TextChoices):
        DEMO_IN_PROGRESS = 'demo_in_progress', 'Demo in progress'
        VIVA_IN_PROGRESS = 'viva_in_progress', 'Viva in progress'
        RECORDING_UPLOADING = 'recording_uploading', 'Recording uploading'
        RECORDING_FAILED = 'recording_failed', 'Recording failed'
        COMPLETED = 'completed', 'Completed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.OneToOneField(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='physical_run',
    )
    kiosk_access = models.ForeignKey(
        PhysicalKioskAccess,
        on_delete=models.PROTECT,
        related_name='runs',
    )
    status = models.CharField(max_length=30, choices=Status.choices)
    recording_started_at = models.DateTimeField()
    viva_started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    recording = models.OneToOneField(
        SessionRecording,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='physical_run',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def is_active(self):
        return self.status in {
            self.Status.DEMO_IN_PROGRESS,
            self.Status.VIVA_IN_PROGRESS,
        }

    def __str__(self):
        return f'Physical run for {self.session_id} ({self.status})'


class PhysicalRecordingUpload(models.Model):
    """Progress of a chunked physical-room recording upload."""

    class Status(models.TextChoices):
        CAPTURING = 'capturing', 'Capturing'
        UPLOADING = 'uploading', 'Uploading'
        FINALIZING = 'finalizing', 'Finalizing'
        READY = 'ready', 'Ready'
        FAILED = 'failed', 'Failed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.OneToOneField(
        PhysicalEvaluationRun,
        on_delete=models.CASCADE,
        related_name='recording_upload',
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.CAPTURING,
    )
    blob_path = models.CharField(max_length=512, blank=True, default='')
    mime_type = models.CharField(max_length=100, blank=True, default='video/webm')
    expected_chunks = models.PositiveIntegerField(null=True, blank=True)
    uploaded_chunk_indices = models.JSONField(default=list, blank=True)
    duration_seconds = models.PositiveIntegerField(null=True, blank=True)
    finalization_requested = models.BooleanField(default=False)
    error_message = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    finalized_at = models.DateTimeField(null=True, blank=True)

    @property
    def uploaded_chunks(self):
        return len(set(self.uploaded_chunk_indices or []))

    def __str__(self):
        return f'Recording upload for {self.run.session_id} ({self.status})'
