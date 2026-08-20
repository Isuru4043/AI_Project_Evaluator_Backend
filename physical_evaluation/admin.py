from django.contrib import admin

from physical_evaluation.models import (
    PhysicalEvaluationRun,
    PhysicalKioskAccess,
    PhysicalProjectConfig,
    PhysicalRecordingUpload,
)


@admin.register(PhysicalProjectConfig)
class PhysicalProjectConfigAdmin(admin.ModelAdmin):
    list_display = ('project', 'location', 'created_by', 'updated_at')
    search_fields = ('project__project_name', 'location')
    exclude = ('panel_pin_hash',)

    def has_add_permission(self, request):
        # Configurations must be created through the project API so a securely
        # hashed PIN is always present.
        return False


@admin.register(PhysicalKioskAccess)
class PhysicalKioskAccessAdmin(admin.ModelAdmin):
    list_display = ('id', 'config', 'opened_by', 'created_at', 'expires_at', 'closed_at')
    readonly_fields = ('token_digest', 'created_at', 'last_activity_at')


@admin.register(PhysicalEvaluationRun)
class PhysicalEvaluationRunAdmin(admin.ModelAdmin):
    list_display = ('session', 'status', 'recording_started_at', 'completed_at')
    list_filter = ('status',)


@admin.register(PhysicalRecordingUpload)
class PhysicalRecordingUploadAdmin(admin.ModelAdmin):
    list_display = ('run', 'status', 'uploaded_chunks', 'expected_chunks', 'updated_at')
    list_filter = ('status',)
    readonly_fields = ('blob_path', 'uploaded_chunk_indices', 'created_at', 'updated_at')
