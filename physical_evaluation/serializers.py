from rest_framework import serializers

from core.models import GroupMember
from physical_evaluation.models import (
    PhysicalEvaluationRun,
    PhysicalProjectConfig,
    PhysicalRecordingUpload,
)


class PhysicalProjectConfigSerializer(serializers.ModelSerializer):
    project_id = serializers.UUIDField(read_only=True)
    project_name = serializers.CharField(source='project.project_name', read_only=True)
    evaluation_mode = serializers.CharField(source='project.evaluation_mode', read_only=True)
    panel_pin_configured = serializers.SerializerMethodField()

    class Meta:
        model = PhysicalProjectConfig
        fields = [
            'project_id', 'project_name', 'evaluation_mode', 'location',
            'panel_pin_configured', 'updated_at',
        ]

    def get_panel_pin_configured(self, obj):
        return bool(obj.panel_pin_hash)


class PhysicalSettingsUpdateSerializer(serializers.Serializer):
    location = serializers.CharField(max_length=255, required=False)
    panel_pin = serializers.CharField(
        min_length=4, max_length=128, required=False, trim_whitespace=False,
    )

    def validate(self, attrs):
        if not attrs:
            raise serializers.ValidationError('Provide location and/or panel_pin.')
        return attrs


class PanelPinSerializer(serializers.Serializer):
    pin = serializers.CharField(max_length=128, trim_whitespace=False)


class IdentityOverrideSerializer(PanelPinSerializer):
    reason = serializers.CharField(min_length=5, max_length=500)


class PhysicalSessionSerializer(serializers.Serializer):
    def to_representation(self, session):
        student = None
        group = None
        if session.student_id:
            student = {
                'student_id': str(session.student_id),
                'full_name': session.student.user.full_name,
                'registration_number': session.student.registration_number,
            }
        if session.group_id:
            members = GroupMember.objects.filter(group=session.group).select_related('student__user')
            group = {
                'group_id': str(session.group_id),
                'group_name': session.group.group_name,
                'members': [
                    {
                        'student_id': str(member.student_id),
                        'full_name': member.student.user.full_name,
                        'registration_number': member.student.registration_number,
                    }
                    for member in members
                ],
            }

        try:
            run = session.physical_run
        except Exception:
            run = None

        submission = session.submission
        if submission is None:
            if session.group_id:
                submission = session.project.submissions.filter(group_id=session.group_id).first()
            elif session.student_id:
                submission = session.project.submissions.filter(student_id=session.student_id).first()
        submission_ready = bool(
            submission
            and hasattr(submission, 'index_status')
            and submission.index_status.status == 'ready'
        )

        return {
            'session_id': str(session.id),
            'project_name': session.project.project_name,
            'student': student,
            'group': group,
            'scheduled_start': session.scheduled_start,
            'scheduled_end': session.scheduled_end,
            'location': session.location_room or session.project.physical_config.location,
            'demo_enabled': session.demo_enabled,
            'status': session.status,
            'physical_status': run.status if run else None,
            'submission_ready': submission_ready,
        }


class PhysicalRunSerializer(serializers.ModelSerializer):
    session = PhysicalSessionSerializer(read_only=True)
    recording_upload = serializers.SerializerMethodField()

    class Meta:
        model = PhysicalEvaluationRun
        fields = [
            'id', 'session', 'status', 'recording_started_at',
            'viva_started_at', 'completed_at', 'recording_upload',
            'identity_status', 'identity_verification', 'identity_verified_at',
            'identity_override_at', 'identity_override_reason',
        ]

    def get_recording_upload(self, obj):
        try:
            upload = obj.recording_upload
        except PhysicalRecordingUpload.DoesNotExist:
            return None
        return PhysicalRecordingUploadSerializer(upload).data


class PhysicalRecordingUploadSerializer(serializers.ModelSerializer):
    session_id = serializers.UUIDField(source='run.session_id', read_only=True)
    uploaded_chunks = serializers.IntegerField(read_only=True)

    class Meta:
        model = PhysicalRecordingUpload
        fields = [
            'id', 'session_id', 'status', 'mime_type', 'expected_chunks',
            'uploaded_chunks', 'uploaded_chunk_indices', 'duration_seconds', 'error_message',
            'created_at', 'updated_at', 'finalized_at',
        ]
