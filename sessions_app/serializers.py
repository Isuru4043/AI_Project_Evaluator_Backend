"""
Serializers for the Sessions app.

Covers: session panel, demo flow, viva questions,
end-viva media upload, and student session status.
"""

from rest_framework import serializers

from core.models import EvaluationSession, GroupMember, SessionRecording, VivaQuestion


class SessionPanelEntrySerializer(serializers.ModelSerializer):
    """Serializes a single session for the examiner panel."""

    session_id = serializers.UUIDField(source='id', read_only=True)
    student_full_name = serializers.CharField(source='student.user.full_name', read_only=True, default=None)
    registration_number = serializers.CharField(source='student.registration_number', read_only=True, default=None)
    group_name = serializers.CharField(source='group.group_name', read_only=True, default=None)
    group_members = serializers.SerializerMethodField()

    class Meta:
        model = EvaluationSession
        fields = [
            'session_id', 'student_full_name', 'registration_number',
            'group_name', 'group_members', 'scheduled_start', 'scheduled_end',
            'location_room', 'status',
        ]

    def get_group_members(self, obj):
        if not obj.group:
            return None
        members = GroupMember.objects.filter(
            group=obj.group,
        ).select_related('student__user')
        return [m.student.user.full_name for m in members]


class EvaluationSessionDetailSerializer(serializers.ModelSerializer):
    """Detailed session serializer for demo start/complete responses."""

    session_id = serializers.UUIDField(source='id', read_only=True)
    student_full_name = serializers.CharField(source='student.user.full_name', read_only=True, default=None)
    registration_number = serializers.CharField(source='student.registration_number', read_only=True, default=None)
    group_name = serializers.CharField(source='group.group_name', read_only=True, default=None)
    project_name = serializers.CharField(source='project.project_name', read_only=True)

    class Meta:
        model = EvaluationSession
        fields = [
            'session_id', 'project_name', 'student_full_name',
            'registration_number', 'group_name', 'scheduled_start',
            'scheduled_end', 'actual_start', 'demo_completed_at',
            'location_room', 'status',
        ]


class VivaQuestionSerializer(serializers.ModelSerializer):
    """Serializes a viva question."""

    question_id = serializers.UUIDField(source='id', read_only=True)

    class Meta:
        model = VivaQuestion
        fields = [
            'question_id', 'question_text', 'blooms_level',
            'question_source', 'question_order', 'generated_at',
        ]


class VivaQuestionCreateSerializer(serializers.Serializer):
    """Validates input for creating an examiner viva question."""

    question_text = serializers.CharField()
    blooms_level = serializers.ChoiceField(
        choices=['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'],
        required=False, allow_null=True, default=None,
    )
    question_order = serializers.IntegerField()


class VivaQuestionUpdateSerializer(serializers.Serializer):
    """Validates input for updating a viva question."""

    question_text = serializers.CharField(required=False)
    blooms_level = serializers.ChoiceField(
        choices=['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'],
        required=False, allow_null=True,
    )
    question_order = serializers.IntegerField(required=False)


class SessionRecordingSerializer(serializers.ModelSerializer):
    """Serializes a session recording."""

    class Meta:
        model = SessionRecording
        fields = [
            'id', 'session', 'video_file_url', 'audio_file_url',
            'duration_seconds', 'recorded_at',
        ]


class StudentSessionStatusSerializer(serializers.ModelSerializer):
    """Serializes session status for the student dashboard."""

    project_name = serializers.CharField(source='project.project_name', read_only=True)
    group_name = serializers.CharField(source='group.group_name', read_only=True, default=None)

    class Meta:
        model = EvaluationSession
        fields = [
            'project_name', 'scheduled_start', 'scheduled_end',
            'location_room', 'status', 'group_name', 'demo_completed_at',
        ]
