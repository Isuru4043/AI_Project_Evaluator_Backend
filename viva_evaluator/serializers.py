from rest_framework import serializers
from core.models import ProjectSubmission
from viva_evaluator.models import SubmissionIndexStatus


class SubmissionUploadSerializer(serializers.ModelSerializer):
    """
    Handles the student report file upload.
    The report_file comes in as a multipart form upload.
    """
    report_file = serializers.FileField(write_only=True)

    class Meta:
        model = ProjectSubmission
        fields = ['id', 'project', 'student', 'group', 'report_file', 'submitted_at']
        read_only_fields = ['id', 'submitted_at']

    def validate_report_file(self, value):
        """Only allow PDF and DOCX files."""
        ext = value.name.split('.')[-1].lower()
        if ext not in ['pdf', 'docx']:
            raise serializers.ValidationError("Only PDF and DOCX files are accepted.")
        return value

    def create(self, validated_data):
        report_file = validated_data.pop('report_file')

        # Create the ProjectSubmission record
        submission = ProjectSubmission.objects.create(**validated_data)

        # Create the index status record and attach the file
        SubmissionIndexStatus.objects.create(
            submission=submission,
            report_file=report_file,
            status=SubmissionIndexStatus.IndexStatus.PENDING,
        )

        return submission


class SubmissionIndexStatusSerializer(serializers.ModelSerializer):
    """Returns the current indexing state of a submission."""

    class Meta:
        model = SubmissionIndexStatus
        fields = ['submission', 'status', 'error_message', 'indexed_at']