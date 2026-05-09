import re

from django.conf import settings
from rest_framework import serializers

from core.models import CodeSubmission, GeneratedVivaQuestion


class CodeSubmissionCreateSerializer(serializers.Serializer):
    project_submission_id = serializers.UUIDField()
    source_type = serializers.ChoiceField(choices=CodeSubmission.SourceType.choices)
    github_url = serializers.URLField(required=False, allow_blank=True, allow_null=True)
    zip_file = serializers.FileField(required=False, allow_null=True)
    build_command = serializers.CharField(required=False, allow_blank=True, allow_null=True)

    def validate(self, attrs):
        source_type = attrs.get("source_type")
        github_url = attrs.get("github_url")
        zip_file = attrs.get("zip_file")

        if source_type == CodeSubmission.SourceType.GITHUB and not github_url:
            raise serializers.ValidationError({
                "github_url": "GitHub URL is required for github source type."
            })

        if source_type == CodeSubmission.SourceType.ZIP and not zip_file:
            raise serializers.ValidationError({
                "zip_file": "ZIP file is required for zip source type."
            })

        if github_url:
            pattern = r"^https://github\.com/[^/]+/[^/]+(?:\.git)?/?$"
            if not re.match(pattern, github_url):
                raise serializers.ValidationError({
                    "github_url": "Invalid GitHub repository URL."
                })

        if zip_file:
            max_mb = getattr(settings, "CODE_ANALYSIS_MAX_ZIP_MB", 100)
            if zip_file.size > max_mb * 1024 * 1024:
                raise serializers.ValidationError({
                    "zip_file": f"ZIP file exceeds {max_mb}MB limit."
                })

        return attrs


class CodeSubmissionStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = CodeSubmission
        fields = [
            "id",
            "analysis_status",
            "analysis_error",
            "language_detected",
            "build_system_detected",
            "build_command",
            "sonar_report_url",
            "quality_status",
            "quality_reason",
            "uploaded_at",
            "analyzed_at",
            "questions_generated_at",
        ]


class GeneratedVivaQuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneratedVivaQuestion
        fields = [
            "id",
            "question_text",
            "blooms_level",
            "source_type",
            "code_reference",
            "sonar_issue_reference",
            "reasoning",
            "created_at",
        ]


class CodeSubmissionSonarSummarySerializer(serializers.ModelSerializer):
    sonar_metrics = serializers.SerializerMethodField()
    sonar_dashboard = serializers.SerializerMethodField()

    class Meta:
        model = CodeSubmission
        fields = [
            "id",
            "sonar_summary",
            "sonar_metrics",
            "sonar_dashboard",
            "sonar_report_url",
            "quality_status",
            "quality_reason",
            "analyzed_at",
        ]

    def get_sonar_metrics(self, obj):
        measures = _sonar_measures_map(obj)

        return {
            "bugs": measures.get("bugs"),
            "vulnerabilities": measures.get("vulnerabilities"),
            "code_smells": measures.get("code_smells"),
            "security_rating": measures.get("security_rating"),
            "reliability_rating": measures.get("reliability_rating"),
            "maintainability_rating": measures.get("sqale_rating"),
            "security_hotspots_reviewed": measures.get("security_hotspots_reviewed"),
            "coverage": measures.get("coverage"),
            "duplicated_lines_density": measures.get("duplicated_lines_density"),
            "ncloc": measures.get("ncloc"),
            "open_issues": (obj.sonar_summary or {}).get("total_issues"),
        }

    def get_sonar_dashboard(self, obj):
        measures = _sonar_measures_map(obj)

        return {
            "security": {
                "rating": _rating_letter(measures.get("security_rating")),
                "rating_value": measures.get("security_rating"),
                "issues": measures.get("vulnerabilities"),
                "open_hotspots": measures.get("security_hotspots"),
            },
            "reliability": {
                "rating": _rating_letter(measures.get("reliability_rating")),
                "rating_value": measures.get("reliability_rating"),
                "issues": measures.get("bugs"),
            },
            "maintainability": {
                "rating": _rating_letter(measures.get("sqale_rating")),
                "rating_value": measures.get("sqale_rating"),
                "issues": measures.get("code_smells"),
            },
            "hotspots_reviewed": {
                "value": measures.get("security_hotspots_reviewed"),
                "open_hotspots": measures.get("security_hotspots"),
            },
            "coverage": {
                "value": measures.get("coverage"),
            },
            "duplications": {
                "value": measures.get("duplicated_lines_density"),
            },
            "open_issues": (obj.sonar_summary or {}).get("total_issues"),
        }


class CodeSubmissionQuestionsSerializer(serializers.ModelSerializer):
    questions = GeneratedVivaQuestionSerializer(many=True, source="generated_questions")

    class Meta:
        model = CodeSubmission
        fields = [
            "id",
            "code_summary",
            "questions",
            "questions_generated_at",
        ]


def _rating_letter(value):
    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    mapping = {
        1.0: "A",
        2.0: "B",
        3.0: "C",
        4.0: "D",
        5.0: "E",
    }
    return mapping.get(numeric)


def _sonar_measures_map(obj):
    measures = {}
    for measure in (obj.sonar_summary or {}).get("measures", []):
        metric = measure.get("metric")
        if metric:
            measures[metric] = measure.get("value")
    return measures