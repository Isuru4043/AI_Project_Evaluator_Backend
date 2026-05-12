from rest_framework import serializers
from core.models import (
    Project, RubricCategory,
    RubricCriteria, ProjectSubmission, EvaluationSession
)
from viva_evaluator.models import (
    SubmissionIndexStatus, CriteriaQuestionHint
)


# =============================================================================
# SUBMISSION SERIALIZERS
# =============================================================================

class SubmissionUploadSerializer(serializers.ModelSerializer):
    report_file = serializers.FileField(write_only=True)

    class Meta:
        model = ProjectSubmission
        fields = ['id', 'project', 'student', 'group', 'report_file', 'submitted_at']
        read_only_fields = ['id', 'submitted_at']

    def validate_report_file(self, value):
        ext = value.name.split('.')[-1].lower()
        if ext not in ['pdf', 'docx']:
            raise serializers.ValidationError("Only PDF and DOCX files are accepted.")
        return value

    def create(self, validated_data):
        report_file = validated_data.pop('report_file')
        submission = ProjectSubmission.objects.create(**validated_data)
        SubmissionIndexStatus.objects.create(
            submission=submission,
            report_file=report_file,
            status=SubmissionIndexStatus.IndexStatus.PENDING,
        )
        return submission


class SubmissionIndexStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubmissionIndexStatus
        fields = ['submission', 'status', 'error_message', 'processed_at']


# =============================================================================
# RUBRIC SERIALIZERS
# =============================================================================

class CriteriaQuestionHintSerializer(serializers.ModelSerializer):
    class Meta:
        model = CriteriaQuestionHint
        fields = ['id', 'hint_text', 'order']
        read_only_fields = ['id']


class RubricCriteriaSerializer(serializers.ModelSerializer):
    question_hints = CriteriaQuestionHintSerializer(many=True, required=False)

    class Meta:
        model = RubricCriteria
        fields = [
            'id', 'criteria_name', 'max_score',
            'weight_in_category', 'description',
            'questions_to_ask', 'question_hints',
        ]
        read_only_fields = ['id']

    def create(self, validated_data):
        hints_data = validated_data.pop('question_hints', [])
        criteria = RubricCriteria.objects.create(**validated_data)
        for hint in hints_data:
            CriteriaQuestionHint.objects.create(criteria=criteria, **hint)
        return criteria


class RubricCategorySerializer(serializers.ModelSerializer):
    criteria = RubricCriteriaSerializer(many=True, required=False)

    # Read-only warning field — not stored, just returned in response
    weight_warning = serializers.SerializerMethodField()

    class Meta:
        model = RubricCategory
        fields = [
            'id', 'category_name', 'weight_percentage',
            'description', 'criteria', 'weight_warning',
        ]
        read_only_fields = ['id', 'weight_warning']

    def get_weight_warning(self, obj):
        """
        Checks if criterion weights within this category add up to 100.
        Returns a warning string if not, None if correct.
        """
        criteria = obj.criteria.all()
        weights = [
            float(c.weight_in_category)
            for c in criteria
            if c.weight_in_category is not None
        ]
        if not weights:
            return None
        total = sum(weights)
        if abs(total - 100.0) > 0.01:
            return (
                f"Criterion weights in '{obj.category_name}' add up to "
                f"{total}% instead of 100%. Please review."
            )
        return None

    def create(self, validated_data):
        criteria_data = validated_data.pop('criteria', [])
        category = RubricCategory.objects.create(**validated_data)
        for criterion_data in criteria_data:
            hints_data = criterion_data.pop('question_hints', [])
            criterion = RubricCriteria.objects.create(
                category=category, **criterion_data
            )
            for hint in hints_data:
                CriteriaQuestionHint.objects.create(
                    criteria=criterion, **hint
                )
        return category


class ProjectCreateSerializer(serializers.ModelSerializer):
    rubric_categories = RubricCategorySerializer(many=True, required=False)

    class Meta:
        model = Project
        fields = [
            'id', 'project_name', 'description',
            'is_group_project', 'submission_deadline',
            'academic_year', 'status', 'rubric_categories',
        ]
        read_only_fields = ['id']

    def validate_rubric_categories(self, categories):
        """
        Checks that all category weights add up to 100.
        Raises a warning but does not block saving.
        Stores warning in context for the view to include in response.
        """
        weights = [
            float(cat.get('weight_percentage', 0))
            for cat in categories
        ]
        total = sum(weights)
        if weights and abs(total - 100.0) > 0.01:
            # Store warning in context — view will include it in response
            if 'warnings' not in self.context:
                self.context['warnings'] = []
            self.context['warnings'].append(
                f"Category weights add up to {total}% instead of 100%. "
                f"Please review your rubric."
            )
        return categories

    def validate(self, data):
        """
        Also validates criterion weights within each category
        before saving.
        """
        categories = data.get('rubric_categories', [])
        warnings = self.context.get('warnings', [])

        for cat in categories:
            cat_name = cat.get('category_name', 'Unknown')
            criteria_list = cat.get('criteria', [])
            weights = [
                float(c.get('weight_in_category', 0))
                for c in criteria_list
                if c.get('weight_in_category') is not None
            ]
            if weights:
                total = sum(weights)
                if abs(total - 100.0) > 0.01:
                    warnings.append(
                        f"Criterion weights in '{cat_name}' add up to "
                        f"{total}% instead of 100%. Please review."
                    )

        if warnings:
            self.context['warnings'] = warnings

        return data

    def create(self, validated_data):
        categories_data = validated_data.pop('rubric_categories', [])
        project = Project.objects.create(**validated_data)

        for category_data in categories_data:
            criteria_data = category_data.pop('criteria', [])
            category = RubricCategory.objects.create(
                project=project, **category_data
            )
            for criterion_data in criteria_data:
                hints_data = criterion_data.pop('question_hints', [])
                criterion = RubricCriteria.objects.create(
                    category=category, **criterion_data
                )
                for hint in hints_data:
                    CriteriaQuestionHint.objects.create(
                        criteria=criterion, **hint
                    )

        return project


class ProjectDetailSerializer(serializers.ModelSerializer):
    rubric_categories = RubricCategorySerializer(many=True, read_only=True)

    class Meta:
        model = Project
        fields = [
            'id', 'project_name', 'description',
            'is_group_project', 'submission_deadline',
            'academic_year', 'status', 'rubric_categories',
            'created_at',
        ]

class EvaluationSessionCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvaluationSession
        fields = [
            'id', 'project', 'student', 'group',
            'submission', 'scheduled_start', 'scheduled_end',
            'location_room', 'status',
        ]
        read_only_fields = ['id', 'status']

    def validate(self, data):
        """
        Ensure either student or group is provided, not both.
        Ensure submission belongs to the project.
        """
        student = data.get('student')
        group = data.get('group')
        project = data.get('project')
        submission = data.get('submission')

        if not student and not group:
            raise serializers.ValidationError(
                "Either student or group must be provided."
            )
        if student and group:
            raise serializers.ValidationError(
                "Provide either student or group, not both."
            )
        if submission and submission.project != project:
            raise serializers.ValidationError(
                "Submission does not belong to the selected project."
            )
        return data


class EvaluationSessionDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvaluationSession
        fields = [
            'id', 'project', 'student', 'group',
            'submission', 'scheduled_start', 'scheduled_end',
            'actual_start', 'location_room', 'status',
        ]

class RubricCategoryUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = RubricCategory
        fields = [
            'category_name', 'weight_percentage', 'description',
        ]

class RubricCriteriaUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = RubricCriteria
        fields = [
            'criteria_name', 'max_score',
            'weight_in_category', 'description', 'questions_to_ask',
        ]