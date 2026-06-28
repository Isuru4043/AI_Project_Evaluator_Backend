"""viva_evaluator views package (split from the former views.py)."""

from viva_evaluator.views.submission_views import (
    SubmissionUploadView,
    SubmissionStatusView,
)

from viva_evaluator.views.session_views import (
    SessionStartView,
    AnswerSubmitView,
    SessionReportView,
    EvaluationSessionCreateView,
    SessionListView,
    SessionStatusView,
)

from viva_evaluator.views.project_views import (
    ProjectCreateView,
    ProjectDetailView,
    ProjectListView,
    StudentListView,
)

from viva_evaluator.views.rubric_views import (
    RubricCategoryCreateView,
    RubricCriteriaCreateView,
    QuestionHintCreateView,
    RubricUploadPreviewView,
    RubricConfirmSaveView,
    RubricCategoryUpdateView,
    RubricCriteriaUpdateView,
    QuestionHintDeleteView,
)

from viva_evaluator.views.scoring_views import (
    FinalScoreSubmitView,
)

from viva_evaluator.views.brief_views import (
    BriefListView,
    BriefDetailView,
    BriefApproveView,
    BriefEditView,
    BriefRejectView,
)

from viva_evaluator.views.ablation_views import (
    AblationRunView,
)


__all__ = [
    'SubmissionUploadView',
    'SubmissionStatusView',
    'SessionStartView',
    'AnswerSubmitView',
    'SessionReportView',
    'EvaluationSessionCreateView',
    'SessionListView',
    'SessionStatusView',
    'ProjectCreateView',
    'ProjectDetailView',
    'ProjectListView',
    'StudentListView',
    'RubricCategoryCreateView',
    'RubricCriteriaCreateView',
    'QuestionHintCreateView',
    'RubricUploadPreviewView',
    'RubricConfirmSaveView',
    'RubricCategoryUpdateView',
    'RubricCriteriaUpdateView',
    'QuestionHintDeleteView',
    'FinalScoreSubmitView',
    'BriefListView',
    'BriefDetailView',
    'BriefApproveView',
    'BriefEditView',
    'BriefRejectView',
    'AblationRunView',
]
