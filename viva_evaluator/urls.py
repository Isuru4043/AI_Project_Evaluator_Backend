from django.urls import path
from viva_evaluator.views import (
    SubmissionUploadView,
    SubmissionStatusView,
    SessionStartView,
    AnswerSubmitView,
    SessionReportView,
    ProjectCreateView,
    ProjectListView,
    ProjectDetailView,
    RubricCategoryCreateView,
    RubricCriteriaCreateView,
    QuestionHintCreateView,
    RubricUploadPreviewView,
    RubricConfirmSaveView,
    EvaluationSessionCreateView,
    StudentListView,
    SessionListView,
    SessionStatusView,
    CurrentQuestionView,
    FinalScoreSubmitView,
    RubricCategoryUpdateView,
    RubricCriteriaUpdateView,
    QuestionHintDeleteView,
    BriefListView,
    BriefDetailView,
    BriefApproveView,
    BriefEditView,
    BriefRejectView,
    AblationRunView,
)

urlpatterns = [
    # Submissions
    path('submissions/upload/', SubmissionUploadView.as_view(), name='submission-upload'),
    path('submissions/<uuid:submission_id>/status/', SubmissionStatusView.as_view(), name='submission-status'),

    # Sessions
    path('sessions/create/', EvaluationSessionCreateView.as_view(), name='session-create'),
    path('sessions/start/', SessionStartView.as_view(), name='session-start'),
    path('sessions/<uuid:session_id>/answer/', AnswerSubmitView.as_view(), name='answer-submit'),
    path('sessions/<uuid:session_id>/report/', SessionReportView.as_view(), name='session-report'),
    path('sessions/<uuid:session_id>/status/', SessionStatusView.as_view(), name='session-status'),
    path('sessions/<uuid:session_id>/current/', CurrentQuestionView.as_view(), name='session-current'),
    path('sessions/<uuid:session_id>/final-scores/', FinalScoreSubmitView.as_view(), name='final-scores'),

    # Projects
    path('projects/', ProjectListView.as_view(), name='project-list-create'),
    path('projects/<uuid:project_id>/', ProjectDetailView.as_view(), name='project-detail'),
    path('projects/<uuid:project_id>/categories/', RubricCategoryCreateView.as_view(), name='category-create'),
    path('projects/<uuid:project_id>/sessions/', SessionListView.as_view(), name='session-list'),

    # Rubric
    path('categories/<uuid:category_id>/criteria/', RubricCriteriaCreateView.as_view(), name='criteria-create'),
    path('criteria/<uuid:criteria_id>/hints/', QuestionHintCreateView.as_view(), name='hint-create'),
    path('rubric/upload-preview/', RubricUploadPreviewView.as_view(), name='rubric-upload-preview'),
    path('rubric/confirm-save/', RubricConfirmSaveView.as_view(), name='rubric-confirm-save'),

    # Students
    path('students/', StudentListView.as_view(), name='student-list'),

    path('categories/<uuid:category_id>/', RubricCategoryUpdateView.as_view(), name='category-update'),
    path('criteria/<uuid:criteria_id>/', RubricCriteriaUpdateView.as_view(), name='criteria-update'),
    path('hints/<uuid:hint_id>/', QuestionHintDeleteView.as_view(), name='hint-delete'),

    # ── WEEK 4 — Examiner-in-the-loop brief review ──────────────────────────
    path('briefs/', BriefListView.as_view(), name='brief-list'),
    path('briefs/<uuid:brief_id>/', BriefDetailView.as_view(), name='brief-detail'),
    path('briefs/<uuid:brief_id>/approve/', BriefApproveView.as_view(), name='brief-approve'),
    path('briefs/<uuid:brief_id>/edit/', BriefEditView.as_view(), name='brief-edit'),
    path('briefs/<uuid:brief_id>/reject/', BriefRejectView.as_view(), name='brief-reject'),

    # ── WEEK 7 — Ablation experiment harness ────────────────────────────────
    path('ablation/run/', AblationRunView.as_view(), name='ablation-run'),
]
