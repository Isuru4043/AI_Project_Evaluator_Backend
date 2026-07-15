from django.urls import path

from .views import (
    CodeSubmissionCreateView,
    CodeSubmissionReportView,
    CodeSubmissionStatusView,
    CodeSubmissionQuestionsView,
    CodeSubmissionSonarSummaryView,
)

app_name = "code_analysis"

urlpatterns = [
    path("submit/", CodeSubmissionCreateView.as_view(), name="code-submit"),
    path(
        "submissions/<uuid:code_submission_id>/status/",
        CodeSubmissionStatusView.as_view(),
        name="code-status",
    ),
    path(
        "submissions/<uuid:code_submission_id>/sonar-summary/",
        CodeSubmissionSonarSummaryView.as_view(),
        name="code-sonar-summary",
    ),
    path(
        "submissions/<uuid:code_submission_id>/questions/",
        CodeSubmissionQuestionsView.as_view(),
        name="code-questions",
    ),
    path(
    "submissions/<uuid:code_submission_id>/report/",
    CodeSubmissionReportView.as_view(),
    name="code-report",
),
]
