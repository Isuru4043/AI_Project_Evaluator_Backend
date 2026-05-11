from django.urls import path
from viva_evaluator.views import (
    SubmissionUploadView,
    SubmissionStatusView,
    SessionStartView,
    AnswerSubmitView,
    SessionReportView,
)

urlpatterns = [
    path('submissions/upload/', SubmissionUploadView.as_view(), name='submission-upload'),
    path('submissions/<uuid:submission_id>/status/', SubmissionStatusView.as_view(), name='submission-status'),
    path('sessions/start/', SessionStartView.as_view(), name='session-start'),
    path('sessions/<uuid:session_id>/answer/', AnswerSubmitView.as_view(), name='answer-submit'),
    path('sessions/<uuid:session_id>/report/', SessionReportView.as_view(), name='session-report'),
]