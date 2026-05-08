from django.urls import path
from viva_evaluator.views import (
    SubmissionUploadView,
    SubmissionIndexView,
    SubmissionStatusView,
)

urlpatterns = [
    path('submissions/upload/', SubmissionUploadView.as_view(), name='submission-upload'),
    path('submissions/<uuid:submission_id>/index/', SubmissionIndexView.as_view(), name='submission-index'),
    path('submissions/<uuid:submission_id>/status/', SubmissionStatusView.as_view(), name='submission-status'),
]