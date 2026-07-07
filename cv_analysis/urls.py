"""URL configuration for CV/behavioral analysis.

Prefixed with /api/ when included in the main urls.py.
"""

from django.urls import path

from cv_analysis.views import (
    CVAnalyzeTriggerView,
    CVRecordingDownloadView,
    CVRecordingUploadView,
    CVSummaryView,
)

app_name = 'cv_analysis'

urlpatterns = [
    path(
        'sessions/<uuid:session_id>/cv/recording/',
        CVRecordingUploadView.as_view(),
        name='cv-recording',
    ),
    path(
        'sessions/<uuid:session_id>/cv/recording/download/',
        CVRecordingDownloadView.as_view(),
        name='cv-recording-download',
    ),
    path(
        'sessions/<uuid:session_id>/cv/analyze/',
        CVAnalyzeTriggerView.as_view(),
        name='cv-analyze',
    ),
    path(
        'sessions/<uuid:session_id>/cv/summary/',
        CVSummaryView.as_view(),
        name='cv-summary',
    ),
]
