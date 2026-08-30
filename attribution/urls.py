"""URL configuration for speaker attribution.

Prefixed with /api/ when included in the main urls.py.
"""

from django.urls import path

from attribution.views import (
    AttributionConfirmView,
    AttributionReconcileView,
    AttributionReviewView,
    EvidenceIngestView,
    SeatBindingView,
    SpeakerDetectionTestView,
    StationArtifactView,
    UnknownSpeakerView,
)

app_name = 'attribution'

urlpatterns = [
    path(
        'attribution/speaker-detection-test/bind/',
        SpeakerDetectionTestView.as_view(),
        name='speaker-detection-test-bind',
    ),
    path(
        'sessions/<uuid:session_id>/attribution/evidence/',
        EvidenceIngestView.as_view(),
        name='attribution-evidence',
    ),
    path(
        'sessions/<uuid:session_id>/attribution/bind/',
        SeatBindingView.as_view(),
        name='attribution-bind',
    ),
    path(
        'sessions/<uuid:session_id>/attribution/artifact/',
        StationArtifactView.as_view(),
        name='attribution-artifact',
    ),
    path(
        'sessions/<uuid:session_id>/attribution/unknown-speakers/',
        UnknownSpeakerView.as_view(),
        name='attribution-unknown-speakers',
    ),
    path(
        'sessions/<uuid:session_id>/attribution/answers/',
        AttributionReviewView.as_view(),
        name='attribution-answers',
    ),
    path(
        'sessions/<uuid:session_id>/attribution/answers/<uuid:answer_id>/confirm/',
        AttributionConfirmView.as_view(),
        name='attribution-confirm',
    ),
    path(
        'sessions/<uuid:session_id>/attribution/reconcile/',
        AttributionReconcileView.as_view(),
        name='attribution-reconcile',
    ),
]
