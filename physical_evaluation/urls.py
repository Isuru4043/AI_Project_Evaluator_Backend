from django.urls import path

from physical_evaluation.views import (
    KioskActiveRunView,
    KioskCloseView,
    KioskDemoCompleteView,
    KioskIdentityOverrideView,
    KioskOpenView,
    KioskSessionCompleteView,
    KioskSessionFinishView,
    KioskSessionListView,
    KioskSessionStartView,
    KioskRecordingChunkUploadView,
    KioskRecordingFinalizeView,
    KioskRecordingStartView,
    KioskRecordingStatusView,
    PhysicalProjectSettingsView,
)


app_name = 'physical_evaluation'

urlpatterns = [
    path(
        'projects/<uuid:project_id>/settings/',
        PhysicalProjectSettingsView.as_view(),
        name='project-settings',
    ),
    path(
        'projects/<uuid:project_id>/kiosk/open/',
        KioskOpenView.as_view(),
        name='kiosk-open',
    ),
    path('kiosk/close/', KioskCloseView.as_view(), name='kiosk-close'),
    path('kiosk/sessions/', KioskSessionListView.as_view(), name='kiosk-sessions'),
    path('kiosk/active/', KioskActiveRunView.as_view(), name='kiosk-active'),
    path(
        'kiosk/sessions/<uuid:session_id>/start/',
        KioskSessionStartView.as_view(),
        name='kiosk-session-start',
    ),
    path(
        'kiosk/sessions/<uuid:session_id>/demo/complete/',
        KioskDemoCompleteView.as_view(),
        name='kiosk-demo-complete',
    ),
    path(
        'kiosk/sessions/<uuid:session_id>/identity/override/',
        KioskIdentityOverrideView.as_view(),
        name='kiosk-identity-override',
    ),
    path(
        'kiosk/sessions/<uuid:session_id>/complete/',
        KioskSessionCompleteView.as_view(),
        name='kiosk-session-complete',
    ),
    path(
        'kiosk/sessions/<uuid:session_id>/finish/',
        KioskSessionFinishView.as_view(),
        name='kiosk-session-finish',
    ),
    path(
        'kiosk/sessions/<uuid:session_id>/recording/start/',
        KioskRecordingStartView.as_view(),
        name='kiosk-recording-start',
    ),
    path(
        'kiosk/sessions/<uuid:session_id>/recording/chunks/<int:chunk_index>/',
        KioskRecordingChunkUploadView.as_view(),
        name='kiosk-recording-chunk',
    ),
    path(
        'kiosk/sessions/<uuid:session_id>/recording/finalize/',
        KioskRecordingFinalizeView.as_view(),
        name='kiosk-recording-finalize',
    ),
    path(
        'kiosk/sessions/<uuid:session_id>/recording/status/',
        KioskRecordingStatusView.as_view(),
        name='kiosk-recording-status',
    ),
]
