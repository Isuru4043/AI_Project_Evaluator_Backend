"""URL configuration for physiological signals.

Prefixed with /api/ when included in the main urls.py.
"""

from django.urls import path

from physiology.views import (
    PhysioBaselineView,
    PhysioDeviceView,
    PhysioSampleView,
    PhysioTimelineView,
)

app_name = 'physiology'

urlpatterns = [
    path(
        'sessions/<uuid:session_id>/physio/device/',
        PhysioDeviceView.as_view(),
        name='physio-device',
    ),
    path(
        'sessions/<uuid:session_id>/physio/samples/',
        PhysioSampleView.as_view(),
        name='physio-samples',
    ),
    path(
        'sessions/<uuid:session_id>/physio/baseline/<str:action>/',
        PhysioBaselineView.as_view(),
        name='physio-baseline',
    ),
    path(
        'sessions/<uuid:session_id>/physio/timeline/',
        PhysioTimelineView.as_view(),
        name='physio-timeline',
    ),
]
