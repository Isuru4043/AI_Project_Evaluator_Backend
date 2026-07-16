"""
URL configuration for the Agora service.

Prefixed with /api/ when included in the main urls.py.
"""

from django.urls import path

from agora_service.views import AgoraTokenView, AgoraRosterView

app_name = 'agora_service'

urlpatterns = [
    path(
        'sessions/<uuid:session_id>/agora-token/',
        AgoraTokenView.as_view(),
        name='agora-token',
    ),
    path(
        'sessions/<uuid:session_id>/agora-roster/',
        AgoraRosterView.as_view(),
        name='agora-roster',
    ),
]
