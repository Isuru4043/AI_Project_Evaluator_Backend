"""
URL configuration for the Sessions app.

Prefixed with /api/ when included in main urls.py.
"""

from django.urls import path

from sessions_app.views import (
    ActiveSessionView,
    CompleteDemoView,
    EndVivaView,
    ExaminerVivaQuestionCreateView,
    ExaminerVivaQuestionDeleteView,
    ExaminerVivaQuestionListView,
    ExaminerVivaQuestionUpdateView,
    SessionPanelOpenView,
    StartDemoView,
    StudentSessionStatusView,
)

app_name = 'sessions_app'

urlpatterns = [
    # ── Session Panel (Examiner) ─────────────────────────────────────────
    path(
        'projects/<uuid:project_id>/session-panel/open/',
        SessionPanelOpenView.as_view(),
        name='session-panel-open',
    ),
    path(
        'projects/<uuid:project_id>/session-panel/active/',
        ActiveSessionView.as_view(),
        name='session-panel-active',
    ),

    # ── Demo Flow ────────────────────────────────────────────────────────
    path(
        'sessions/<uuid:session_id>/start-demo/',
        StartDemoView.as_view(),
        name='start-demo',
    ),
    path(
        'sessions/<uuid:session_id>/complete-demo/',
        CompleteDemoView.as_view(),
        name='complete-demo',
    ),
    path(
        'sessions/<uuid:session_id>/end-viva/',
        EndVivaView.as_view(),
        name='end-viva',
    ),

    # ── Examiner Viva Questions ──────────────────────────────────────────
    path(
        'projects/<uuid:project_id>/viva/questions/create/',
        ExaminerVivaQuestionCreateView.as_view(),
        name='viva-question-create',
    ),
    path(
        'projects/<uuid:project_id>/viva/questions/',
        ExaminerVivaQuestionListView.as_view(),
        name='viva-question-list',
    ),
    path(
        'projects/<uuid:project_id>/viva/questions/<uuid:question_id>/update/',
        ExaminerVivaQuestionUpdateView.as_view(),
        name='viva-question-update',
    ),
    path(
        'projects/<uuid:project_id>/viva/questions/<uuid:question_id>/delete/',
        ExaminerVivaQuestionDeleteView.as_view(),
        name='viva-question-delete',
    ),

    # ── Student Session Status ───────────────────────────────────────────
    path(
        'sessions/my-status/',
        StudentSessionStatusView.as_view(),
        name='student-session-status',
    ),
]
