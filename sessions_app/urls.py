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
    StudentEndDemoView,
    StudentSessionStatusView,
    StudentStartDemoView,
    StudentStartVivaView,
    PresencePingView,
)
from sessions_app.views_live import (
    LiveQuestionAnswerView,
    LiveQuestionCreateView,
    LiveQuestionListView,
    LiveQuestionPendingView,
)
from sessions_app.views_demo import (
    DemoAudioUploadView,
    DemoQueueStatusView,
    DemoScreenshotUploadView,
    StartWarmupView,
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
        'sessions/<uuid:session_id>/student/start-demo/',
        StudentStartDemoView.as_view(),
        name='student-start-demo',
    ),
    path(
        'sessions/<uuid:session_id>/student/start-viva/',
        StudentStartVivaView.as_view(),
        name='student-start-viva',
    ),
    path(
        'sessions/<uuid:session_id>/end-demo/',
        StudentEndDemoView.as_view(),
        name='student-end-demo',
    ),
    path(
        'sessions/<uuid:session_id>/end-viva/',
        EndVivaView.as_view(),
        name='end-viva',
    ),
    path(
        'sessions/<uuid:session_id>/presence/',
        PresencePingView.as_view(),
        name='presence-ping',
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

    # ── Live examiner interjection (during in-progress viva) ─────────────
    path(
        'sessions/<uuid:session_id>/live-questions/',
        LiveQuestionCreateView.as_view(),
        name='live-question-create',
    ),
    path(
        'sessions/<uuid:session_id>/live-questions/list/',
        LiveQuestionListView.as_view(),
        name='live-question-list',
    ),
    path(
        'sessions/<uuid:session_id>/live-questions/pending/',
        LiveQuestionPendingView.as_view(),
        name='live-question-pending',
    ),
    path(
        'sessions/<uuid:session_id>/live-questions/<uuid:question_id>/answer/',
        LiveQuestionAnswerView.as_view(),
        name='live-question-answer',
    ),

    # ── Demo Capture (presentation monitoring) ───────────────────────────
    path(
        'sessions/<uuid:session_id>/start-warmup/',
        StartWarmupView.as_view(),
        name='demo-start-warmup',
    ),
    path(
        'sessions/<uuid:session_id>/demo-audio/',
        DemoAudioUploadView.as_view(),
        name='demo-audio-upload',
    ),
    path(
        'sessions/<uuid:session_id>/demo-screenshot/',
        DemoScreenshotUploadView.as_view(),
        name='demo-screenshot-upload',
    ),
    path(
        'sessions/<uuid:session_id>/demo-queue-status/',
        DemoQueueStatusView.as_view(),
        name='demo-queue-status',
    ),
]

