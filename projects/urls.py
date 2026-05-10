"""
URL configuration for the Projects app.

All routes are prefixed with /api/projects/ when included
in the main project urls.py.
"""

from django.urls import path

from projects.views.project_views import (
    AddExaminerView, AvailableProjectsView, EnrollInProjectView,
    ListExaminersView, MyEnrollmentsView, ProjectActivateView,
    ProjectCreateView, ProjectDetailView, ProjectListView,
    ProjectUpdateView, RemoveExaminerView, SubmissionDetailView,
    SubmitProjectView,
)
from projects.views.rubric_views import (
    RubricCategoryCreateView, RubricCategoryDeleteView,
    RubricCategoryUpdateView, RubricCriteriaCreateView,
    RubricCriteriaDeleteView, RubricCriteriaUpdateView, RubricListView,
)
from projects.views.session_views import (
    AutoScheduleView, ManualScheduleView, MySessionView,
    SessionListView, SessionResetView, SessionUpdateView,
)

app_name = 'projects'

urlpatterns = [
    # ── Project Management (Examiner) ────────────────────────────────────
    path('create/', ProjectCreateView.as_view(), name='project-create'),
    path('', ProjectListView.as_view(), name='project-list'),
    path('<uuid:project_id>/', ProjectDetailView.as_view(), name='project-detail'),
    path('<uuid:project_id>/update/', ProjectUpdateView.as_view(), name='project-update'),
    path('<uuid:project_id>/activate/', ProjectActivateView.as_view(), name='project-activate'),

    # ── Examiner Assignment ──────────────────────────────────────────────
    path('<uuid:project_id>/examiners/', ListExaminersView.as_view(), name='examiner-list'),
    path('<uuid:project_id>/examiners/add/', AddExaminerView.as_view(), name='examiner-add'),
    path('<uuid:project_id>/examiners/remove/', RemoveExaminerView.as_view(), name='examiner-remove'),

    # ── Student Enrollment ───────────────────────────────────────────────
    path('available/', AvailableProjectsView.as_view(), name='available-projects'),
    path('my-enrollments/', MyEnrollmentsView.as_view(), name='my-enrollments'),
    path('<uuid:project_id>/enroll/', EnrollInProjectView.as_view(), name='enroll'),

    # ── Submissions ──────────────────────────────────────────────────────
    path('<uuid:project_id>/submit/', SubmitProjectView.as_view(), name='submit'),
    path('<uuid:project_id>/submission/', SubmissionDetailView.as_view(), name='submission-detail'),

    # ── Rubrics ──────────────────────────────────────────────────────────
    path('<uuid:project_id>/rubrics/', RubricListView.as_view(), name='rubric-list'),
    path('<uuid:project_id>/rubrics/categories/create/', RubricCategoryCreateView.as_view(), name='rubric-category-create'),
    path('<uuid:project_id>/rubrics/categories/<uuid:category_id>/update/', RubricCategoryUpdateView.as_view(), name='rubric-category-update'),
    path('<uuid:project_id>/rubrics/categories/<uuid:category_id>/delete/', RubricCategoryDeleteView.as_view(), name='rubric-category-delete'),
    path('<uuid:project_id>/rubrics/categories/<uuid:category_id>/criteria/create/', RubricCriteriaCreateView.as_view(), name='rubric-criteria-create'),
    path('<uuid:project_id>/rubrics/categories/<uuid:category_id>/criteria/<uuid:criteria_id>/update/', RubricCriteriaUpdateView.as_view(), name='rubric-criteria-update'),
    path('<uuid:project_id>/rubrics/categories/<uuid:category_id>/criteria/<uuid:criteria_id>/delete/', RubricCriteriaDeleteView.as_view(), name='rubric-criteria-delete'),

    # ── Session Scheduling ───────────────────────────────────────────────
    path('<uuid:project_id>/sessions/', SessionListView.as_view(), name='session-list'),
    path('<uuid:project_id>/sessions/my-session/', MySessionView.as_view(), name='my-session'),
    path('<uuid:project_id>/sessions/schedule/manual/', ManualScheduleView.as_view(), name='schedule-manual'),
    path('<uuid:project_id>/sessions/schedule/auto/', AutoScheduleView.as_view(), name='schedule-auto'),
    path('<uuid:project_id>/sessions/<uuid:session_id>/update/', SessionUpdateView.as_view(), name='session-update'),
    path('<uuid:project_id>/sessions/reset/', SessionResetView.as_view(), name='session-reset'),
]
