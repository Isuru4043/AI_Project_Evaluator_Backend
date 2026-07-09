"""
URL configuration for the Authentication app.

All routes are prefixed with /api/auth/ when included
in the main project urls.py.
"""

from django.urls import path

from .views import (
    CookieTokenRefreshView,
    ExaminerRegisterView,
    LoginView,
    LogoutView,
    StudentRegisterView,
    UserProfileView,
)

app_name = 'authentication'

urlpatterns = [
    # Registration
    path('examiner/register/', ExaminerRegisterView.as_view(), name='examiner-register'),
    path('student/register/', StudentRegisterView.as_view(), name='student-register'),

    # Login / Logout
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),

    # Token refresh (cookie-based)
    path('token/refresh/', CookieTokenRefreshView.as_view(), name='token-refresh'),

    # Current user profile
    path('me/', UserProfileView.as_view(), name='user-profile'),
]
