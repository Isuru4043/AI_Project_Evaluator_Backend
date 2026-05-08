"""
URL configuration for the Authentication app.

All routes are prefixed with /api/auth/ when included
in the main project urls.py.
"""

from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .views import (
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

    # Token refresh (built-in SimpleJWT view)
    path('token/refresh/', TokenRefreshView.as_view(), name='token-refresh'),

    # Current user profile
    path('me/', UserProfileView.as_view(), name='user-profile'),
]
