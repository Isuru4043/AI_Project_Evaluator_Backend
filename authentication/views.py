"""
Views for the Authentication app.

Provides endpoints for examiner/student registration, login,
logout, token refresh, and fetching the current user profile.
"""

from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError

from django.conf import settings

from .cookies import clear_auth_cookies, set_auth_cookies
from .serializers import (
    ExaminerRegisterSerializer,
    LoginSerializer,
    StudentRegisterSerializer,
    UserProfileResponseSerializer,
)


def build_tokens_for_user(user):
    """
    Create a refresh/access pair for ``user`` and embed the ``role`` claim so
    the Next.js middleware can make role-based routing decisions by decoding
    the access token (which lives in an HttpOnly cookie) without a DB hit.
    """
    refresh = RefreshToken.for_user(user)
    refresh['role'] = user.role
    return refresh, refresh.access_token


# =============================================================================
# Examiner Registration
# =============================================================================

class ExaminerRegisterView(APIView):
    """
    POST /api/auth/examiner/register/

    Register a new examiner account. Creates both the User
    (with role='examiner') and the linked ExaminerProfile.
    """

    permission_classes = [AllowAny]

    def post(self, request):
        try:
            serializer = ExaminerRegisterSerializer(data=request.data)

            if not serializer.is_valid():
                return Response(
                    {
                        'success': False,
                        'message': 'Validation failed.',
                        'errors': serializer.errors,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            serializer.save()

            return Response(
                {
                    'success': True,
                    'message': 'Examiner registered successfully.',
                    'data': None,
                },
                status=status.HTTP_201_CREATED,
            )

        except Exception as e:
            return Response(
                {
                    'success': False,
                    'message': f'An unexpected error occurred: {str(e)}',
                    'errors': {},
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# =============================================================================
# Student Registration
# =============================================================================

class StudentRegisterView(APIView):
    """
    POST /api/auth/student/register/

    Register a new student account. Creates both the User
    (with role='student') and the linked StudentProfile.
    """

    permission_classes = [AllowAny]

    def post(self, request):
        try:
            serializer = StudentRegisterSerializer(data=request.data)

            if not serializer.is_valid():
                return Response(
                    {
                        'success': False,
                        'message': 'Validation failed.',
                        'errors': serializer.errors,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            serializer.save()

            return Response(
                {
                    'success': True,
                    'message': 'Student registered successfully.',
                    'data': None,
                },
                status=status.HTTP_201_CREATED,
            )

        except Exception as e:
            return Response(
                {
                    'success': False,
                    'message': f'An unexpected error occurred: {str(e)}',
                    'errors': {},
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# =============================================================================
# Login (Shared — examiner & student)
# =============================================================================

class LoginView(APIView):
    """
    POST /api/auth/login/

    Authenticate a user (examiner or student) and return
    JWT access + refresh tokens along with basic user info.
    """

    permission_classes = [AllowAny]

    def post(self, request):
        try:
            serializer = LoginSerializer(data=request.data)

            if not serializer.is_valid():
                return Response(
                    {
                        'success': False,
                        'message': 'Login failed.',
                        'errors': serializer.errors,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            user = serializer.validated_data['user']

            # Generate JWT tokens (with role claim for middleware routing)
            refresh, access = build_tokens_for_user(user)

            # Tokens are delivered as HttpOnly cookies, never in the body, so
            # client-side JavaScript can't read them.
            response = Response(
                {
                    'success': True,
                    'message': 'Login successful.',
                    'data': {
                        'access': str(access),
                        'refresh': str(refresh),
                        'user': {
                            'id': str(user.id),
                            'full_name': user.full_name,
                            'email': user.email,
                            'role': user.role,
                        },
                    },
                },
                status=status.HTTP_200_OK,
            )
            return set_auth_cookies(response, str(access), str(refresh))

        except Exception as e:
            return Response(
                {
                    'success': False,
                    'message': f'An unexpected error occurred: {str(e)}',
                    'errors': {},
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# =============================================================================
# Logout
# =============================================================================

class LogoutView(APIView):
    """
    POST /api/auth/logout/

    Blacklist the refresh token (read from the HttpOnly cookie) and clear the
    auth cookies. Allowed even with an expired access token so the browser
    session can always be terminated.
    """

    permission_classes = [AllowAny]

    def post(self, request):
        # Read the refresh token from the cookie (falling back to the body for
        # non-browser API clients).
        refresh_token = (
            request.COOKIES.get(settings.AUTH_COOKIE_REFRESH_NAME)
            or request.data.get('refresh')
        )

        try:
            if refresh_token:
                token = RefreshToken(refresh_token)
                token.blacklist()
        except TokenError:
            # Token already invalid/blacklisted — logging out is still success.
            pass
        except Exception as e:
            response = Response(
                {
                    'success': False,
                    'message': f'An unexpected error occurred: {str(e)}',
                    'errors': {},
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
            return clear_auth_cookies(response)

        # Always clear the cookies so the browser session ends regardless.
        response = Response(
            {
                'success': True,
                'message': 'Logout successful.',
                'data': None,
            },
            status=status.HTTP_200_OK,
        )
        return clear_auth_cookies(response)


# =============================================================================
# Current User Profile
# =============================================================================

class UserProfileView(APIView):
    """
    GET /api/auth/me/

    Return the authenticated user's info together with their
    role-specific profile (ExaminerProfile or StudentProfile).
    Requires: Bearer <access_token> in the Authorization header.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            serializer = UserProfileResponseSerializer(request.user)

            return Response(
                {
                    'success': True,
                    'message': 'User profile retrieved successfully.',
                    'data': serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {
                    'success': False,
                    'message': f'An unexpected error occurred: {str(e)}',
                    'errors': {},
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# =============================================================================
# Token Refresh (cookie-based)
# =============================================================================

class CookieTokenRefreshView(APIView):
    """
    POST /api/auth/token/refresh/

    Read the refresh token from the HttpOnly cookie, rotate it, and set fresh
    access (and refresh) cookies. No tokens are read from or written to the
    response body.
    """

    permission_classes = [AllowAny]

    def post(self, request):
        refresh_token = (
            request.COOKIES.get(settings.AUTH_COOKIE_REFRESH_NAME)
            or request.data.get('refresh')
        )

        if not refresh_token:
            return Response(
                {
                    'success': False,
                    'message': 'No refresh token provided.',
                    'errors': {},
                },
                status=status.HTTP_401_UNAUTHORIZED,
            )

        try:
            refresh = RefreshToken(refresh_token)
            access = refresh.access_token

            new_refresh_value = None
            # Honour ROTATE_REFRESH_TOKENS / BLACKLIST_AFTER_ROTATION.
            if settings.SIMPLE_JWT.get('ROTATE_REFRESH_TOKENS'):
                if settings.SIMPLE_JWT.get('BLACKLIST_AFTER_ROTATION'):
                    try:
                        refresh.blacklist()
                    except AttributeError:
                        pass
                refresh.set_jti()
                refresh.set_exp()
                refresh.set_iat()
                new_refresh_value = str(refresh)

        except TokenError:
            response = Response(
                {
                    'success': False,
                    'message': 'Refresh token is invalid or expired.',
                    'errors': {},
                },
                status=status.HTTP_401_UNAUTHORIZED,
            )
            return clear_auth_cookies(response)

        response = Response(
            {
                'success': True,
                'message': 'Token refreshed.',
                'data': None,
            },
            status=status.HTTP_200_OK,
        )
        return set_auth_cookies(response, str(access), new_refresh_value)
