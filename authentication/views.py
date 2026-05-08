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

from .serializers import (
    ExaminerRegisterSerializer,
    LoginSerializer,
    StudentRegisterSerializer,
    UserProfileResponseSerializer,
)


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

            # Generate JWT tokens
            refresh = RefreshToken.for_user(user)

            return Response(
                {
                    'success': True,
                    'message': 'Login successful.',
                    'data': {
                        'access': str(refresh.access_token),
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

    Blacklist the provided refresh token so it can no longer
    be used to obtain new access tokens.
    Requires: Bearer <access_token> in the Authorization header.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get('refresh')

            if not refresh_token:
                return Response(
                    {
                        'success': False,
                        'message': 'Refresh token is required.',
                        'errors': {'refresh': ['This field is required.']},
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            token = RefreshToken(refresh_token)
            token.blacklist()

            return Response(
                {
                    'success': True,
                    'message': 'Logout successful. Token has been blacklisted.',
                    'data': None,
                },
                status=status.HTTP_205_RESET_CONTENT,
            )

        except TokenError:
            return Response(
                {
                    'success': False,
                    'message': 'Token is invalid or already blacklisted.',
                    'errors': {},
                },
                status=status.HTTP_400_BAD_REQUEST,
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
