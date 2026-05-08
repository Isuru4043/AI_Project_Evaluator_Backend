"""
Serializers for the Authentication app.

Handles input validation for registration, login, logout,
and user profile responses.
"""

import re

from django.contrib.auth import authenticate
from rest_framework import serializers

from core.models import ExaminerProfile, StudentProfile, User


# =============================================================================
# Password validation helper
# =============================================================================

def validate_password_strength(password):
    """
    Enforce password policy:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one digit
    """
    if len(password) < 8:
        raise serializers.ValidationError(
            "Password must be at least 8 characters long."
        )
    if not re.search(r'[A-Z]', password):
        raise serializers.ValidationError(
            "Password must contain at least one uppercase letter."
        )
    if not re.search(r'[0-9]', password):
        raise serializers.ValidationError(
            "Password must contain at least one number."
        )


# =============================================================================
# Examiner Registration Serializer
# =============================================================================

class ExaminerRegisterSerializer(serializers.Serializer):
    """Validates and creates an examiner user + profile."""

    full_name = serializers.CharField(max_length=255)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    confirm_password = serializers.CharField(write_only=True)
    employee_id = serializers.CharField(max_length=100, required=False, allow_blank=True)
    department = serializers.CharField(max_length=255, required=False, allow_blank=True)
    designation = serializers.CharField(max_length=255, required=False, allow_blank=True)

    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError(
                "A user with this email already exists."
            )
        return value

    def validate_password(self, value):
        validate_password_strength(value)
        return value

    def validate(self, attrs):
        if attrs['password'] != attrs['confirm_password']:
            raise serializers.ValidationError({
                'confirm_password': "Passwords do not match."
            })
        return attrs

    def create(self, validated_data):
        # Pop profile-specific fields
        employee_id = validated_data.pop('employee_id', None)
        department = validated_data.pop('department', None)
        designation = validated_data.pop('designation', None)
        validated_data.pop('confirm_password')

        # Create the user
        user = User.objects.create_user(
            email=validated_data['email'],
            password=validated_data['password'],
            full_name=validated_data['full_name'],
            role=User.Role.EXAMINER,
        )

        # Create the examiner profile
        ExaminerProfile.objects.create(
            user=user,
            employee_id=employee_id or None,
            department=department or None,
            designation=designation or None,
        )

        return user


# =============================================================================
# Student Registration Serializer
# =============================================================================

class StudentRegisterSerializer(serializers.Serializer):
    """Validates and creates a student user + profile."""

    full_name = serializers.CharField(max_length=255)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    confirm_password = serializers.CharField(write_only=True)
    registration_number = serializers.CharField(max_length=100)
    degree_program = serializers.CharField(max_length=255, required=False, allow_blank=True)
    academic_year = serializers.IntegerField(required=False, allow_null=True)
    batch = serializers.CharField(max_length=100, required=False, allow_blank=True)

    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError(
                "A user with this email already exists."
            )
        return value

    def validate_registration_number(self, value):
        if StudentProfile.objects.filter(registration_number=value).exists():
            raise serializers.ValidationError(
                "A student with this registration number already exists."
            )
        return value

    def validate_password(self, value):
        validate_password_strength(value)
        return value

    def validate(self, attrs):
        if attrs['password'] != attrs['confirm_password']:
            raise serializers.ValidationError({
                'confirm_password': "Passwords do not match."
            })
        return attrs

    def create(self, validated_data):
        # Pop profile-specific fields
        registration_number = validated_data.pop('registration_number')
        degree_program = validated_data.pop('degree_program', None)
        academic_year = validated_data.pop('academic_year', None)
        batch = validated_data.pop('batch', None)
        validated_data.pop('confirm_password')

        # Create the user
        user = User.objects.create_user(
            email=validated_data['email'],
            password=validated_data['password'],
            full_name=validated_data['full_name'],
            role=User.Role.STUDENT,
        )

        # Create the student profile
        StudentProfile.objects.create(
            user=user,
            registration_number=registration_number,
            degree_program=degree_program or None,
            academic_year=academic_year,
            batch=batch or None,
        )

        return user


# =============================================================================
# Login Serializer
# =============================================================================

class LoginSerializer(serializers.Serializer):
    """Validates login credentials for both examiners and students."""

    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        email = attrs.get('email')
        password = attrs.get('password')

        # Check if user exists
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            raise serializers.ValidationError({
                'email': "No account found with this email address."
            })

        # Check if user is active
        if not user.is_active:
            raise serializers.ValidationError({
                'email': "This account has been deactivated."
            })

        # Authenticate
        authenticated_user = authenticate(email=email, password=password)
        if authenticated_user is None:
            raise serializers.ValidationError({
                'password': "Invalid password."
            })

        attrs['user'] = authenticated_user
        return attrs


# =============================================================================
# Profile Serializers (for the /me/ endpoint response)
# =============================================================================

class ExaminerProfileSerializer(serializers.ModelSerializer):
    """Serializes examiner profile details."""

    class Meta:
        model = ExaminerProfile
        fields = ['employee_id', 'department', 'designation']


class StudentProfileSerializer(serializers.ModelSerializer):
    """Serializes student profile details."""

    class Meta:
        model = StudentProfile
        fields = ['registration_number', 'degree_program', 'academic_year', 'batch']


class UserProfileResponseSerializer(serializers.ModelSerializer):
    """
    Serializes the full user profile including role-specific details.
    The 'profile' field is dynamically populated based on user role.
    """

    profile = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ['id', 'full_name', 'email', 'role', 'profile']

    def get_profile(self, obj):
        if obj.role == User.Role.EXAMINER:
            try:
                return ExaminerProfileSerializer(obj.examiner_profile).data
            except ExaminerProfile.DoesNotExist:
                return None
        elif obj.role == User.Role.STUDENT:
            try:
                return StudentProfileSerializer(obj.student_profile).data
            except StudentProfile.DoesNotExist:
                return None
        return None
