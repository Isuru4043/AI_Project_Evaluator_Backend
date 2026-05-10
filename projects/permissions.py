"""
Custom permission classes for the Projects app.
"""

from rest_framework.permissions import BasePermission

from core.models import ProjectExaminer


class IsExaminer(BasePermission):
    """Allow access only to users with role = 'examiner'."""

    message = 'Only examiners can access this resource.'

    def has_permission(self, request, view):
        return (
            request.user.is_authenticated
            and request.user.role == 'examiner'
        )


class IsStudent(BasePermission):
    """Allow access only to users with role = 'student'."""

    message = 'Only students can access this resource.'

    def has_permission(self, request, view):
        return (
            request.user.is_authenticated
            and request.user.role == 'student'
        )


class IsExaminerOrStudent(BasePermission):
    """Allow access to both examiners and students."""

    def has_permission(self, request, view):
        return (
            request.user.is_authenticated
            and request.user.role in ('examiner', 'student')
        )


class IsProjectLead(BasePermission):
    """Allow access only to the lead examiner of the specific project."""

    message = 'Only the lead examiner of this project can perform this action.'

    def has_permission(self, request, view):
        if not request.user.is_authenticated or request.user.role != 'examiner':
            return False
        project_id = view.kwargs.get('project_id')
        if not project_id:
            return False
        try:
            return ProjectExaminer.objects.filter(
                project_id=project_id,
                examiner=request.user.examiner_profile,
                role_in_project='lead',
            ).exists()
        except Exception:
            return False
