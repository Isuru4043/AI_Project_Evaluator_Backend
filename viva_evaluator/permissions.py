"""Object-scoped authorization for viva sessions and project resources."""

from __future__ import annotations

from typing import Optional, Set

from rest_framework.permissions import BasePermission

from core.models import (
    EvaluationSession,
    GroupMember,
    Project,
    ProjectExaminer,
    ProjectSubmission,
    RubricCategory,
    RubricCriteria,
)
from physical_evaluation.models import PhysicalEvaluationRun, PhysicalKioskAccess


PARTICIPANT = "participant"
EXAMINER = "examiner"


def _session_id(request, view):
    data = getattr(request, "data", {}) or {}
    return view.kwargs.get("session_id") or data.get("session_id")


def _project_id(request, view) -> Optional[str]:
    data = getattr(request, "data", {}) or {}
    project_id = (
        view.kwargs.get("project_id")
        or data.get("project_id")
        or data.get("project")
    )
    if project_id:
        return str(project_id)

    category_id = view.kwargs.get("category_id")
    if category_id:
        return (
            RubricCategory.objects.filter(id=category_id)
            .values_list("project_id", flat=True)
            .first()
        )
    criteria_id = view.kwargs.get("criteria_id")
    if criteria_id:
        return (
            RubricCriteria.objects.filter(id=criteria_id)
            .values_list("category__project_id", flat=True)
            .first()
        )
    hint_id = view.kwargs.get("hint_id")
    if hint_id:
        from viva_evaluator.models import CriteriaQuestionHint

        return (
            CriteriaQuestionHint.objects.filter(id=hint_id)
            .values_list("criteria__category__project_id", flat=True)
            .first()
        )
    submission_id = view.kwargs.get("submission_id")
    if submission_id:
        return (
            ProjectSubmission.objects.filter(id=submission_id)
            .values_list("project_id", flat=True)
            .first()
        )
    return None


def session_roles_for_user(user, session: EvaluationSession) -> Set[str]:
    """Return the caller's independently verified roles for one session."""
    if getattr(user, "is_superuser", False):
        return {PARTICIPANT, EXAMINER}

    roles: Set[str] = set()
    if getattr(user, "role", None) == "student":
        try:
            student = user.student_profile
        except Exception:
            student = None
        if student is not None and (
            session.student_id == student.id
            or (
                session.group_id
                and GroupMember.objects.filter(
                    group_id=session.group_id,
                    student_id=student.id,
                ).exists()
            )
        ):
            roles.add(PARTICIPANT)

    if getattr(user, "role", None) == "examiner":
        try:
            examiner = user.examiner_profile
        except Exception:
            examiner = None
        if examiner is not None and ProjectExaminer.objects.filter(
            project_id=session.project_id,
            examiner_id=examiner.id,
        ).exists():
            roles.add(EXAMINER)
    return roles


def user_is_assigned_project_examiner(user, project_id) -> bool:
    if getattr(user, "is_superuser", False):
        return True
    if getattr(user, "role", None) != "examiner":
        return False
    try:
        examiner_id = user.examiner_profile.id
    except Exception:
        return False
    return ProjectExaminer.objects.filter(
        project_id=project_id,
        examiner_id=examiner_id,
    ).exists()


class VivaSessionPermission(BasePermission):
    """Authorize a participant, assigned examiner, or active physical kiosk."""

    message = "You are not authorized to access this viva session."
    allowed_roles = {PARTICIPANT, EXAMINER}

    def has_permission(self, request, view):
        session_id = _session_id(request, view)
        if not session_id:
            return True  # Let request-schema validation report the missing ID.

        if isinstance(request.auth, PhysicalKioskAccess):
            return PhysicalEvaluationRun.objects.filter(
                session_id=session_id,
                kiosk_access=request.auth,
                status=PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS,
            ).exists()

        session = EvaluationSession.objects.only(
            "id", "project_id", "student_id", "group_id"
        ).filter(id=session_id).first()
        if session is None:
            return True  # Preserve the endpoint's existing 404 response.
        return bool(session_roles_for_user(request.user, session) & self.allowed_roles)


class CanParticipateInVivaSession(VivaSessionPermission):
    allowed_roles = {PARTICIPANT}


class IsAssignedSessionExaminer(VivaSessionPermission):
    allowed_roles = {EXAMINER}


class IsAssignedProjectExaminer(BasePermission):
    message = "You are not an assigned examiner for this project."

    def has_permission(self, request, view):
        project_id = _project_id(request, view)
        if not project_id:
            return False
        if not Project.objects.filter(id=project_id).exists():
            return True  # Preserve the view's existing 404 response.
        return user_is_assigned_project_examiner(request.user, project_id)


class CanAccessProjectSubmission(BasePermission):
    """Allow the owning student/group or an examiner assigned to its project."""

    message = "You are not authorized to access this project submission."

    def has_permission(self, request, view):
        submission_id = view.kwargs.get("submission_id")
        if not submission_id:
            return False
        submission = ProjectSubmission.objects.only(
            "project_id", "student_id", "group_id"
        ).filter(id=submission_id).first()
        if submission is None:
            return True
        if user_is_assigned_project_examiner(request.user, submission.project_id):
            return True
        if getattr(request.user, "role", None) != "student":
            return False
        try:
            student_id = request.user.student_profile.id
        except Exception:
            return False
        return submission.student_id == student_id or bool(
            submission.group_id
            and GroupMember.objects.filter(
                group_id=submission.group_id,
                student_id=student_id,
            ).exists()
        )
