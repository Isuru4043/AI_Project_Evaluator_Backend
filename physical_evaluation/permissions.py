from rest_framework.permissions import BasePermission

from physical_evaluation.models import PhysicalEvaluationRun, PhysicalKioskAccess
from core.models import EvaluationSession, Project


class CanAccessSharedVivaSession(BasePermission):
    """Scope kiosk credentials to their currently active physical viva only."""

    message = 'This kiosk is not authorized for the requested viva session.'

    def has_permission(self, request, view):
        if not isinstance(request.auth, PhysicalKioskAccess):
            # Physical start/answer operations must come through a scoped kiosk
            # credential, never through the examiner's still-open login cookie.
            if request.method == 'POST':
                session_id = view.kwargs.get('session_id') or request.data.get('session_id')
                if session_id and EvaluationSession.objects.filter(
                    id=session_id,
                    project__evaluation_mode=Project.EvaluationMode.PHYSICAL,
                ).exists():
                    return False
            return True

        session_id = view.kwargs.get('session_id')
        if session_id is None:
            session_id = request.data.get('session_id')
        if not session_id:
            return False

        return PhysicalEvaluationRun.objects.filter(
            session_id=session_id,
            kiosk_access=request.auth,
            status=PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS,
        ).exists()
