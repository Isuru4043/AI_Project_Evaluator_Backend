"""Session manifest builder — seam 1 between the platform and exam-station-cv.

Produces the exact ``SessionManifest`` shape the CV module's contracts define
(see exam-station-cv/src/exam_cv/contracts/schemas.py). The CV module never
mints its own session identity; this is where the platform hands it one.
"""

from django.utils import timezone

from core.models import EvaluationSession, GroupMember

SCHEMA_VERSION = '1.0'


def _display_name(student_profile) -> str:
    user = student_profile.user
    full = f"{user.first_name} {user.last_name}".strip()
    return full or user.username or user.email


def build_manifest(session: EvaluationSession) -> dict:
    """EvaluationSession → SessionManifest dict (JSON-serializable)."""
    if session.group_id:
        members = (
            GroupMember.objects
            .filter(group_id=session.group_id)
            .select_related('student__user')
        )
        roster = [
            {
                'student_id': str(m.student.id),
                'display_name': _display_name(m.student),
            }
            for m in members
        ]
        mode = 'group'
    elif session.student_id:
        roster = [
            {
                'student_id': str(session.student.id),
                'display_name': _display_name(session.student),
            }
        ]
        mode = 'individual'
    else:
        raise ValueError(
            f"Session {session.id} has neither a student nor a group — "
            "cannot build a CV manifest."
        )

    t0 = session.actual_start or session.scheduled_start or timezone.now()
    return {
        'schema_version': SCHEMA_VERSION,
        'session_id': str(session.id),
        'mode': mode,
        'roster': roster,
        't0_utc': t0.isoformat(),
        'notes': f"project={session.project_id}",
    }
