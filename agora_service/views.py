"""
Agora token generation API view.

Endpoint:
    POST /api/sessions/<session_id>/agora-token/

Returns an Agora RTC token so the frontend can join the video channel
for the given evaluation session.
"""

import logging

from django.conf import settings
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import (
    EvaluationSession,
    ExaminerProfile,
    GroupMember,
    ProjectExaminer,
)
from agora_service.token_builder import (
    build_rtc_token,
    ROLE_PUBLISHER,
    _uid_from_user_id,
)

logger = logging.getLogger(__name__)


class AgoraTokenView(APIView):
    """
    POST /api/sessions/<session_id>/agora-token/

    Generates a temporary Agora RTC token for the authenticated user to
    join the video call for the given session.

    Permissions:
        - Examiners assigned to the session's project.
        - Students who are the session's student or a member of the
          session's group.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        # ── Fetch the session ────────────────────────────────────────────
        session = (
            EvaluationSession.objects
            .filter(id=session_id)
            .select_related('project', 'student__user', 'group')
            .first()
        )
        if not session:
            return Response(
                {'success': False, 'message': 'Session not found.'},
                status=404,
            )

        # ── Authorization ────────────────────────────────────────────────
        user = request.user

        if user.role == 'examiner':
            try:
                ep = user.examiner_profile
            except ExaminerProfile.DoesNotExist:
                return Response(
                    {'success': False, 'message': 'Examiner profile not found.'},
                    status=403,
                )
            if not ProjectExaminer.objects.filter(
                project=session.project, examiner=ep,
            ).exists():
                return Response(
                    {'success': False, 'message': 'You are not assigned to this project.'},
                    status=403,
                )

        elif user.role == 'student':
            is_session_student = (
                session.student and session.student.user_id == user.id
            )
            is_group_member = (
                session.group
                and GroupMember.objects.filter(
                    group=session.group,
                    student__user=user,
                ).exists()
            )
            if not is_session_student and not is_group_member:
                return Response(
                    {'success': False, 'message': 'You are not part of this session.'},
                    status=403,
                )
        else:
            return Response(
                {'success': False, 'message': 'Invalid user role.'},
                status=403,
            )

        # ── Ensure the session has a channel name ────────────────────────
        if not session.agora_channel_name:
            if session.group_id:
                session.agora_channel_name = f"group_{session.group_id}"
            else:
                session.agora_channel_name = str(session.id)
            session.save(update_fields=['agora_channel_name'])

        # ── Build the RTC token ──────────────────────────────────────────
        uid = _uid_from_user_id(user.id)
        screen_share_uid = uid + 1000000000
        channel = session.agora_channel_name

        try:
            token = build_rtc_token(
                channel_name=channel,
                uid=uid,
                role=ROLE_PUBLISHER,
                expire_seconds=86400,   # 24 hours
            )
            screen_share_token = build_rtc_token(
                channel_name=channel,
                uid=screen_share_uid,
                role=ROLE_PUBLISHER,
                expire_seconds=86400,   # 24 hours
            )
        except (ValueError, ImportError) as exc:
            logger.error('agora: Token generation failed: %s', exc)
            return Response(
                {'success': False, 'message': str(exc)},
                status=500,
            )

        return Response({
            'success': True,
            'message': 'Agora token generated.',
            'data': {
                'app_id': settings.AGORA_APP_ID,
                'channel': channel,
                'token': token,
                'uid': uid,
                'screen_share_token': screen_share_token,
                'screen_share_uid': screen_share_uid,
            },
        })


class AgoraRosterView(APIView):
    """
    GET /api/sessions/<session_id>/agora-roster/

    Returns a map of {agora_numeric_uid: user_full_name} for all potential
    participants in the session call (students, group members, examiners).
    """
    permission_classes = [IsAuthenticated]

    def _get_display_name(self, user) -> str:
        # Custom User model has no first_name, last_name, or username. It uses full_name and email.
        if user.full_name and user.full_name.strip() and user.full_name.lower() != "none":
            return user.full_name.strip()
        return user.email

    def get(self, request, session_id):
        session = (
            EvaluationSession.objects
            .filter(id=session_id)
            .select_related('project', 'student__user', 'group')
            .first()
        )
        if not session:
            return Response(
                {'success': False, 'message': 'Session not found.'},
                status=404,
            )

        roster = {}

        # 1. Main student
        if session.student:
            uid = _uid_from_user_id(session.student.user_id)
            name = self._get_display_name(session.student.user)
            roster[uid] = name

        # 2. Group members (if group project)
        if session.group:
            for member in GroupMember.objects.filter(group=session.group).select_related('student__user'):
                uid = _uid_from_user_id(member.student.user_id)
                name = self._get_display_name(member.student.user)
                roster[uid] = name

        # 3. Examiners assigned to this project
        for pe in ProjectExaminer.objects.filter(project=session.project).select_related('examiner__user'):
            uid = _uid_from_user_id(pe.examiner.user_id)
            name = self._get_display_name(pe.examiner.user)
            roster[uid] = f"{name} (Examiner)"

        return Response({
            'success': True,
            'roster': roster
        })

