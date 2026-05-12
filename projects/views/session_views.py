"""
Views for Session Scheduling (Feature 5).
"""

from datetime import datetime, timedelta

from django.db import transaction
from django.utils import timezone
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from core.models import (
    EvaluationSession, GroupMember, Project, ProjectExaminer,
    ProjectSubmission, StudentGroup, StudentProfile,
)
from projects.permissions import IsExaminer, IsProjectLead, IsStudent
from projects.serializers import (
    AutoScheduleSerializer, EvaluationSessionSerializer,
    ManualScheduleSerializer, SessionUpdateSerializer,
)
from projects.views.project_views import _err, _get_examiner_profile, _get_student_profile, _is_assigned, _ok, _500


class ManualScheduleView(APIView):
    """POST /api/projects/<project_id>/sessions/schedule/manual/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not _is_assigned(ep, project):
                return _err('You are not assigned to this project.', code=403)

            ser = ManualScheduleSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            created = []
            with transaction.atomic():
                for entry in ser.validated_data['sessions']:
                    if not project.is_group_project:
                        # Individual project
                        sid = entry.get('student_id')
                        if not sid:
                            return _err('student_id is required for individual projects.')
                        sp = StudentProfile.objects.filter(id=sid).first()
                        if not sp:
                            return _err(f'Student {sid} not found.')
                        if not ProjectSubmission.objects.filter(project=project, student=sp).exists():
                            return _err(f'Student {sp.user.full_name} is not enrolled in this project.')

                        session = EvaluationSession.objects.create(
                            project=project, student=sp, group=None,
                            scheduled_start=entry['scheduled_start'],
                            scheduled_end=entry['scheduled_end'],
                            location_room=entry.get('location_room', ''),
                            status='scheduled',
                        )
                        created.append(session)
                    else:
                        # Group project
                        gid = entry.get('group_id')
                        if not gid:
                            return _err('group_id is required for group projects.')
                        group = StudentGroup.objects.filter(id=gid, project=project).first()
                        if not group:
                            return _err(f'Group {gid} not found in this project.')

                        members = GroupMember.objects.filter(group=group).select_related('student')
                        for member in members:
                            session = EvaluationSession.objects.create(
                                project=project, student=member.student, group=group,
                                scheduled_start=entry['scheduled_start'],
                                scheduled_end=entry['scheduled_end'],
                                location_room=entry.get('location_room', ''),
                                status='scheduled',
                            )
                            created.append(session)

            data = EvaluationSessionSerializer(created, many=True).data
            return _ok('Sessions scheduled successfully.', data, 201)
        except Exception as e:
            return _500(e)


class AutoScheduleView(APIView):
    """POST /api/projects/<project_id>/sessions/schedule/auto/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not _is_assigned(ep, project):
                return _err('You are not assigned to this project.', code=403)

            ser = AutoScheduleSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            date_ranges = ser.validated_data['date_ranges']
            duration = ser.validated_data['duration_per_slot_minutes']
            room = ser.validated_data.get('location_room', '')

            # Build time slots
            slots = []
            for dr in date_ranges:
                d = dr['date']
                current = timezone.make_aware(
                    datetime.combine(d, dr['start_time']),
                )
                end_dt = timezone.make_aware(
                    datetime.combine(d, dr['end_time']),
                )
                while current + timedelta(minutes=duration) <= end_dt:
                    slots.append((current, current + timedelta(minutes=duration)))
                    current += timedelta(minutes=duration)

            # Determine entities to schedule
            if not project.is_group_project:
                entities = list(
                    ProjectSubmission.objects.filter(
                        project=project, student__isnull=False,
                    ).order_by('submitted_at').select_related('student')
                )
                n = len(entities)
            else:
                entities = list(
                    StudentGroup.objects.filter(project=project).order_by('created_at')
                )
                n = len(entities)

            total_available = len(slots) * duration
            total_required = n * duration

            if n == 0:
                return _err('No students/groups enrolled in this project.')

            if total_required > total_available:
                return _err(
                    f'Time range is not enough for all the students/groups. '
                    f'Required: {total_required} minutes, Available: {total_available} minutes. '
                    f'Please extend the time range or reduce the duration per slot.'
                )

            created = []
            with transaction.atomic():
                slot_idx = 0
                for entity in entities:
                    if slot_idx >= len(slots):
                        break
                    s_start, s_end = slots[slot_idx]
                    slot_idx += 1

                    if not project.is_group_project:
                        session = EvaluationSession.objects.create(
                            project=project, student=entity.student, group=None,
                            scheduled_start=s_start, scheduled_end=s_end,
                            location_room=room, status='scheduled',
                        )
                        created.append(session)
                    else:
                        members = GroupMember.objects.filter(
                            group=entity,
                        ).select_related('student')
                        for member in members:
                            session = EvaluationSession.objects.create(
                                project=project, student=member.student, group=entity,
                                scheduled_start=s_start, scheduled_end=s_end,
                                location_room=room, status='scheduled',
                            )
                            created.append(session)

            data = EvaluationSessionSerializer(created, many=True).data
            return _ok('Sessions auto-scheduled successfully.', data, 201)
        except Exception as e:
            return _500(e)


class SessionListView(APIView):
    """GET /api/projects/<project_id>/sessions/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def get(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            sessions = EvaluationSession.objects.filter(
                project=project,
            ).select_related('student__user', 'group').order_by('scheduled_start')

            status_filter = request.query_params.get('status')
            if status_filter:
                sessions = sessions.filter(status=status_filter)

            if project.is_group_project:
                # Group sessions by group
                groups = {}
                for s in sessions:
                    gid = str(s.group_id) if s.group_id else 'ungrouped'
                    if gid not in groups:
                        groups[gid] = {
                            'group_id': gid,
                            'group_name': s.group.group_name if s.group else None,
                            'scheduled_start': s.scheduled_start,
                            'scheduled_end': s.scheduled_end,
                            'location_room': s.location_room,
                            'status': s.status,
                            'students': [],
                        }
                    groups[gid]['students'].append({
                        'student_name': s.student.user.full_name if s.student else None,
                        'student_reg_no': s.student.registration_number if s.student else None,
                    })
                data = list(groups.values())
            else:
                data = EvaluationSessionSerializer(sessions, many=True).data

            return _ok('Sessions retrieved.', data)
        except Exception as e:
            return _500(e)


class MySessionView(APIView):
    """GET /api/projects/<project_id>/sessions/my-session/"""
    permission_classes = [IsAuthenticated, IsStudent]

    def get(self, request, project_id):
        try:
            sp = _get_student_profile(request.user)
            if not sp:
                return _err('Student profile not found.', code=404)

            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            session = EvaluationSession.objects.filter(
                project=project, student=sp,
            ).select_related('group').first()

            if not session:
                return _err('No session found for you in this project.', code=404)

            data = {
                'session_id': str(session.id),
                'scheduled_start': session.scheduled_start,
                'scheduled_end': session.scheduled_end,
                'location_room': session.location_room,
                'status': session.status,
            }

            if session.group:
                members = GroupMember.objects.filter(
                    group=session.group,
                ).select_related('student__user').exclude(student=sp)
                data['group_name'] = session.group.group_name
                data['group_members'] = [
                    m.student.user.full_name for m in members
                ]

            return _ok('Session retrieved.', data)
        except Exception as e:
            return _500(e)


class SessionUpdateView(APIView):
    """PUT /api/projects/<project_id>/sessions/<session_id>/update/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def put(self, request, project_id, session_id):
        try:
            session = EvaluationSession.objects.filter(
                id=session_id, project_id=project_id,
            ).first()
            if not session:
                return _err('Session not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not _is_assigned(ep, session.project):
                return _err('You are not assigned to this project.', code=403)

            ser = SessionUpdateSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            data = ser.validated_data
            update_fields = {}
            for f in ('scheduled_start', 'scheduled_end', 'location_room'):
                if f in data:
                    update_fields[f] = data[f]

            if not update_fields:
                return _err('No fields to update.')

            if session.group:
                # Update all sessions for the same group + project
                EvaluationSession.objects.filter(
                    project_id=project_id, group=session.group,
                ).update(**update_fields)
            else:
                for k, v in update_fields.items():
                    setattr(session, k, v)
                session.save()

            session.refresh_from_db()
            return _ok('Session updated.', EvaluationSessionSerializer(session).data)
        except Exception as e:
            return _500(e)


class SessionResetView(APIView):
    """DELETE /api/projects/<project_id>/sessions/reset/"""
    permission_classes = [IsAuthenticated, IsProjectLead]

    def delete(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            has_active = EvaluationSession.objects.filter(
                project=project, status__in=['in_progress', 'completed'],
            ).exists()
            if has_active:
                return _err(
                    'Cannot reset sessions — some sessions are in progress or completed.'
                )

            EvaluationSession.objects.filter(project=project).delete()
            return _ok('All sessions have been reset. You can now reschedule.')
        except Exception as e:
            return _500(e)


class NextSessionView(APIView):
    """GET /api/projects/sessions/next/"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            now = timezone.now()
            next_session = None

            if request.user.role == 'student':
                sp = _get_student_profile(request.user)
                if not sp:
                    return _err('Student profile not found.', code=404)
                next_session = EvaluationSession.objects.filter(
                    student=sp,
                    status='scheduled',
                    scheduled_start__gte=now
                ).select_related('project', 'group').order_by('scheduled_start').first()

            elif request.user.role == 'examiner':
                ep = _get_examiner_profile(request.user)
                if not ep:
                    return _err('Examiner profile not found.', code=404)
                project_ids = ProjectExaminer.objects.filter(examiner=ep).values_list('project_id', flat=True)
                next_session = EvaluationSession.objects.filter(
                    project_id__in=project_ids,
                    status='scheduled',
                    scheduled_start__gte=now
                ).select_related('project', 'student__user', 'group').order_by('scheduled_start').first()
            else:
                return _err('Invalid role.', code=403)

            if not next_session:
                return _err('No upcoming sessions.', code=404)

            data = {
                'session_id': str(next_session.id),
                'project_id': str(next_session.project_id),
                'project_name': next_session.project.project_name,
                'scheduled_start': next_session.scheduled_start,
                'scheduled_end': next_session.scheduled_end,
                'location_room': next_session.location_room,
                'status': next_session.status,
            }

            if next_session.group:
                data['group_name'] = next_session.group.group_name
            if next_session.student:
                data['student_name'] = next_session.student.user.full_name
                data['student_reg_no'] = next_session.student.registration_number

            return _ok('Next session retrieved.', data)
        except Exception as e:
            return _500(e)
