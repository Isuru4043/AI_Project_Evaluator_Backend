from datetime import timedelta
from unittest.mock import patch

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from core.models import (
    EvaluationSession,
    ExaminerProfile,
    Project,
    ProjectExaminer,
    StudentProfile,
    User,
)


class AgoraRosterMetadataTests(TestCase):
    def setUp(self):
        self.examiner_user = User.objects.create_user(
            email='agora-examiner@example.com',
            password='password',
            full_name='Agora Examiner',
            role=User.Role.EXAMINER,
        )
        examiner = ExaminerProfile.objects.create(
            user=self.examiner_user,
            employee_id='AGORA-EXAMINER',
        )
        self.student_user = User.objects.create_user(
            email='agora-student@example.com',
            password='password',
            full_name='Agora Student',
            role=User.Role.STUDENT,
        )
        student = StudentProfile.objects.create(
            user=self.student_user,
            registration_number='AGORA-STUDENT',
        )
        project = Project.objects.create(project_name='Agora roster test')
        ProjectExaminer.objects.create(
            project=project,
            examiner=examiner,
            role_in_project=ProjectExaminer.RoleInProject.LEAD,
        )
        now = timezone.now()
        self.session = EvaluationSession.objects.create(
            project=project,
            student=student,
            scheduled_start=now - timedelta(minutes=5),
            scheduled_end=now + timedelta(minutes=55),
            status=EvaluationSession.Status.IN_PROGRESS,
        )
        self.client = APIClient()
        self.client.force_authenticate(self.examiner_user)

    @patch('agora_service.views._uid_from_user_id')
    def test_high_normal_uid_is_distinct_from_explicit_screen_uid(self, uid):
        uid.side_effect = lambda user_id: (
            1_500_000_000
            if user_id == self.student_user.id
            else 400_000_000
        )

        response = self.client.get(
            f'/api/sessions/{self.session.id}/agora-roster/',
        )

        self.assertEqual(response.status_code, 200, response.data)
        self.assertIn(2_500_000_000, response.data['screen_share_uids'])
        self.assertNotIn(1_500_000_000, response.data['screen_share_uids'])

        student_streams = [
            stream for stream in response.data['participants']
            if stream['display_name'] == 'Agora Student'
        ]
        self.assertEqual(
            {(stream['uid'], stream['stream_type']) for stream in student_streams},
            {
                (1_500_000_000, 'camera'),
                (2_500_000_000, 'screen_share'),
            },
        )
