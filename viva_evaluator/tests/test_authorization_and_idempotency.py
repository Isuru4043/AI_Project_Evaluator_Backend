import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")

import django

django.setup()

from viva_evaluator.permissions import (
    EXAMINER,
    PARTICIPANT,
    session_roles_for_user,
    user_is_assigned_project_examiner,
)
from viva_evaluator.services.answer_idempotency import (
    request_fingerprint,
    speaker_key,
)


class SessionAuthorizationTests(unittest.TestCase):
    def test_direct_student_is_session_participant(self):
        student = SimpleNamespace(id="student-1")
        user = SimpleNamespace(
            role="student", student_profile=student, is_superuser=False
        )
        session = SimpleNamespace(
            student_id="student-1", group_id=None, project_id="project-1"
        )
        self.assertEqual(session_roles_for_user(user, session), {PARTICIPANT})

    @patch("viva_evaluator.permissions.GroupMember.objects.filter")
    def test_group_member_is_session_participant(self, filter_mock):
        filter_mock.return_value.exists.return_value = True
        student = SimpleNamespace(id="student-1")
        user = SimpleNamespace(
            role="student", student_profile=student, is_superuser=False
        )
        session = SimpleNamespace(
            student_id=None, group_id="group-1", project_id="project-1"
        )
        self.assertEqual(session_roles_for_user(user, session), {PARTICIPANT})
        filter_mock.assert_called_once_with(
            group_id="group-1", student_id="student-1"
        )

    @patch("viva_evaluator.permissions.ProjectExaminer.objects.filter")
    def test_only_assigned_examiner_gets_examiner_role(self, filter_mock):
        filter_mock.return_value.exists.return_value = True
        examiner = SimpleNamespace(id="examiner-1")
        user = SimpleNamespace(
            role="examiner", examiner_profile=examiner, is_superuser=False
        )
        session = SimpleNamespace(
            student_id=None, group_id=None, project_id="project-1"
        )
        self.assertEqual(session_roles_for_user(user, session), {EXAMINER})

        filter_mock.return_value.exists.return_value = False
        self.assertEqual(session_roles_for_user(user, session), set())

    @patch("viva_evaluator.permissions.ProjectExaminer.objects.filter")
    def test_project_access_uses_assignment_table(self, filter_mock):
        filter_mock.return_value.exists.return_value = True
        user = SimpleNamespace(
            role="examiner",
            examiner_profile=SimpleNamespace(id="examiner-1"),
            is_superuser=False,
        )
        self.assertTrue(user_is_assigned_project_examiner(user, "project-1"))


class AnswerIdempotencyTests(unittest.TestCase):
    def test_fingerprint_is_stable_across_metric_key_order(self):
        first = request_fingerprint(
            answer_text="An answer",
            speech_metrics={"pause": 2, "clarity": 0.9},
            speaker_id="student-1",
        )
        second = request_fingerprint(
            answer_text="An answer",
            speech_metrics={"clarity": 0.9, "pause": 2},
            speaker_id="student-1",
        )
        self.assertEqual(first, second)

    def test_fingerprint_changes_when_answer_changes(self):
        first = request_fingerprint(
            answer_text="First", speech_metrics=None, speaker_id="group"
        )
        second = request_fingerprint(
            answer_text="Second", speech_metrics=None, speaker_id="group"
        )
        self.assertNotEqual(first, second)

    def test_speaker_key_distinguishes_group_and_student(self):
        self.assertEqual(speaker_key("group"), "group")
        self.assertEqual(speaker_key("abc"), "student:abc")


if __name__ == "__main__":
    unittest.main()
