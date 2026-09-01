"""Who may submit an answer in a remote group viva.

Regression cover for the bug where one group member could submit and another
could not. The attribution engine names whichever member it heard speaking;
that inference was then checked as if the caller had claimed to BE that member,
so the student actually holding the keyboard was refused with
"You cannot submit an answer for another participant."

Attribution is evidence about who spoke. It routes marks. It must never decide
who is allowed to press Submit.
"""

import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")

import django

django.setup()

from viva_evaluator.views.session_views import AnswerSubmitView


STUDENT_A = "11111111-1111-1111-1111-111111111111"
STUDENT_B = "22222222-2222-2222-2222-222222222222"
SESSION_ID = "33333333-3333-3333-3333-333333333333"
QUESTION_ID = "44444444-4444-4444-4444-444444444444"


def _request(caller_id, data):
    """A DRF-ish request from a signed-in student (never a kiosk)."""
    return SimpleNamespace(
        data=data,
        auth=None,
        user=SimpleNamespace(student_profile=SimpleNamespace(id=caller_id)),
    )


class GroupAnswerSubmissionTests(TestCase):
    """Each test stops the view right after the authorisation decision.

    The pipeline beyond that point needs a real submission and database, which
    this suite deliberately does not build: the question under test is only
    whether the caller gets past the permission gate.
    """

    def _run(self, caller_id, data, attributed_to):
        session = SimpleNamespace(
            id=SESSION_ID, student=None, student_id=None, group_id="group-1",
        )
        question = SimpleNamespace(id=QUESTION_ID)

        with (
            patch("core.models.EvaluationSession.objects.get", return_value=session),
            patch("core.models.VivaQuestion.objects.get", return_value=question),
            patch(
                "viva_evaluator.views.session_views._resolve_session_submission",
                # None short-circuits with 400 immediately AFTER the
                # authorisation check, which is the boundary under test.
                return_value=None,
            ),
            patch(
                "attribution.services.engine.resolve_speaker_id",
                return_value=attributed_to,
            ),
            patch(
                "core.models.StudentProfile.objects.filter",
            ) as profile_filter,
        ):
            profile_filter.return_value.first.return_value = (
                SimpleNamespace(id=attributed_to)
                if attributed_to != "group"
                else None
            )
            return AnswerSubmitView().post(_request(caller_id, data), SESSION_ID)

    def test_member_may_submit_when_attribution_names_someone_else(self):
        """The reported bug: A submits, the engine heard B, A was refused."""
        response = self._run(
            caller_id=STUDENT_A,
            data={"question_id": QUESTION_ID, "answer_text": "My answer."},
            attributed_to=STUDENT_B,
        )
        self.assertNotEqual(response.status_code, 403)

    def test_member_may_submit_when_attribution_names_them(self):
        """The case that always worked — must keep working."""
        response = self._run(
            caller_id=STUDENT_B,
            data={"question_id": QUESTION_ID, "answer_text": "My answer."},
            attributed_to=STUDENT_B,
        )
        self.assertNotEqual(response.status_code, 403)

    def test_member_may_submit_when_attribution_is_uncertain(self):
        response = self._run(
            caller_id=STUDENT_A,
            data={"question_id": QUESTION_ID, "answer_text": "My answer."},
            attributed_to="group",
        )
        self.assertNotEqual(response.status_code, 403)

    def test_naming_another_participant_is_still_refused(self):
        """The protection this check exists for must survive the fix.

        A web client naming someone is rejected earlier, by the rule that only
        the physical kiosk may choose a speaker — so the claim never reaches
        the pipeline either way.
        """
        response = self._run(
            caller_id=STUDENT_A,
            data={
                "question_id": QUESTION_ID,
                "answer_text": "My answer.",
                "speaker_id": STUDENT_B,
            },
            attributed_to=STUDENT_B,
        )
        self.assertEqual(response.status_code, 403)
