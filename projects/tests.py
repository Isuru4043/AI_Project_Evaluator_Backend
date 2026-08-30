import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase

from core.models import (
    EvaluationSession,
    ExaminerProfile,
    Project,
    RubricGroupingCache,
    StudentProfile,
    User,
)
from viva_evaluator.models import SubmissionIndexStatus


class CreateReadySessionCommandTests(TestCase):
    def setUp(self):
        examiner_user = User.objects.create_user(
            email='examiner@example.com',
            password='unused',
            full_name='Test Examiner',
            role=User.Role.EXAMINER,
        )
        ExaminerProfile.objects.create(user=examiner_user)

        self.student_emails = []
        for number in (1, 2):
            email = f'student{number}@example.com'
            user = User.objects.create_user(
                email=email,
                password='unused',
                full_name=f'Test Student {number}',
                role=User.Role.STUDENT,
            )
            StudentProfile.objects.create(
                user=user,
                registration_number=f'TEST-{number}',
            )
            self.student_emails.append(email)

    def test_creates_complete_physical_group_session(self):
        rubric_data = {
            'rubric_categories': [{
                'category_name': 'Architecture',
                'weight_percentage': 100,
                'description': 'Architecture quality',
                'criteria': [{
                    'criteria_name': 'Trust boundaries',
                    'max_score': 10,
                    'weight_in_category': 100,
                    'description': 'Explains all trust boundaries.',
                    'questions_to_ask': 2,
                    'question_hints': [{
                        'hint_text': 'Where is identity verified?',
                        'order': 1,
                    }],
                }],
            }],
        }

        def finish_indexing(submission_id, _report_bytes):
            SubmissionIndexStatus.objects.filter(
                submission_id=submission_id,
            ).update(status=SubmissionIndexStatus.IndexStatus.READY)

        def create_grouping(project, max_questions):
            return RubricGroupingCache.objects.create(
                project=project,
                max_questions=max_questions,
                grouped_criteria={'viva_topics': []},
            )

        with tempfile.TemporaryDirectory() as directory:
            rubric_path = Path(directory) / 'rubric.pdf'
            report_path = Path(directory) / 'report.pdf'
            rubric_path.write_bytes(b'fake rubric')
            report_path.write_bytes(b'fake report')

            with (
                patch(
                    'core.utils.document_parser.extract_text_from_file',
                    return_value='rubric text',
                ),
                patch(
                    'core.utils.document_parser.extract_text_from_bytes',
                    return_value='report text',
                ),
                patch(
                    'viva_evaluator.services.rubric_extractor.extract_rubric_from_text',
                    return_value=rubric_data,
                ),
                patch(
                    'AI_Evaluator_Backend.azure_storage.upload_report_to_blob',
                    return_value='https://storage.example/reports/project/group/report.pdf',
                ),
                patch(
                    'viva_evaluator.services.indexing.indexing_runner.run_report_indexing',
                    side_effect=finish_indexing,
                ),
                patch(
                    'viva_evaluator.services.rubric_extractor.generate_viva_grouping',
                    side_effect=create_grouping,
                ),
            ):
                output = StringIO()
                call_command(
                    'create_ready_session',
                    name='Automated Zero Trust Test',
                    rubric=str(rubric_path),
                    report=str(report_path),
                    examiner_email='examiner@example.com',
                    student_emails=self.student_emails,
                    stdout=output,
                )

        project = Project.objects.get(project_name='Automated Zero Trust Test')
        session = EvaluationSession.objects.get(project=project)
        submission = project.submissions.get()

        self.assertEqual(project.status, Project.Status.ACTIVE)
        self.assertEqual(project.evaluation_mode, Project.EvaluationMode.PHYSICAL)
        self.assertEqual(project.physical_config.location, 'LT1')
        self.assertEqual(project.student_groups.get().members.count(), 2)
        self.assertEqual(project.rubric_categories.count(), 1)
        self.assertEqual(project.rubric_categories.get().criteria.count(), 1)
        self.assertEqual(submission.index_status.status, SubmissionIndexStatus.IndexStatus.READY)
        self.assertEqual(session.submission, submission)
        self.assertEqual(session.group, project.student_groups.get())
        self.assertIn('Ready session created successfully', output.getvalue())
