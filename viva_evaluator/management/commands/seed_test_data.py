from django.core.management.base import BaseCommand
from django.utils import timezone


class Command(BaseCommand):
    help = 'Seeds test data for viva evaluator demo'

    def handle(self, *args, **options):
        from core.models import (
            User, StudentProfile, ExaminerProfile,
            Project, ProjectExaminer, RubricCategory,
            RubricCriteria, ProjectSubmission, EvaluationSession
        )
        from viva_evaluator.models import SubmissionIndexStatus

        self.stdout.write('Creating test data...')

        # --- Examiner ---
        examiner_user, _ = User.objects.get_or_create(
            email='examiner@test.com',
            defaults={
                'full_name': 'Dr. Smith',
                'role': User.Role.EXAMINER,
                'is_active': True,
            }
        )
        examiner_user.set_password('test1234')
        examiner_user.save()

        examiner_profile, _ = ExaminerProfile.objects.get_or_create(
            user=examiner_user,
            defaults={
                'employee_id': 'EMP001',
                'department': 'Computer Science',
                'designation': 'Senior Lecturer',
            }
        )

        # --- Student ---
        student_user, _ = User.objects.get_or_create(
            email='student@test.com',
            defaults={
                'full_name': 'John Doe',
                'role': User.Role.STUDENT,
                'is_active': True,
            }
        )
        student_user.set_password('test1234')
        student_user.save()

        student_profile, _ = StudentProfile.objects.get_or_create(
            user=student_user,
            defaults={
                'registration_number': 'STU2024001',
                'degree_program': 'BSc Computer Science',
                'academic_year': 4,
                'batch': '2024',
            }
        )

        # --- Project ---
        project, _ = Project.objects.get_or_create(
            project_name='Plant Disease Detection using CNN',
            defaults={
                'description': 'A machine learning project to detect plant diseases using convolutional neural networks.',
                'is_group_project': False,
                'status': Project.Status.ACTIVE,
                'academic_year': '2024/2025',
                'submission_deadline': timezone.now() + timezone.timedelta(days=30),
            }
        )

        # --- Project Examiner Link ---
        ProjectExaminer.objects.get_or_create(
            project=project,
            examiner=examiner_profile,
            defaults={'role_in_project': ProjectExaminer.RoleInProject.LEAD}
        )

        # --- Rubric Categories and Criteria ---
        cat1, _ = RubricCategory.objects.get_or_create(
            project=project,
            category_name='Problem Formulation',
            defaults={
                'weight_percentage': 30,
                'description': 'How well the student identifies and defines the research problem.',
            }
        )

        RubricCriteria.objects.get_or_create(
            category=cat1,
            criteria_name='Problem Identification',
            defaults={
                'max_score': 10,
                'description': 'Student clearly identifies the problem being solved and explains why it matters.',
            }
        )

        RubricCriteria.objects.get_or_create(
            category=cat1,
            criteria_name='Research Gap',
            defaults={
                'max_score': 10,
                'description': 'Student demonstrates awareness of existing solutions and the gap their project fills.',
            }
        )

        cat2, _ = RubricCategory.objects.get_or_create(
            project=project,
            category_name='Methodology',
            defaults={
                'weight_percentage': 40,
                'description': 'The approach taken to solve the problem.',
            }
        )

        RubricCriteria.objects.get_or_create(
            category=cat2,
            criteria_name='Technical Approach',
            defaults={
                'max_score': 10,
                'description': 'Student explains the technical method used and justifies why it was chosen.',
            }
        )

        # --- Submission ---
        submission, _ = ProjectSubmission.objects.get_or_create(
            project=project,
            student=student_profile,
            defaults={
                'github_repo_url': 'https://github.com/student/plant-disease',
            }
        )

        # --- Index Status with sample extracted text ---
        sample_text = """
        This project develops a machine learning model to detect diseases in plant leaves
        using convolutional neural networks (CNNs). The problem was formulated based on the
        need for automated agricultural monitoring. Farmers in developing regions lose up to
        40% of their crops annually due to undetected plant diseases.

        The methodology involves four main stages: dataset collection from the PlantVillage
        dataset containing 54,000 images, preprocessing using image augmentation techniques,
        model training using ResNet50 with transfer learning, and evaluation using accuracy,
        precision, and recall metrics.

        The research gap identified is that existing manual inspection methods are slow,
        expensive, and require domain expertise. Current automated solutions have limited
        accuracy on real-world field images due to varying lighting and background conditions.

        Our proposed solution achieves 94.7% accuracy on the test set, outperforming existing
        approaches. The system uses a webcam interface to allow farmers to photograph leaves
        and receive instant disease diagnosis with treatment recommendations.
        """

        index_status, _ = SubmissionIndexStatus.objects.get_or_create(
            submission=submission,
            defaults={
                'extracted_text': sample_text,
                'status': SubmissionIndexStatus.IndexStatus.READY,
                'processed_at': timezone.now(),
            }
        )

        if not index_status.extracted_text:
            index_status.extracted_text = sample_text
            index_status.status = SubmissionIndexStatus.IndexStatus.READY
            index_status.processed_at = timezone.now()
            index_status.save()

        # --- Evaluation Session ---
        session, _ = EvaluationSession.objects.get_or_create(
            project=project,
            student=student_profile,
            submission=submission,
            defaults={
                'scheduled_start': timezone.now(),
                'scheduled_end': timezone.now() + timezone.timedelta(hours=1),
                'status': EvaluationSession.Status.SCHEDULED,
            }
        )

        self.stdout.write(self.style.SUCCESS('\nTest data created successfully!\n'))
        self.stdout.write(f'Examiner login:  examiner@test.com / test1234')
        self.stdout.write(f'Student login:   student@test.com / test1234')
        self.stdout.write(f'Project ID:      {project.id}')
        self.stdout.write(f'Submission ID:   {submission.id}')
        self.stdout.write(f'Session ID:      {session.id}')
        self.stdout.write(f'\nUse the Session ID to start a viva via POST /api/viva/sessions/start/')