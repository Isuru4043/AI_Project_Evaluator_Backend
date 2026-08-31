"""Provision a complete, ready-to-run evaluation session from local files."""

from datetime import timedelta
from pathlib import Path
from urllib.parse import unquote, urlparse

from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone


DEFAULT_RUBRIC = r'C:\Users\shami\Desktop\Test\Zero_Trust_Project_Evaluation_Rubric.pdf'
DEFAULT_REPORT = r'C:\Users\shami\Desktop\Test\zerotrust.pdf'
DEFAULT_EXAMINER = 'examiner@university.edu'
DEFAULT_STUDENTS = (
    'student@university.edu',
    'isuru.akalanka8058@gmail.com',
)


class Command(BaseCommand):
    help = (
        'Create a complete group project, extract its rubric, enrol students, '
        'upload and index the report, activate the project, and schedule one session.'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--name', required=True,
            help='Unique project/session display name. Change this for every run.',
        )
        parser.add_argument('--rubric', default=DEFAULT_RUBRIC)
        parser.add_argument('--report', default=DEFAULT_REPORT)
        parser.add_argument('--examiner-email', default=DEFAULT_EXAMINER)
        parser.add_argument(
            '--student-email', action='append', dest='student_emails', default=[],
            help=(
                'Student email to enrol. Repeat this option for every member. '
                'When omitted, the two configured test students are used.'
            ),
        )
        parser.add_argument(
            '--mode', choices=('physical', 'remote'), default='physical',
        )
        parser.add_argument('--location', default='LT1')
        parser.add_argument('--panel-pin', default='1234')
        parser.add_argument('--start-in-minutes', type=int, default=1)
        parser.add_argument('--duration-minutes', type=int, default=60)
        parser.add_argument('--max-questions', type=int, default=6)
        parser.add_argument('--viva-weight', type=int, default=100)
        parser.add_argument('--academic-year', default=str(timezone.now().year))
        parser.add_argument(
            '--no-demo', action='store_false', dest='demo_enabled', default=True,
            help='Send students directly to the viva instead of starting with demo mode.',
        )

    def handle(self, *args, **options):
        from core.models import (
            EvaluationSession,
            ExaminerProfile,
            GroupMember,
            Project,
            ProjectExaminer,
            ProjectSubmission,
            RubricCategory,
            RubricCriteria,
            StudentGroup,
            StudentProfile,
        )
        from core.utils.document_parser import (
            extract_text_from_bytes,
            extract_text_from_file,
        )
        from viva_evaluator.models import CriteriaQuestionHint, SubmissionIndexStatus
        from viva_evaluator.services.indexing.indexing_runner import run_report_indexing
        from viva_evaluator.services.rubric_extractor import (
            extract_rubric_from_text,
            generate_viva_grouping,
        )

        name = options['name'].strip()
        if not name:
            raise CommandError('--name cannot be empty.')
        if len(name) > 255:
            raise CommandError('--name must contain at most 255 characters.')

        rubric_path = self._validated_file(options['rubric'], 'rubric', 10)
        report_path = self._validated_file(options['report'], 'report', 50)
        if rubric_path.suffix.lower() not in {'.pdf', '.docx', '.md', '.markdown', '.txt'}:
            raise CommandError('Rubric must be PDF, DOCX, MD, Markdown, or TXT.')
        if report_path.suffix.lower() != '.pdf':
            raise CommandError('The project report must be a PDF file.')

        duration = options['duration_minutes']
        max_questions = options['max_questions']
        viva_weight = options['viva_weight']
        if duration <= 0:
            raise CommandError('--duration-minutes must be greater than zero.')
        if max_questions <= 0:
            raise CommandError('--max-questions must be greater than zero.')
        if not 0 <= viva_weight <= 100:
            raise CommandError('--viva-weight must be between 0 and 100.')
        if options['mode'] == Project.EvaluationMode.PHYSICAL:
            if not options['location'].strip():
                raise CommandError('--location is required for a physical session.')
            if len(options['panel_pin']) < 4:
                raise CommandError('--panel-pin must contain at least four characters.')

        if Project.objects.filter(project_name__iexact=name).exists():
            raise CommandError(
                f'A project named "{name}" already exists. Use a different --name.'
            )

        examiner = ExaminerProfile.objects.filter(
            user__email__iexact=options['examiner_email'].strip(),
            user__role='examiner',
        ).select_related('user').first()
        if not examiner:
            raise CommandError(
                f'Examiner account not found: {options["examiner_email"]}'
            )

        student_emails = options['student_emails'] or list(DEFAULT_STUDENTS)
        student_emails = list(dict.fromkeys(email.strip().lower() for email in student_emails))
        if not student_emails or any(not email for email in student_emails):
            raise CommandError('At least one valid --student-email is required.')

        students_by_email = {
            profile.user.email.lower(): profile
            for profile in StudentProfile.objects.filter(
                user__email__in=student_emails,
                user__role='student',
            ).select_related('user')
        }
        missing_students = [email for email in student_emails if email not in students_by_email]
        if missing_students:
            raise CommandError(
                'Student account(s) not found: ' + ', '.join(missing_students)
            )
        students = [students_by_email[email] for email in student_emails]

        self.stdout.write(f'Reading rubric: {rubric_path}')
        rubric_text = extract_text_from_file(str(rubric_path))
        if not rubric_text.strip():
            raise CommandError('No readable text was found in the rubric file.')

        report_bytes = report_path.read_bytes()
        report_text = extract_text_from_bytes(report_bytes, report_path.name)
        if not report_text.strip():
            raise CommandError('No readable text was found in the project report.')

        self.stdout.write('Extracting rubric structure with the configured LLM...')
        extracted_rubric = extract_rubric_from_text(rubric_text)
        if extracted_rubric.get('error'):
            raise CommandError(f'Rubric extraction failed: {extracted_rubric["error"]}')
        categories_data = extracted_rubric.get('rubric_categories') or []
        if not categories_data:
            raise CommandError('No rubric categories were found in the rubric file.')

        project = None
        report_url = None
        try:
            with transaction.atomic():
                project = Project.objects.create(
                    project_name=name,
                    description=(
                        f'Automatically provisioned evaluation for {report_path.stem}.'
                    ),
                    is_group_project=True,
                    submission_deadline=timezone.now() + timedelta(days=30),
                    status=Project.Status.DRAFT,
                    academic_year=options['academic_year'],
                    evaluation_mode=options['mode'],
                )
                ProjectExaminer.objects.create(
                    project=project,
                    examiner=examiner,
                    role_in_project=ProjectExaminer.RoleInProject.LEAD,
                )

                if options['mode'] == Project.EvaluationMode.PHYSICAL:
                    from physical_evaluation.models import PhysicalProjectConfig

                    physical_config = PhysicalProjectConfig(
                        project=project,
                        location=options['location'].strip(),
                        created_by=examiner,
                    )
                    physical_config.set_panel_pin(options['panel_pin'])
                    physical_config.save()

                group = StudentGroup.objects.create(
                    project=project,
                    group_name=f'{name} Group',
                )
                GroupMember.objects.bulk_create([
                    GroupMember(group=group, student=student) for student in students
                ])
                submission = ProjectSubmission.objects.create(
                    project=project,
                    group=group,
                )
                index_status = SubmissionIndexStatus.objects.create(
                    submission=submission,
                    extracted_text=report_text,
                    status=SubmissionIndexStatus.IndexStatus.PROCESSING,
                )
                self._create_rubric(
                    project,
                    categories_data,
                    RubricCategory,
                    RubricCriteria,
                    CriteriaQuestionHint,
                )

            self.stdout.write('Uploading the report to private Azure storage...')
            report_file = ContentFile(report_bytes, name=report_path.name)
            from AI_Evaluator_Backend.azure_storage import upload_report_to_blob

            report_url = upload_report_to_blob(
                report_file,
                str(project.id),
                group_id=str(group.id),
            )
            submission.report_file_url = report_url
            submission.save(update_fields=['report_file_url'])

            self.stdout.write('Building the report index; this can take a few minutes...')
            run_report_indexing(str(submission.id), report_bytes)
            index_status.refresh_from_db()
            if index_status.status != SubmissionIndexStatus.IndexStatus.READY:
                raise RuntimeError(
                    index_status.error_message or 'Report indexing did not reach READY status.'
                )

            self.stdout.write('Preparing adaptive viva topics...')
            grouping_cache = generate_viva_grouping(project, max_questions)

            scheduled_start = timezone.now() + timedelta(
                minutes=options['start_in_minutes'],
            )
            with transaction.atomic():
                project.status = Project.Status.ACTIVE
                project.save(update_fields=['status'])
                session = EvaluationSession.objects.create(
                    project=project,
                    group=group,
                    submission=submission,
                    scheduled_start=scheduled_start,
                    scheduled_end=scheduled_start + timedelta(minutes=duration),
                    location_room=(
                        options['location'].strip()
                        if options['mode'] == Project.EvaluationMode.PHYSICAL
                        else ''
                    ),
                    status=EvaluationSession.Status.SCHEDULED,
                    demo_enabled=options['demo_enabled'],
                    max_total_questions=max_questions,
                    viva_weight_percentage=viva_weight,
                    grouping_cache=grouping_cache,
                    agora_channel_name=f'group_{group.id}',
                )
        except Exception as exc:
            if project is not None:
                Project.objects.filter(id=project.id).delete()
            if report_url:
                self._delete_uploaded_report(report_url)
            raise CommandError(
                f'Session creation failed and the partial database setup was removed: {exc}'
            ) from exc

        self.stdout.write(self.style.SUCCESS('\nReady session created successfully.'))
        self.stdout.write(f'Project:       {project.project_name}')
        self.stdout.write(f'Project ID:    {project.id}')
        self.stdout.write(f'Group:         {group.group_name}')
        self.stdout.write(f'Session ID:    {session.id}')
        self.stdout.write(f'Mode:          {project.evaluation_mode}')
        self.stdout.write(f'Scheduled:     {session.scheduled_start:%Y-%m-%d %H:%M} - {session.scheduled_end:%H:%M}')
        self.stdout.write(f'Index status:  {index_status.status}')
        self.stdout.write('Students:      ' + ', '.join(student.user.full_name for student in students))

    @staticmethod
    def _validated_file(raw_path, label, max_size_mb):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise CommandError(f'{label.title()} file not found: {path}')
        if path.stat().st_size > max_size_mb * 1024 * 1024:
            raise CommandError(
                f'{label.title()} file exceeds the {max_size_mb} MB limit: {path}'
            )
        return path

    @staticmethod
    def _create_rubric(
        project,
        categories_data,
        rubric_category_model,
        rubric_criteria_model,
        hint_model,
    ):
        for category_data in categories_data:
            category = rubric_category_model.objects.create(
                project=project,
                category_name=str(category_data.get('category_name') or 'Untitled')[:255],
                weight_percentage=category_data.get('weight_percentage') or 0,
                description=category_data.get('description') or '',
            )
            for criteria_data in category_data.get('criteria') or []:
                criteria = rubric_criteria_model.objects.create(
                    category=category,
                    criteria_name=str(criteria_data.get('criteria_name') or 'Untitled')[:255],
                    max_score=criteria_data.get('max_score') or 10,
                    weight_in_category=criteria_data.get('weight_in_category'),
                    description=criteria_data.get('description') or '',
                    questions_to_ask=int(criteria_data.get('questions_to_ask') or 3),
                    is_individual=bool(criteria_data.get('is_individual', True)),
                )
                hint_model.objects.bulk_create([
                    hint_model(
                        criteria=criteria,
                        hint_text=str(hint.get('hint_text') or '').strip(),
                        order=int(hint.get('order') or index),
                    )
                    for index, hint in enumerate(
                        criteria_data.get('question_hints') or [], start=1,
                    )
                    if str(hint.get('hint_text') or '').strip()
                ])

    @staticmethod
    def _delete_uploaded_report(report_url):
        try:
            from AI_Evaluator_Backend.azure_storage import (
                AZURE_CONTAINER_REPORTS,
                delete_blob,
            )

            path = unquote(urlparse(report_url).path).lstrip('/')
            container, separator, blob_path = path.partition('/')
            if separator and container == AZURE_CONTAINER_REPORTS and blob_path:
                delete_blob(container, blob_path)
        except Exception:
            # Cleanup is best-effort; retain the original provisioning error.
            pass
