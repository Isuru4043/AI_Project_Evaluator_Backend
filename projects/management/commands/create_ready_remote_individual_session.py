"""Provision ready-to-run individual remote evaluation sessions."""

from .create_ready_session import (
    DEFAULT_REPORT as BASE_DEFAULT_REPORT,
    DEFAULT_RUBRIC as BASE_DEFAULT_RUBRIC,
    Command as ReadySessionCommand,
)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_RUBRIC = (
    r'C:\Users\pavit\Desktop\Vivasense documents\simplified_rubric.pdf'
)

DEFAULT_REPORT = (
    r'C:\Users\pavit\Desktop\Vivasense documents\vivasense_test_report.pdf'
)

DEFAULT_EXAMINER = 'examiner@university.edu'

# Default students to provision when no --student-email is specified.
DEFAULT_STUDENTS = [
    'student@university.edu',
    'Pavith@gmail.com',
    'isuru.akalanka8058@gmail.com',
]


# ---------------------------------------------------------------------------
# Management command
# ---------------------------------------------------------------------------

class Command(ReadySessionCommand):
    help = (
        'Create complete individual remote project and evaluation sessions '
        'for the configured students using the Vivasense rubric and report '
        'documents.'
    )

    def add_arguments(self, parser):
        # Keep all arguments from create_ready_session.
        super().add_arguments(parser)

        # Override defaults for this command.
        parser.set_defaults(
            rubric=DEFAULT_RUBRIC,
            report=DEFAULT_REPORT,
            mode='remote',
            individual=True,
        )

    def handle(self, *args, **options):
        # -------------------------------------------------------------------
        # Force this command to create individual remote sessions.
        # -------------------------------------------------------------------
        options['individual'] = True
        options['mode'] = 'remote'

        # -------------------------------------------------------------------
        # Students
        # -------------------------------------------------------------------
        student_emails = options.get('student_emails')
        if not student_emails:
            student_emails = DEFAULT_STUDENTS.copy()
        elif isinstance(student_emails, str):
            student_emails = [student_emails]

        # -------------------------------------------------------------------
        # Examiner
        # -------------------------------------------------------------------
        if not str(options.get('examiner_email') or '').strip():
            options['examiner_email'] = DEFAULT_EXAMINER

        # -------------------------------------------------------------------
        # Rubric
        # -------------------------------------------------------------------
        if (
            not options.get('rubric')
            or options.get('rubric') == BASE_DEFAULT_RUBRIC
        ):
            options['rubric'] = DEFAULT_RUBRIC

        # -------------------------------------------------------------------
        # Report
        # -------------------------------------------------------------------
        if (
            not options.get('report')
            or options.get('report') == BASE_DEFAULT_REPORT
        ):
            options['report'] = DEFAULT_REPORT

        base_name = options.get('name', '').strip()

        # If a single student is provided, run standard provisioning directly.
        if len(student_emails) == 1:
            options['student_emails'] = student_emails
            return super().handle(*args, **options)

        # If multiple students are provided, provision an individual session for each student.
        self.stdout.write(
            self.style.NOTICE(
                f'Creating {len(student_emails)} individual remote sessions for: '
                + ', '.join(student_emails)
            )
        )
        for i, email in enumerate(student_emails, start=1):
            single_options = options.copy()
            single_options['student_emails'] = [email]
            single_options['name'] = f'{base_name} - {email}'
            self.stdout.write(
                self.style.NOTICE(
                    f'\n[{i}/{len(student_emails)}] Provisioning individual session for {email}...'
                )
            )
            super().handle(*args, **single_options)
