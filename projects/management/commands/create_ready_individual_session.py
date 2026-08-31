"""Provision a ready-to-run individual physical session for Pavith."""

from .create_ready_session import Command as ReadySessionCommand


DEFAULT_EXAMINER = 'examiner@university.edu'
DEFAULT_STUDENT = 'Pavith@gmail.com'


class Command(ReadySessionCommand):
    help = (
        'Create a complete individual physical project and session. Defaults '
        'to Pavith and examiner@university.edu.'
    )

    def handle(self, *args, **options):
        options['individual'] = True
        options['mode'] = 'physical'
        if not options.get('student_emails'):
            options['student_emails'] = [DEFAULT_STUDENT]
        if not str(options.get('examiner_email') or '').strip():
            options['examiner_email'] = DEFAULT_EXAMINER
        return super().handle(*args, **options)
