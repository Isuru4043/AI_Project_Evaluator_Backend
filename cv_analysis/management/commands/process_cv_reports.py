"""CV analysis worker — claims PENDING behavioral-analysis jobs and runs them.

Intended to run on a machine that HAS the exam-station-cv engine venv (the
department HPC, or a laptop for demos). The web backend (Azure App Service)
can leave CV_ANALYSIS_ENABLED off and just accumulate PENDING jobs; this
worker — sharing the same NeonDB + Azure blob — processes them.

Usage:
    python manage.py process_cv_reports            # loop forever
    python manage.py process_cv_reports --once      # drain then exit
    python manage.py process_cv_reports --interval 5

Multiple workers are safe: each job is claimed with an atomic conditional
UPDATE (PENDING → PROCESSING), so only one worker ever grabs a given report.
"""

import time

from django.core.management.base import BaseCommand
from django.db import transaction

from cv_analysis.models import CVSessionReport
from cv_analysis.services.runner import run_cv_analysis


class Command(BaseCommand):
    help = "Process pending CV/behavioral analysis jobs."

    def add_arguments(self, parser):
        parser.add_argument('--once', action='store_true',
                            help='Drain all pending jobs then exit.')
        parser.add_argument('--interval', type=int, default=10,
                            help='Seconds to sleep when idle (loop mode).')

    def handle(self, *args, **options):
        once = options['once']
        interval = options['interval']
        self.stdout.write("CV worker started.")

        while True:
            session_id = self._claim_one()
            if session_id is not None:
                self.stdout.write(f"Processing session {session_id}…")
                try:
                    report = run_cv_analysis(session_id)
                    self.stdout.write(
                        self.style.SUCCESS(f"  → {report.status}")
                    )
                except Exception as exc:  # run_cv_analysis records failures itself
                    self.stderr.write(f"  → error: {exc}")
                continue  # immediately look for the next job

            if once:
                self.stdout.write("No pending jobs. Exiting (--once).")
                return
            time.sleep(interval)

    @staticmethod
    def _claim_one():
        """Atomically flip the oldest PENDING report to PROCESSING and return
        its session_id, or None if there are none. The conditional filter on
        status prevents two workers claiming the same job."""
        with transaction.atomic():
            report = (
                CVSessionReport.objects
                .filter(status=CVSessionReport.Status.PENDING)
                .order_by('created_at')
                .first()
            )
            if report is None:
                return None
            claimed = (
                CVSessionReport.objects
                .filter(pk=report.pk, status=CVSessionReport.Status.PENDING)
                .update(status=CVSessionReport.Status.PROCESSING)
            )
            if not claimed:
                return None  # another worker beat us; try again next loop
            return report.session_id
