"""
Async report indexing (D1).

Moves the SLOW part of report processing — image captioning (Gemini Vision)
and SBERT embedding / FAISS build — OUT of the HTTP request and out of the DB
transaction, into a background thread.

WHY:
    Doing this work synchronously inside `transaction.atomic()` held a DB
    connection open for minutes. On serverless Postgres (Neon) the idle
    connection gets dropped → "SSL connection has been closed unexpectedly",
    and the upload request blocked for ~8 minutes.

PATTERN:
    Mirrors code_analysis.services.analysis_runner — a module-level
    ThreadPoolExecutor, gated by settings.REPORT_INDEX_ASYNC (default True),
    with a synchronous fallback for tests.

LIFECYCLE / STATUS:
    The request marks SubmissionIndexStatus = PROCESSING and returns fast.
    This worker sets it to READY when the FAISS index is built, or FAILED on
    error. The frontend polls SubmissionStatusView until READY; SessionStartView
    refuses to start a viva until then.

NOTE (honest limitation):
    This is an in-process executor — a server restart mid-indexing loses the
    job (the submission stays PROCESSING). A durable queue (Celery + Redis)
    is the production-grade upgrade; the threadpool is a proportionate FYP
    choice and is consistent with how code analysis already runs.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from django.conf import settings

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def enqueue_report_indexing(submission_id, report_bytes: bytes,
                            code_submission_id=None) -> None:
    """
    Schedule FAISS indexing for a submission. Returns immediately when async.

    If code_submission_id is given, code analysis is CHAINED to run *after*
    report indexing completes — code analysis appends code chunks to the report
    index and reads report chunks to build the KG, so it must not race the
    report index build.
    """
    logger.info('enqueue_report_indexing submission=%s bytes=%d code_submission=%s',
                submission_id, len(report_bytes or b''), code_submission_id)

    if not getattr(settings, 'REPORT_INDEX_ASYNC', True):
        logger.info('REPORT_INDEX_ASYNC is False — indexing synchronously.')
        _run_report_indexing(submission_id, report_bytes, code_submission_id)
        return

    _EXECUTOR.submit(_run_report_indexing, submission_id, report_bytes, code_submission_id)


def _run_report_indexing(submission_id, report_bytes: bytes,
                         code_submission_id=None) -> None:
    """
    Background worker: build the FAISS index and update status. Runs in its own
    thread, so it MUST close its DB connections when done to avoid leaking
    connections to the (serverless) database.

    After the indexing attempt (success OR failure), if a code submission is
    attached, code analysis is enqueued — guaranteeing report chunks are in the
    index before code analysis appends to it and builds the knowledge graph.
    """
    from django.db import connections
    from django.utils import timezone
    from core.models import ProjectSubmission
    from viva_evaluator.models import SubmissionIndexStatus

    try:
        try:
            submission = ProjectSubmission.objects.get(id=submission_id)
        except ProjectSubmission.DoesNotExist:
            logger.error('report indexing: submission %s no longer exists', submission_id)
            return

        index_status, _ = SubmissionIndexStatus.objects.get_or_create(submission=submission)

        try:
            from viva_evaluator.services.indexing import index_report
            from viva_evaluator.services.rag.vector_store import save_index_for_submission

            result = index_report(report_bytes, enable_image_captions=True)
            chunks = result.get('chunks', [])
            num_chunks, _ = save_index_for_submission(submission, chunks)

            index_status.status = SubmissionIndexStatus.IndexStatus.READY
            index_status.processed_at = timezone.now()
            index_status.error_message = None
            index_status.save(update_fields=['status', 'processed_at', 'error_message'])
            logger.info(
                'report indexing done submission=%s chunks=%d images_captioned=%d',
                submission_id, num_chunks, result.get('images_captioned', 0),
            )
        except Exception as exc:
            logger.exception('report indexing FAILED submission=%s', submission_id)
            index_status.status = SubmissionIndexStatus.IndexStatus.FAILED
            index_status.error_message = str(exc)[:2000]
            index_status.processed_at = timezone.now()
            index_status.save(update_fields=['status', 'error_message', 'processed_at'])
    finally:
        # Threads get their own DB connections that Django does NOT auto-close.
        connections.close_all()

    # Chain code analysis AFTER report indexing so it doesn't race the index
    # build (it appends code chunks + reads report chunks for the KG).
    if code_submission_id:
        try:
            from code_analysis.services.analysis_runner import enqueue_code_analysis
            logger.info('report indexing: chaining code analysis for %s', code_submission_id)
            enqueue_code_analysis(code_submission_id)
        except Exception:
            logger.exception('report indexing: failed to enqueue code analysis for %s',
                             code_submission_id)
