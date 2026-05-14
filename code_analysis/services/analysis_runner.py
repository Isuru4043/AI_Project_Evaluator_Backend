import logging
from concurrent.futures import ThreadPoolExecutor

from django.conf import settings

from .analysis_service import CodeAnalysisService

logger = logging.getLogger(__name__)


_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def enqueue_code_analysis(code_submission_id):
    logger.info(f"enqueue_code_analysis called for submission_id: {code_submission_id}")
    if not getattr(settings, "CODE_ANALYSIS_ASYNC", True):
        logger.info("CODE_ANALYSIS_ASYNC is False. Running synchronously.")
        CodeAnalysisService().analyze_submission(code_submission_id)
        return

    logger.info(f"Submitting analyze_submission for {code_submission_id} to ThreadPoolExecutor.")
    _EXECUTOR.submit(CodeAnalysisService().analyze_submission, code_submission_id)
