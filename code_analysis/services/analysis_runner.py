from concurrent.futures import ThreadPoolExecutor

from django.conf import settings

from .analysis_service import CodeAnalysisService


_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def enqueue_code_analysis(code_submission_id):
    if not getattr(settings, "CODE_ANALYSIS_ASYNC", True):
        CodeAnalysisService().analyze_submission(code_submission_id)
        return

    _EXECUTOR.submit(CodeAnalysisService().analyze_submission, code_submission_id)
