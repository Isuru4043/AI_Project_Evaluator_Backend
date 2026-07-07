"""Post-hoc CV analysis runner.

Downloads the session recording from blob storage, writes the session
manifest (seam 1), invokes the exam-station-cv engine's ``analyze`` CLI in
its own virtualenv (heavy CV deps stay out of the Django venv / cloud
deploys), and stores the resulting summary artifact (seam 3).

Deploys without the CV toolchain simply leave CV_ANALYSIS_ENABLED off:
recordings are still stored, and analysis can be triggered later from a
machine that has the engine.
"""

import json
import logging
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlparse

from django.conf import settings

from core.models import EvaluationSession, SessionRecording
from cv_analysis.models import CVSessionReport
from .manifest import build_manifest

logger = logging.getLogger(__name__)

# CV analysis is CPU-bound (face mesh over every frame) — one at a time.
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def enqueue_cv_analysis(session_id) -> bool:
    """Queue post-hoc analysis for a session's recording.

    Returns True if the job was queued/ran, False if CV analysis is disabled.
    """
    if not getattr(settings, 'CV_ANALYSIS_ENABLED', False):
        logger.info("CV_ANALYSIS_ENABLED is off — skipping analysis for %s", session_id)
        return False

    if not getattr(settings, 'CV_ANALYSIS_ASYNC', True):
        run_cv_analysis(session_id)
        return True

    logger.info("Queueing CV analysis for session %s", session_id)
    _EXECUTOR.submit(_run_safely, session_id)
    return True


def _run_safely(session_id):
    try:
        run_cv_analysis(session_id)
    except Exception:
        logger.exception("CV analysis crashed for session %s", session_id)


def run_cv_analysis(session_id) -> CVSessionReport:
    session = (
        EvaluationSession.objects
        .select_related('project', 'student__user', 'group')
        .get(id=session_id)
    )
    report, _ = CVSessionReport.objects.get_or_create(session=session)
    report.status = CVSessionReport.Status.PROCESSING
    report.error_message = ''
    report.save(update_fields=['status', 'error_message', 'updated_at'])

    try:
        recording = (
            SessionRecording.objects
            .filter(session=session)
            .exclude(video_file_url__isnull=True)
            .exclude(video_file_url='')
            .order_by('-recorded_at')
            .first()
        )
        if recording is None:
            raise RuntimeError("No video recording found for this session.")

        report.recording_url = recording.video_file_url

        with tempfile.TemporaryDirectory(prefix='cv_analysis_') as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / _blob_filename(recording.video_file_url)
            _download_blob(recording.video_file_url, video_path)

            manifest_path = tmp_path / 'manifest.json'
            manifest_path.write_text(
                json.dumps(build_manifest(session)), encoding='utf-8',
            )

            out_dir = tmp_path / 'out'
            summary = _run_engine(video_path, manifest_path, out_dir, session_id)

        report.artifact = summary
        report.status = CVSessionReport.Status.COMPLETED
        report.save(update_fields=[
            'artifact', 'status', 'recording_url', 'updated_at',
        ])
        logger.info("CV analysis completed for session %s", session_id)
    except Exception as e:
        logger.exception("CV analysis failed for session %s", session_id)
        report.status = CVSessionReport.Status.FAILED
        report.error_message = str(e)[:2000]
        report.save(update_fields=[
            'status', 'error_message', 'recording_url', 'updated_at',
        ])
    return report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _blob_filename(blob_url: str) -> str:
    name = unquote(urlparse(blob_url).path.rsplit('/', 1)[-1]) or 'recording'
    return name


def _download_blob(blob_url: str, dest: Path) -> None:
    """Download an Azure blob URL (as stored by azure_storage upload helpers)
    to a local file using the account credentials."""
    from AI_Evaluator_Backend.azure_storage import _get_blob_service_client

    parsed = urlparse(blob_url)
    container, _, blob_path = unquote(parsed.path).lstrip('/').partition('/')
    if not container or not blob_path:
        raise RuntimeError(f"Unrecognized blob URL format: {blob_url}")

    client = _get_blob_service_client().get_blob_client(
        container=container, blob=blob_path,
    )
    with open(dest, 'wb') as f:
        client.download_blob().readinto(f)
    logger.info("Downloaded recording (%d bytes) to %s", dest.stat().st_size, dest)


def _run_engine(video_path: Path, manifest_path: Path, out_dir: Path, session_id):
    """Invoke `exam_cv.analyze` in the engine's own venv and load the summary."""
    python_bin = getattr(settings, 'CV_ANALYSIS_PYTHON', '')
    if not python_bin or not Path(python_bin).exists():
        raise RuntimeError(
            "CV engine python not found. Set CV_ANALYSIS_PYTHON to the "
            "exam-station-cv virtualenv's python executable."
        )

    cmd = [
        str(python_bin), '-m', 'exam_cv.analyze',
        '--video', str(video_path),
        '--manifest', str(manifest_path),
        '--output-dir', str(out_dir),
    ]
    timeout = int(getattr(settings, 'CV_ANALYSIS_TIMEOUT', 1800))
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or '')[-1500:]
        raise RuntimeError(f"exam_cv.analyze failed (rc={result.returncode}): {tail}")

    summary_path = out_dir / f"session_{session_id}_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Engine finished but summary not found at {summary_path}")
    return json.loads(summary_path.read_text(encoding='utf-8'))
