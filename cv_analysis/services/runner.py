"""Post-hoc CV analysis runner.

Drives analysis of a session recording and stores the resulting summary
artifact (seam 3). The recording itself is written straight to Azure Blob by
Agora Cloud Recording (see agora_service/cloud_recording.py) — nothing here
ever handles video bytes on the web tier.

Two backends:

* ``modal`` (default, cloud): hands Modal a short-lived SAS URL for the
  recording plus the session manifest, and — for group sessions — one SAS per
  student enrollment photo so faces can be matched to roster members. Analysis
  is submit/poll because a 20-min viva takes ~5-10 min to process, far longer
  than one HTTP request should live.
* ``subprocess`` (local dev): the original path — invokes the exam-station-cv
  engine's CLI in its own virtualenv (CV_ANALYSIS_PYTHON).

The polling thread is best-effort: ``modal_call_id`` is persisted, so if the
gunicorn worker running it is recycled the result is still claimable later via
``poll_modal_result`` (the summary endpoint calls it for stuck reports).
"""

import json
import logging
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from django.conf import settings

from core.models import EvaluationSession, GroupMember, SessionRecording
from cv_analysis.models import CVSessionReport
from .manifest import build_manifest
from .storage import is_local_recording

logger = logging.getLogger(__name__)

# CV analysis is CPU-bound (face mesh over every frame) — one at a time.
# With the modal backend this thread mostly just waits on polling.
_EXECUTOR = ThreadPoolExecutor(max_workers=1)

# How long a SAS handed to Modal stays valid. Comfortably longer than the
# analysis itself so a queued job doesn't expire before it starts.
_SAS_EXPIRY_HOURS = 6

_POLL_INTERVAL_S = 20


def enqueue_cv_analysis(session_id) -> bool:
    """Queue post-hoc analysis for a session's recording.

    Returns True if this process will run it now, False if it was left as a
    PENDING job for a separate worker to claim.

    When CV_ANALYSIS_ENABLED is off we still create a PENDING CVSessionReport
    so the recording becomes a claimable job — it is NOT silently dropped.
    """
    if not getattr(settings, 'CV_ANALYSIS_ENABLED', False):
        report, _ = CVSessionReport.objects.get_or_create(session_id=session_id)
        if report.status in (
            CVSessionReport.Status.COMPLETED,
            CVSessionReport.Status.PROCESSING,
        ):
            return False
        report.status = CVSessionReport.Status.PENDING
        report.error_message = ''
        report.save(update_fields=['status', 'error_message', 'updated_at'])
        logger.info(
            "CV disabled here — session %s left PENDING for a worker.", session_id,
        )
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

        ref = recording.video_file_url
        report.recording_url = ref
        manifest = build_manifest(session)

        if _backend() == 'modal':
            _run_via_modal(report, session, ref, manifest)
        else:
            _run_via_subprocess(report, session, ref, manifest, session_id)

    except Exception as e:
        logger.exception("CV analysis failed for session %s", session_id)
        report.status = CVSessionReport.Status.FAILED
        report.error_message = str(e)[:2000]
        report.save(update_fields=[
            'status', 'error_message', 'recording_url', 'updated_at',
        ])
    return report


def _backend() -> str:
    return getattr(settings, 'CV_ANALYSIS_BACKEND', 'modal').lower()


# ---------------------------------------------------------------------------
# Modal backend
# ---------------------------------------------------------------------------


def _run_via_modal(report, session, recording_ref, manifest) -> None:
    if is_local_recording(recording_ref):
        raise RuntimeError(
            "The modal backend needs a blob URL, but this recording is a local "
            f"path ({recording_ref}). Use CV_ANALYSIS_BACKEND=subprocess for "
            "local files."
        )

    submit_url = getattr(settings, 'MODAL_CV_SUBMIT_URL', '')
    token = getattr(settings, 'MODAL_CV_TOKEN', '')
    if not submit_url or not token:
        raise RuntimeError(
            "Modal CV endpoint not configured. Set MODAL_CV_SUBMIT_URL, "
            "MODAL_CV_RESULT_URL and MODAL_CV_TOKEN."
        )

    payload = {
        'token': token,
        'video_url': _sas_for(recording_ref),
        'manifest': manifest,
    }
    photos = _enrollment_photo_urls(session)
    if photos:
        payload['enrollment_photos'] = photos
        logger.info(
            "CV analysis for session %s: %d enrollment photo(s) for face ID.",
            session.id, len(photos),
        )

    response = requests.post(submit_url, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"Modal submit failed ({response.status_code}): {response.text[:500]}"
        )
    call_id = response.json().get('call_id')
    if not call_id:
        raise RuntimeError(f"Modal submit returned no call_id: {response.text[:300]}")

    report.modal_call_id = call_id
    report.save(update_fields=['modal_call_id', 'recording_url', 'updated_at'])
    logger.info("CV analysis for session %s submitted to Modal (%s).",
                session.id, call_id)

    deadline = time.monotonic() + int(getattr(settings, 'CV_ANALYSIS_TIMEOUT', 3600))
    while time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)
        if poll_modal_result(report):
            return
    raise RuntimeError(
        f"Modal analysis still running after CV_ANALYSIS_TIMEOUT ({call_id})."
    )


def poll_modal_result(report) -> bool:
    """Check an in-flight Modal job once; True when the report is terminal.

    Safe to call from anywhere (the summary endpoint uses it to resume a job
    whose polling thread died with its gunicorn worker). Transport errors are
    swallowed — the job is still running on Modal, so the next poll retries.
    """
    if not report.modal_call_id:
        return False

    result_url = getattr(settings, 'MODAL_CV_RESULT_URL', '')
    token = getattr(settings, 'MODAL_CV_TOKEN', '')
    if not result_url or not token:
        return False

    try:
        response = requests.get(
            result_url,
            params={'call_id': report.modal_call_id, 'token': token},
            timeout=30,
        )
    except requests.RequestException as e:
        logger.warning("CV poll transport error for session %s: %s",
                       report.session_id, e)
        return False

    if response.status_code == 202:
        return False
    if response.status_code != 200:
        logger.warning("CV poll got %d for session %s: %s",
                       response.status_code, report.session_id,
                       response.text[:300])
        return False

    body = response.json()
    status = body.get('status')

    if status == 'done':
        report.artifact = body.get('summary')
        report.status = CVSessionReport.Status.COMPLETED
        report.modal_call_id = ''
        report.save(update_fields=[
            'artifact', 'status', 'modal_call_id', 'updated_at',
        ])
        logger.info("CV analysis completed for session %s", report.session_id)
        return True

    if status == 'failed':
        report.status = CVSessionReport.Status.FAILED
        report.error_message = str(body.get('error', 'Modal job failed'))[:2000]
        report.modal_call_id = ''
        report.save(update_fields=[
            'status', 'error_message', 'modal_call_id', 'updated_at',
        ])
        logger.error("CV analysis failed on Modal for session %s: %s",
                     report.session_id, report.error_message)
        return True

    return False


def _enrollment_photo_urls(session) -> dict:
    """student_id -> SAS URL of that student's enrollment face photo.

    Group sessions only: a composite recording frames every member at once, so
    faces must be matched against enrolled photos to know who is speaking.
    Individual sessions need no gallery (roster of one). Members without a
    photo are omitted — they resolve to unknown, never guessed.
    """
    if not session.group_id:
        return {}

    members = (
        GroupMember.objects
        .filter(group_id=session.group_id)
        .select_related('student')
    )
    photos = {}
    for member in members:
        ref = getattr(member.student, 'face_photo_url', '')
        if not ref:
            logger.info(
                "Session %s: student %s has no enrollment photo — their turns "
                "will show as unknown.", session.id, member.student_id,
            )
            continue
        try:
            photos[str(member.student_id)] = _sas_for(ref)
        except Exception:
            logger.exception("Could not sign enrollment photo for student %s",
                             member.student_id)
    return photos


def _sas_for(blob_url: str) -> str:
    """Short-lived read URL for a blob we stored (video or face photo)."""
    from AI_Evaluator_Backend.azure_storage import generate_sas_url

    container, blob_path = _split_blob_url(blob_url)
    return generate_sas_url(container, blob_path, expiry_hours=_SAS_EXPIRY_HOURS)


def _split_blob_url(blob_url: str):
    parsed = urlparse(blob_url)
    container, _, blob_path = unquote(parsed.path).lstrip('/').partition('/')
    if not container or not blob_path:
        raise RuntimeError(f"Unrecognized blob URL format: {blob_url}")
    return container, blob_path


# ---------------------------------------------------------------------------
# Subprocess backend (local dev)
# ---------------------------------------------------------------------------


def _run_via_subprocess(report, session, recording_ref, manifest, session_id) -> None:
    with tempfile.TemporaryDirectory(prefix='cv_analysis_') as tmp:
        tmp_path = Path(tmp)

        if is_local_recording(recording_ref):
            video_path = Path(recording_ref)
            if not video_path.exists():
                raise RuntimeError(f"Local recording not found: {video_path}")
        else:
            video_path = tmp_path / _blob_filename(recording_ref)
            _download_blob(recording_ref, video_path)

        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

        enrollment_dir = _download_enrollment_photos(session, tmp_path)

        out_dir = tmp_path / 'out'
        summary = _run_engine(
            video_path, manifest_path, out_dir, session_id, enrollment_dir,
        )

    report.artifact = summary
    report.status = CVSessionReport.Status.COMPLETED
    report.save(update_fields=[
        'artifact', 'status', 'recording_url', 'updated_at',
    ])
    logger.info("CV analysis completed for session %s", session_id)


def _download_enrollment_photos(session, tmp_path: Path):
    """Fetch roster face photos into <tmp>/enroll for the engine, or None."""
    if not session.group_id:
        return None

    members = (
        GroupMember.objects
        .filter(group_id=session.group_id)
        .select_related('student')
    )
    enrollment_dir = tmp_path / 'enroll'
    enrollment_dir.mkdir(exist_ok=True)
    found = False
    for member in members:
        ref = getattr(member.student, 'face_photo_url', '')
        if not ref:
            continue
        dest = enrollment_dir / f"{member.student_id}.jpg"
        try:
            if is_local_recording(ref):
                dest.write_bytes(Path(ref).read_bytes())
            else:
                _download_blob(ref, dest)
            found = True
        except Exception:
            logger.exception("Could not fetch enrollment photo for student %s",
                             member.student_id)
    return enrollment_dir if found else None


def _blob_filename(blob_url: str) -> str:
    name = unquote(urlparse(blob_url).path.rsplit('/', 1)[-1]) or 'recording'
    return name


def _download_blob(blob_url: str, dest: Path) -> None:
    """Download an Azure blob URL to a local file using account credentials."""
    from AI_Evaluator_Backend.azure_storage import _get_blob_service_client

    container, blob_path = _split_blob_url(blob_url)
    client = _get_blob_service_client().get_blob_client(
        container=container, blob=blob_path,
    )
    with open(dest, 'wb') as f:
        client.download_blob().readinto(f)
    logger.info("Downloaded %s (%d bytes)", dest.name, dest.stat().st_size)


def _run_engine(video_path, manifest_path, out_dir, session_id, enrollment_dir):
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
    if enrollment_dir is not None:
        cmd += ['--enrollment-dir', str(enrollment_dir)]

    timeout = int(getattr(settings, 'CV_ANALYSIS_TIMEOUT', 3600))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or '')[-1500:]
        raise RuntimeError(f"exam_cv.analyze failed (rc={result.returncode}): {tail}")

    summary_path = out_dir / f"session_{session_id}_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Engine finished but summary not found at {summary_path}")
    return json.loads(summary_path.read_text(encoding='utf-8'))
