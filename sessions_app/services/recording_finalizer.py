"""Finalize online Agora recordings exactly once.

The adaptive viva pipeline can complete a session without an examiner pressing
the manual "End Viva" button.  Recording shutdown therefore belongs in a
shared service that both completion paths call.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
from typing import Optional

from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from core.models import EvaluationSession, Project, SessionRecording


logger = logging.getLogger(__name__)


class RecordingFinalizationError(RuntimeError):
    """Raised when an online viva finishes without a usable video."""


@dataclass(frozen=True)
class RecordingFinalizationResult:
    recording: SessionRecording
    already_finalized: bool


def recording_owner(session: EvaluationSession) -> EvaluationSession:
    """Return the deterministic row that owns a shared Agora recording."""
    if not session.group_id or not session.agora_channel_name:
        return session
    try:
        owner = EvaluationSession.objects.filter(
            id=session.agora_channel_name,
        ).first()
    except (ValueError, ValidationError):
        return session
    return owner or session


def _locked_session_family(session_id):
    """Lock a group in stable order so sibling finalizers cannot deadlock."""
    preview = EvaluationSession.objects.only(
        "id", "project_id", "group_id",
    ).get(id=session_id)
    # `group` is nullable. PostgreSQL rejects SELECT ... FOR UPDATE when the
    # query joins the nullable side of an outer join. Only the concrete
    # session rows need locking; project is non-null and safe to join.
    sessions = EvaluationSession.objects.select_for_update().select_related(
        "project",
    )
    if preview.group_id:
        locked = list(
            sessions.filter(
                project_id=preview.project_id,
                group_id=preview.group_id,
            ).order_by("id")
        )
        return next(item for item in locked if item.id == preview.id), locked
    session = sessions.get(id=session_id)
    return session, [session]


def _locked_recording_owner(
    session: EvaluationSession,
    locked_family,
) -> EvaluationSession:
    if not session.group_id or not session.agora_channel_name:
        return session
    owner = next(
        (
            item for item in locked_family
            if str(item.id) == session.agora_channel_name
        ),
        None,
    )
    if owner is not None:
        return owner
    try:
        owner = (
            EvaluationSession.objects.select_for_update()
            .filter(id=session.agora_channel_name)
            .first()
        )
    except (ValueError, ValidationError):
        return session
    return owner or session


def _stop_stt_later(session: EvaluationSession) -> None:
    """Stop STT without holding up the final answer response."""
    from agora_service.stt_manager import is_enabled, stop_stt

    if not (is_enabled() and session.agora_stt_task_id):
        return
    threading.Thread(target=stop_stt, args=(session,), daemon=True).start()


def _mark_sessions_completed(session: EvaluationSession) -> None:
    if session.group_id:
        EvaluationSession.objects.filter(
            project_id=session.project_id,
            group_id=session.group_id,
        ).update(status=EvaluationSession.Status.COMPLETED)
        return
    if session.status != EvaluationSession.Status.COMPLETED:
        session.status = EvaluationSession.Status.COMPLETED
        session.save(update_fields=["status"])


def _enqueue_analysis(session_id) -> None:
    try:
        from cv_analysis.services.runner import enqueue_cv_analysis

        enqueue_cv_analysis(session_id)
    except Exception as exc:  # Recording is safe even if the worker is down.
        logger.exception(
            "Could not queue behavior analysis for session %s", session_id,
        )
        _safe_mark_recording_failure(
            session_id,
            f"Recording saved, but behavior analysis could not be queued: {exc}",
        )


def mark_recording_failure(session_id, message: str) -> None:
    """Expose finalization failure to the behavior-analysis status endpoint."""
    from cv_analysis.models import CVSessionReport

    report, _ = CVSessionReport.objects.get_or_create(session_id=session_id)
    if report.status == CVSessionReport.Status.COMPLETED:
        return
    report.status = CVSessionReport.Status.FAILED
    report.error_message = str(message)[:2000]
    report.save(update_fields=["status", "error_message", "updated_at"])


def _safe_mark_recording_failure(session_id, message: str) -> None:
    try:
        mark_recording_failure(session_id, message)
    except Exception:
        logger.exception(
            "Could not persist recording failure for session %s", session_id,
        )


def finalize_online_recording(
    session_id,
    *,
    fallback_video_url: Optional[str] = None,
    fallback_audio_url: Optional[str] = None,
) -> RecordingFinalizationResult:
    """Stop Agora, persist its video, and complete an online session.

    A row lock serializes automatic completion, manual completion, and client
    retries.  The lock intentionally spans the Agora stop call: it is a single
    end-of-session network request and prevents two web workers from stopping
    the same recording or creating duplicate recording rows.
    """
    should_enqueue = False
    stt_session = None
    finalization_error = None
    result = None

    with transaction.atomic():
        session, locked_family = _locked_session_family(session_id)
        if session.project.evaluation_mode == Project.EvaluationMode.PHYSICAL:
            raise RecordingFinalizationError(
                "Physical sessions must use the physical recording workflow."
            )
        if session.status not in (
            EvaluationSession.Status.IN_PROGRESS,
            EvaluationSession.Status.COMPLETED,
        ):
            raise RecordingFinalizationError(
                "This session has not started and cannot be finalized."
            )

        owner = _locked_recording_owner(session, locked_family)
        stt_session = owner if owner.agora_stt_task_id else session

        recording = (
            SessionRecording.objects.select_for_update()
            .filter(session=session)
            .order_by("-recorded_at")
            .first()
        )
        existing_video_url = (
            str(recording.video_file_url or "") if recording else ""
        )

        # A previous successful request is a replay, unless handles show that
        # Agora is somehow still recording and must be stopped to avoid cost.
        if existing_video_url and not owner.agora_recording_sid:
            _mark_sessions_completed(session)
            result = RecordingFinalizationResult(recording, True)
        else:
            cloud = None
            from agora_service.cloud_recording import (
                is_enabled as recording_enabled,
                stop_recording,
            )

            if recording_enabled() and owner.agora_recording_sid:
                cloud = stop_recording(owner)

            cloud_video_url = str((cloud or {}).get("url") or "")
            video_url = cloud_video_url or fallback_video_url or existing_video_url
            if not video_url:
                if owner.agora_recording_sid:
                    detail = "Agora could not stop or return the recording file."
                else:
                    # start_recording() stores Agora's exact REST error. Keep it;
                    # replacing it here would hide actionable causes such as
                    # invalid_appid or a disabled Cloud Recording product.
                    from cv_analysis.models import CVSessionReport

                    start_report = CVSessionReport.objects.filter(
                        session=owner,
                        status=CVSessionReport.Status.FAILED,
                    ).first()
                    stored_detail = str(
                        getattr(start_report, "error_message", "") or ""
                    )
                    if stored_detail.startswith(
                        "Agora Cloud Recording failed to start:"
                    ):
                        detail = stored_detail
                    else:
                        detail = (
                            "Agora Cloud Recording did not start for this session."
                        )
                finalization_error = (
                    f"{detail} No video recording was saved."
                )
            else:
                duration_seconds = None
                if session.actual_start:
                    duration_seconds = max(
                        0,
                        int((timezone.now() - session.actual_start).total_seconds()),
                    )
                recording_started_at = (cloud or {}).get("started_at")
                if cloud_video_url and not recording_started_at:
                    recording_started_at = (
                        session.demo_completed_at or session.actual_start
                    )

                if recording is None:
                    recording = SessionRecording.objects.create(
                        session=session,
                        video_file_url=video_url,
                        audio_file_url=fallback_audio_url,
                        duration_seconds=duration_seconds,
                        recording_started_at=recording_started_at,
                    )
                else:
                    recording.video_file_url = video_url
                    if fallback_audio_url:
                        recording.audio_file_url = fallback_audio_url
                    recording.duration_seconds = duration_seconds
                    if recording_started_at:
                        recording.recording_started_at = recording_started_at
                    recording.save(update_fields=[
                        "video_file_url",
                        "audio_file_url",
                        "duration_seconds",
                        "recording_started_at",
                    ])

                _mark_sessions_completed(session)
                should_enqueue = not existing_video_url or bool(cloud_video_url)
                result = RecordingFinalizationResult(recording, False)

    if finalization_error:
        raise RecordingFinalizationError(finalization_error)
    if stt_session is not None:
        _stop_stt_later(stt_session)
    if should_enqueue:
        _enqueue_analysis(session_id)
    return result


def finalize_completed_online_session(
    session: EvaluationSession,
    payload: dict,
) -> dict:
    """Finalize an automatically terminated remote viva without losing answer."""
    if not payload.get("session_complete"):
        return payload
    if session.project.evaluation_mode == Project.EvaluationMode.PHYSICAL:
        return payload

    try:
        result = finalize_online_recording(session.id)
        payload["recording_finalization"] = {
            "status": "completed",
            "recording_id": str(result.recording.id),
        }
    except RecordingFinalizationError as exc:
        logger.error(
            "Automatic recording finalization failed for session %s: %s",
            session.id,
            exc,
        )
        _safe_mark_recording_failure(session.id, str(exc))
        payload["recording_finalization"] = {
            "status": "failed",
            "message": str(exc),
        }
    except Exception as exc:
        logger.exception(
            "Unexpected automatic recording finalization failure for %s",
            session.id,
        )
        message = f"Unexpected recording finalization error: {exc}"
        _safe_mark_recording_failure(session.id, message)
        payload["recording_finalization"] = {
            "status": "failed",
            "message": message,
        }
    return payload
