"""Speculative ElevenLabs speech generation with deterministic storage keys.

Question text is never held up by this module.  A worker starts after Tier 1,
runs alongside the Critic, and writes audio to shared Django storage.  Only the
cache key belonging to the final accepted candidate is exposed by the API.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage


logger = logging.getLogger(__name__)
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCK = threading.RLock()
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="viva-tts")
_JOBS: "OrderedDict[str, _TTSJob]" = OrderedDict()
_METRICS = {
    "requests": 0,
    "jobs_started": 0,
    "cache_hits": 0,
    "generated": 0,
    "failed": 0,
    "stale_candidates": 0,
    "characters_requested": 0,
    "characters_generated": 0,
    "generation_latency_ms": 0,
}


@dataclass(frozen=True)
class TTSTicket:
    cache_key: str
    candidate_hash: str
    text_chars: int


@dataclass(frozen=True)
class TTSResult:
    status: str
    storage_path: str
    mime_type: str
    latency_ms: int
    cache_hit: bool
    text_chars: int
    error: str = ""


@dataclass
class _TTSJob:
    ticket: TTSTicket
    future: Future
    keep: bool = False


def _config() -> Dict[str, Any]:
    return {
        "enabled": bool(getattr(settings, "ELEVENLABS_TTS_ENABLED", False)),
        "api_key": str(getattr(settings, "ELEVENLABS_API_KEY", "") or ""),
        "voice_id": str(getattr(settings, "ELEVENLABS_VOICE_ID", "") or ""),
        "model_id": str(
            getattr(settings, "ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
            or "eleven_flash_v2_5"
        ),
        "output_format": str(
            getattr(settings, "ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
            or "mp3_44100_128"
        ),
        "timeout": max(1.0, float(getattr(settings, "ELEVENLABS_TIMEOUT_SECONDS", 20))),
        "max_jobs": max(8, int(getattr(settings, "ELEVENLABS_TTS_CACHE_MAX_JOBS", 256))),
    }


def _enabled(config: Dict[str, Any]) -> bool:
    return bool(config["enabled"] and config["api_key"] and config["voice_id"])


def _cache_key(candidate_hash: str, config: Dict[str, Any]) -> str:
    identity = ":".join(
        (
            candidate_hash,
            config["voice_id"],
            config["model_id"],
            config["output_format"],
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _storage_path(cache_key: str) -> str:
    return f"viva-tts/{cache_key}.mp3"


def start_speculative_tts(question_text: str, candidate_hash: str) -> Optional[TTSTicket]:
    """Start or join a TTS job and return immediately."""
    config = _config()
    if not _enabled(config):
        logger.info("[TTS Backend] ElevenLabs TTS disabled in config (ELEVENLABS_TTS_ENABLED=False or missing key/voice).")
        return None
    if not question_text.strip() or not candidate_hash:
        return None

    key = _cache_key(candidate_hash, config)
    ticket = TTSTicket(key, candidate_hash, len(question_text))
    with _LOCK:
        _METRICS["requests"] += 1
        _METRICS["characters_requested"] += len(question_text)
        existing = _JOBS.get(key)
        if existing is not None:
            _JOBS.move_to_end(key)
            logger.info("[TTS Backend] Existing job reused for key=%s...", key[:12])
            return existing.ticket

        logger.info("[TTS Backend] Starting speculative TTS for key=%s... (len=%d chars)", key[:12], len(question_text))
        future = _EXECUTOR.submit(_generate_audio, ticket, question_text, config)
        _JOBS[key] = _TTSJob(ticket=ticket, future=future)
        _METRICS["jobs_started"] += 1
        _trim_jobs(config["max_jobs"])
    return ticket


def finalize_question_tts(
    question_text: str,
    candidate_hash: str,
    speculative_ticket: Optional[TTSTicket],
) -> Dict[str, Any]:
    """Keep matching speculation or start final audio after a regeneration."""
    config = _config()
    if not _enabled(config):
        return {"status": "disabled", "enabled": False}

    wasted = bool(
        speculative_ticket
        and speculative_ticket.candidate_hash != candidate_hash
    )
    if wasted:
        logger.info("[TTS Backend] Speculative audio wasted due to candidate hash mismatch.")
        discard_speculative_tts(speculative_ticket)

    ticket = (
        speculative_ticket
        if speculative_ticket and not wasted
        else start_speculative_tts(question_text, candidate_hash)
    )
    if ticket is None:
        return {"status": "failed", "enabled": True, "error": "tts_not_started"}

    with _LOCK:
        job = _JOBS.get(ticket.cache_key)
        if job is not None:
            job.keep = True
    descriptor = _describe_ticket(ticket)
    descriptor["speculative"] = not wasted
    descriptor["speculative_wasted"] = wasted
    return descriptor


def discard_speculative_tts(ticket: Optional[TTSTicket]) -> None:
    if ticket is None:
        return
    with _LOCK:
        job = _JOBS.get(ticket.cache_key)
        if job is None or job.keep:
            return
        _METRICS["stale_candidates"] += 1
        job.future.add_done_callback(
            lambda future, key=ticket.cache_key: _delete_stale_result(key, future)
        )


def _describe_ticket(ticket: TTSTicket) -> Dict[str, Any]:
    descriptor: Dict[str, Any] = {
        "enabled": True,
        "status": "pending",
        "cache_key": ticket.cache_key,
        "candidate_hash": ticket.candidate_hash,
        "characters": ticket.text_chars,
        "model_id": _config()["model_id"],
        "cache_hit": None,
        "generation_latency_ms": None,
    }
    with _LOCK:
        job = _JOBS.get(ticket.cache_key)
    if job is None or not job.future.done():
        return descriptor
    result = job.future.result()
    descriptor.update(
        status=result.status,
        cache_hit=result.cache_hit,
        generation_latency_ms=result.latency_ms,
    )
    if result.error:
        descriptor["error"] = result.error[:160]
    return descriptor


def _generate_audio(ticket: TTSTicket, question_text: str, config: Dict[str, Any]) -> TTSResult:
    started = time.perf_counter()
    path = _storage_path(ticket.cache_key)
    mime_type = "audio/mpeg"
    try:
        if default_storage.exists(path):
            latency = int((time.perf_counter() - started) * 1000)
            logger.info("[TTS Backend] Cache HIT in storage for key=%s (%d ms)", ticket.cache_key[:12], latency)
            result = TTSResult(
                "ready", path, mime_type,
                latency, True,
                ticket.text_chars,
            )
            _record_result(result)
            return result

        logger.info(
            "[TTS Backend] Calling ElevenLabs API (model=%s, voice=%s, chars=%d) for: '%s'...",
            config["model_id"], config["voice_id"], len(question_text), question_text[:45],
        )
        t_req = time.perf_counter()
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{config['voice_id']}",
            params={"output_format": config["output_format"]},
            headers={
                "xi-api-key": config["api_key"],
                "Content-Type": "application/json",
                "Accept": mime_type,
            },
            json={"text": question_text, "model_id": config["model_id"]},
            timeout=config["timeout"],
        )
        response.raise_for_status()
        if not response.content:
            raise RuntimeError("ElevenLabs returned empty audio")

        eleven_ms = int((time.perf_counter() - t_req) * 1000)
        logger.info("[TTS Backend] ElevenLabs responded in %d ms (%d bytes)", eleven_ms, len(response.content))

        t_save = time.perf_counter()
        saved_path = default_storage.save(path, ContentFile(response.content))
        save_ms = int((time.perf_counter() - t_save) * 1000)
        total_ms = int((time.perf_counter() - started) * 1000)
        logger.info("[TTS Backend] Saved to storage (%s) in %d ms (total=%d ms)", saved_path, save_ms, total_ms)

        result = TTSResult(
            "ready", saved_path, mime_type,
            total_ms, False,
            ticket.text_chars,
        )
        _record_result(result)
        return result
    except Exception as exc:
        total_ms = int((time.perf_counter() - started) * 1000)
        logger.warning("[TTS Backend] ElevenLabs TTS FAILED (%s: %s) after %d ms", type(exc).__name__, str(exc), total_ms)
        result = TTSResult(
            "failed", path, mime_type,
            total_ms, False,
            ticket.text_chars, type(exc).__name__,
        )
        _record_result(result)
        return result


def _record_result(result: TTSResult) -> None:
    with _LOCK:
        if result.status == "ready":
            if result.cache_hit:
                _METRICS["cache_hits"] += 1
            else:
                _METRICS["generated"] += 1
                _METRICS["characters_generated"] += result.text_chars
        else:
            _METRICS["failed"] += 1
        _METRICS["generation_latency_ms"] += result.latency_ms
    logger.info(
        "tts_generation status=%s cache_hit=%s chars=%d latency_ms=%d",
        result.status,
        result.cache_hit,
        result.text_chars,
        result.latency_ms,
    )


def _delete_stale_result(cache_key: str, future: Future) -> None:
    try:
        result = future.result()
        with _LOCK:
            job = _JOBS.get(cache_key)
            keep = bool(job and job.keep)
        if not keep and result.status == "ready" and not result.cache_hit:
            default_storage.delete(result.storage_path)
            logger.info("tts_stale_audio_discarded cache_key=%s", cache_key[:12])
        if not keep:
            with _LOCK:
                _JOBS.pop(cache_key, None)
    except Exception:
        logger.exception("failed to discard stale TTS audio")


def _trim_jobs(max_jobs: int) -> None:
    while len(_JOBS) > max_jobs:
        removable = next(
            (key for key, job in _JOBS.items() if job.future.done()),
            None,
        )
        if removable is None:
            break
        _JOBS.pop(removable, None)


def get_tts_status(cache_key: str) -> Dict[str, Any]:
    """Resolve a persisted cache key without waiting for generation."""
    if not _KEY_RE.fullmatch(str(cache_key or "")):
        return {"status": "unavailable"}
    with _LOCK:
        job = _JOBS.get(cache_key)
    if job is not None:
        return _describe_ticket(job.ticket)

    path = _storage_path(cache_key)
    try:
        if default_storage.exists(path):
            return {
                "enabled": True,
                "status": "ready",
                "cache_key": cache_key,
                "cache_hit": True,
                "generation_latency_ms": None,
            }
    except Exception:
        logger.exception("failed to inspect persisted TTS audio")
        return {"enabled": True, "status": "failed", "cache_key": cache_key}
    return {"enabled": True, "status": "pending", "cache_key": cache_key}


def get_tts_audio(cache_key: str) -> Optional[tuple[bytes, str, Dict[str, Any]]]:
    status = get_tts_status(cache_key)
    if status.get("status") != "ready":
        return None
    path = _storage_path(cache_key)
    with _LOCK:
        job = _JOBS.get(cache_key)
        if job is not None and job.future.done():
            path = job.future.result().storage_path
    try:
        with default_storage.open(path, "rb") as audio_file:
            return audio_file.read(), "audio/mpeg", status
    except Exception:
        logger.exception("failed to read persisted TTS audio")
        return None


def generate_instant_tts_signed_url(cache_key: str) -> Optional[str]:
    """Pre-compute the Azure SAS URL instantly via local cryptography.

    Since the storage path is deterministic (viva-tts/{cache_key}.mp3),
    generate_blob_sas computes the signed URL in 0.001 ms without network calls,
    allowing the frontend to receive the audio URL directly in the turn response.
    """
    config = _config()
    if not _enabled(config) or not _KEY_RE.fullmatch(str(cache_key or "")):
        return None
    try:
        import os
        from AI_Evaluator_Backend.azure_storage import generate_sas_url

        container = os.getenv("AZURE_CONTAINER", "media")
        path = _storage_path(cache_key)
        signed_url = generate_sas_url(container, path, expiry_hours=1)
        logger.info("[TTS Backend] Instant SAS URL pre-computed for key=%s...", cache_key[:12])
        return signed_url
    except Exception as e:
        logger.warning("[TTS Backend] Could not pre-generate instant SAS URL: %s", str(e))
        return None


def get_tts_signed_url(cache_key: str) -> Optional[Dict[str, Any]]:
    """Return a short-lived signed Azure URL for direct browser streaming.

    This avoids proxying the full MP3 through Django, letting the browser's
    ``<audio>`` element stream directly from Azure Blob Storage so playback
    can start after the first few buffered kilobytes instead of waiting for
    the entire file to transfer through the backend.
    """
    tts_status = get_tts_status(cache_key)
    signed_url = generate_instant_tts_signed_url(cache_key)
    if signed_url is None:
        return None
    return {
        "audio_url": signed_url,
        "cache_hit": tts_status.get("cache_hit") is True,
        "tts_status": tts_status.get("status", "pending"),
    }


def tts_metrics_snapshot() -> Dict[str, int]:
    with _LOCK:
        return dict(_METRICS)


def bind_question_tts_audit(question) -> None:
    """Persist the eventual background result without delaying the response."""
    try:
        audit = dict(question.extension.generation_audit or {})
    except Exception:
        return
    tts = dict(audit.get("tts") or {})
    cache_key = str(tts.get("cache_key") or "")
    candidate_hash = str(audit.get("candidate_hash") or "")
    if not cache_key or tts.get("candidate_hash") != candidate_hash:
        return

    with _LOCK:
        job = _JOBS.get(cache_key)
    if job is None:
        return

    question_id = str(question.id)

    def schedule_persistence(future: Future) -> None:
        try:
            _EXECUTOR.submit(
                _persist_tts_result,
                question_id,
                candidate_hash,
                cache_key,
                future,
            )
        except RuntimeError:
            logger.warning("could not schedule persisted TTS audit update")

    job.future.add_done_callback(schedule_persistence)


def _persist_tts_result(
    question_id: str,
    candidate_hash: str,
    cache_key: str,
    future: Future,
) -> None:
    """Update only the matching content-addressed audit record."""
    try:
        from django.utils import timezone
        from viva_evaluator.models import VivaQuestionExtension

        result = future.result()
        extension = VivaQuestionExtension.objects.filter(
            question_id=question_id
        ).first()
        if extension is None:
            return
        audit = dict(extension.generation_audit or {})
        tts = dict(audit.get("tts") or {})
        if (
            audit.get("candidate_hash") != candidate_hash
            or tts.get("cache_key") != cache_key
        ):
            return
        tts.update(
            status=result.status,
            cache_hit=result.cache_hit,
            generation_latency_ms=result.latency_ms,
            characters=result.text_chars,
            completed_at=timezone.now().isoformat(),
        )
        if result.error:
            tts["error"] = result.error[:160]
        audit["tts"] = tts
        VivaQuestionExtension.objects.filter(pk=extension.pk).update(
            generation_audit=audit
        )
    except Exception:
        logger.exception("failed to persist completed TTS audit")
