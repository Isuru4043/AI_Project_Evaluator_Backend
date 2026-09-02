"""ElevenLabs Scribe speech-to-text for student answers.

The browser records short utterances and posts them to an authenticated
route; the API key never leaves the server.  Transcription is synchronous
because a turn cannot proceed until the answer text exists, but every call is
bounded by a timeout so a provider outage degrades to typing rather than
hanging the viva.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from django.conf import settings


logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_METRICS = {
    "requests": 0,
    "transcribed": 0,
    "failed": 0,
    "empty": 0,
    "audio_bytes": 0,
    "characters_returned": 0,
    "latency_ms": 0,
}

# ElevenLabs accepts common browser recording containers.  The map exists so an
# unlabelled blob still reaches the API with a filename it can dispatch on.
_EXTENSION_BY_MIME = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "mp4",
    "audio/mpeg": "mp3",
    "audio/mpga": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/flac": "flac",
    "audio/aac": "aac",
    "video/webm": "webm",
    "video/mp4": "mp4",
}


@dataclass(frozen=True)
class STTResult:
    status: str
    text: str
    language_code: str
    language_probability: Optional[float]
    latency_ms: int
    audio_bytes: int
    error: str = ""


def _config() -> Dict[str, Any]:
    return {
        "enabled": bool(getattr(settings, "ELEVENLABS_STT_ENABLED", False)),
        "api_key": str(getattr(settings, "ELEVENLABS_API_KEY", "") or ""),
        "model_id": str(
            getattr(settings, "ELEVENLABS_STT_MODEL_ID", "scribe_v1")
            or "scribe_v1"
        ),
        "language_code": str(
            getattr(settings, "ELEVENLABS_STT_LANGUAGE_CODE", "") or ""
        ),
        "timeout": max(
            1.0, float(getattr(settings, "ELEVENLABS_STT_TIMEOUT_SECONDS", 25))
        ),
        "max_bytes": max(
            1024, int(getattr(settings, "ELEVENLABS_STT_MAX_AUDIO_BYTES", 12_000_000))
        ),
        "min_bytes": max(
            0, int(getattr(settings, "ELEVENLABS_STT_MIN_AUDIO_BYTES", 1_200))
        ),
    }


def is_enabled() -> bool:
    config = _config()
    return bool(config["enabled"] and config["api_key"])


def max_audio_bytes() -> int:
    return int(_config()["max_bytes"])


def _filename_for(content_type: str, fallback_name: str) -> str:
    base_type = (content_type or "").split(";")[0].strip().lower()
    extension = _EXTENSION_BY_MIME.get(base_type)
    if extension:
        return f"answer.{extension}"
    if "." in fallback_name:
        return fallback_name
    return "answer.webm"


def transcribe_answer_audio(
    audio_bytes: bytes,
    content_type: str = "",
    filename: str = "",
) -> STTResult:
    """Transcribe one recorded utterance with ElevenLabs Scribe."""
    config = _config()
    started = time.perf_counter()
    size = len(audio_bytes or b"")

    if not (config["enabled"] and config["api_key"]):
        logger.info("[STT Backend] ElevenLabs STT disabled in config.")
        return STTResult("disabled", "", "", None, 0, size, "stt_disabled")
    if size < config["min_bytes"]:
        # A blob this small is silence or a clipped recorder start, not speech.
        logger.info("[STT Backend] Ignoring %d-byte clip below the minimum.", size)
        return STTResult("empty", "", "", None, 0, size)
    if size > config["max_bytes"]:
        return STTResult(
            "failed", "", "", None, 0, size, "audio_too_large",
        )

    upload_name = _filename_for(content_type, filename)
    payload: Dict[str, Any] = {"model_id": config["model_id"]}
    if config["language_code"]:
        payload["language_code"] = config["language_code"]
    # Audio-event tags such as "(laughter)" would land in the graded answer.
    payload["tag_audio_events"] = "false"
    payload["diarize"] = "false"

    with _LOCK:
        _METRICS["requests"] += 1
        _METRICS["audio_bytes"] += size

    try:
        logger.info(
            "[STT Backend] Calling ElevenLabs Scribe (model=%s, %d bytes, %s)",
            config["model_id"], size, upload_name,
        )
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": config["api_key"], "Accept": "application/json"},
            data=payload,
            files={
                "file": (
                    upload_name,
                    audio_bytes,
                    (content_type or "application/octet-stream").split(";")[0],
                ),
            },
            timeout=config["timeout"],
        )
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        latency = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "[STT Backend] ElevenLabs STT FAILED (%s: %s) after %d ms",
            type(exc).__name__, str(exc)[:200], latency,
        )
        result = STTResult(
            "failed", "", "", None, latency, size, type(exc).__name__,
        )
        _record_result(result)
        return result

    latency = int((time.perf_counter() - started) * 1000)
    text = str(body.get("text") or "").strip()
    probability = body.get("language_probability")
    result = STTResult(
        "ready" if text else "empty",
        text,
        str(body.get("language_code") or ""),
        float(probability) if isinstance(probability, (int, float)) else None,
        latency,
        size,
    )
    logger.info(
        "[STT Backend] Scribe returned %d chars in %d ms (lang=%s)",
        len(text), latency, result.language_code or "?",
    )
    _record_result(result)
    return result


def _record_result(result: STTResult) -> None:
    with _LOCK:
        if result.status == "ready":
            _METRICS["transcribed"] += 1
            _METRICS["characters_returned"] += len(result.text)
        elif result.status == "empty":
            _METRICS["empty"] += 1
        else:
            _METRICS["failed"] += 1
        _METRICS["latency_ms"] += result.latency_ms
    logger.info(
        "stt_transcription status=%s chars=%d audio_bytes=%d latency_ms=%d",
        result.status, len(result.text), result.audio_bytes, result.latency_ms,
    )


def stt_metrics_snapshot() -> Dict[str, int]:
    with _LOCK:
        return dict(_METRICS)
