"""Speech-to-text services for student viva answers."""

from .elevenlabs import (
    STTResult,
    is_enabled,
    max_audio_bytes,
    stt_metrics_snapshot,
    transcribe_answer_audio,
)

__all__ = [
    "STTResult",
    "is_enabled",
    "max_audio_bytes",
    "stt_metrics_snapshot",
    "transcribe_answer_audio",
]
