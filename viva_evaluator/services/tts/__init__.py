"""Non-blocking text-to-speech services for viva questions."""

from .elevenlabs import (
    bind_question_tts_audit,
    discard_speculative_tts,
    finalize_question_tts,
    generate_instant_tts_signed_url,
    get_tts_audio,
    get_tts_signed_url,
    get_tts_status,
    start_speculative_tts,
    tts_metrics_snapshot,
)

__all__ = [
    "bind_question_tts_audit",
    "discard_speculative_tts",
    "finalize_question_tts",
    "generate_instant_tts_signed_url",
    "get_tts_audio",
    "get_tts_signed_url",
    "get_tts_status",
    "start_speculative_tts",
    "tts_metrics_snapshot",
]

