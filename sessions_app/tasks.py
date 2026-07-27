"""
Background tasks executed by the Django-Q2 worker cluster.

These functions are never called directly — they are enqueued via
``async_task()`` in the upload views and picked up by the ``qcluster``
workers running alongside the Django process.

Each task receives a lightweight ``segment_id`` (UUID), fetches
the media bytes from database/Azure Blob, processes them using
Gemini Multimodal via ``viva_evaluator.services.llm_service``, and writes the result back.
"""

import logging
from core.models import DemoCapturedSegment
from viva_evaluator.services.llm_service import llm_call_with_media

logger = logging.getLogger(__name__)


def _infer_mime_type(filename: str, default: str) -> str:
    fn = (filename or '').lower()
    if fn.endswith('.webm'):
        return 'audio/webm'
    elif fn.endswith('.mp3'):
        return 'audio/mp3'
    elif fn.endswith('.wav'):
        return 'audio/wav'
    elif fn.endswith('.m4a') or fn.endswith('.mp4'):
        return 'audio/mp4'
    elif fn.endswith('.png'):
        return 'image/png'
    elif fn.endswith('.jpg') or fn.endswith('.jpeg'):
        return 'image/jpeg'
    return default


def transcribe_audio_task(segment_id: str) -> None:
    """Send an audio chunk to Gemini Multimodal and save the transcript."""
    try:
        segment = DemoCapturedSegment.objects.get(id=segment_id)
    except DemoCapturedSegment.DoesNotExist:
        logger.error('Segment %s not found — skipping transcription.', segment_id)
        return

    try:
        segment.file.open('rb')
        audio_bytes = segment.file.read()
        segment.file.close()

        filename = segment.file.name.split('/')[-1] if segment.file.name else 'audio.webm'
        mime_type = _infer_mime_type(filename, 'audio/webm')

        prompt = (
            "You are a precise speech-to-text transcriber for academic presentation oral exams.\n"
            "Transcribe the audio clip verbatim accurately. Do NOT summarize or add commentary.\n"
            "Output ONLY the plain transcribed text."
        )

        transcript = llm_call_with_media(
            prompt=prompt,
            media_bytes=audio_bytes,
            mime_type=mime_type,
            model='fast',
            fallback='',
        )

        segment.processed_text = str(transcript).strip()
        segment.is_processed = True
        segment.save(update_fields=['processed_text', 'is_processed'])

        logger.info(
            'Transcribed audio segment %s (%.1fs–%.1fs) via Gemini Multimodal — %d chars.',
            segment_id, segment.start_time, segment.end_time,
            len(segment.processed_text),
        )

    except Exception as exc:
        segment.error_message = str(exc)[:500]
        segment.save(update_fields=['error_message'])
        logger.exception('Failed to transcribe audio segment %s via Gemini', segment_id)


def analyze_slide_task(segment_id: str) -> None:
    """Send a slide screenshot to Gemini Multimodal Vision and save the analysis."""
    try:
        segment = DemoCapturedSegment.objects.get(id=segment_id)
    except DemoCapturedSegment.DoesNotExist:
        logger.error('Segment %s not found — skipping slide analysis.', segment_id)
        return

    try:
        segment.file.open('rb')
        image_bytes = segment.file.read()
        segment.file.close()

        filename = segment.file.name.split('/')[-1] if segment.file.name else 'slide.jpg'
        mime_type = _infer_mime_type(filename, 'image/jpeg')

        prompt = (
            "You are analyzing a slide screenshot from an academic computer science project presentation.\n"
            "Identify and extract:\n"
            "1. Main Slide Title\n"
            "2. Key Bullet Points & Conceptual Claims\n"
            "3. Architecture Diagram / System Components / Database Models shown\n"
            "4. Any visible Code Snippets or Formulas\n\n"
            "Be concise, structured, and factual. Focus on technical content."
        )

        analysis = llm_call_with_media(
            prompt=prompt,
            media_bytes=image_bytes,
            mime_type=mime_type,
            model='fast',
            fallback='',
        )

        segment.processed_text = str(analysis).strip()
        segment.is_processed = True
        segment.save(update_fields=['processed_text', 'is_processed'])

        logger.info(
            'Analyzed slide segment %s (offset %.1fs) via Gemini Multimodal — %d chars.',
            segment_id, segment.start_time,
            len(segment.processed_text),
        )

    except Exception as exc:
        segment.error_message = str(exc)[:500]
        segment.save(update_fields=['error_message'])
        logger.exception('Failed to analyze slide segment %s via Gemini', segment_id)
