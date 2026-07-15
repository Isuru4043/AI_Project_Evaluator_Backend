
"""
Background tasks executed by the Django-Q2 worker cluster.

These functions are never called directly — they are enqueued via
``async_task()`` in the upload views and picked up by the ``qcluster``
workers running alongside the Django process.

Each task receives only a lightweight ``segment_id`` (UUID), fetches
the media bytes from the database/Azure Blob, POSTs them to the
appropriate Modal endpoint, and writes the result back.
"""

import logging

import requests
from django.conf import settings

from core.models import DemoCapturedSegment

logger = logging.getLogger(__name__)


def transcribe_audio_task(segment_id: str) -> None:
    """Send an audio chunk to Canary-Qwen on Modal and save the transcript."""
    try:
        segment = DemoCapturedSegment.objects.get(id=segment_id)
    except DemoCapturedSegment.DoesNotExist:
        logger.error('Segment %s not found — skipping transcription.', segment_id)
        return

    try:
        # Read the audio bytes from the Azure Blob-backed FileField
        segment.file.open('rb')
        audio_bytes = segment.file.read()
        segment.file.close()

        filename = segment.file.name.split('/')[-1] if segment.file.name else 'audio.webm'

        resp = requests.post(
            settings.MODAL_CANARY_URL,
            files={'audio': (filename, audio_bytes)},
            timeout=120,
        )
        resp.raise_for_status()

        data = resp.json()
        segment.processed_text = data.get('transcript', '')
        segment.is_processed = True
        segment.save(update_fields=['processed_text', 'is_processed'])

        logger.info(
            'Transcribed audio segment %s (%.1fs–%.1fs) — %d chars.',
            segment_id, segment.start_time, segment.end_time,
            len(segment.processed_text),
        )

    except Exception as exc:
        segment.error_message = str(exc)[:500]
        segment.save(update_fields=['error_message'])
        logger.exception('Failed to transcribe segment %s', segment_id)


def analyze_slide_task(segment_id: str) -> None:
    """Send a slide screenshot to Qwen2.5-VL on Modal and save the analysis."""
    try:
        segment = DemoCapturedSegment.objects.get(id=segment_id)
    except DemoCapturedSegment.DoesNotExist:
        logger.error('Segment %s not found — skipping slide analysis.', segment_id)
        return

    try:
        # Read the image bytes from the Azure Blob-backed FileField
        segment.file.open('rb')
        image_bytes = segment.file.read()
        segment.file.close()

        filename = segment.file.name.split('/')[-1] if segment.file.name else 'slide.jpg'

        resp = requests.post(
            settings.MODAL_QWEN_VL_URL,
            files={'image': (filename, image_bytes)},
            timeout=120,
        )
        resp.raise_for_status()

        data = resp.json()
        segment.processed_text = data.get('result', '')
        segment.is_processed = True
        segment.save(update_fields=['processed_text', 'is_processed'])

        logger.info(
            'Analysed slide segment %s (offset %.1fs) — %d chars.',
            segment_id, segment.start_time,
            len(segment.processed_text),
        )

    except Exception as exc:
        segment.error_message = str(exc)[:500]
        segment.save(update_fields=['error_message'])
        logger.exception('Failed to analyse slide segment %s', segment_id)
