"""Question timeline — chapter markers for the examiner's recording player.

The AI examiner asks its questions through the student's browser speech
synthesiser, which no recording can capture: the audio never exists as a
stream. So instead of hearing the question, the examiner reads it — each
question is placed at its offset into the session recording, and the player
renders those as clickable chapters / a caption overlay.

The offset is pure arithmetic over two wall clocks:

    offset_ms = question.generated_at - recording.recording_started_at

``recording_started_at`` is Agora's sliceStartTime, i.e. the instant that maps
to video position 00:00:00 (see agora_service/cloud_recording.py). Without it
there is no shared origin and the timeline is omitted rather than guessed.
"""

import logging

from core.models import SessionRecording, VivaQuestion

logger = logging.getLogger(__name__)


def build_question_timeline(session) -> list:
    """[{question_id, question_text, offset_ms, source, order}] for a session.

    Empty when the recording's t0 is unknown, or before any question is asked.
    Questions generated before the recording started (i.e. during the demo
    phase, which the recording deliberately excludes) are dropped — a negative
    offset would seek nowhere.
    """
    recording = (
        SessionRecording.objects
        .filter(session=session)
        .exclude(recording_started_at__isnull=True)
        .order_by('-recorded_at')
        .first()
    )
    if recording is None:
        return []

    t0 = recording.recording_started_at
    questions = (
        VivaQuestion.objects
        .filter(session=session)
        .order_by('question_order', 'generated_at')
    )

    timeline = []
    for question in questions:
        if not question.generated_at:
            continue
        offset_ms = int((question.generated_at - t0).total_seconds() * 1000)
        if offset_ms < 0:
            continue
        timeline.append({
            'question_id': str(question.id),
            'question_text': question.question_text,
            'offset_ms': offset_ms,
            'source': question.question_source,
            'order': question.question_order,
        })
    return timeline
