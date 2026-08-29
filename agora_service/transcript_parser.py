"""
Transcript Parser — converts WebVTT (.vtt) caption files into
RAG-compatible chunks that plug directly into the existing
viva_evaluator retrieval pipeline.

Chunk format matches ``viva_evaluator.services.rag.chunking``:
    {
        "text":       "What the student said...",
        "source":     "transcript",
        "section":    "live_presentation",
        "chunk_idx":  0,
        "char_start": 0,
        "char_end":   500,
    }
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# Chunking parameters — tuned for spoken transcripts (shorter than report
# text because speech is less dense).
TRANSCRIPT_CHUNK_SIZE = 600       # characters
TRANSCRIPT_OVERLAP = 100          # characters
MIN_CHUNK_SIZE = 50               # discard tiny fragments


def parse_vtt_to_text(vtt_content: str) -> str:
    """
    Extract plain text from a WebVTT file, stripping timestamps and
    cue headers.

    Args:
        vtt_content: Raw .vtt file contents as a string.

    Returns:
        Concatenated plain text of all captions.
    """
    lines = vtt_content.strip().splitlines()
    text_parts = []

    for line in lines:
        line = line.strip()
        # Skip the WEBVTT header
        if line.upper().startswith('WEBVTT'):
            continue
        # Skip NOTE lines
        if line.upper().startswith('NOTE'):
            continue
        # Skip blank lines
        if not line:
            continue
        # Skip cue identifiers (numeric lines)
        if line.isdigit():
            continue
        # Skip timestamp lines (e.g. "00:00:01.000 --> 00:00:04.000")
        if '-->' in line:
            continue
        # Strip HTML tags (e.g. <v Speaker>) commonly found in VTT
        clean = re.sub(r'<[^>]+>', '', line).strip()
        if clean:
            text_parts.append(clean)

    return ' '.join(text_parts)


def parse_vtt_to_speaker_turns(vtt_content: str) -> List[Dict]:
    """
    Extract WHO spoke WHEN from a WebVTT transcript.

    Agora's STT bot labels each cue with the publisher it heard, as a WebVTT
    voice span::

        00:00:04.120 --> 00:00:09.480
        <v 1274918203>The architecture uses a message queue because...</v>

    ``parse_vtt_to_text`` deliberately strips those tags — it feeds the RAG
    pipeline, which wants prose. This function keeps them, because the tag is
    the speaker's Agora UID, and ``attribution.services.ingest`` maps that UID
    back to a student. Same file, two readings: what was said, and who said it.

    Cues with no voice tag are returned with ``speaker=None`` rather than
    dropped, so a window shows as contested rather than silent.

    Args:
        vtt_content: Raw .vtt file contents as a string.

    Returns:
        [{'speaker': str|None, 't_start_ms': int, 't_end_ms': int, 'text': str}]
        ordered by start time. Offsets are relative to the caption clock, which
        shares its origin with the recording.
    """
    turns: List[Dict] = []
    pending_start = pending_end = None
    buffer: List[str] = []
    speaker = None

    def flush():
        nonlocal pending_start, pending_end, buffer, speaker
        if pending_start is not None and buffer:
            text = ' '.join(buffer).strip()
            if text:
                turns.append({
                    'speaker': speaker,
                    't_start_ms': pending_start,
                    't_end_ms': pending_end,
                    'text': text,
                })
        pending_start = pending_end = None
        buffer = []
        speaker = None

    for raw in vtt_content.strip().splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.upper().startswith(('WEBVTT', 'NOTE')) or line.isdigit():
            continue

        if '-->' in line:
            flush()
            start, end = _parse_cue_timing(line)
            if start is None:
                continue
            pending_start, pending_end = start, end
            continue

        if pending_start is None:
            continue

        voice = _VOICE_TAG_RE.match(line)
        if voice:
            speaker = voice.group(1).strip() or None
            line = line[voice.end():]

        clean = re.sub(r'<[^>]+>', '', line).strip()
        if clean:
            buffer.append(clean)

    flush()
    turns.sort(key=lambda t: t['t_start_ms'])
    return turns


# <v 1274918203> / <v.loud Speaker Name> — the UID or name is the payload.
_VOICE_TAG_RE = re.compile(r'<v[^\s>]*\s+([^>]+)>')

_TIMESTAMP_RE = re.compile(
    r'(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})'
)


def _parse_cue_timing(line: str):
    """'00:00:04.120 --> 00:00:09.480' -> (4120, 9480). (None, None) if unparseable."""
    parts = line.split('-->')
    if len(parts) != 2:
        return None, None
    start = _timestamp_to_ms(parts[0])
    end = _timestamp_to_ms(parts[1])
    if start is None or end is None or end <= start:
        return None, None
    return start, end


def _timestamp_to_ms(fragment: str):
    match = _TIMESTAMP_RE.search(fragment)
    if not match:
        return None
    hours, minutes, seconds, millis = match.groups()
    return (
        int(hours or 0) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1000
        + int(millis.ljust(3, '0'))
    )


def parse_vtt_to_chunks(
    vtt_content: str,
    session_id: str = '',
    chunk_size: int = TRANSCRIPT_CHUNK_SIZE,
    overlap: int = TRANSCRIPT_OVERLAP,
) -> List[Dict]:
    """
    Parse WebVTT transcript into chunks compatible with the existing
    RAG retrieval pipeline.

    Args:
        vtt_content:  Raw .vtt file content.
        session_id:   Session UUID for logging/tracking.
        chunk_size:   Target characters per chunk.
        overlap:      Overlap between consecutive chunks.

    Returns:
        List of chunk dicts in the standard format used by
        ``viva_evaluator.services.rag.chunking``.
    """
    full_text = parse_vtt_to_text(vtt_content)

    if not full_text or len(full_text) < MIN_CHUNK_SIZE:
        logger.info(
            'transcript_parser: No usable text from transcript (session=%s, len=%d)',
            session_id, len(full_text),
        )
        if full_text:
            return [{
                'text': full_text,
                'source': 'transcript',
                'section': 'live_presentation',
                'chunk_idx': 0,
                'char_start': 0,
                'char_end': len(full_text),
            }]
        return []

    chunks: List[Dict] = []
    step = max(chunk_size - overlap, 1)
    start = 0
    chunk_idx = 0

    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        # Try to break at a sentence boundary
        end = _adjust_to_boundary(full_text, start, end)
        fragment = full_text[start:end].strip()

        if len(fragment) >= MIN_CHUNK_SIZE:
            chunks.append({
                'text': fragment,
                'source': 'transcript',
                'section': 'live_presentation',
                'chunk_idx': chunk_idx,
                'char_start': start,
                'char_end': end,
            })
            chunk_idx += 1

        if end >= len(full_text):
            break
        start = max(start + step, end - overlap)

    logger.info(
        'transcript_parser: Parsed %d chunks from transcript (session=%s, total_chars=%d)',
        len(chunks), session_id, len(full_text),
    )
    return chunks


def _adjust_to_boundary(text: str, start: int, target_end: int) -> int:
    """Shift end to land on a sentence boundary when possible."""
    if target_end >= len(text):
        return len(text)

    search_start = max(start + 1, target_end - 80)
    best = -1
    for ending in ('. ', '? ', '! ', '\n'):
        pos = text.rfind(ending, search_start, target_end)
        if pos > best:
            best = pos + len(ending)
    return best if best != -1 else target_end
