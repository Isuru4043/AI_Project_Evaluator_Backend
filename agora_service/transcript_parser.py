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
