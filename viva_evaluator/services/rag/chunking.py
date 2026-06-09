"""
Chunking — splits submission text into retrievable pieces.

WEEK 1: Simple sliding-window chunker.
WEEK 2: Section-aware chunking (heading detection) replaces this.

A chunk is a dict with the following structure:
    {
        "text":       "...",          # raw text content
        "source":     "report",       # 'report' or 'code'
        "section":    "Methodology",  # logical section (Week 2: real sections)
        "chunk_idx":  0,              # position within source
        "char_start": 0,              # offset in original text
        "char_end":   800,
    }

Chunks are kept in this dict form (no Pydantic) for easy JSON persistence
to the SubmissionIndexStatus.faiss_chunks_json field.
"""

from typing import List, Dict


# =============================================================================
# Chunking parameters — tuned for academic reports + code summaries.
# =============================================================================

DEFAULT_CHUNK_SIZE = 800       # characters (≈ 200 tokens for English)
DEFAULT_OVERLAP = 150          # characters of overlap between chunks
MIN_CHUNK_SIZE = 100           # discard fragments smaller than this


def chunk_text(
    text: str,
    source: str = 'report',
    section: str = 'unknown',
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict]:
    """
    Split text into overlapping windows.

    Args:
        text:       Raw text to chunk.
        source:     'report' or 'code' — used by retrieval filtering.
        section:    Logical section name (Week 2: derived from headings).
        chunk_size: Target characters per chunk.
        overlap:    Characters shared between consecutive chunks.

    Returns:
        List of chunk dicts. Empty list if text is too short.
    """
    text = (text or '').strip()
    if len(text) < MIN_CHUNK_SIZE:
        # Whole text fits in one chunk
        if text:
            return [{
                'text':       text,
                'source':     source,
                'section':    section,
                'chunk_idx':  0,
                'char_start': 0,
                'char_end':   len(text),
            }]
        return []

    chunks: List[Dict] = []
    start = 0
    chunk_idx = 0
    step = chunk_size - overlap
    if step <= 0:
        # Defensive: prevent infinite loop on bad params
        step = max(chunk_size // 2, 1)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a sentence boundary near `end` if possible
        end = _adjust_to_boundary(text, start, end)
        chunk_content = text[start:end].strip()

        if len(chunk_content) >= MIN_CHUNK_SIZE:
            chunks.append({
                'text':       chunk_content,
                'source':     source,
                'section':    section,
                'chunk_idx':  chunk_idx,
                'char_start': start,
                'char_end':   end,
            })
            chunk_idx += 1

        if end >= len(text):
            break
        start = max(start + step, end - overlap)

    return chunks


def chunk_report_text(report_text: str) -> List[Dict]:
    """
    Convenience wrapper for treating raw text as a single section.

    Note: as of Week 2, the preferred path is via
    viva_evaluator.services.indexing.index_report(), which performs
    section-aware chunking. This function remains for use cases
    where only plain text is available (e.g., DOCX or fallback paths).
    """
    return chunk_text(report_text, source='report', section='full_report')


def chunk_code_summaries(summaries: List[Dict]) -> List[Dict]:
    """
    Wrapper for code processing — used in Week 3 once AST batching is in place.

    Args:
        summaries: List of {file_path, function_name, summary, source_code}

    Returns:
        Chunk dicts ready for embedding.
    """
    chunks: List[Dict] = []
    for idx, item in enumerate(summaries):
        text_blob = (
            f"File: {item.get('file_path', 'unknown')}\n"
            f"Function: {item.get('function_name', 'unknown')}\n"
            f"Summary: {item.get('summary', '')}\n"
            f"Code:\n{item.get('source_code', '')[:500]}"
        )
        chunks.append({
            'text':       text_blob,
            'source':     'code',
            'section':    item.get('file_path', 'unknown'),
            'chunk_idx':  idx,
            'char_start': 0,
            'char_end':   len(text_blob),
        })
    return chunks


# =============================================================================
# Helpers
# =============================================================================

_SENTENCE_ENDINGS = ('. ', '? ', '! ', '\n\n')


def _adjust_to_boundary(text: str, start: int, target_end: int) -> int:
    """
    Try to shift `target_end` slightly to land on a sentence boundary,
    so chunks don't end mid-sentence. Searches up to 100 chars back.
    """
    if target_end >= len(text):
        return len(text)

    search_start = max(start + 1, target_end - 100)
    best = -1
    for ending in _SENTENCE_ENDINGS:
        pos = text.rfind(ending, search_start, target_end)
        if pos > best:
            best = pos + len(ending)
    return best if best != -1 else target_end
