"""
Section detector — extracts logical sections from a PDF using font-size
analysis and heading patterns.

STRATEGY:
    1. Parse the PDF page by page using PyMuPDF's structured "dict" mode,
       which gives us per-span font size + text + bbox.
    2. Compute the document's body-text font size (the most common size).
    3. Anything noticeably larger than body text + a regex match against
       common heading words ("Introduction", "Methodology", ...) becomes
       a section boundary.
    4. Group all body text under the most recent heading.

OUTPUT FORMAT:
    [
        {
            "title":        "Methodology",
            "level":        1,                      # 1 = top, 2 = sub-section
            "page_start":   3,
            "page_end":     6,
            "text":         "...full section text...",
        },
        ...
    ]

If detection fails (image-only PDF, unusual layout), we fall back to a single
'full_report' section so the rest of the pipeline still works.
"""


import logging
import re
from collections import Counter
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Heading detection rules.
# =============================================================================

# Common section names in academic / FYP reports. Case-insensitive.
HEADING_KEYWORDS = [
    r'abstract', r'executive summary',
    r'introduction', r'background', r'motivation',
    r'literature review', r'related work',
    r'problem statement', r'problem formulation',
    r'objectives', r'aims', r'goals',
    r'requirements', r'scope',
    r'methodology', r'approach', r'research design',
    r'system design', r'architecture', r'design',
    r'implementation', r'development',
    r'threat model', r'security analysis',
    r'evaluation', r'experiments', r'results', r'discussion',
    r'testing', r'validation', r'verification',
    r'conclusion', r'conclusions',
    r'future work', r'limitations',
    r'references', r'bibliography', r'appendix',
    r'acknowledgements', r'acknowledgments',
]
_HEADING_KEYWORD_RE = re.compile(
    r'^\s*(?:\d+(?:\.\d+)*\s+)?(?:' + '|'.join(HEADING_KEYWORDS) + r')\b',
    re.IGNORECASE,
)

# Numbered headings like "3.2 System Design" or "4.1.2 Encryption Layer".
# We only treat these as headings if they're also large font.
_NUMBERED_HEADING_RE = re.compile(r'^\s*\d+(?:\.\d+){0,3}\.?\s+[A-Z]')

# Body font size detection threshold — heading must be at least this much
# bigger than the document's body text size (in pt).
HEADING_SIZE_DELTA = 1.5

# Minimum span length to consider as a heading (filters out page numbers etc.)
MIN_HEADING_LENGTH = 4
MAX_HEADING_LENGTH = 120


# =============================================================================
# Public API
# =============================================================================

def detect_sections(pdf_bytes: bytes) -> List[Dict]:
    """
    Detect sections in a PDF document.

    Args:
        pdf_bytes: Raw PDF file bytes.

    Returns:
        List of section dicts (see module docstring). Falls back to a single
        section containing all text if heading detection fails.
    """
    try:
        spans = _extract_spans(pdf_bytes)
    except Exception as exc:
        logger.warning('Section detection: PDF parsing failed (%s). Falling back.', exc)
        return _fallback_single_section(pdf_bytes)

    if not spans:
        return _fallback_single_section(pdf_bytes)

    body_size = _estimate_body_font_size(spans)
    heading_threshold = body_size + HEADING_SIZE_DELTA

    sections = _group_into_sections(spans, body_size, heading_threshold)

    if not sections or sum(len(s['text']) for s in sections) < 200:
        # Detection produced nothing useful — fall back
        logger.info('Section detection: no useful sections, falling back.')
        return _fallback_single_section(pdf_bytes)

    logger.info(
        'Section detection: found %d sections (body=%spt, heading_threshold=%spt)',
        len(sections), round(body_size, 1), round(heading_threshold, 1),
    )
    return sections


# =============================================================================
# Internals
# =============================================================================

def _extract_spans(pdf_bytes: bytes) -> List[Dict]:
    """
    Extract every text span from the PDF with its font size and page number.
    A 'span' is PyMuPDF's smallest text unit with consistent formatting.
    """
    import fitz

    spans: List[Dict] = []
    with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
        for page_num, page in enumerate(doc, start=1):
            page_dict = page.get_text('dict')
            for block in page_dict.get('blocks', []):
                if block.get('type') != 0:  # 0 = text block
                    continue
                for line in block.get('lines', []):
                    # Concatenate adjacent spans on the same line if they share a size.
                    line_text_parts: List[str] = []
                    line_size: Optional[float] = None
                    for span in line.get('spans', []):
                        text = span.get('text', '').strip()
                        if not text:
                            continue
                        size = round(float(span.get('size', 0)), 1)
                        line_text_parts.append(text)
                        line_size = size if line_size is None else max(line_size, size)
                    if not line_text_parts:
                        continue
                    spans.append({
                        'text': ' '.join(line_text_parts),
                        'size': line_size or 0,
                        'page': page_num,
                    })
    return spans


def _estimate_body_font_size(spans: List[Dict]) -> float:
    """
    Body font size = the most common (modal) font size weighted by total
    character count at that size. This avoids being thrown off by long
    headings or short page numbers.
    """
    char_count_by_size: Counter = Counter()
    for span in spans:
        char_count_by_size[span['size']] += len(span['text'])

    if not char_count_by_size:
        return 11.0  # safe default for academic PDFs

    body_size, _ = char_count_by_size.most_common(1)[0]
    return float(body_size)


def _is_heading(span: Dict, body_size: float, heading_threshold: float) -> Tuple[bool, int]:
    """
    Decide whether a span is a section heading. Returns (is_heading, level).
    level: 1 = top-level section, 2 = subsection.
    """
    text = span['text'].strip()
    size = span['size']
    length = len(text)

    if length < MIN_HEADING_LENGTH or length > MAX_HEADING_LENGTH:
        return False, 0

    # Body-text-sized lines aren't headings — even if they match keywords,
    # those are body sentences mentioning the keyword.
    is_large = size >= heading_threshold
    has_keyword = bool(_HEADING_KEYWORD_RE.match(text))
    is_numbered = bool(_NUMBERED_HEADING_RE.match(text))

    # Top-level: keyword OR (large + numbered)
    if is_large and (has_keyword or is_numbered):
        # Subsection if numbered with depth > 1 (e.g., "3.2 ...")
        if is_numbered and text.split()[0].count('.') >= 1:
            return True, 2
        return True, 1

    # Catch unnumbered keyword headings even if font analysis failed
    if has_keyword and length < 60 and size > body_size:
        return True, 1

    return False, 0


def _group_into_sections(
    spans: List[Dict],
    body_size: float,
    heading_threshold: float,
) -> List[Dict]:
    """
    Walk through spans in document order, starting a new section whenever
    a heading is encountered, accumulating body text otherwise.
    """
    sections: List[Dict] = []
    current: Optional[Dict] = None

    for span in spans:
        is_heading, level = _is_heading(span, body_size, heading_threshold)

        if is_heading:
            # Close previous section
            if current is not None:
                current['text'] = current['text'].strip()
                if current['text']:
                    sections.append(current)

            # Start new section
            current = {
                'title':       _normalize_title(span['text']),
                'level':       level,
                'page_start':  span['page'],
                'page_end':    span['page'],
                'text':        '',
            }
        else:
            if current is None:
                # Pre-heading content — bucket as 'preface'
                current = {
                    'title':       'preface',
                    'level':       1,
                    'page_start':  span['page'],
                    'page_end':    span['page'],
                    'text':        '',
                }
            current['text'] += span['text'] + ' '
            current['page_end'] = span['page']

    # Close the last section
    if current is not None:
        current['text'] = current['text'].strip()
        if current['text']:
            sections.append(current)

    return sections


def _normalize_title(raw: str) -> str:
    """Clean up a heading: strip leading numbering, collapse whitespace."""
    text = raw.strip()
    # Strip leading "3.2 ", "3.2.1 ", "3) ", "Chapter 3: " patterns
    text = re.sub(r'^\s*\d+(?:\.\d+){0,3}\.?\s+', '', text)
    text = re.sub(r'^\s*chapter\s+\d+\s*:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*section\s+\d+\s*:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text or 'untitled'


# =============================================================================
# Fallback — used when detection fails or PDF is too unusual.
# =============================================================================

def _fallback_single_section(pdf_bytes: bytes) -> List[Dict]:
    """Return one big section with all extractable text."""
    from core.utils.document_parser import extract_text_from_bytes

    try:
        text = extract_text_from_bytes(pdf_bytes, 'fallback.pdf')
    except Exception:
        text = ''

    if not text.strip():
        return []

    return [{
        'title':      'full_report',
        'level':      1,
        'page_start': 1,
        'page_end':   1,
        'text':       text.strip(),
    }]
