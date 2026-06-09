"""
Image captioner — turns extracted figures into searchable text via the
multimodal LLM.

Each caption becomes a chunk in FAISS tagged source='figure'. The Questioner
never sees the image directly — it sees the caption text alongside the
report's text chunks. This means questions can reference diagrams without
the LLM needing image input at runtime.

A good caption answers three things:
    1. What KIND of diagram is this? (architecture, sequence, ER, flowchart, ...)
    2. What components / actors / nodes are visible?
    3. What relationships / data flows are shown?
"""

import logging
from typing import List, Dict

from viva_evaluator.services.llm_service import llm_call_with_image

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt — kept short, asks for structured but flexible output.
# =============================================================================

_CAPTION_PROMPT_TEMPLATE = """\
You are analysing a figure from a final-year Computer Science project report.

NEARBY TEXT (from the same page, may include the figure caption):
{nearby_text}

CONTEXT: This figure appears in the section titled "{section_hint}".

TASK:
Describe this figure in 2-4 sentences for use as searchable text in a viva
examination system. The student will be asked questions about this figure
without seeing it again.

Cover whichever of these are relevant:
- What kind of diagram is it? (architecture, sequence, ER, flowchart,
  screenshot, chart, dataflow, deployment, UI mockup, etc.)
- What are the main components, actors, or entities visible?
- What relationships, flows, or interactions does it show?
- Any specific labels, technologies, or terms shown in the diagram.

Write naturally as if describing it to someone over the phone. Do NOT
start with "This image shows" — start directly with the content.
"""


# =============================================================================
# Public API
# =============================================================================

def caption_images(images: List[Dict]) -> List[Dict]:
    """
    Generate captions for each extracted image.

    Args:
        images: Output of image_extractor.extract_images().

    Returns:
        List of dicts ready to be turned into FAISS chunks. Each dict has:
            text:         the caption (multi-sentence description)
            source:       'figure'
            section:      derived from section_hint
            page:         page number in the original PDF
            image_idx:    sequential index across the document
    """
    if not images:
        return []

    captioned: List[Dict] = []
    for idx, img in enumerate(images):
        caption = _caption_one(img)
        if not caption:
            continue

        captioned.append({
            'text':       caption,
            'source':     'figure',
            'section':    img.get('section_hint', 'unknown'),
            'page':       img.get('page', 0),
            'image_idx':  idx,
        })

    logger.info('Image captioner: produced %d captions from %d images',
                len(captioned), len(images))
    return captioned


# =============================================================================
# Internals
# =============================================================================

def _caption_one(img: Dict) -> str:
    """Generate a single caption. Returns '' on any failure."""
    nearby = (img.get('nearby_text') or '').strip()
    if not nearby:
        nearby = '(no nearby text available)'

    section_hint = img.get('section_hint') or 'unknown'

    prompt = _CAPTION_PROMPT_TEMPLATE.format(
        nearby_text=nearby[:500],
        section_hint=section_hint,
    )

    try:
        caption = llm_call_with_image(
            prompt=prompt,
            image_bytes=img['image_bytes'],
            image_mime='image/png',
            model='fast',         # use the cheaper model — vision is more expensive
            expect_json=False,
            max_retries=1,        # don't retry too aggressively at indexing time
            fallback='',
        )
    except Exception as exc:
        logger.warning(
            'Caption failed for page=%s img_idx=%s: %s',
            img.get('page'), img.get('image_idx'), exc,
        )
        return ''

    return (caption or '').strip()
