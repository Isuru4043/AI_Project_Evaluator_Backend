"""
Image extractor — pulls embedded figures out of a PDF.

STRATEGY:
    PyMuPDF gives us page.get_images() which lists every embedded image.
    We extract each one as PNG bytes plus metadata: page number, bounding
    box, and a tentative caption from text near the image (for context
    when the captioner runs).

OUTPUT FORMAT:
    [
        {
            "image_bytes":  b"<PNG data>",
            "page":         3,
            "image_idx":    0,                    # nth image on the page
            "nearby_text":  "Figure 2: ...",      # text right below the image
            "section_hint": "System Design",      # name of section this image
                                                  # falls in (matched against
                                                  # section_detector output)
        },
        ...
    ]
"""

import io
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Skip tiny decorative images (icons, separators, watermarks).
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

# Cap how many images we caption per submission to keep cost predictable.
# Most FYP reports have 5-15 figures; this is a safety net.
MAX_IMAGES_PER_SUBMISSION = 25

# How much surrounding text to grab as context for each image (chars).
NEARBY_TEXT_WINDOW = 250


# =============================================================================
# Public API
# =============================================================================

def extract_images(
    pdf_bytes: bytes,
    sections: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Pull every meaningful figure out of the PDF.

    Args:
        pdf_bytes: Raw PDF bytes.
        sections:  Section detector output, used to attribute each image
                   to a section by page range. Optional.

    Returns:
        List of image dicts (see module docstring). Empty list if no images
        or extraction fails.
    """
    try:
        return _extract(pdf_bytes, sections or [])
    except Exception as exc:
        logger.warning('Image extraction failed: %s', exc)
        return []


# =============================================================================
# Internals
# =============================================================================

def _extract(pdf_bytes: bytes, sections: List[Dict]) -> List[Dict]:
    import fitz

    images: List[Dict] = []

    with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text('text') or ''

            # get_images returns (xref, smask, w, h, bpc, colorspace, ...)
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                if len(images) >= MAX_IMAGES_PER_SUBMISSION:
                    logger.info(
                        'Image extractor: hit cap of %d images, stopping.',
                        MAX_IMAGES_PER_SUBMISSION,
                    )
                    return images

                xref = img_info[0]
                width = img_info[2]
                height = img_info[3]

                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    continue

                try:
                    pix = fitz.Pixmap(doc, xref)
                    # Convert CMYK or other color spaces to RGB for Gemini
                    if pix.n - pix.alpha >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    image_bytes = pix.tobytes('png')
                    pix = None  # free
                except Exception as exc:
                    logger.debug('Skipping image xref=%s on page %s: %s',
                                 xref, page_num, exc)
                    continue

                section_hint = _section_for_page(sections, page_num)
                nearby_text = _nearby_text(page_text, img_idx)

                images.append({
                    'image_bytes':  image_bytes,
                    'page':         page_num,
                    'image_idx':    img_idx,
                    'width':        width,
                    'height':       height,
                    'nearby_text':  nearby_text,
                    'section_hint': section_hint,
                })

    logger.info('Image extractor: collected %d images', len(images))
    return images


def _section_for_page(sections: List[Dict], page: int) -> str:
    """Return the title of whichever section the page falls inside."""
    for section in sections:
        if section.get('page_start', 0) <= page <= section.get('page_end', 0):
            return section.get('title', 'unknown')
    return 'unknown'


def _nearby_text(page_text: str, img_idx: int) -> str:
    """
    Try to find caption-like text on the page. We look for "Figure N",
    "Fig. N", "Diagram N" patterns close to the start of the page text
    or just take the first NEARBY_TEXT_WINDOW chars as fallback.
    """
    import re

    if not page_text:
        return ''

    pattern = re.compile(
        r'((?:figure|fig\.?|diagram)\s+\d+[\.\:]?\s*[^\n]{0,200})',
        re.IGNORECASE,
    )
    matches = pattern.findall(page_text)
    if matches and img_idx < len(matches):
        return matches[img_idx].strip()[:NEARBY_TEXT_WINDOW]

    return page_text.strip()[:NEARBY_TEXT_WINDOW]
