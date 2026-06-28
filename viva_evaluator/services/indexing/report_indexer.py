"""
Report indexer — single entry point that turns a submission's raw bytes
into a list of FAISS-ready chunks.

PIPELINE:
    PDF bytes
        ↓
    section_detector  → list of sections with text + page ranges
        ↓
    For each section:
        chunking.chunk_text(...) → text chunks tagged with the section title
        ↓
    image_extractor   → list of figures with bytes + page + section hint
        ↓
    image_captioner   → list of caption text chunks tagged source='figure'
        ↓
    Combine: [text chunks ...] + [figure chunks ...]
        ↓
    Return chunk list (vector_store handles embedding + persistence)

This is the function the upload views call.
"""

import logging
from typing import List, Dict

from viva_evaluator.services.rag.chunking import chunk_text
from viva_evaluator.services.indexing.section_detector import detect_sections
from viva_evaluator.services.indexing.image_extractor import extract_images
from viva_evaluator.services.indexing.image_captioner import caption_images

logger = logging.getLogger(__name__)


def index_report(
    pdf_bytes: bytes,
    enable_image_captions: bool = True,
) -> Dict:
    """
    Build the full chunk list for a report.

    Args:
        pdf_bytes:             Raw PDF bytes.
        enable_image_captions: If False, skip the image-captioning pass
                               (faster, no LLM calls during indexing).
                               Useful for tests.

    Returns:
        {
            "chunks":          [chunk dicts ready for vector_store.save()],
            "sections":        [section dicts],
            "images_found":    int,
            "images_captioned": int,
        }
    """
    # Step 1 — section detection
    sections = detect_sections(pdf_bytes)
    logger.info('index_report: detected %d sections', len(sections))

    # Step 2 — chunk each section
    text_chunks: List[Dict] = []
    for section in sections:
        section_chunks = chunk_text(
            text=section['text'],
            source='report',
            section=section['title'],
        )
        # Annotate with page range so retrieval can show provenance
        for ch in section_chunks:
            ch['page_start'] = section.get('page_start')
            ch['page_end'] = section.get('page_end')
            ch['section_level'] = section.get('level', 1)
        text_chunks.extend(section_chunks)

    # Renumber chunk_idx globally so each chunk has a unique id
    for i, ch in enumerate(text_chunks):
        ch['chunk_idx'] = i

    # Step 3 — extract + caption images
    figure_chunks: List[Dict] = []
    images_found = 0
    images_captioned = 0

    if enable_image_captions:
        try:
            images = extract_images(pdf_bytes, sections=sections)
            images_found = len(images)

            captions = caption_images(images)
            images_captioned = len(captions)

            # Renumber within figures so they have stable ids alongside text
            for i, cap in enumerate(captions):
                cap['chunk_idx'] = len(text_chunks) + i
                # mirror the text chunk shape for retrieval
                cap.setdefault('char_start', 0)
                cap.setdefault('char_end', len(cap.get('text', '')))
            figure_chunks = captions
        except Exception as exc:
            logger.warning('index_report: image pipeline failed: %s', exc)

    all_chunks = text_chunks + figure_chunks

    logger.info(
        'index_report: %d text chunks + %d figure chunks (from %d images)',
        len(text_chunks), len(figure_chunks), images_found,
    )

    return {
        'chunks':           all_chunks,
        'sections':         sections,
        'images_found':     images_found,
        'images_captioned': images_captioned,
    }
