"""
Indexing pipeline — turns a raw submission file into FAISS-ready chunks.

WEEK 2:
    - section_detector: heading-aware PDF parsing
    - image_extractor:  pulls figures out of the PDF
    - image_captioner:  Gemini Vision generates a caption for each figure
    - report_indexer:   orchestrates text + figures into a single chunk list

WEEK 3 (planned):
    - code_indexer: AST → batched LLM summaries → chunks
"""

from viva_evaluator.services.indexing.report_indexer import index_report

__all__ = ['index_report']
