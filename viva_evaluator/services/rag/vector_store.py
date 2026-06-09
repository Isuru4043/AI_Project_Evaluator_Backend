"""
Vector store — FAISS indices persisted in PostgreSQL.

STORAGE MODEL:
    Each ProjectSubmission has ONE FAISS index serialized as binary,
    plus a parallel JSON list of chunk metadata. Both stored on the
    SubmissionIndexStatus row.

LIFECYCLE:
    1. After upload + text extraction:
         chunks = chunk_text(...)
         vectors = embed_texts([c['text'] for c in chunks])
         save_index_for_submission(submission, chunks, vectors)
    2. At query time during a viva session:
         store = load_index_for_submission(submission)
         hits = store.search(query_vector, top_k=3)
"""

import io
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

from viva_evaluator.services.rag.embeddings import EMBEDDING_DIM, embed_texts

logger = logging.getLogger(__name__)


# =============================================================================
# In-process cache: submission_id -> SubmissionVectorStore
#
# The index never changes during a viva (the submission is frozen after the
# deadline). Re-deserializing the blob from the remote DB every turn cost
# ~13s. We cache the deserialized store and invalidate it only when the
# index is re-saved or appended to (i.e., between sessions, never during one).
# =============================================================================
_INDEX_CACHE: dict = {}


class SubmissionVectorStore:
    """
    In-memory FAISS index bound to a single submission's chunks.

    Construction is private — use load_index_for_submission() or
    save_index_for_submission() instead.
    """

    def __init__(self, faiss_index, chunks: List[Dict]):
        self._index = faiss_index
        self._chunks = chunks

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Find the top_k chunks most similar to query_vector.

        Args:
            query_vector:  (384,) float32, L2-normalized.
            top_k:         How many results to return after filtering.
            source_filter: If set ('report' or 'code'), restrict results.

        Returns:
            List of chunk dicts, each with an added 'score' key (cosine, 0..1).
        """
        if self._index.ntotal == 0:
            return []

        # Search a wider net if filtering — we may discard some results
        search_k = top_k * 4 if source_filter else top_k
        search_k = min(search_k, self._index.ntotal)

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self._index.search(query, search_k)

        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = dict(self._chunks[idx])  # shallow copy
            if source_filter and chunk.get('source') != source_filter:
                continue
            chunk['score'] = float(score)
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)


# =============================================================================
# Persistence — FAISS index <-> bytes
# =============================================================================

def _index_to_bytes(faiss_index) -> bytes:
    """Serialize a FAISS index to raw bytes for DB storage."""
    import faiss
    writer = faiss.PyCallbackIOWriter(_BytesAccumulator())
    return faiss.serialize_index(faiss_index).tobytes()


def _bytes_to_index(blob: bytes):
    """Deserialize a FAISS index from raw bytes."""
    import faiss
    arr = np.frombuffer(blob, dtype=np.uint8)
    return faiss.deserialize_index(arr)


class _BytesAccumulator:
    """Helper for FAISS PyCallbackIOWriter; not used in current impl but kept for future."""
    def __init__(self):
        self.buf = io.BytesIO()

    def __call__(self, data):
        self.buf.write(data)
        return len(data)


def _build_index(vectors: np.ndarray):
    """
    Build a FAISS IndexFlatIP (inner product == cosine since vectors are normalized).
    Flat index is fine for ≤ ~10K chunks per submission. No training needed.
    """
    import faiss
    if vectors.shape[0] == 0:
        # Empty index — search() returns nothing
        return faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors.astype(np.float32))
    return index


# =============================================================================
# Public persistence API — what the indexing pipeline and viva loop call.
# =============================================================================

def save_index_for_submission(submission, chunks: List[Dict]) -> Tuple[int, int]:
    """
    Embed all chunks, build a FAISS index, persist to the submission's
    SubmissionIndexStatus row.

    Args:
        submission: ProjectSubmission instance.
        chunks:     List of chunk dicts from chunking.py.

    Returns:
        (num_chunks_indexed, embedding_dim)
    """
    from viva_evaluator.models import SubmissionIndexStatus

    if not chunks:
        logger.warning('save_index_for_submission: no chunks for submission=%s', submission.id)
        # Still persist an empty index so retrieval doesn't fail later
        chunks = []

    texts = [c['text'] for c in chunks]
    vectors = embed_texts(texts) if texts else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    index = _build_index(vectors)

    # Serialize FAISS index
    import faiss
    serialized = faiss.serialize_index(index)
    blob = bytes(serialized)

    index_status, _ = SubmissionIndexStatus.objects.get_or_create(submission=submission)
    index_status.faiss_index_blob = blob
    index_status.faiss_chunks_json = chunks
    index_status.save(update_fields=['faiss_index_blob', 'faiss_chunks_json'])

    # Index changed — drop any stale cache entry
    _INDEX_CACHE.pop(str(submission.id), None)

    logger.info(
        'save_index_for_submission: submission=%s chunks=%d bytes=%d',
        submission.id, len(chunks), len(blob),
    )
    return len(chunks), EMBEDDING_DIM


def load_index_for_submission(submission) -> Optional[SubmissionVectorStore]:
    """
    Load the FAISS index + chunks for a submission. Returns None if not indexed.

    Uses the in-process cache (_INDEX_CACHE) to avoid re-deserializing the
    blob from the remote DB on every turn. Cache is invalidated on save/append.
    """
    sub_id = str(submission.id)
    cached = _INDEX_CACHE.get(sub_id)
    if cached is not None:
        return cached

    from viva_evaluator.models import SubmissionIndexStatus

    try:
        index_status = SubmissionIndexStatus.objects.get(submission=submission)
    except SubmissionIndexStatus.DoesNotExist:
        return None

    blob = index_status.faiss_index_blob
    chunks = index_status.faiss_chunks_json or []

    if not blob:
        logger.info('load_index_for_submission: no index for submission=%s', submission.id)
        return None

    index = _bytes_to_index(blob)
    store = SubmissionVectorStore(index, chunks)
    _INDEX_CACHE[sub_id] = store
    logger.info(
        'load_index_for_submission: loaded + cached submission=%s (chunks=%d)',
        sub_id, len(chunks),
    )
    return store


def invalidate_index_cache(submission) -> None:
    """Drop the cached store for a submission. Call after re-indexing."""
    _INDEX_CACHE.pop(str(submission.id), None)


# =============================================================================
# Merge code chunks into an existing index — used after code analysis runs.
# =============================================================================

def append_chunks_to_submission(submission, new_chunks: List[Dict]) -> int:
    """
    Append additional chunks (e.g., code chunks from Week 3 indexer) to an
    existing FAISS index without rebuilding the report index.

    Args:
        submission: ProjectSubmission.
        new_chunks: Chunk dicts to add.

    Returns:
        Total number of chunks in the index after append.
    """
    from viva_evaluator.models import SubmissionIndexStatus
    import faiss

    if not new_chunks:
        return 0

    try:
        index_status = SubmissionIndexStatus.objects.get(submission=submission)
    except SubmissionIndexStatus.DoesNotExist:
        # No prior index — fall back to creating a fresh one
        return save_index_for_submission(submission, new_chunks)[0]

    existing_chunks = index_status.faiss_chunks_json or []
    existing_blob = index_status.faiss_index_blob

    # Renumber the new chunks to live after the existing ones
    base_idx = len(existing_chunks)
    for i, chunk in enumerate(new_chunks):
        chunk['chunk_idx'] = base_idx + i

    # Embed the new chunks
    new_texts = [c['text'] for c in new_chunks]
    new_vectors = embed_texts(new_texts)

    # Load existing index, add vectors, re-serialize
    if existing_blob:
        existing_index = _bytes_to_index(existing_blob)
    else:
        existing_index = _build_index(np.zeros((0, EMBEDDING_DIM), dtype=np.float32))

    if new_vectors.shape[0] > 0:
        existing_index.add(new_vectors.astype(np.float32))

    # Persist combined state
    serialized = faiss.serialize_index(existing_index)
    blob = bytes(serialized)
    combined_chunks = list(existing_chunks) + new_chunks

    index_status.faiss_index_blob = blob
    index_status.faiss_chunks_json = combined_chunks
    index_status.save(update_fields=['faiss_index_blob', 'faiss_chunks_json'])

    # Index changed — drop any stale cache entry
    _INDEX_CACHE.pop(str(submission.id), None)

    logger.info(
        'append_chunks_to_submission: submission=%s added=%d total=%d',
        submission.id, len(new_chunks), len(combined_chunks),
    )
    return len(combined_chunks)
