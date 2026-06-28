"""
Embeddings — turns chunks into vectors via SBERT.

DESIGN:
    - Single global model instance (lazy-loaded).
    - all-MiniLM-L6-v2: 384 dims, ~80MB, fast on CPU, good quality.
    - Used for both indexing (chunks) and querying (live questions).
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384


# =============================================================================
# Lazy singleton — model load is ~5 seconds, do it once.
# =============================================================================

_model: Optional[object] = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info('Loading SBERT model: %s', EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info('SBERT model loaded.')
    return _model


# =============================================================================
# Public API
# =============================================================================

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings into a (N, 384) float32 numpy array.

    Args:
        texts: List of input strings. Empty list returns shape (0, 384).

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM), dtype=float32.
        Vectors are L2-normalized so dot product == cosine similarity.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # critical: enables cosine via inner product
    )
    return embeddings.astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single string into a (384,) float32 vector.
    """
    return embed_texts([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two pre-normalized vectors.
    If vectors aren't normalized, compute it from scratch.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
