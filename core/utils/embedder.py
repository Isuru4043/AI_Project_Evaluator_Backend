import numpy as np
from sentence_transformers import SentenceTransformer

# Model is downloaded once and cached locally by HuggingFace
# all-mpnet-base-v2 gives the best semantic accuracy for technical text
MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singleton — model loads once when the server starts
_model = None


def get_model() -> SentenceTransformer:
    """Returns the shared SBERT model instance, loading it if necessary."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Embeds a single string into a vector.

    Args:
        text: Any string — a chunk, a question, or a student answer.

    Returns:
        A numpy float32 array of shape (768,)
    """
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embeds a list of strings in a single batched call (more efficient).

    Args:
        texts: List of strings to embed.

    Returns:
        A numpy float32 array of shape (len(texts), 768)
    """
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two embedding vectors.
    Returns a value between 0.0 and 1.0.
    Used to score student answers against ideal context.
    """
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm == 0:
        return 0.0
    return float(dot / norm)