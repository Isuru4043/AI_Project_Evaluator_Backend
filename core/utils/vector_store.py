import os
import json
import faiss
import numpy as np
from core.utils.embedder import embed_texts, embed_text


class VectorStore:
    """
    Wraps a FAISS index for a single submission.

    Each submission gets its own VectorStore — built once after upload,
    saved to disk, and loaded back at the start of each viva session.

    The index maps integer positions to chunk embeddings.
    The chunks themselves are stored alongside the index in a JSON file
    so you can retrieve the original text after a search.
    """

    def __init__(self):
        self.index = None
        self.chunks = []  # Parallel list to the FAISS index positions

    def build(self, chunks: list[str]) -> None:
        """
        Embeds all chunks and builds a FAISS index from them.

        Args:
            chunks: List of text chunks from the parsed report.
        """
        if not chunks:
            raise ValueError("Cannot build a vector store from empty chunks.")

        self.chunks = chunks
        embeddings = embed_texts(chunks)

        dimension = embeddings.shape[1]  # 384 for all-mpnet-base-v2

        # IndexFlatIP = Inner Product (cosine similarity when vectors are normalized)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def query(self, query_text: str, top_k: int = 3) -> list[str]:
        """
        Finds the top_k most semantically similar chunks to the query.

        Args:
            query_text: A rubric criterion description or question topic.
            top_k: Number of chunks to retrieve.

        Returns:
            List of the most relevant text chunks.
        """
        if self.index is None:
            raise RuntimeError("Vector store has not been built yet. Call build() first.")

        query_vec = embed_text(query_text).reshape(1, -1)
        faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results

    def save(self, directory: str, submission_id: str) -> str:
        """
        Saves the FAISS index and chunk list to disk.

        Args:
            directory: Folder to save into (e.g. MEDIA_ROOT/faiss_indexes/)
            submission_id: Used to name the files uniquely.

        Returns:
            The path to the saved index file.
        """
        os.makedirs(directory, exist_ok=True)

        index_path = os.path.join(directory, f"{submission_id}.index")
        chunks_path = os.path.join(directory, f"{submission_id}.chunks.json")

        faiss.write_index(self.index, index_path)

        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        return index_path

    @classmethod
    def load(cls, directory: str, submission_id: str) -> "VectorStore":
        """
        Loads a previously saved VectorStore from disk.

        Args:
            directory: Folder where the index was saved.
            submission_id: Used to locate the correct files.

        Returns:
            A fully loaded VectorStore ready to query.
        """
        index_path = os.path.join(directory, f"{submission_id}.index")
        chunks_path = os.path.join(directory, f"{submission_id}.chunks.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        store = cls()
        store.index = faiss.read_index(index_path)

        with open(chunks_path, 'r', encoding='utf-8') as f:
            store.chunks = json.load(f)

        return store