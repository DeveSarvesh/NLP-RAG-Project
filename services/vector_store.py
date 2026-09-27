"""
FAISS Vector Store Service
Manages the FAISS index (IndexFlatIP) and maintains synchronized
metadata mapping for fast cosine similarity nearest-neighbor search.
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss


class VectorStore:
    """
    FAISS-backed vector store that maintains dense vector embeddings
    and associated chunk metadata for similarity retrieval.
    """

    def __init__(self, dimension: int = 384):
        """
        Args:
            dimension: Dimensionality of embeddings (384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self._index: Optional[faiss.Index] = None
        self._metadata: List[Dict[str, Any]] = []
        self._init_index()

    def _init_index(self):
        """Initializes an exact inner-product (cosine similarity on normalized vectors) FAISS index."""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._metadata = []

    @property
    def total_chunks(self) -> int:
        """Returns the number of indexed chunks."""
        return self._index.ntotal if self._index is not None else 0

    @property
    def total_documents(self) -> int:
        """Returns the number of unique documents indexed."""
        docs = {item.get("document") for item in self._metadata if "document" in item}
        return len(docs)

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Adds text chunks and their corresponding embeddings into the FAISS index.

        Args:
            chunks: List of chunk metadata dictionaries from TextChunker.
            embeddings: 2D numpy array of shape (N, dimension), dtype float32.
        """
        if len(chunks) == 0:
            return

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings shape {embeddings.shape} incompatible with index dimension {self.dimension}."
            )

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )

        # Ensure embeddings are contiguous float32
        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Add vectors to FAISS index
        self._index.add(embeddings_f32)

        # Append metadata in identical order
        self._metadata.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches the FAISS index for the Top-K most similar chunks to the query vector.

        Args:
            query_embedding: 1D or 2D numpy array of shape (dimension,) or (1, dimension).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of tuples: [(chunk_metadata_dict, similarity_score), ...]
            sorted by descending similarity score.
        """
        if self.total_chunks == 0:
            return []

        # Ensure query is 2D float32 of shape (1, dimension)
        if query_embedding.ndim == 1:
            query_vec = query_embedding.reshape(1, -1)
        else:
            query_vec = query_embedding

        query_f32 = np.ascontiguousarray(query_vec, dtype=np.float32)

        k = min(top_k, self.total_chunks)
        scores, indices = self._index.search(query_f32, k)

        results: List[Tuple[Dict[str, Any], float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self._metadata):
                chunk_meta = self._metadata[idx].copy()
                results.append((chunk_meta, float(score)))

        return results

    def save(self, index_dir: str = "data/indexes", index_name: str = "faiss_index"):
        """
        Serializes the FAISS index and metadata to disk.

        Args:
            index_dir: Target directory.
            index_name: Base name for index and metadata files.
        """
        os.makedirs(index_dir, exist_ok=True)
        index_file = os.path.join(index_dir, f"{index_name}.bin")
        meta_file = os.path.join(index_dir, f"{index_name}_meta.json")

        faiss.write_index(self._index, index_file)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2)

    def load(self, index_dir: str = "data/indexes", index_name: str = "faiss_index"):
        """
        Loads the FAISS index and metadata from disk.

        Args:
            index_dir: Directory containing saved files.
            index_name: Base name for index and metadata files.
        """
        index_file = os.path.join(index_dir, f"{index_name}.bin")
        meta_file = os.path.join(index_dir, f"{index_name}_meta.json")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            raise FileNotFoundError(f"FAISS index files not found in '{index_dir}' with name '{index_name}'.")

        self._index = faiss.read_index(index_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

    def clear(self):
        """Resets the FAISS index and metadata storage for a fresh session."""
        self._init_index()

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on the current vector store."""
        return {
            "total_chunks": self.total_chunks,
            "total_documents": self.total_documents,
            "dimension": self.dimension,
            "is_trained": self._index.is_trained if self._index is not None else False
        }
