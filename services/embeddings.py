"""
Embedding Service
Generates dense vector embeddings using Sentence Transformers (all-MiniLM-L6-v2).
Provides batch encoding, L2 normalization for fast cosine similarity,
and query vectorization.
"""

import os
from typing import List, Dict, Any, Union, Optional
import numpy as np


class EmbeddingService:
    """
    Singleton/cached service for converting text into dense vector embeddings
    using the Sentence Transformers library.
    """

    _instance: Optional["EmbeddingService"] = None
    _model = None

    def __new__(cls, model_name: Optional[str] = None):
        """Ensures a single instance across the application to prevent reloading weights."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            chosen_model = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            cls._instance.model_name = chosen_model
            cls._instance._model = None
            cls._instance._dimension = None
        return cls._instance

    def _load_model(self):
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            try:
                # pyrefly: ignore [missing-import]
                from sentence_transformers import SentenceTransformer
                # Load model onto CPU (or GPU if available)
                self._model = SentenceTransformer(self.model_name)
                # Determine embedding dimension
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                else:
                    self._dimension = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}'. "
                    f"Ensure 'sentence-transformers' and 'torch' are installed: {e}"
                )

    @property
    def dimension(self) -> int:
        """Returns the embedding dimension (384 for all-MiniLM-L6-v2)."""
        self._load_model()
        return self._dimension

    def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Embeds a list of text strings into a 2D numpy array of shape (N, dimension).

        Args:
            texts: List of text strings to embed.
            normalize: If True, vectors are L2-normalized so that inner product equals cosine similarity.
            batch_size: Batch size for encoding.

        Returns:
            np.ndarray of shape (len(texts), dimension) and dtype float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._load_model()

        # Generate embeddings
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embeddings.astype(np.float32)

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Convenience method to embed a list of chunk dictionaries.
        Extracts the 'text' key from each chunk dict.

        Args:
            chunks: List of chunk dictionaries from TextChunker.
            normalize: If True, L2-normalizes output vectors.
            batch_size: Batch size for encoding.

        Returns:
            np.ndarray of shape (len(chunks), dimension).
        """
        texts = [chunk.get("text", "") for chunk in chunks]
        return self.embed_texts(texts, normalize=normalize, batch_size=batch_size)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Embeds a single query string into a 1D vector of shape (dimension,).

        Args:
            query: Natural language query string.
            normalize: If True, L2-normalizes the output vector.

        Returns:
            1D np.ndarray of shape (dimension,) and dtype float32.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        self._load_model()

        embedding = self._model.encode(
            query.strip(),
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embedding.astype(np.float32).flatten()
