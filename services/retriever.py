"""
Semantic Retriever Service
Performs end-to-end semantic search by converting user queries to embeddings,
querying the FAISS vector store, applying relevance thresholds, and formatting
citations and context blocks for grounded LLM generation.
"""

import os
from typing import List, Dict, Any, Optional
from services.embeddings import EmbeddingService
from services.vector_store import VectorStore


class SemanticRetriever:
    """
    High-level retriever that bridges user queries, dense vector embedding,
    and FAISS similarity search with threshold filtering and citation formatting.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        default_top_k: int = 5,
        default_threshold: float = 0.15
    ):
        """
        Args:
            vector_store: Instance of VectorStore (or newly instantiated if None).
            embedding_service: Instance of EmbeddingService (or singleton if None).
            default_top_k: Default number of top chunks to retrieve.
            default_threshold: Minimum cosine similarity score for a chunk to be considered relevant.
        """
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedding_service or EmbeddingService()
        self.default_top_k = int(os.getenv("DEFAULT_TOP_K", default_top_k))
        self.default_threshold = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", default_threshold))

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes semantic retrieval for a natural language query.

        Args:
            query: User's question or search query.
            top_k: Number of chunks to retrieve (defaults to self.default_top_k).
            threshold: Minimum relevance similarity score (defaults to self.default_threshold).

        Returns:
            Dictionary containing query metadata, ranked chunks with similarity scores,
            relevance flags, and formatted citation sources.
        """
        k = top_k if top_k is not None else self.default_top_k
        min_thresh = threshold if threshold is not None else self.default_threshold

        if not query or not query.strip():
            return {
                "query": "",
                "total_retrieved": 0,
                "has_relevant_context": False,
                "threshold": min_thresh,
                "results": [],
                "context_text": "",
                "sources": []
            }

        # 1. Embed query into normalized 384-d vector
        query_vec = self.embedder.embed_query(query.strip(), normalize=True)

        # 2. Search FAISS index
        raw_results = self.vector_store.search(query_vec, top_k=k)

        # 3. Format and evaluate relevance
        structured_results: List[Dict[str, Any]] = []
        relevant_chunks: List[Dict[str, Any]] = []
        unique_sources: List[str] = []

        for rank, (chunk, score) in enumerate(raw_results, 1):
            is_rel = score >= min_thresh
            citation_str = f"{chunk.get('document', 'Unknown')} - Page {chunk.get('page', 1)}"

            item = {
                "rank": rank,
                "chunk_id": chunk.get("chunk_id", f"chunk_{rank}"),
                "document": chunk.get("document", "Unknown"),
                "page": chunk.get("page", 1),
                "score": round(score, 4),
                "is_relevant": is_rel,
                "text": chunk.get("text", ""),
                "citation": citation_str,
                "word_count": chunk.get("word_count", 0)
            }
            structured_results.append(item)

            if is_rel:
                relevant_chunks.append(item)
                if citation_str not in unique_sources:
                    unique_sources.append(citation_str)

        has_relevant = len(relevant_chunks) > 0

        # 4. Construct context string for LLM injection
        context_text = self.format_context_for_llm(
            relevant_chunks if has_relevant else structured_results
        )

        return {
            "query": query.strip(),
            "retrieval_mode": "semantic_faiss",
            "top_k": k,
            "threshold": min_thresh,
            "total_retrieved": len(structured_results),
            "relevant_count": len(relevant_chunks),
            "has_relevant_context": has_relevant,
            "results": structured_results,
            "context_text": context_text,
            "sources": unique_sources
        }

    def format_context_for_llm(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks into a clean, structured context block with clear
        document and page boundaries for LLM synthesis.

        Args:
            chunks: List of chunk result dictionaries.

        Returns:
            Formatted context string.
        """
        if not chunks:
            return ""

        context_blocks = []
        for ch in chunks:
            doc = ch.get("document", "Unknown")
            page = ch.get("page", 1)
            text = ch.get("text", "").strip()
            score = ch.get("score", 0.0)

            block = f"--- [Document: {doc} | Page: {page} | Similarity: {score:.4f}] ---\n{text}"
            context_blocks.append(block)

        return "\n\n".join(context_blocks)
