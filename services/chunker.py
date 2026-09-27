"""
Text Chunking & Metadata Preservation Service
Segments extracted page text into overlapping semantic chunks
while preserving document, page, and chunk index provenance metadata.
"""

import os
import re
from typing import List, Dict, Any, Optional


class TextChunker:
    """
    Splits document text into fixed-size overlapping chunks.
    Ensures metadata (document name, page number, chunk ID) is preserved.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        """
        Args:
            chunk_size: Target maximum number of words per chunk (e.g., 500-800 words).
            chunk_overlap: Number of overlapping words between consecutive chunks (e.g., 50-100 words).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _sanitize_doc_name(self, filename: str) -> str:
        """Sanitizes filename for use in chunk IDs (removes extension and special chars)."""
        base = os.path.splitext(os.path.basename(filename))[0]
        # Replace non-alphanumeric characters with underscores
        return re.sub(r'[^a-zA-Z0-9_-]', '_', base)

    def chunk_text(
        self,
        text: str,
        document: str = "document.pdf",
        page: int = 1,
        start_chunk_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Splits a raw or page text string into overlapping chunks using a sliding window.

        Args:
            text: Cleaned text string to split.
            document: Source document filename.
            page: Source page number (1-indexed).
            start_chunk_idx: Initial chunk index offset.

        Returns:
            List of chunk dictionaries with metadata.
        """
        if not text or not text.strip():
            return []

        words = text.split()
        total_words = len(words)

        if total_words == 0:
            return []

        # If text is smaller than chunk_size, return single chunk
        if total_words <= self.chunk_size:
            doc_slug = self._sanitize_doc_name(document)
            chunk_id = f"{doc_slug}_p{page}_c{start_chunk_idx}"
            return [{
                "chunk_id": chunk_id,
                "document": document,
                "page": page,
                "chunk_index": start_chunk_idx,
                "text": text.strip(),
                "word_count": total_words,
                "char_count": len(text.strip()),
                "start_word_idx": 0,
                "end_word_idx": total_words
            }]

        chunks: List[Dict[str, Any]] = []
        step = self.chunk_size - self.chunk_overlap
        current_idx = 0
        chunk_counter = start_chunk_idx
        doc_slug = self._sanitize_doc_name(document)

        while current_idx < total_words:
            end_idx = min(current_idx + self.chunk_size, total_words)
            chunk_words = words[current_idx:end_idx]
            chunk_text = " ".join(chunk_words).strip()

            chunk_id = f"{doc_slug}_p{page}_c{chunk_counter}"

            chunks.append({
                "chunk_id": chunk_id,
                "document": document,
                "page": page,
                "chunk_index": chunk_counter,
                "text": chunk_text,
                "word_count": len(chunk_words),
                "char_count": len(chunk_text),
                "start_word_idx": current_idx,
                "end_word_idx": end_idx
            })

            chunk_counter += 1
            current_idx += step

            # Prevent infinite loop if step is somehow 0
            if step <= 0:
                break

        return chunks

    def chunk_page(self, page_data: Dict[str, Any], start_chunk_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Chunks an individual page dictionary from PDFProcessor.

        Args:
            page_data: Page dictionary with keys 'document', 'page', 'text'.
            start_chunk_idx: Starting chunk counter.

        Returns:
            List of chunk dictionaries for this page.
        """
        return self.chunk_text(
            text=page_data.get("text", ""),
            document=page_data.get("document", "document.pdf"),
            page=page_data.get("page", 1),
            start_chunk_idx=start_chunk_idx
        )

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a list of extracted pages (e.g. from PDFProcessor) into a unified list of chunks.
        Preserves page-level source attribution for every chunk.

        Args:
            pages: List of page dictionaries from PDFProcessor.

        Returns:
            Unified list of all chunk dictionaries across the document.
        """
        all_chunks: List[Dict[str, Any]] = []
        global_chunk_idx = 0

        for page in pages:
            page_chunks = self.chunk_page(page, start_chunk_idx=global_chunk_idx)
            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return all_chunks
