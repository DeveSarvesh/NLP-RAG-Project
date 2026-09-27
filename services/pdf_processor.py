"""
PDF Processing & Text Preprocessing Service
Extracts text and metadata page-by-page from PDFs using PyMuPDF (fitz)
and performs text cleaning and validation.
"""

import os
import re
from typing import List, Dict, Any, Union
import pymupdf  # PyMuPDF


class PDFProcessor:
    """
    Handles PDF validation, text extraction, page tracking,
    and text normalization for RAG ingestion.
    """

    def __init__(self, min_char_threshold: int = 30):
        """
        Args:
            min_char_threshold: Minimum total characters expected across the document.
                                Documents below this are flagged as scanned/image-only.
        """
        self.min_char_threshold = min_char_threshold

    def clean_text(self, raw_text: str) -> str:
        """
        Performs text cleaning and normalization:
        1. Fixes hyphenated line breaks (e.g. 'trans-\nformer' -> 'transformer').
        2. Normalizes non-printable and control characters.
        3. Collapses excessive spaces and tabs into single spaces.
        4. Normalizes multiple blank lines into double newlines.
        5. Strips leading and trailing whitespace.

        Args:
            raw_text: Raw text string extracted directly from PDF page.

        Returns:
            Cleaned and normalized text string.
        """
        if not raw_text or not isinstance(raw_text, str):
            return ""

        # 1. Remove non-printable control characters (keep standard \n, \t)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', raw_text)

        # 2. Fix hyphenation at line breaks (e.g., 'archi-\n    tecture' -> 'architecture')
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # 3. Replace irregular whitespace characters (non-breaking spaces, form feeds)
        text = text.replace('\xa0', ' ').replace('\r\n', '\n').replace('\r', '\n')

        # 4. Collapse multiple spaces / horizontal tabs into a single space
        text = re.sub(r'[ \t]+', ' ', text)

        # 5. Collapse 3 or more consecutive newlines into 2 (paragraph separation)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 6. Remove leading/trailing space per line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text.strip()

    def extract_pages(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts cleaned text and metadata from a PDF file on disk.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            List of page dictionaries:
            [
                {
                    "document": "sample.pdf",
                    "page": 1,
                    "text": "Cleaned page text...",
                    "raw_text": "Original text...",
                    "char_count": 520,
                    "word_count": 85
                },
                ...
            ]

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is invalid, empty, or a scanned/image-only PDF.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            raise ValueError(f"File '{file_path}' is not a valid PDF file.")

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError(f"PDF file '{file_path}' is empty (0 bytes).")

        filename = os.path.basename(file_path)

        try:
            doc = pymupdf.open(file_path)
        except Exception as e:
            raise ValueError(f"Failed to open PDF '{filename}'. Corrupted or invalid file format: {e}")

        return self._process_doc(doc, filename)

    def extract_from_bytes(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Extracts cleaned text and metadata from PDF bytes (e.g. from Flask upload).

        Args:
            file_bytes: Raw bytes of the uploaded PDF file.
            filename: Original filename of the document.

        Returns:
            List of page dictionaries with page numbers and cleaned text.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError(f"Uploaded file '{filename}' is empty.")

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Could not parse uploaded PDF '{filename}'. Invalid or corrupted file: {e}")

        return self._process_doc(doc, filename)

    def _process_doc(self, doc: pymupdf.Document, filename: str) -> List[Dict[str, Any]]:
        """Internal helper to iterate over pages and construct page metadata objects."""
        if doc.is_encrypted:
            doc.close()
            raise ValueError(f"PDF '{filename}' is password protected / encrypted.")

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ValueError(f"PDF '{filename}' contains 0 pages.")

        extracted_pages: List[Dict[str, Any]] = []
        total_extracted_chars = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-indexed for human readability and citations
            page = doc.load_page(page_idx)
            raw_text = page.get_text("text") or ""
            cleaned_text = self.clean_text(raw_text)

            char_count = len(cleaned_text)
            word_count = len(cleaned_text.split())
            total_extracted_chars += char_count

            extracted_pages.append({
                "document": filename,
                "page": page_num,
                "total_pages": total_pages,
                "text": cleaned_text,
                "raw_text": raw_text,
                "char_count": char_count,
                "word_count": word_count
            })

        doc.close()

        # Check for scanned / image-only PDFs
        if total_extracted_chars < self.min_char_threshold:
            raise ValueError(
                f"PDF '{filename}' appears to be a scanned or image-only document "
                f"({total_extracted_chars} total characters found). "
                f"Text extraction is unavailable without OCR."
            )

        return extracted_pages
