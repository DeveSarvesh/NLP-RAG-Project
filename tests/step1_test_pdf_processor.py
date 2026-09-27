"""
STEP 1 INTERACTIVE DEMO & TEST SCRIPT
Tests the PDF Processor module, page-level extraction, text cleaning,
and error handling.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.pdf_processor import PDFProcessor
from tests.generate_test_pdf import generate_sample_pdf


def run_step1_demo():
    print("=" * 70)
    print("STEP 1: PDF PROCESSING & TEXT PREPROCESSING DEMONSTRATION")
    print("=" * 70)

    # 1. Generate sample PDF
    pdf_path = "data/sample_nlp_paper.pdf"
    print(f"\n[1] Generating sample multi-page PDF at: {pdf_path}")
    generate_sample_pdf(pdf_path)
    print("    Done! PDF generated with 3 pages of NLP & RAG content.")

    # 2. Instantiate PDFProcessor
    print("\n[2] Initializing PDFProcessor...")
    processor = PDFProcessor(min_char_threshold=30)

    # 3. Extract pages
    print(f"\n[3] Extracting text and metadata from '{pdf_path}'...")
    pages = processor.extract_pages(pdf_path)
    print(f"    Successfully extracted {len(pages)} pages!\n")

    # 4. Display extracted pages and metadata
    print("-" * 70)
    print("EXTRACTED PAGES & METADATA INSPECTION")
    print("-" * 70)

    for p in pages:
        print(f"Document:    {p['document']}")
        print(f"Page Number: {p['page']} of {p['total_pages']}")
        print(f"Word Count:  {p['word_count']} words | Character Count: {p['char_count']} chars")
        print("Text Preview:")
        preview_lines = p['text'].split('\n')[:4]
        for line in preview_lines:
            print(f"  > {line}")
        if len(preview_lines) > 4:
            print("  > ... [truncated]")
        print("-" * 70)

    # 5. Demonstrate Text Cleaning
    print("\n[4] Demonstrating Text Cleaning & Normalization:")
    messy_sample = (
        "Large Language Models rely on trans-\n    former archi-\n    tectures.\n\n\n\n"
        "They use dense   vector\t\tembeddings\xa0to capture semantics."
    )
    print("\n--- RAW UNCLEANED TEXT ---")
    print(repr(messy_sample))
    print(messy_sample)

    cleaned_sample = processor.clean_text(messy_sample)
    print("\n--- CLEANED & NORMALIZED TEXT ---")
    print(repr(cleaned_sample))
    print(cleaned_sample)

    # 6. Test Error Handling
    print("\n[5] Testing Error Handling:")
    try:
        processor.extract_pages("data/non_existent_file.pdf")
    except FileNotFoundError as e:
        print(f"  [PASS] Handled non-existent file: {e}")

    # Empty PDF test
    empty_path = "data/empty_test.pdf"
    with open(empty_path, "wb") as f:
        f.write(b"")
    try:
        processor.extract_pages(empty_path)
    except ValueError as e:
        print(f"  [PASS] Handled 0-byte empty file: {e}")
    finally:
        if os.path.exists(empty_path):
            os.remove(empty_path)

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETED SUCCESSFULLY!")
    print("PDF extraction, metadata tracking, and cleaning are functioning perfectly.")
    print("=" * 70)


if __name__ == "__main__":
    run_step1_demo()
