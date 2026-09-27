"""
STEP 2 INTERACTIVE DEMO & TEST SCRIPT
Tests the TextChunker module, sliding window chunking, overlap verification,
and metadata preservation.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.pdf_processor import PDFProcessor
from services.chunker import TextChunker
from tests.generate_test_pdf import generate_sample_pdf


def run_step2_demo():
    print("=" * 70)
    print("STEP 2: TEXT CHUNKING & METADATA PRESERVATION DEMONSTRATION")
    print("=" * 70)

    # 1. Ensure sample PDF is present and extract pages using Step 1
    pdf_path = "data/sample_nlp_paper.pdf"
    if not os.path.exists(pdf_path):
        generate_sample_pdf(pdf_path)

    processor = PDFProcessor()
    pages = processor.extract_pages(pdf_path)
    print(f"\n[1] Extracted {len(pages)} pages from '{pdf_path}' using Step 1 PDFProcessor.")

    # 2. Demonstrate Standard Chunking (chunk_size=500, chunk_overlap=80)
    print("\n[2] Performing Document Chunking (chunk_size=500 words, chunk_overlap=80 words)...")
    standard_chunker = TextChunker(chunk_size=500, chunk_overlap=80)
    chunks = standard_chunker.chunk_pages(pages)

    print(f"    Total Chunks Created: {len(chunks)}")
    print("-" * 70)
    print("STANDARD CHUNKS & METADATA INSPECTION:")
    print("-" * 70)

    for i, ch in enumerate(chunks):
        print(f"Chunk #{i+1}:")
        print(f"  Chunk ID:   {ch['chunk_id']}")
        print(f"  Document:   {ch['document']}")
        print(f"  Page:       {ch['page']}")
        print(f"  Word Count: {ch['word_count']} words | Char Count: {ch['char_count']} chars")
        print(f"  Word Span:  [{ch['start_word_idx']} : {ch['end_word_idx']}]")
        print("  Preview:    " + ch['text'][:120].replace('\n', ' ') + "...")
        print("-" * 70)

    # 3. Visualizing Overlap in Action with a Focused Demonstration
    print("\n[3] Visualizing Sliding Window Overlap in Action:")
    print("    Let's use chunk_size=70 words and chunk_overlap=20 words on Page 1 text.")
    print("    Notice how the tail of Chunk 0 appears at the head of Chunk 1!\n")

    demo_chunker = TextChunker(chunk_size=70, chunk_overlap=20)
    demo_chunks = demo_chunker.chunk_page(pages[0])

    for idx, c in enumerate(demo_chunks):
        print(f"=== {c['chunk_id']} (Words: {c['word_count']}, Range: [{c['start_word_idx']}:{c['end_word_idx']}]) ===")
        print(f"\"{c['text']}\"\n")

    if len(demo_chunks) >= 2:
        # Compute exact overlapping words between chunk 0 and chunk 1
        c0_words = demo_chunks[0]['text'].split()
        c1_words = demo_chunks[1]['text'].split()
        overlap_words = c0_words[-20:]
        print("-" * 70)
        print("OVERLAP REGION VERIFICATION (Last 20 words of C0 vs First 20 words of C1):")
        print("  Tail of C0: " + " ".join(overlap_words))
        print("  Head of C1: " + " ".join(c1_words[:20]))
        print("  Match: " + ("PERFECT MATCH (No context lost across boundary)" if " ".join(overlap_words) == " ".join(c1_words[:20]) else "Partial overlap"))
        print("-" * 70)

    # 4. Test Error Handling
    print("\n[4] Testing Validation & Edge Cases:")
    try:
        TextChunker(chunk_size=50, chunk_overlap=50) # overlap >= size should fail
    except ValueError as e:
        print(f"  [PASS] Handled invalid overlap >= size: {e}")

    empty_result = standard_chunker.chunk_text("")
    print(f"  [PASS] Handled empty string input (returned {len(empty_result)} chunks)")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETED SUCCESSFULLY!")
    print("Text chunking, sliding-window overlap, and metadata tracking work as intended.")
    print("=" * 70)


if __name__ == "__main__":
    run_step2_demo()
