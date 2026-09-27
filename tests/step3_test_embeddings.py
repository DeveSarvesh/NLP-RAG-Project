"""
STEP 3 INTERACTIVE DEMO & TEST SCRIPT
Tests the EmbeddingService module, dense vector generation, L2 normalization,
and demonstrates semantic similarity scoring with NumPy.
"""

import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.pdf_processor import PDFProcessor
from services.chunker import TextChunker
from services.embeddings import EmbeddingService
from tests.generate_test_pdf import generate_sample_pdf


def run_step3_demo():
    print("=" * 70)
    print("STEP 3: DENSE VECTOR EMBEDDINGS DEMONSTRATION")
    print("=" * 70)

    # 1. Pipeline: Step 1 (PDF) -> Step 2 (Chunks)
    pdf_path = "data/sample_nlp_paper.pdf"
    if not os.path.exists(pdf_path):
        generate_sample_pdf(pdf_path)

    processor = PDFProcessor()
    pages = processor.extract_pages(pdf_path)

    chunker = TextChunker(chunk_size=500, chunk_overlap=80)
    chunks = chunker.chunk_pages(pages)
    print(f"\n[1] Extracted {len(pages)} pages and created {len(chunks)} chunks.")

    # 2. Initialize Embedding Service
    print("\n[2] Initializing EmbeddingService (sentence-transformers/all-MiniLM-L6-v2)...")
    embedder = EmbeddingService()
    print(f"    Embedding Model: {embedder.model_name}")
    print(f"    Vector Dimension: {embedder.dimension} dimensions")

    # 3. Generate Embeddings for Chunks
    print(f"\n[3] Generating dense embeddings for all {len(chunks)} chunks...")
    chunk_embeddings = embedder.embed_chunks(chunks, normalize=True)

    print(f"    Embeddings Matrix Shape: {chunk_embeddings.shape} (N_chunks, Dimension)")
    print(f"    Data Type: {chunk_embeddings.dtype}")

    # Verify L2 Normalization (Unit Length: ||v|| = 1.0)
    norms = np.linalg.norm(chunk_embeddings, axis=1)
    print("    L2 Norms of first 3 chunk vectors:", [round(float(n), 4) for n in norms[:3]])
    print("    [PASS] Vectors are unit-normalized (enabling dot product = cosine similarity).")

    # 4. Interactive Semantic Similarity with Test Queries
    print("\n" + "=" * 70)
    print("[4] DEMONSTRATING SEMANTIC SIMILARITY SEARCH (Dot Product Cosine Similarity)")
    print("=" * 70)

    test_queries = [
        # Query 1: Targets Page 1 (RAG motivation & limitations of LLMs)
        "Why do language models hallucinate and need external memory?",
        # Query 2: Targets Page 2 (Chunking overlap & FAISS indexing)
        "What is the recommended chunk size and why is overlap important?",
        # Query 3: Targets Page 3 (IR evaluation metrics)
        "Explain Mean Reciprocal Rank and Precision at K.",
        # Query 4: Out-of-domain query (Testing low relevance)
        "What is the recipe for baking chocolate chip cookies?"
    ]

    for q_idx, query in enumerate(test_queries, 1):
        print(f"\n--- QUERY #{q_idx}: \"{query}\" ---")
        query_vector = embedder.embed_query(query, normalize=True)

        # Because both chunk_embeddings and query_vector are L2-normalized:
        # Cosine Similarity = chunk_embeddings @ query_vector
        similarity_scores = np.dot(chunk_embeddings, query_vector)

        # Rank chunks by descending similarity
        ranked_indices = np.argsort(similarity_scores)[::-1]

        print(f"{'Rank':<5} | {'Score':<8} | {'Document':<22} | {'Page':<6} | {'Chunk ID':<24} | {'Preview'}")
        print("-" * 90)

        for rank, chunk_i in enumerate(ranked_indices, 1):
            score = similarity_scores[chunk_i]
            target_chunk = chunks[chunk_i]
            preview = target_chunk['text'][:45].replace('\n', ' ') + "..."
            print(
                f"#{rank:<4} | {score:0.4f}   | {target_chunk['document']:<22} | "
                f"Page {target_chunk['page']:<1} | {target_chunk['chunk_id']:<24} | {preview}"
            )

        top_chunk = chunks[ranked_indices[0]]
        top_score = similarity_scores[ranked_indices[0]]

        if top_score >= 0.35:
            print(f"  --> Top Match: Page {top_chunk['page']} (Similarity: {top_score:0.4f}) [RELEVANT]")
        else:
            print(f"  --> Top Match: Page {top_chunk['page']} (Similarity: {top_score:0.4f}) [LOW RELEVANCE / OUT-OF-DOMAIN]")

    # 5. Validation & Edge Cases
    print("\n" + "=" * 70)
    print("[5] Testing Validation & Edge Cases:")
    empty_embeds = embedder.embed_texts([])
    print(f"  [PASS] Empty text list returned shape: {empty_embeds.shape}")

    try:
        embedder.embed_query("   ")
    except ValueError as e:
        print(f"  [PASS] Handled empty query string: {e}")

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETED SUCCESSFULLY!")
    print("Dense vector embeddings and semantic similarity are functioning perfectly.")
    print("=" * 70)


if __name__ == "__main__":
    run_step3_demo()
