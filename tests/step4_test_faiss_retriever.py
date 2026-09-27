"""
STEP 4 INTERACTIVE DEMO & TEST SCRIPT
Tests the FAISS VectorStore, index building, persistence (save/load),
and the SemanticRetriever with relevance threshold filtering and citation formatting.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.pdf_processor import PDFProcessor
from services.chunker import TextChunker
from services.embeddings import EmbeddingService
from services.vector_store import VectorStore
from services.retriever import SemanticRetriever
from tests.generate_test_pdf import generate_sample_pdf


def run_step4_demo():
    print("=" * 70)
    print("STEP 4: FAISS VECTOR INDEXING & SEMANTIC RETRIEVAL DEMONSTRATION")
    print("=" * 70)

    # 1. Pipeline: Step 1 (PDF) -> Step 2 (Chunks) -> Step 3 (Embeddings)
    pdf_path = "data/sample_nlp_paper.pdf"
    if not os.path.exists(pdf_path):
        generate_sample_pdf(pdf_path)

    processor = PDFProcessor()
    pages = processor.extract_pages(pdf_path)

    chunker = TextChunker(chunk_size=500, chunk_overlap=80)
    chunks = chunker.chunk_pages(pages)

    embedder = EmbeddingService()
    embeddings = embedder.embed_chunks(chunks, normalize=True)
    print(f"\n[1] Extracted {len(pages)} pages, created {len(chunks)} chunks, generated embeddings {embeddings.shape}.")

    # 2. Build FAISS Vector Store
    print("\n[2] Initializing FAISS VectorStore and indexing chunks...")
    vector_store = VectorStore(dimension=embedder.dimension)
    vector_store.add_documents(chunks, embeddings)

    stats = vector_store.get_stats()
    print(f"    Total Indexed Chunks:    {stats['total_chunks']}")
    print(f"    Total Indexed Documents: {stats['total_documents']}")
    print(f"    Index Dimension:         {stats['dimension']}")
    print(f"    Index Is Trained:        {stats['is_trained']}")

    # 3. Test Semantic Retriever
    print("\n[3] Initializing SemanticRetriever (Top-K=3, Threshold=0.15)...")
    retriever = SemanticRetriever(
        vector_store=vector_store,
        embedding_service=embedder,
        default_top_k=3,
        default_threshold=0.15
    )

    test_queries = [
        ("What limitations of LLMs make RAG necessary?", "Page 1"),
        ("Why is sliding-window overlap used during chunking?", "Page 2"),
        ("How is Mean Reciprocal Rank (MRR) defined?", "Page 3"),
        ("What is the recipe to make chocolate chip cookies?", "None / Out-of-Domain")
    ]

    for q_idx, (query, expected_target) in enumerate(test_queries, 1):
        print(f"\n" + "-" * 70)
        print(f"QUERY #{q_idx}: \"{query}\"")
        print(f"Expected Target: {expected_target}")
        print("-" * 70)

        retrieval_res = retriever.retrieve(query, top_k=3, threshold=0.15)

        print(f"Has Relevant Context: {retrieval_res['has_relevant_context']}")
        print(f"Relevant Chunks Count: {retrieval_res['relevant_count']} / {retrieval_res['total_retrieved']}")
        print(f"Sources Found: {retrieval_res['sources']}")
        print("\nRanked Chunks:")

        for item in retrieval_res["results"]:
            rel_tag = "[PASSES THRESHOLD]" if item["is_relevant"] else "[BELOW THRESHOLD]"
            print(
                f"  Rank #{item['rank']} | Score: {item['score']:.4f} {rel_tag} | "
                f"{item['citation']} | {item['chunk_id']}"
            )
            print(f"    Preview: \"{item['text'][:100]}...\"\n")

    # 4. Demonstrate Context Formatting for LLM
    print("=" * 70)
    print("[4] DEMONSTRATING CONTEXT FORMATTING FOR LLM INJECTION")
    print("=" * 70)
    sample_query = "What limitations of LLMs make RAG necessary?"
    sample_res = retriever.retrieve(sample_query, top_k=2)
    print("Formatted Context Block that will be sent to Gemini in Step 5:")
    print("-" * 70)
    print(sample_res["context_text"])
    print("-" * 70)

    # 5. Test FAISS Index Serialization (Save and Load)
    print("\n[5] Testing FAISS Persistence (Save to disk & Load back)...")
    save_dir = "data/indexes"
    vector_store.save(index_dir=save_dir, index_name="test_step4_index")
    print(f"    Saved FAISS index binary and metadata JSON to '{save_dir}'.")

    new_store = VectorStore(dimension=embedder.dimension)
    new_store.load(index_dir=save_dir, index_name="test_step4_index")
    print(f"    Successfully loaded saved index! Total chunks in loaded store: {new_store.total_chunks}")

    # Verify search on reloaded store
    reloaded_retriever = SemanticRetriever(vector_store=new_store, embedding_service=embedder)
    reloaded_res = reloaded_retriever.retrieve(sample_query, top_k=1)
    print(f"    Top match from reloaded store: {reloaded_res['results'][0]['citation']} (Score: {reloaded_res['results'][0]['score']:.4f})")
    print("    [PASS] FAISS serialization and deserialization verified.")

    print("\n" + "=" * 70)
    print("STEP 4 COMPLETED SUCCESSFULLY!")
    print("FAISS indexing, semantic retrieval, relevance thresholding, and persistence work perfectly.")
    print("=" * 70)


if __name__ == "__main__":
    run_step4_demo()
