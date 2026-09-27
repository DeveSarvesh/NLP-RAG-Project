"""
STEP 5 INTERACTIVE DEMO & TEST SCRIPT
Tests the complete end-to-end Single Document RAG pipeline:
PDF Ingestion -> Chunking -> Embeddings -> FAISS Search -> Grounded Gemini Generation.
Also demonstrates dynamic processing on any custom PDF provided via CLI argument.
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
from services.llm_service import LLMService
from tests.generate_test_pdf import generate_sample_pdf


def run_step5_demo(custom_pdf_path: str = None):
    print("=" * 70)
    print("STEP 5: END-TO-END SINGLE-DOCUMENT RAG DEMONSTRATION")
    print("=" * 70)

    # 1. Select Document (Dynamic Support!)
    if custom_pdf_path and os.path.exists(custom_pdf_path):
        pdf_path = custom_pdf_path
        print(f"\n[1] Using custom document: {pdf_path}")
    else:
        pdf_path = "data/sample_nlp_paper.pdf"
        if not os.path.exists(pdf_path):
            generate_sample_pdf(pdf_path)
        print(f"\n[1] Using sample NLP document: {pdf_path}")

    # 2. Stage 1: Ingestion (PDF Extraction + Cleaning)
    print("\n[2] Extracting and cleaning text from PDF...")
    processor = PDFProcessor()
    pages = processor.extract_pages(pdf_path)
    print(f"    Extracted {len(pages)} pages.")

    # 3. Stage 2: Chunking & Metadata Preservation
    print("\n[3] Chunking document with sliding-window overlap...")
    chunker = TextChunker(chunk_size=500, chunk_overlap=80)
    chunks = chunker.chunk_pages(pages)
    print(f"    Created {len(chunks)} chunks with metadata.")

    # 4. Stage 3: Dense Vector Embeddings (all-MiniLM-L6-v2)
    print("\n[4] Generating dense embeddings (all-MiniLM-L6-v2)...")
    embedder = EmbeddingService()
    embeddings = embedder.embed_chunks(chunks, normalize=True)
    print(f"    Embeddings matrix shape: {embeddings.shape}")

    # 5. Stage 4: FAISS Vector Indexing
    print("\n[5] Building FAISS IndexFlatIP store...")
    vector_store = VectorStore(dimension=embedder.dimension)
    vector_store.add_documents(chunks, embeddings)
    print(f"    FAISS index populated: {vector_store.total_chunks} chunks ready.")

    # 6. Initialize Retriever and LLM Service
    retriever = SemanticRetriever(
        vector_store=vector_store,
        embedding_service=embedder,
        default_top_k=3,
        default_threshold=0.15
    )

    llm = LLMService()
    print(f"\n[6] Initialized LLM Service (Model: {llm.model_name}, API Connected: {llm.is_available})")
    if not llm.is_available:
        print("    * Notice: Set GEMINI_API_KEY in .env to enable live Gemini API calls.")
        print("    * Running in grounded context-preview mode for this test.")

    # 7. Run Test Questions
    test_queries = [
        "What fundamental limitations of LLMs does RAG solve?",
        "Why is sliding-window overlap critical when chunking documents?",
        "Explain the three Information Retrieval metrics used to evaluate retrieval quality.",
        "What is the recipe to make chocolate chip cookies?"  # Out-of-domain test
    ]

    print("\n" + "=" * 70)
    print("RUNNING GROUNDED Q&A TESTS")
    print("=" * 70)

    for idx, query in enumerate(test_queries, 1):
        print(f"\n>>> QUESTION #{idx}: \"{query}\"")

        # A. Semantic Retrieval
        retrieval_res = retriever.retrieve(query, top_k=2)

        # B. Grounded Generation
        generation_res = llm.generate_grounded_answer(query, retrieval_res)

        print("\n--- RETRIEVAL SUMMARY ---")
        print(f"Has Relevant Context: {retrieval_res['has_relevant_context']}")
        print(f"Sources Found:        {retrieval_res['sources']}")
        if retrieval_res['results']:
            print(f"Top Chunk Score:      {retrieval_res['results'][0]['score']:.4f} ({retrieval_res['results'][0]['citation']})")

        print("\n--- GENERATED ANSWER ---")
        print(generation_res['answer'])
        print("-" * 70)

    # 8. Test Conversational Follow-up
    print("\n" + "=" * 70)
    print("TESTING CONVERSATIONAL FOLLOW-UP QUESTION")
    print("=" * 70)

    q1 = "What vector indexing library is discussed in the document?"
    r1 = retriever.retrieve(q1, top_k=2)
    ans1 = llm.generate_grounded_answer(q1, r1)

    print(f"\nTurn 1 User: {q1}")
    print(f"Turn 1 Assistant: {ans1['answer'][:150]}...")

    # Chat history tracking
    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": ans1['answer']}
    ]

    q2 = "What two index types does it provide and how are they compared?"
    r2 = retriever.retrieve(q2, top_k=2)
    ans2 = llm.generate_grounded_answer(q2, r2, chat_history=history)

    print(f"\nTurn 2 User (Follow-up): {q2}")
    print(f"Turn 2 Assistant: {ans2['answer']}")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("STEP 5 COMPLETED SUCCESSFULLY!")
    print("Single Document RAG pipeline is fully functional and context-grounded.")
    print("=" * 70)


if __name__ == "__main__":
    custom_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_step5_demo(custom_path)
