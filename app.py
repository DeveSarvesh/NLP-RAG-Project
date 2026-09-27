"""
Intelligent Multi-Document RAG Assistant - Flask Application
Serves the web interface and REST API endpoints for document ingestion,
FAISS semantic retrieval, grounded Gemini generation, and session management.
"""

import os
import time
import json
from typing import List, Dict, Any
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from services.pdf_processor import PDFProcessor
from services.chunker import TextChunker
from services.embeddings import EmbeddingService
from services.vector_store import VectorStore
from services.retriever import SemanticRetriever
from services.llm_service import LLMService
from tests.generate_test_pdf import generate_sample_pdf

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "nlp_rag_assistant_secret_2026")
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "data", "uploads")
INDEX_FOLDER = os.path.join(os.path.dirname(__file__), "data", "indexes")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# Initialize core services
pdf_processor = PDFProcessor()
chunker = TextChunker(
    chunk_size=int(os.getenv("DEFAULT_CHUNK_SIZE", 500)),
    chunk_overlap=int(os.getenv("DEFAULT_CHUNK_OVERLAP", 80))
)
embedding_service = EmbeddingService()
vector_store = VectorStore(dimension=embedding_service.dimension)
retriever = SemanticRetriever(
    vector_store=vector_store,
    embedding_service=embedding_service,
    default_top_k=int(os.getenv("DEFAULT_TOP_K", 3)),
    default_threshold=float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", 0.15))
)
llm_service = LLMService()

# In-memory document registry for session tracking
document_registry: Dict[str, Dict[str, Any]] = {}
chat_history: List[Dict[str, str]] = []


def auto_load_sample_if_empty():
    """Automatically loads the sample NLP document if the index is currently empty."""
    if vector_store.total_chunks == 0:
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_nlp_paper.pdf")
        if not os.path.exists(sample_path):
            generate_sample_pdf(sample_path)
        
        try:
            start_t = time.time()
            pages = pdf_processor.extract_pages(sample_path)
            chunks = chunker.chunk_pages(pages)
            embeddings = embedding_service.embed_chunks(chunks, normalize=True)
            vector_store.add_documents(chunks, embeddings)

            doc_name = os.path.basename(sample_path)
            document_registry[doc_name] = {
                "filename": doc_name,
                "filepath": sample_path,
                "pages": len(pages),
                "chunks": len(chunks),
                "words": sum(p.get("word_count", 0) for p in pages),
                "processing_time": round(time.time() - start_t, 2),
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"[Startup] Could not auto-load sample document: {e}")


# Run initial auto-load
auto_load_sample_if_empty()


@app.route("/")
def index():
    """Renders the main single-page application interface."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns current system status, stats, and indexed document details."""
    stats = vector_store.get_stats()
    latest_doc = list(document_registry.values())[-1] if document_registry else None

    return jsonify({
        "status": "ready",
        "total_documents": len(document_registry),
        "total_chunks": stats["total_chunks"],
        "index_dimension": stats["dimension"],
        "is_trained": stats["is_trained"],
        "llm_available": llm_service.is_available,
        "llm_model": llm_service.model_name,
        "documents": list(document_registry.values()),
        "latest_document": latest_doc,
        "default_top_k": retriever.default_top_k,
        "default_threshold": retriever.default_threshold
    })


@app.route("/api/upload", methods=["POST"])
def upload_documents():
    """
    Handles single or multiple PDF file uploads.
    Extracts text, chunks content, creates embeddings, and updates FAISS index.
    """
    if "files" not in request.files and "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    files = request.files.getlist("files") or [request.files["file"]]
    if not files or files[0].filename == "":
        return jsonify({"error": "No file selected for upload."}), 400

    processed_docs = []
    total_new_chunks = 0
    start_total_t = time.time()

    for file in files:
        filename = file.filename
        if not filename.lower().endswith(".pdf"):
            continue

        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        try:
            start_doc_t = time.time()
            pages = pdf_processor.extract_pages(save_path)
            chunks = chunker.chunk_pages(pages)

            if chunks:
                embeddings = embedding_service.embed_chunks(chunks, normalize=True)
                vector_store.add_documents(chunks, embeddings)
                total_new_chunks += len(chunks)

            proc_time = round(time.time() - start_doc_t, 2)
            doc_meta = {
                "filename": filename,
                "filepath": save_path,
                "pages": len(pages),
                "chunks": len(chunks),
                "words": sum(p.get("word_count", 0) for p in pages),
                "processing_time": proc_time,
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            document_registry[filename] = doc_meta
            processed_docs.append(doc_meta)

        except Exception as e:
            return jsonify({
                "error": f"Failed to process '{filename}': {str(e)}"
            }), 400

    return jsonify({
        "success": True,
        "message": f"Successfully processed and indexed {len(processed_docs)} document(s).",
        "processed_documents": processed_docs,
        "total_new_chunks": total_new_chunks,
        "total_chunks_in_index": vector_store.total_chunks,
        "total_processing_time": round(time.time() - start_total_t, 2)
    })


@app.route("/api/query", methods=["POST"])
def query_rag():
    """
    Executes grounded RAG search and answer generation.
    Accepts question, top_k, threshold, and returns answer, citations, and ranked chunks.
    """
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    top_k = int(data.get("top_k", retriever.default_top_k))
    threshold = float(data.get("threshold", retriever.default_threshold))

    if not question:
        return jsonify({"error": "Question parameter is required."}), 400

    if vector_store.total_chunks == 0:
        return jsonify({
            "error": "No documents are currently indexed. Please upload a PDF first."
        }), 400

    start_t = time.time()

    # 1. Semantic Retrieval with FAISS
    retrieval_res = retriever.retrieve(
        query=question,
        top_k=top_k,
        threshold=threshold
    )

    # 2. Grounded Answer Generation with Gemini
    generation_res = llm_service.generate_grounded_answer(
        query=question,
        retrieval_result=retrieval_res,
        chat_history=chat_history
    )

    elapsed_time = round(time.time() - start_t, 2)

    # Update in-memory chat history
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": generation_res.get("answer", "")})

    return jsonify({
        "question": question,
        "answer": generation_res.get("answer", ""),
        "sources": generation_res.get("sources", []),
        "has_relevant_context": generation_res.get("has_relevant_context", False),
        "is_grounded": generation_res.get("is_grounded", True),
        "model_used": generation_res.get("model_used", llm_service.model_name),
        "retrieval_results": retrieval_res.get("results", []),
        "total_retrieved": len(retrieval_res.get("results", [])),
        "execution_time_seconds": elapsed_time,
        "timestamp": time.strftime("%I:%M %p")
    })


@app.route("/api/clear", methods=["POST"])
def clear_session():
    """Resets the vector index, document registry, and chat history."""
    vector_store.clear()
    document_registry.clear()
    chat_history.clear()
    return jsonify({
        "success": True,
        "message": "Index, document registry, and conversation history cleared."
    })


@app.route("/api/load-sample", methods=["POST"])
def load_sample():
    """Loads or reloads the sample NLP research paper into the index."""
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_nlp_paper.pdf")
    generate_sample_pdf(sample_path)
    
    start_t = time.time()
    pages = pdf_processor.extract_pages(sample_path)
    chunks = chunker.chunk_pages(pages)
    embeddings = embedding_service.embed_chunks(chunks, normalize=True)
    vector_store.add_documents(chunks, embeddings)

    doc_name = os.path.basename(sample_path)
    doc_meta = {
        "filename": doc_name,
        "filepath": sample_path,
        "pages": len(pages),
        "chunks": len(chunks),
        "words": sum(p.get("word_count", 0) for p in pages),
        "processing_time": round(time.time() - start_t, 2),
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    document_registry[doc_name] = doc_meta

    return jsonify({
        "success": True,
        "document": doc_meta,
        "total_chunks": vector_store.total_chunks
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Starting Intelligent RAG Assistant on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
