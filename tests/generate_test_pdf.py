"""
Test PDF Generator
Generates a realistic 3-page academic mini-paper PDF for testing the RAG pipeline.
"""

import os
from fpdf import FPDF


class AcademicPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 8, 'NLP Research Notes: Dense Retrieval & RAG Foundations', 0, 1, 'R')
        self.line(10, 15, 200, 15)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def generate_sample_pdf(output_path: str = "data/sample_nlp_paper.pdf") -> str:
    """Generates a 3-page sample PDF containing structured NLP content."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pdf = AcademicPDF()
    pdf.set_auto_page_break(auto=True, margin=18)

    # PAGE 1: Introduction to RAG & Vector Embeddings
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(20, 35, 60)
    pdf.cell(0, 10, 'Foundations of Retrieval-Augmented Generation (RAG)', 0, 1, 'L')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(0, 7, '1. Introduction and Architectural Motivation', 0, 1, 'L')

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 30, 30)
    p1_text = (
        "Large Language Models (LLMs) have achieved remarkable natural language generation capabilities. "
        "However, they suffer from fundamental limitations including knowledge cutoffs, inability to access "
        "private corporate or domain-specific data, and severe hallucinations when reasoning over uncertain facts. "
        "Retrieval-Augmented Generation (RAG) resolves these issues by separating the external knowledge store "
        "from the parametric memory of the model.\n\n"
        "In a standard RAG architecture, when a user poses a natural-language query, the system first retrieves "
        "semantically relevant passages from an external corpus using dense vector embeddings. These retrieved passages "
        "are then dynamically injected into the LLM context prompt alongside the original query. The language model is "
        "specifically instructed to synthesize its final answer grounded strictly on the supplied context."
    )
    pdf.multi_cell(0, 6, p1_text)
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(0, 7, '2. Semantic Dense Vector Embeddings', 0, 1, 'L')

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 30, 30)
    p1_text2 = (
        "Traditional keyword search methods such as Term Frequency-Inverse Document Frequency (TF-IDF) rely on exact "
        "lexical overlap. Consequently, they fail when users ask questions using synonyms or paraphrase concepts. "
        "Modern dense retrieval solves this by using pre-trained transformer encoders, such as all-MiniLM-L6-v2, "
        "to map text chunks into a shared 384-dimensional continuous vector space. In this latent space, semantically "
        "similar passages are positioned close to one another based on cosine similarity."
    )
    pdf.multi_cell(0, 6, p1_text2)

    # PAGE 2: Chunking Strategies and FAISS Indexing
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(20, 35, 60)
    pdf.cell(0, 10, 'Document Chunking and Vector Indexing', 0, 1, 'L')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(0, 7, '3. Chunking Strategies and Granularity Trade-offs', 0, 1, 'L')

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 30, 30)
    p2_text = (
        "Raw documents are typically too long to embed as a single vector without severe information dilution. "
        "Therefore, text segmentation or chunking is performed. Typical chunk sizes range between 500 and 800 words "
        "with an overlap of 50 to 100 words.\n\n"
        "Sliding-window overlap is critical because it ensures that semantic continuity and cross-sentence dependencies "
        "are not severed at arbitrary chunk boundaries. Moreover, every individual chunk must preserve its provenance "
        "metadata, including source document name, page number, and chunk index, to enable reliable citations."
    )
    pdf.multi_cell(0, 6, p2_text)
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(0, 7, '4. Vector Indexing with FAISS', 0, 1, 'L')

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 30, 30)
    p2_text2 = (
        "Facebook AI Similarity Search (FAISS) is an open-source library optimized for high-performance dense vector "
        "similarity search. For exact nearest-neighbor search, FAISS provides IndexFlatIP (Inner Product) and "
        "IndexFlatL2 (Euclidean distance). When embeddings are L2-normalized to unit length, the inner product is "
        "mathematically identical to Cosine Similarity. Query latency with IndexFlatIP on CPU for small to medium "
        "corpora is sub-millisecond, providing scalable Top-K candidate retrieval."
    )
    pdf.multi_cell(0, 6, p2_text2)

    # PAGE 3: Grounded Generation and Experimental Evaluation
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(20, 35, 60)
    pdf.cell(0, 10, 'Grounded LLM Generation & Evaluation Metrics', 0, 1, 'L')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(0, 7, '5. Grounded Prompting and Hallucination Mitigation', 0, 1, 'L')

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 30, 30)
    p3_text = (
        "Sending retrieved context to the LLM requires strict prompt framing. The system prompt instructs the model: "
        "'Answer the question using ONLY the provided context chunks. If the context does not contain sufficient facts, "
        "clearly state that the information is not available.'\n\n"
        "Additionally, a relevance similarity threshold (such as 0.35 cosine similarity) filters out unrelated chunks "
        "before they reach the LLM, preventing the model from answering out-of-domain queries with ungrounded speculation."
    )
    pdf.multi_cell(0, 6, p3_text)
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(0, 7, '6. Retrieval Evaluation Metrics (Precision@K, Recall@K, MRR)', 0, 1, 'L')

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 30, 30)
    p3_text2 = (
        "To quantitatively benchmark retrieval quality, three core Information Retrieval metrics are evaluated:\n"
        "1. Precision@K: The proportion of retrieved Top-K chunks that are truly relevant to the query.\n"
        "2. Recall@K: The proportion of all known ground-truth relevant chunks that appear in the Top-K results.\n"
        "3. Mean Reciprocal Rank (MRR): The average of reciprocal ranks (1/rank) of the first relevant chunk retrieved.\n\n"
        "In empirical experiments, dense semantic retrieval consistently outperforms lexical TF-IDF on complex, "
        "synonym-heavy, and conceptual queries."
    )
    pdf.multi_cell(0, 6, p3_text2)

    pdf.output(output_path)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    path = generate_sample_pdf()
    print(f"Generated sample PDF at: {path}")
