"""
Grounded LLM Generation Service
Integrates Google Gemini API with strict context-grounded prompting,
anti-hallucination rules, citation formatting, and information-not-found guards.
"""

import os
import re
from typing import List, Dict, Any, Optional
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMService:
    """
    Manages interaction with the Gemini API to generate factual,
    context-grounded answers with precise document and page citations.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an Intelligent NLP Document Assistant.\n"
        "Your task is to answer the user's question using ONLY the provided context passages below.\n\n"
        "STRICT INSTRUCTIONS:\n"
        "1. Base your answer STRICTLY on the facts directly stated in the context passages.\n"
        "2. Do NOT use external pre-trained knowledge or make speculative claims.\n"
        "3. If the provided context does not contain enough facts to answer the question, respond EXACTLY with:\n"
        "   \"I couldn't find sufficient information about this question in the uploaded documents.\"\n"
        "4. Explicitly cite the source document and page number for every key point using the format:\n"
        "   [Doc: <filename>, Page: <page_number>]\n"
        "5. Keep your response clear, structured, and factual."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Args:
            api_key: Gemini API Key. Defaults to GEMINI_API_KEY environment variable.
            model_name: Gemini model name (e.g., 'gemini-2.5-flash' or 'gemini-1.5-flash').
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initializes the google.genai Client."""
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            self._client = None
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"[LLMService] Warning: Failed to initialize Google GenAI client: {e}")
            self._client = None

    @property
    def is_available(self) -> bool:
        """Returns True if the Gemini API client is configured and ready."""
        return self._client is not None

    def generate_grounded_answer(
        self,
        query: str,
        retrieval_result: Dict[str, Any],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generates a context-grounded response using the retrieved passages and Gemini.

        Args:
            query: User's question string.
            retrieval_result: Dictionary returned by SemanticRetriever.retrieve().
            chat_history: Optional list of previous turns: [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            Dictionary containing the generated answer, source citations,
            relevance flags, and metadata.
        """
        if not query or not query.strip():
            return {
                "query": query,
                "answer": "Please ask a question about the uploaded document.",
                "sources": [],
                "has_relevant_context": False,
                "is_grounded": True,
                "model_used": self.model_name
            }

        # 1. Check if retriever found any relevant context
        has_relevant = retrieval_result.get("has_relevant_context", False)
        context_text = retrieval_result.get("context_text", "").strip()
        sources = retrieval_result.get("sources", [])

        # If similarity threshold was not met for any chunk, reject immediately
        if not has_relevant or not context_text:
            return {
                "query": query,
                "answer": "I couldn't find sufficient information about this question in the uploaded documents.",
                "sources": [],
                "has_relevant_context": False,
                "is_grounded": True,
                "model_used": self.model_name,
                "retrieval_mode": retrieval_result.get("retrieval_mode", "semantic_faiss")
            }

        # 2. If Gemini API key is missing, return a structured preview response
        if not self.is_available:
            mock_answer = (
                "**[Notice: GEMINI_API_KEY not configured in .env]**\n\n"
                f"Relevant context was successfully retrieved for your query: *\"{query}\"*\n\n"
                "**Retrieved Sources:**\n" +
                "\n".join([f"- {s}" for s in sources]) +
                "\n\n**Retrieved Context Passages:**\n" +
                context_text +
                "\n\n*To enable real-time Gemini generation, please add your `GEMINI_API_KEY` to the `.env` file.*"
            )
            return {
                "query": query,
                "answer": mock_answer,
                "sources": sources,
                "has_relevant_context": True,
                "is_grounded": True,
                "model_used": "offline_preview",
                "retrieval_mode": retrieval_result.get("retrieval_mode", "semantic_faiss")
            }

        # 3. Build Prompt for Gemini
        prompt_content = self._build_prompt(query, context_text, chat_history)

        # Candidate models to try in order if the primary experiences high demand / 503
        candidate_models = [
            self.model_name,
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash"
        ]
        # Remove duplicate model names while preserving order
        unique_models = []
        for m in candidate_models:
            if m and m not in unique_models:
                unique_models.append(m)

        last_error = None
        for current_model in unique_models:
            try:
                response = self._client.models.generate_content(
                    model=current_model,
                    contents=prompt_content,
                )

                generated_text = response.text.strip() if response.text else ""

                # Check if model reported information not found
                is_not_found = (
                    "couldn't find sufficient information" in generated_text.lower() or
                    "could not find sufficient information" in generated_text.lower() or
                    "information is not available" in generated_text.lower()
                )

                final_sources = [] if is_not_found else sources

                return {
                    "query": query,
                    "answer": generated_text,
                    "sources": final_sources,
                    "has_relevant_context": not is_not_found,
                    "is_grounded": True,
                    "model_used": current_model,
                    "retrieval_mode": retrieval_result.get("retrieval_mode", "semantic_faiss")
                }

            except Exception as e:
                last_error = e
                # If high demand or 503, try next candidate model
                continue

        # If all candidate models failed
        return {
            "query": query,
            "answer": f"Error communicating with Gemini API: {str(last_error)}",
            "sources": sources,
            "has_relevant_context": True,
            "is_grounded": False,
            "error": str(last_error),
            "model_used": self.model_name
        }

    def _build_prompt(
        self,
        query: str,
        context_text: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Constructs the full grounded prompt including context and optional conversation history."""
        prompt_parts = [self.DEFAULT_SYSTEM_PROMPT, "\n\n=== RETRIEVED DOCUMENT CONTEXT ==="]
        prompt_parts.append(context_text)
        prompt_parts.append("=== END OF CONTEXT ===\n")

        # Include prior conversation history if available
        if chat_history and len(chat_history) > 0:
            prompt_parts.append("=== RECENT CONVERSATION HISTORY ===")
            # Limit to last 4 messages to stay focused
            for msg in chat_history[-4:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("=== END OF HISTORY ===\n")

        prompt_parts.append(f"User Question: {query}\n\nAnswer:")
        return "\n".join(prompt_parts)

    def summarize_document(self, chunks: List[Dict[str, Any]], document_name: str) -> Dict[str, Any]:
        """Generates a structured grounded summary of an entire document."""
        if not chunks:
            return {
                "document": document_name,
                "summary": "No document content available to summarize.",
                "sources": []
            }

        # Combine chunks into full context (or top representative chunks)
        combined_text = "\n\n".join([
            f"[Page {c.get('page', 1)}]: {c.get('text', '')}" for c in chunks[:10]
        ])

        if not self.is_available:
            return {
                "document": document_name,
                "summary": f"Summary preview for '{document_name}' ({len(chunks)} chunks indexed). Add GEMINI_API_KEY in .env to generate AI summary.",
                "sources": [f"{document_name} - All Pages"]
            }

        prompt = (
            f"You are an NLP Research Assistant. Summarize the following document '{document_name}' based ONLY on the provided text.\n\n"
            f"TEXT:\n{combined_text}\n\n"
            f"Provide a structured summary with:\n"
            f"1. Main Objective / Core Topic\n"
            f"2. Key Findings & Technical Details (with page citations)\n"
            f"3. Practical Applications / Takeaways"
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return {
                "document": document_name,
                "summary": response.text.strip(),
                "sources": [f"{document_name} - Pages 1-{max(c.get('page', 1) for c in chunks)}"]
            }
        except Exception as e:
            return {
                "document": document_name,
                "summary": f"Failed to generate summary: {e}",
                "sources": []
            }
