"""
rag_pipeline.py — End-to-end RAG pipeline orchestrator.

Ties together: PDF loading → chunking → indexing → retrieval → answer generation.
"""
import os
import logging
from src.config import CHROMA_DIR
from src.pdf_loader import load_pdfs, chunk_documents
from src.vector_store import build_vector_store, load_vector_store, Retriever
# from src.llm_chain import build_llm, build_chain, generate_answer
from src.llm_chain import build_llm, generate_answer

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main pipeline class — initialise once, then call .ask() repeatedly."""

    def __init__(self):
        logger.info("=" * 50)
        logger.info("IntelliSense RAG — Initialising pipeline")
        logger.info("=" * 50)

        # Step 1: Build or load vector index
        if os.path.exists(CHROMA_DIR):
            logger.info("Found existing index — loading from disk")
            vs = load_vector_store()
        else:
            logger.info("No index found — building from PDFs")
            docs = load_pdfs()
            chunks = chunk_documents(docs)
            vs = build_vector_store(chunks)

        # Step 2: Create retriever
        self.retriever = Retriever(vs)

        # Step 3: Create LLM chain
        self.llm = build_llm()

        logger.info("Pipeline ready — waiting for queries")

    def ask(self, question: str) -> str:
        """Answer a question using the full RAG pipeline."""
        # Retrieve relevant chunks
        results = self.retriever.query(question)

        # Combine chunk texts into a single context string
        context = "\n\n".join(doc.page_content for doc, _score in results)

        # Generate grounded answer
        answer = generate_answer(self.llm, context, question)
        return answer
