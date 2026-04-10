"""
vector_store.py — Embed chunks, store in ChromaDB, and retrieve.

Pipeline stage: Document chunks  →  ChromaDB index  →  relevant chunks
"""
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME, TOP_K

logger = logging.getLogger(__name__)


def build_vector_store(chunks):
    """Embed chunks and persist to ChromaDB."""
    logger.info(f"Embedding {len(chunks)} chunks with '{EMBEDDING_MODEL}'...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    logger.info(f"Vector store built — {len(chunks)} vectors stored in '{CHROMA_DIR}'")
    return vector_store


def load_vector_store():
    """Load an existing ChromaDB vector store from disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    logger.info("Loaded existing vector store from disk")
    return vector_store


# ── Retriever abstraction ──
class Retriever:
    """Thin abstraction over ChromaDB similarity search.

    Provides a clean .query() interface so the retrieval backend
    can be swapped later (e.g., FAISS, Pinecone) without touching
    the rest of the pipeline.
    """

    def __init__(self, vector_store, top_k: int = TOP_K):
        self.vector_store = vector_store
        self.top_k = top_k
        logger.info(f"Retriever initialised (top_k={top_k})")

    def query(self, question: str, top_k: int = None):
        """Return the top-k most relevant chunks for a question."""
        k = top_k or self.top_k
        results = self.vector_store.similarity_search_with_score(question, k=k)
        logger.debug(f"Retrieved {len(results)} chunk(s) for: '{question[:60]}...'")
        for i, (doc, score) in enumerate(results):
            logger.debug(f"  [{i}] score={score:.4f} | {doc.page_content[:80]}...")
        return results
