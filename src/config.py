"""
config.py — Central configuration for IntelliSense RAG.
All tuneable parameters live here. No magic numbers elsewhere.
"""
import os

# ── Paths ──
PDF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# ── Chunking ──
CHUNK_SIZE = 512          # tokens per chunk  (sweet spot for technical docs)
CHUNK_OVERLAP = 50        # overlap to preserve context across chunk boundaries

# ── Embedding model  (runs locally, no API key needed) ──
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Retrieval ──
TOP_K = 3                 # number of chunks to retrieve per query
COLLECTION_NAME = "sensor_docs"

# ── LLM (Qwen via HuggingFace Inference API — free tier) ──
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
LLM_TEMPERATURE = 0.3     # low = more factual, less creative
LLM_MAX_TOKENS = 512

# ── HuggingFace token (set via environment variable) ──
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── System prompt — grounds answers in retrieved context only ──
SYSTEM_PROMPT = (
    "You are IntelliSense, an internal Continental AG assistant. "
    "Answer questions about BLE tire sensors ONLY using the provided context. "
    "If the context does not contain the answer, say 'I could not find this "
    "information in the sensor documentation.' Be concise and precise."
)
