"""
pdf_loader.py — Load PDF documents and split into chunks.

Pipeline stage: PDF file  →  list of LangChain Document chunks
"""
import glob
import logging
from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_pdfs(pdf_dir: str = PDF_DIR):
    """Load all PDFs from the given directory and return raw Document list."""
    pdf_files = glob.glob(f"{pdf_dir}/*.pdf")
    logger.info(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}")

    all_docs = []
    for path in pdf_files:
        loader = PyPDFLoader(path)
        docs = loader.load()
        logger.info(f"  Loaded '{path}' — {len(docs)} page(s)")
        all_docs.extend(docs)
    return all_docs


def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} page(s) into {len(chunks)} chunk(s) "
                f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks
