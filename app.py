"""
app.py — Entry point for IntelliSense RAG.

Usage:
    python app.py
"""
import sys
import logging
from PySide6.QtWidgets import QApplication
from src.rag_pipeline import RAGPipeline
from src.ui import ChatWindow

# ── Configure logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    # Step 1: Build the RAG pipeline (loads PDFs, builds index, connects LLM)
    pipeline = RAGPipeline()

    # Step 2: Launch PySide6 chat UI
    app = QApplication(sys.argv)
    window = ChatWindow(pipeline)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
