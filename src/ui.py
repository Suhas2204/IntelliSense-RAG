"""
ui.py — PySide6 Chat Interface for IntelliSense RAG.

Clean chat window: scrollable message history + input field.
Runs the RAG pipeline in a background thread to keep UI responsive.
"""
import logging
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor

logger = logging.getLogger(__name__)


# ── Background worker so the UI doesn't freeze during LLM calls ──
class QueryWorker(QThread):
    """Runs RAG pipeline.ask() in a separate thread."""
    finished = Signal(str)

    def __init__(self, pipeline, question):
        super().__init__()
        self.pipeline = pipeline
        self.question = question

    def run(self):
        try:
            answer = self.pipeline.ask(self.question)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            answer = f"Error: {e}"
        self.finished.emit(answer)


class ChatWindow(QMainWindow):
    """Main chat window for IntelliSense RAG."""

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("IntelliSense RAG — Continental Sensor Assistant")
        self.setMinimumSize(700, 500)

        # ── Central widget ──
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # ── Header ──
        header = QLabel("IntelliSense RAG")
        header.setFont(QFont("Segoe UI", 18, QFont.Bold))
        header.setStyleSheet("color: #D4500F;")
        layout.addWidget(header)

        subtitle = QLabel("Ask questions about Continental BLE tire sensor documentation")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: #666;")
        layout.addWidget(subtitle)

        # ── Chat history ──
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 10))
        self.chat_display.setStyleSheet(
            "QTextEdit {"
            "  background-color: #1E1E1E;"
            "  color: #D4D4D4;"
            "  border: 1px solid #333;"
            "  border-radius: 8px;"
            "  padding: 12px;"
            "}"
        )
        layout.addWidget(self.chat_display, stretch=1)

        # ── Input row ──
        input_row = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your question here...")
        self.input_field.setFont(QFont("Segoe UI", 11))
        self.input_field.setStyleSheet(
            "QLineEdit {"
            "  padding: 10px;"
            "  border: 1px solid #CCC;"
            "  border-radius: 6px;"
            "}"
        )
        self.input_field.returnPressed.connect(self._on_send)
        input_row.addWidget(self.input_field, stretch=1)

        self.send_btn = QPushButton("Send")
        self.send_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.send_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #D4500F;"
            "  color: white;"
            "  padding: 10px 24px;"
            "  border: none;"
            "  border-radius: 6px;"
            "}"
            "QPushButton:hover { background-color: #B8440D; }"
            "QPushButton:disabled { background-color: #999; }"
        )
        self.send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self.send_btn)

        layout.addLayout(input_row)

        # ── Welcome message ──
        self._append_message("IntelliSense", "Ready! Ask me anything about the sensor documentation.", "#D4500F")

    def _on_send(self):
        question = self.input_field.text().strip()
        if not question:
            return

        self._append_message("You", question, "#569CD6")
        self.input_field.clear()

        # Disable input while processing
        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)
        self._append_message("IntelliSense", "Thinking...", "#888")

        # Run in background thread
        self.worker = QueryWorker(self.pipeline, question)
        self.worker.finished.connect(self._on_answer)
        self.worker.start()

    def _on_answer(self, answer: str):
        # Remove "Thinking..." line
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        cursor.movePosition(cursor.StartOfBlock, cursor.KeepAnchor)
        cursor.movePosition(cursor.PreviousBlock, cursor.KeepAnchor)
        cursor.removeSelectedText()

        self._append_message("IntelliSense", answer, "#D4500F")
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()

    def _append_message(self, sender: str, text: str, color: str):
        self.chat_display.append(
            f'<p style="margin:4px 0;">'
            f'<span style="color:{color}; font-weight:bold;">{sender}:</span> '
            f'<span style="color:#D4D4D4;">{text}</span>'
            f'</p>'
        )
        # Auto-scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
