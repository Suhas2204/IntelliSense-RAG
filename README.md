# IntelliSense RAG

**RAG-based Conversational Analytics Platform for Continental BLE Tire Sensor Documentation**

Internal tool to help new team members understand sensor systems without reading dozens of PDFs manually. Ask questions in natural language — get grounded, accurate answers.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Sensor PDF  │────▶│  PDF Loader  │────▶│   Text Chunks  │────▶│  Embeddings  │
│  Documents   │     │  (PyPDF)     │     │  (512 tokens,  │     │  (MiniLM-L6) │
└─────────────┘     └──────────────┘     │   50 overlap)  │     └──────┬───────┘
                                          └────────────────┘            │
                                                                        ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│   Answer     │◀────│  Qwen LLM    │◀────│  Retrieved     │◀────│  ChromaDB    │
│  (grounded)  │     │  (HF API)    │     │  Context       │     │  Vector DB   │
└──────┬──────┘     └──────────────┘     │  (top-k=3)     │     └──────────────┘
       │                                  └────────────────┘
       ▼
┌─────────────┐
│  PySide6    │
│  Chat UI    │
└─────────────┘
```

## Project Structure

```
intellisense-rag/
├── docs/
│   └── sensor_manual.pdf       # Continental BLE sensor technical manual
├── src/
│   ├── config.py               # All tuneable parameters (paths, models, thresholds)
│   ├── pdf_loader.py           # PDF loading + recursive chunking
│   ├── vector_store.py         # Embedding + ChromaDB + Retriever class
│   ├── llm_chain.py            # Qwen LLM + LangChain prompt template
│   ├── rag_pipeline.py         # RAGPipeline class — orchestrates all stages
│   └── ui.py                   # PySide6 chat window
├── app.py                      # Entry point
├── pyproject.toml
└── README.md
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install dependencies
uv sync

# 3. Set HuggingFace token (free — get one at huggingface.co/settings/tokens)
export HF_TOKEN="hf_your_token_here"

# 4. Run
python app.py
```

## Example Queries

| Question | What it tests |
|----------|---------------|
| "What sensor variants does Continental offer?" | Basic retrieval |
| "What is the BLE packet format?" | Technical detail retrieval |
| "What pressure is considered critical?" | Threshold lookup |
| "How does data flow from sensor to cloud?" | Pipeline understanding |
| "What should I do if there's no BLE signal?" | Troubleshooting retrieval |
| "What does RSSI stand for?" | Glossary retrieval |

## Key Design Decisions

- **Chunk size 512 with 50 overlap**: Optimal for technical sensor docs — large enough to capture full table rows, small enough for precise retrieval.
- **Retriever abstraction**: Backend-agnostic `.query()` interface — can swap ChromaDB for FAISS or Pinecone without touching pipeline logic.
- **Grounded system prompt**: LLM instructed to answer ONLY from retrieved context, preventing hallucination about sensor specs.
- **Background threading**: QThread keeps the UI responsive during LLM inference.

## Tech Stack

- **LangChain** — Pipeline orchestration & prompt engineering
- **ChromaDB** — Local vector database
- **sentence-transformers** — Free local embeddings (all-MiniLM-L6-v2)
- **Qwen2.5-7B** — LLM via HuggingFace Inference API (free)
- **PySide6** — Desktop chat UI
- **PyPDF** — PDF parsing
