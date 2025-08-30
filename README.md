
# 📚 CLI RAG – PDF Question Answering with Gemini & Ollama

A lightweight **Retrieval-Augmented Generation (RAG) system** that lets you load PDFs, embed them into a FAISS vector store, and query them interactively using either **Google Gemini** (cloud) or **Ollama + Mistral/Gemma** (local).

---

## ✨ Features
- Load and chunk PDFs (PyMuPDF → OCR fallback with Tesseract).
- Store embeddings in SQLite + FAISS.
- Query interactively via CLI.
- Switch between **Gemini API** or **Ollama local LLM** with one environment variable.
- Skips already-processed PDFs (no duplicate work).
- Minimal dependencies & easy to extend.

---

## 📂 Project Structure
```
cli_rag/
│── __init__.py
│── agent.py        # RAG agent logic (retrieval + LLM)
│── cli.py          # Main CLI entrypoint
│── config.py       # Global config (models, API keys, defaults)
│── llm.py          # LLM backends: Gemini / Ollama
│── pdf_loader.py   # PDF → chunks (PyMuPDF → OCR fallback)
│── retrieval.py    # Hybrid retrieval + answer synthesis
│── store.py        # SQLite + FAISS vector store (persistence)
│── memory.py       # (optional) conversational memory helpers
│
├─ generated at runtime (repo root):
│   ├─ rag_store.db        # SQLite DB
│   └─ rag_index.faiss     # FAISS index

````

---

## ⚡ Setup

### 1. Clone repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
````

### 2. Install dependencies

```bash
pip install -q langgraph langchain langchain-community google-generativeai langchain-google-genai pypdf PyPDF2 sentence-transformers scikit-learn faiss-cpu

```

Typical requirements:

```txt
PyMuPDF
pdf2image
pytesseract
PyPDF2
faiss-cpu
sentence-transformers
google-generativeai
requests
```

(plus `poppler` and `tesseract-ocr` system packages if OCR is needed)

### 3. Configure API keys

Set your **Gemini API key** (if using Gemini):

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 4. (Optional) Run Ollama locally

Install [Ollama](https://ollama.ai/) and pull a model, e.g.:

```bash
ollama pull mistral
```

---

## 🚀 Usage

### Run with PDFs

```bash
python -m cli_rag.cli /path/to/paper.pdf /path/to/book.pdf
```

### Run without new PDFs (use existing DB)

```bash
python -m cli_rag.cli
```

### Example session

```
Loading PDFs into database...
📄 Added /content/NIPS-2017-attention-is-all-you-need-Paper.pdf (doc_id=51bfe776f7a7, 11 chunks)

Interactive RAG CLI. Type 'exit' to quit.

> What is the key contribution of "Attention is All You Need"?
...
```

---

## 🔀 Switching Backends

Choose between **Gemini** or **Ollama** by setting `LLM_BACKEND`:

```bash
# Use Gemini (default)
export LLM_BACKEND=gemini

# Use Ollama (local)
export LLM_BACKEND=ollama
```

---

## 🛠️ Development

### Branching

* `main` → stable (Gemini by default)
* `local-llm` → experiments with Ollama / Mistral

### Update dependencies

```bash
pip freeze > requirements.txt
```

---

## 📌 Roadmap

* [ ] Add multi-PDF querying (combine results from multiple docs).
* [ ] Add web UI (Streamlit / FastAPI).
* [ ] Improve OCR pipeline (parallel processing).
* [ ] Support more LLMs (Claude, LLaMA, OpenAI GPT).

---


