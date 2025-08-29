# cli_rag/pdf_loader.py
import hashlib
from typing import List, Dict
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract

from .store import (
    init_db,
    has_doc,
    add_chunks,
    add_embeddings,
    embedder,
)


def extract_text_pymupdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text.strip()


def extract_text_ocr(pdf_path: str) -> str:
    """Extract text from PDF using OCR (slower)."""
    pages = convert_from_path(pdf_path)
    text = ""
    for img in pages:
        text += pytesseract.image_to_string(img)
    return text.strip()


def load_pdf_chunks(pdf_path: str, chunk_size_words=1000, overlap_words=500) -> List[Dict]:
    """Split PDF into overlapping text chunks (fallbacks: PyMuPDF → OCR)."""
    text = extract_text_pymupdf(pdf_path)

    if not text:
        print(f"⚠️ No text via PyMuPDF, falling back to OCR for {pdf_path}")
        text = extract_text_ocr(pdf_path)

    if not text:
        print(f"❌ Still no text extracted from {pdf_path}")
        return []

    words = text.split()
    chunks, cid = [], 0
    for i in range(0, len(words), chunk_size_words - overlap_words):
        piece = " ".join(words[i:i+chunk_size_words]).strip()
        if piece:
            chunks.append({"id": cid, "page": None, "text": piece})
            cid += 1
    return chunks


def compute_doc_id(chunks: List[Dict]) -> str:
    """Stable hash based on content to identify a PDF."""
    raw_text = "\n".join(c["text"] for c in chunks)
    return hashlib.md5(raw_text.encode("utf-8")).hexdigest()[:12]


def load_pdfs(pdf_paths: List[str]) -> List[Dict]:
    """
    Load and persist PDFs into the DB + FAISS index.
    Returns metadata for all newly added documents.
    """
    init_db()
    new_docs = []

    for path in pdf_paths:
        chunks = load_pdf_chunks(path)
        if not chunks:
            continue

        doc_id = compute_doc_id(chunks)

        # Skip if already stored
        if has_doc(doc_id):
            continue

        # Add doc_id field
        for c in chunks:
            c["doc_id"] = doc_id

        # Store in DB + FAISS
        add_chunks(doc_id, chunks)
        add_embeddings(doc_id, chunks)

        new_docs.append({"doc_id": doc_id, "n_chunks": len(chunks), "path": path})

    return new_docs
