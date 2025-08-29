# cli_rag/pdf_loader.py
import hashlib
from typing import List, Dict
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader

from .store import (
    init_db,
    has_doc,
    add_chunks,
    add_embeddings,
    embedder,
)


def extract_text_pymupdf(pdf_path: str) -> str:
    """Extract text using PyMuPDF (fast & reliable)."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text.strip()
    except Exception as e:
        print(f"⚠️ PyMuPDF failed for {pdf_path}: {e}")
        return ""


def extract_text_pypdf2(pdf_path: str) -> str:
    """Fallback extractor using PyPDF2."""
    try:
        reader = PdfReader(pdf_path)
        text = "".join([p.extract_text() or "" for p in reader.pages])
        return text.strip()
    except Exception as e:
        print(f"⚠️ PyPDF2 failed for {pdf_path}: {e}")
        return ""


def extract_text_ocr(pdf_path: str) -> str:
    """Fallback extractor using OCR (very slow)."""
    try:
        pages = convert_from_path(pdf_path)
        text = ""
        for img in pages:
            text += pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"⚠️ OCR failed for {pdf_path}: {e}")
        return ""


def extract_text(pdf_path: str) -> str:
    """Unified text extraction with fallbacks."""
    text = extract_text_pymupdf(pdf_path)
    if text:
        print(f"✔️ Extracted text with PyMuPDF: {pdf_path}")
        return text

    text = extract_text_pypdf2(pdf_path)
    if text:
        print(f"✔️ Extracted text with PyPDF2: {pdf_path}")
        return text

    print(f"⚠️ No text via PyMuPDF/PyPDF2, falling back to OCR for {pdf_path}")
    text = extract_text_ocr(pdf_path)
    if text:
        print(f"✔️ Extracted text with OCR: {pdf_path}")
    else:
        print(f"❌ Failed to extract any text from {pdf_path}")
    return text


def chunk_text(text: str, chunk_size_words=1000, overlap_words=500) -> List[Dict]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks, cid = [], 0
    for i in range(0, len(words), chunk_size_words - overlap_words):
        piece = " ".join(words[i:i+chunk_size_words]).strip()
        if piece:
            chunks.append({"id": cid, "page": None, "text": piece})
            cid += 1
    return chunks


def compute_doc_id(text: str) -> str:
    """Stable hash based on raw text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def load_pdfs(pdf_paths: List[str]) -> List[Dict]:
    """
    Load and persist PDFs into the DB + FAISS index.
    Returns metadata for all newly added documents.
    """
    init_db()
    new_docs = []

    for path in pdf_paths:
        text = extract_text(path)
        if not text:
            print(f"⚠️ Skipping {path}, no text extracted.")
            continue

        doc_id = compute_doc_id(text)

        # Skip if already in DB
        if has_doc(doc_id):
            print(f"✔️ Already processed: {path} (doc_id={doc_id})")
            continue

        # Now chunk properly
        chunks = chunk_text(text)
        for c in chunks:
            c["doc_id"] = doc_id

        add_chunks(doc_id, chunks)
        add_embeddings(doc_id, chunks)

        new_docs.append({"doc_id": doc_id, "n_chunks": len(chunks), "path": path})
        print(f"📄 Added {path} (doc_id={doc_id}, {len(chunks)} chunks)")

    return new_docs
