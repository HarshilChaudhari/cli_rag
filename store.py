# cli_rag/store.py
import os
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from .config import EMBED_MODEL_NAME

DB_PATH = "rag_store.db"
INDEX_PATH = "rag_index.faiss"

# Global embedder
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# ---------- SQLite Helpers ----------

def get_db():
    """Return a connection to the SQLite database with row dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT,
        page INTEGER,
        text TEXT
    )
    """)
    conn.commit()
    conn.close()


def add_chunks(doc_id: str, chunks: List[Dict]):
    """Insert chunks into SQLite if not already present."""
    conn = get_db()
    cur = conn.cursor()

    for c in chunks:
        cur.execute("INSERT INTO chunks (doc_id, page, text) VALUES (?, ?, ?)",
                    (doc_id, c["page"], c["text"]))

    conn.commit()
    conn.close()


def get_chunks_by_ids(ids: List[int]) -> List[Dict]:
    if not ids:
        return []
    conn = get_db()
    cur = conn.cursor()
    q_marks = ",".join("?" * len(ids))
    cur.execute(f"SELECT id, doc_id, page, text FROM chunks WHERE id IN ({q_marks})", ids)
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "doc_id": r[1], "page": r[2], "text": r[3]} for r in rows]


def get_chunks_for_doc(doc_id: str) -> List[Dict]:
    """Fetch all chunks belonging to a given document."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, doc_id, page, text FROM chunks WHERE doc_id=?", (doc_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "doc_id": r[1], "page": r[2], "text": r[3]} for r in rows]


def has_doc(doc_id: str) -> bool:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM chunks WHERE doc_id=? LIMIT 1", (doc_id,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def is_doc_processed(doc_id: str) -> bool:
    """Alias of has_doc, used by cli.py for clarity."""
    return has_doc(doc_id)


# ---------- FAISS Helpers ----------

def init_faiss(dim: int = 768) -> faiss.Index:
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        base_index = faiss.IndexFlatIP(dim)  # cosine similarity (after normalization)
        index = faiss.IndexIDMap(base_index)  # supports add_with_ids
    return index


def save_faiss(index: faiss.Index):
    faiss.write_index(index, INDEX_PATH)


def add_embeddings(doc_id: str, chunks: List[Dict]):
    texts = [c["text"] for c in chunks]
    if not texts:
        return
    embs = embedder.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    index = init_faiss(embs.shape[1])

    # Map FAISS vector IDs to SQLite row IDs
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))
    row_ids = [r[0] for r in cur.fetchall()]
    conn.close()

    if len(row_ids) != len(embs):
        raise ValueError("Mismatch: row count != embeddings count")

    index.add_with_ids(np.array(embs, dtype="float32"), np.array(row_ids, dtype="int64"))
    save_faiss(index)


def search_embeddings(query: str, top_k: int = 6) -> List[Dict]:
    q_emb = embedder.encode([query], convert_to_tensor=False, normalize_embeddings=True)[0]
    index = init_faiss(len(q_emb))
    if index.ntotal == 0:
        return []

    q_emb = np.array([q_emb], dtype="float32")
    scores, ids = index.search(q_emb, top_k)
    ids = ids[0].tolist()
    scores = scores[0].tolist()

    results = get_chunks_by_ids([i for i in ids if i != -1])
    id2score = {i: s for i, s in zip(ids, scores)}
    for r in results:
        r["score"] = float(id2score.get(r["id"], 0.0))
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
