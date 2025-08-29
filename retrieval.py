# cli_rag/retrieval.py
from typing import List, Dict, Any
import json, textwrap, sqlite3

from .store import search_embeddings, get_db, get_chunks_for_doc, embedder
from .llm import llm_complete


def keyword_search(query: str, top_k: int = 6) -> List[Dict]:
    """Naive keyword search fallback using SQLite LIKE."""
    conn = get_db()
    cur = conn.cursor()
    like_q = f"%{query}%"
    cur.execute(
        "SELECT id, doc_id, page, text FROM chunks WHERE text LIKE ? LIMIT ?",
        (like_q, top_k),
    )
    rows = cur.fetchall()
    return [
        {"id": r[0], "doc_id": r[1], "page": r[2], "text": r[3], "bm25": 1.0}
        for r in rows
    ]


def hybrid_search(query: str, top_k: int = 6) -> List[Dict]:
    """Combine semantic (FAISS) + keyword (SQLite LIKE) search."""
    sem_hits = search_embeddings(query, top_k=top_k)
    bm25_hits = keyword_search(query, top_k=top_k)

    pool = {}

    # Semantic results
    for h in sem_hits:
        hid = h["id"]
        if hid not in pool:
            pool[hid] = h.copy()
            pool[hid]["sem"] = h.get("score", 0.0)
            pool[hid]["bm25"] = 0.0
        else:
            pool[hid]["sem"] = max(pool[hid].get("sem", 0.0), h.get("score", 0.0))

    # Keyword results
    for h in bm25_hits:
        hid = h["id"]
        if hid not in pool:
            pool[hid] = h.copy()
            pool[hid]["bm25"] = h.get("bm25", 0.0)
            pool[hid]["sem"] = 0.0
        else:
            pool[hid]["bm25"] = max(pool[hid].get("bm25", 0.0), h.get("bm25", 0.0))

    # Fusion scoring
    results = list(pool.values())
    for r in results:
        r["score"] = 0.6 * r.get("sem", 0.0) + 0.4 * r.get("bm25", 0.0)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# -------- Smarter query expansion --------

def multi_query_expand(question: str, history: List[Dict[str, str]], facts: List[str]) -> List[str]:
    """
    Expands a user query into diverse variations.
    """
    hist_snips = []
    for turn in history[-4:]:
        hist_snips.append(f"Q: {turn['q']}\nA: {turn['a'][:200]}")
    facts_snip = "\n".join(f"- {f}" for f in facts[:6])

    prompt = f"""
Expand the user's question into 4–6 different short search queries.

User's question:
{question}

Recent conversation:
{chr(10).join(hist_snips) if hist_snips else "(no prior turns)"}

Known facts:
{facts_snip if facts_snip else "(none)"}

Rules:
- Each query < 15 words
- Must be diverse:
  1. direct rephrase
  2. keyword-only version
  3. synonyms/paraphrase
  4. context-aware version (use history/facts if useful)
  5. optional broader/narrower scope
- Return only a JSON array of strings.
"""
    try:
        raw = llm_complete(prompt, temperature=0.3)
        start, end = raw.find("["), raw.rfind("]")
        arr = json.loads(raw[start:end+1])
        out = [q.strip() for q in arr if isinstance(q, str) and q.strip()]

        if question not in out:
            out = [question] + out

        uniq, seen = [], set()
        for q in out:
            if q.lower() not in seen:
                uniq.append(q)
                seen.add(q.lower())

        return uniq[:6]

    except Exception:
        kw = " ".join([w for w in question.split() if len(w) > 3])
        return [question, kw]


ANSWER_SYSTEM = """You are a helpful assistant that answers ONLY using provided passages."""

def synthesize_answer(question: str, passages: List[Dict[str, Any]], memory_facts: List[str], history: List[Dict[str, str]]) -> str:
    history_snips = []
    for turn in history[-4:]:
        history_snips.append(f"Q: {turn['q']}\nA: {turn['a'][:200]}")

    passages_text = []
    for i, p in enumerate(passages, start=1):
        excerpt = textwrap.shorten(" ".join(p["text"].split()), width=1200, placeholder=" …")
        passages_text.append(f"[P{i} | Doc {p['doc_id']} | Page {p['page']}]\n{excerpt}")

    facts_snip = "\n".join(f"- {f}" for f in memory_facts[:6]) if memory_facts else "(none)"
    history_text = "\n".join(history_snips) if history_snips else "(none)"

    prompt = f"""
You are a helpful assistant.

Current question:
{question}

Conversation history:
{history_text}

Relevant passages from PDFs:
{chr(10).join(passages_text) if passages_text else "(none)"}

Known document facts:
{facts_snip}

Instructions:
- Prefer answering from history if question refers to prior turns.
- Otherwise, answer using the provided passages.
- Keep it concise and clear.
"""
    return llm_complete(prompt, temperature=0.2)


def memory_update_from_answer(question: str, answer_text: str) -> List[str]:
    prompt = f"""
From the answer below, extract up to 3 short, document-specific facts.
JSON array only.

Answer:
{answer_text}
"""
    try:
        raw = llm_complete(prompt)
        start, end = raw.find("["), raw.rfind("]")
        arr = json.loads(raw[start:end+1])
        return [s.strip() for s in arr if isinstance(s, str) and s.strip()][:3]
    except Exception:
        return []
