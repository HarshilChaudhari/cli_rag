# cli_rag/memory.py
import os, json, time
from typing import List, Dict, Any
from .config import MEMORY_PATH

def _load_memory() -> Dict[str, Any]:
    if os.path.exists(MEMORY_PATH) and os.path.getsize(MEMORY_PATH) > 0:
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    return {}

def _save_memory(mem: Dict[str, Any]):
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w") as f:
        json.dump(mem, f, indent=2)

memory_store = _load_memory()
memory_store.setdefault("GLOBAL", {"doc_id": "GLOBAL", "facts": [], "history": []})
_save_memory(memory_store)

def add_memory_facts(doc_id: str, facts: List[str]):
    mem = _load_memory()
    mem.setdefault(doc_id, {"doc_id": doc_id, "facts": [], "history": []})
    existing = set(mem[doc_id]["facts"])
    for f in facts:
        f = f.strip()
        if f and f not in existing:
            mem[doc_id]["facts"].append(f)
            existing.add(f)
    _save_memory(mem)

def add_history_turn(doc_id: str, user_q: str, answer: str):
    mem = _load_memory()
    mem.setdefault(doc_id, {"doc_id": doc_id, "facts": [], "history": []})
    mem[doc_id]["history"].append({"q": user_q, "a": answer, "ts": time.time()})
    mem[doc_id]["history"] = mem[doc_id]["history"][-10:]
    _save_memory(mem)

def get_doc_memory(doc_id: str):
    return _load_memory().get(doc_id, {"facts": [], "history": []})

def get_global_memory():
    return _load_memory().get("GLOBAL", {"facts": [], "history": []})
