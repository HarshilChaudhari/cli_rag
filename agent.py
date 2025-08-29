# cli_rag/agent.py
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END

from .retrieval import hybrid_search, multi_query_expand, synthesize_answer, memory_update_from_answer
from .memory import get_global_memory, add_history_turn, add_memory_facts


class AgentState(TypedDict):
    question: str
    queries: List[str]
    passages: List[Dict[str, Any]]
    answer: str
    memory: Dict[str, Any]
    doc_ids: List[str]


def node_expand(state: AgentState) -> AgentState:
    global_mem = get_global_memory()
    queries = multi_query_expand(state["question"], global_mem["history"], global_mem["facts"])
    state["queries"] = queries
    return state


def node_search(state: AgentState) -> AgentState:
    pool: Dict[int, Dict[str, Any]] = {}

    for i, q in enumerate(state["queries"]):
        hits = hybrid_search(q)  # now queries DB via store.py

        # Weight: original query > expansions
        weight = 1.0 if i == 0 else 0.85
        for h in hits:
            h = h.copy()  # avoid mutating shared dicts
            h["score"] *= weight

            if h["id"] not in pool or h["score"] > pool[h["id"]]["score"]:
                pool[h["id"]] = h

    # Pick top passages
    passages = sorted(pool.values(), key=lambda x: x["score"], reverse=True)[:8]

    # Update agent state
    state["passages"] = passages
    state["doc_ids"] = list({p["doc_id"] for p in passages})
    return state


def node_answer(state: AgentState) -> AgentState:
    global_mem = get_global_memory()
    answer = synthesize_answer(
        state["question"],
        state["passages"],
        global_mem["facts"],
        global_mem["history"],
    )
    state["answer"] = answer
    return state


def node_mem_update(state: AgentState) -> AgentState:
    add_history_turn("GLOBAL", state["question"], state["answer"])
    facts = memory_update_from_answer(state["question"], state["answer"])
    if facts:
        add_memory_facts("GLOBAL", facts)
        for doc_id in state["doc_ids"]:
            add_memory_facts(doc_id, facts)
    state["memory"] = get_global_memory()
    return state


# Build graph
graph = StateGraph(AgentState)
graph.add_node("expand", node_expand)
graph.add_node("search", node_search)
graph.add_node("answer", node_answer)
graph.add_node("mem_update", node_mem_update)

graph.add_edge(START, "expand")
graph.add_edge("expand", "search")
graph.add_edge("search", "answer")
graph.add_edge("answer", "mem_update")
graph.add_edge("mem_update", END)

agent_app = graph.compile()


def ask(question: str, show_sources: bool = True, excerpt_chars: int = 500) -> str:
    state: AgentState = {
        "question": question,
        "queries": [],
        "passages": [],
        "answer": "",
        "memory": {},
        "doc_ids": [],
    }
    out = agent_app.invoke(state)
    answer = out["answer"].strip()

    if show_sources:
        src_lines = ["", "Sources:"]
        for i, p in enumerate(out["passages"], start=1):
            snippet = " ".join(p["text"].split())
            if excerpt_chars:
                snippet = (snippet[:excerpt_chars] + "…") if len(snippet) > excerpt_chars else snippet
            src_lines.append(f"- [P{i}] Doc {p['doc_id']} | Page {p['page']} | score={p['score']:.3f}")
            src_lines.append(f'  "{snippet}"')
        answer += "\n" + "\n".join(src_lines)

    return answer
