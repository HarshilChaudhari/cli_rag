# cli_rag/llm.py
import os
import requests
import json
import google.generativeai as genai

from .config import (
    LLM_BACKEND,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_MODEL,
    OLLAMA_URL,
)

# =========================
# Setup LLM Backends
# =========================
if LLM_BACKEND == "gemini":
    if not GEMINI_API_KEY:
        raise ValueError("❌ GEMINI_API_KEY is missing. Set it in your environment.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)


def llm_complete(prompt: str, temperature: float = 0.2) -> str:
    """Send prompt to the configured LLM backend."""
    if LLM_BACKEND == "gemini":
        resp = gemini_model.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        return resp.text.strip()

    elif LLM_BACKEND == "ollama":
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "options": {"temperature": temperature},
        }
        response = requests.post(OLLAMA_URL, json=payload, stream=True)

        output = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    output.append(data["response"])
            except Exception:
                pass
        return "".join(output).strip()

    else:
        raise ValueError(f"❌ Unknown LLM_BACKEND: {LLM_BACKEND}")
