# cli_rag/llm.py
import os
import requests

# Pick backend: "gemini" or "ollama"
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini").lower()

# For Gemini (Google Generative AI)
import google.generativeai as genai
if LLM_BACKEND == "gemini":
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")


def llm_complete(prompt: str, temperature: float = 0.2) -> str:
    if LLM_BACKEND == "gemini":
        resp = gemini_model.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        return resp.text.strip()

    elif LLM_BACKEND == "ollama":
        # call local Ollama HTTP API
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "options": {"temperature": temperature},
        }
        response = requests.post(url, json=payload, stream=True)

        output = []
        for line in response.iter_lines():
            if not line:
                continue
            part = line.decode("utf-8")
            try:
                data = eval(part)  # JSON lines
                if "response" in data:
                    output.append(data["response"])
            except Exception:
                pass
        return "".join(output).strip()

    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")
