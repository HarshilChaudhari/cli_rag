# cli_rag/llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
from .config import LLM_MODEL, os

llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=os.environ["GOOGLE_API_KEY"])

def llm_complete(prompt: str, temperature: float = 0.2) -> str:
    out = llm.invoke(prompt)
    return out.content if hasattr(out, "content") else str(out)
