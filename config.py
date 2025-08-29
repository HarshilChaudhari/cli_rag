# cli_rag/config.py
import os
import google.generativeai as genai

# =========================
# CONFIG
# =========================
os.environ["GOOGLE_API_KEY"] = "AIzaSyDHSqtbTgyu2DuAHhIHYB_H8Nzs8_Hf1hA"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

LLM_MODEL = "gemini-2.0-flash"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Default paths (can override from CLI args)
PDF_PATHS = []
MEMORY_PATH = "./data/agent_memory.json"
