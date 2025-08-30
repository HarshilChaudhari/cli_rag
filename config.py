# cli_rag/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# =========================
# CONFIG
# =========================

# LLM backend: "gemini" or "ollama"
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini").lower()

# GEMINI setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# OLLAMA setup
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Embedding model
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# Default paths (can override from CLI args)
PDF_PATHS = []
MEMORY_PATH = "./data/agent_memory.json"
