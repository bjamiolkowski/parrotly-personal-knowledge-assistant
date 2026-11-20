from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"

INDEX_PATH = STORAGE_DIR / "index.faiss"
CHUNKS_PATH = STORAGE_DIR / "chunks.pkl"

EMBED_MODEL = "nomic-embed-text"

DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "ollama")

OLLAMA_BASE_URL =  OLLAMA_BASE_URL = "http://localhost:11434" #"http://host.docker.internal:11434"

OLLAMA_GEN_MODEL = "llama3.1:8b"
OPENAI_GEN_MODEL = "gpt-4o-mini"

GEN_MODEL = os.getenv("GEN_MODEL", OLLAMA_GEN_MODEL)

DEFAULT_TOP_K = 5
DEFAULT_FAISS_K = 20
DEFAULT_TFIDF_K = 20
DEFAULT_ALPHA = 0.6

MIN_RESULTS = 2
MIN_HYBRID_SCORE = 0.15
MIN_CONTEXT_SCORE = 0.10
MAX_CONTEXT_CHARS = 1800
MAX_HISTORY_TURNS = 4

MODEL_PRICING_USD_PER_1M_TOKENS = {
    OLLAMA_GEN_MODEL: {
        "input": 0.0,
        "output": 0.0,
    },

    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },

    "gpt-4.1-mini": {
        "input": 0.40,
        "output": 1.60,
    },
}

GENERATION_MODES = {
    "cheap": {
        "top_k": 3,
        "max_context_chars": 1000,
        "provider": DEFAULT_PROVIDER,
        "model": GEN_MODEL,
    },
    "balanced": {
        "top_k": 5,
        "max_context_chars": 1800,
        "provider": DEFAULT_PROVIDER,
        "model": GEN_MODEL,
    },
    "accurate": {
        "top_k": 8,
        "max_context_chars": 3000,
        "provider": DEFAULT_PROVIDER,
        "model": GEN_MODEL,
    },
}