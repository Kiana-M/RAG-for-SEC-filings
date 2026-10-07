"""Settings: paths, model names, chunking. Secrets come from .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
load_dotenv(REPO / ".env")

DATA_DIR = REPO / "data" / "edgar_corpus"
MANIFEST = REPO / "data" / "manifest.json"
COMPANIES = ROOT / "companies.json"
COVERAGE_CSV = ROOT / "coverage.csv"

# Chunking
CHUNK_TOKENS = 800
CHUNK_OVERLAP = 100
SECTIONS = ["business", "risk_factors", "legal", "mdna", "market_risk", "financials", "controls", "other"]

# Index
INDEX_DIR = ROOT / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_PATH = INDEX_DIR / "bm25.pkl"
COLLECTION = "filings"

# Models (swappable; see llm.py)
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")  # "local" (sentence-transformers), "gemini" or "openai"
EMBED_MODEL = os.getenv("EMBED_MODEL", "Alibaba-NLP/gte-modernbert-base")
EMBED_MAX_TOKENS = 1024  # model tokens; an 800-token (cl100k) chunk fits whole
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "")  # e.g. "search_query: " for nomic models
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))  # gemini only
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))  # chunks per upsert
DEFAULT_LLM_MODELS = {"gemini": "gemini-flash-latest", "anthropic": "claude-opus-5-5", "openai": ""}
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" (GEMINI_API_KEY), "anthropic" (ANTHROPIC_API_KEY), "openai"
LLM_MODEL = os.getenv("LLM_MODEL") or DEFAULT_LLM_MODELS[LLM_PROVIDER]
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "medium")  # low | medium | high | xhigh | max
LLM_FALLBACK_MODELS = [m for m in os.getenv("LLM_FALLBACK_MODELS", "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-flash-lite-latest,gemini-3.1-flash-lite").split(",") if m]
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))  # seconds to wait for a Gemini reply before retrying / falling back
LLM_CACHE = os.getenv("LLM_CACHE", "1") == "1"  # cache responses on disk (identical prompt -> no API call)
CACHE_DIR = ROOT / "cache"
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", LLM_PROVIDER)
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or (LLM_MODEL if JUDGE_PROVIDER == LLM_PROVIDER else DEFAULT_LLM_MODELS[JUDGE_PROVIDER])

# Answering
ANSWER_MODE = os.getenv("ANSWER_MODE", "single")  # "single": one LLM call per answer; "multi": per-company + compare

# Retrieval
TOP_K = 20  # per retriever (dense and BM25) before fusion
RRF_K = 60
PER_TICKER = 5  # chunks kept per ticker (per ticker and year for temporal questions)
RERANK = os.getenv("RERANK", "0") == "1"  # Cohere rerank, needs COHERE_API_KEY
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-v3.5")
