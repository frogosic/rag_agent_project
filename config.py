from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
CHROMA_PATH = str(BASE_DIR / "chroma_data")
BOOKS_DIR = str(BASE_DIR / "books")
POLICIES_DIR = str(BASE_DIR / "policies")

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
GPT_MODEL_ID = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# LLM generation params
# ---------------------------------------------------------------------------
TEMPERATURE = 0.3
MAX_TOKENS = 2048

# ---------------------------------------------------------------------------
# Retrieval params — per domain
# ---------------------------------------------------------------------------

# QA policies — short, dense, factual
# strict similarity, small chunks, low k
QA_CHUNK_SIZE = 512
QA_CHUNK_OVERLAP = 80
QA_SIMILARITY_THRESHOLD = 0.70
QA_TOP_K = 3

# Fiction — narrative, contextual, interpretive
# looser similarity, larger chunks, higher k to capture context
FICTION_CHUNK_SIZE = 1024
FICTION_CHUNK_OVERLAP = 150
FICTION_SIMILARITY_THRESHOLD = 0.65
FICTION_TOP_K = 4

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
INTENT_CONFIDENCE_THRESHOLD = 0.7  # below this → hybrid branch

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVALUATION_SAMPLE_SIZE = 10  # number of queries to evaluate per domain
