import os

# API Configuration
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
LLM_MODEL       = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
BYPASS_MODEL    = os.getenv("BYPASS_MODEL", "minimaxai/minimax-m2.1")

# Chunking Configuration
CHUNK_SIZE           = 800   # Увеличен для сохранения семантики
CHUNK_OVERLAP        = 200
MIN_CHUNK_CHARS      = 100
MAX_CHUNKS_PER_DOC   = 2000
PENDING_TTL_SECONDS  = 3600
MIN_RELEVANCE_SCORE  = 0.15  # Чуть строже для базового фильтра

MAX_CONTEXT_TOKENS   = 12_000 
CHARS_PER_TOKEN      = 3.5
MAX_CHUNKS_PER_FILE  = 4
RRF_K                = 60    # Оптимальная константа RRF согласно исследованиям
MIN_BM25_SCORE       = 0.5

BM25_DUMMY_DIST      = 1.0  

# Feature Flags для Advanced RAG
ENABLE_SELF_CHECK      = os.getenv("ENABLE_SELF_CHECK", "true").lower() == "true"
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
ENABLE_LLM_GRADER      = os.getenv("ENABLE_LLM_GRADER", "true").lower() == "true"