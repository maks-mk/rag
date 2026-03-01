import os

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL     = "nvidia/nv-embedqa-e5-v5"
LLM_MODEL       = "openai/gpt-oss-20b"

CHUNK_SIZE           = 600
CHUNK_OVERLAP        = 120
MIN_CHUNK_CHARS      = 100
MAX_CHUNKS_PER_DOC   = 1000
PENDING_TTL_SECONDS  = 3600
MIN_RELEVANCE_SCORE  = 0.10

MAX_CONTEXT_TOKENS   = 10_000
CHARS_PER_TOKEN      = 3.5
MAX_CHUNKS_PER_FILE  = 3
RRF_K                = 20
MIN_BM25_SCORE       = 0.5

# FIX: Вынесено в константу для предотвращения случайной фильтрации в grade_node
BM25_DUMMY_DIST      = 1.0  
# FIX: Флаг для отключения дорогого Self-Check (по умолчанию включен)
ENABLE_SELF_CHECK    = os.getenv("ENABLE_SELF_CHECK", "true").lower() == "true"
