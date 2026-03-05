# --- START OF FILE config.py ---

import os

# API Configuration
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

# ВАЖНО: Используем модель, которая 100% подтверждена вашим ключом
LLM_MODEL       = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")
BYPASS_MODEL    = os.getenv("BYPASS_MODEL", "meta/llama-3.1-8b-instruct")
RERANK_MODEL    = os.getenv("RERANK_MODEL", "nvidia/llama-3.2-nv-rerankqa-1b-v2")

# Chunking Configuration
CHUNK_SIZE           = 800
CHUNK_OVERLAP        = 200
MIN_CHUNK_CHARS      = 100
MAX_CHUNKS_PER_DOC   = 2000
PENDING_TTL_SECONDS  = 3600
MIN_RELEVANCE_SCORE  = 0.15  

MAX_CONTEXT_TOKENS   = 12_000 
CHARS_PER_TOKEN      = 3.5
MAX_CHUNKS_PER_FILE  = 4
RRF_K                = 60
MIN_BM25_SCORE       = 0.5

BM25_DUMMY_DIST      = 1.0  

# Feature Flags для Advanced RAG
ENABLE_SELF_CHECK      = os.getenv("ENABLE_SELF_CHECK", "true").lower() == "true"
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"