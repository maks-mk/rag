from typing import TypedDict

class RAGState(TypedDict):
    question:         str
    top_k:            int
    chunks:           list[tuple[str, dict, float, str, float]]
    answer:           str
    confidence:       float
    sources:          list[dict]
    tokens_used:      int | None
    retrieval_type:   str | None
    attempts:         int
    is_hallucinating: bool
