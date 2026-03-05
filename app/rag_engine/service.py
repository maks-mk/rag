# --- START OF FILE service.py ---

import os
import time
import logging
import asyncio
import re
import httpx
from datetime import datetime, timezone
from typing import List, Optional

import chromadb
import numpy as np
from openai import AsyncOpenAI, APITimeoutError, APIStatusError
from rank_bm25 import BM25Okapi
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from .config import (
    NVIDIA_BASE_URL,
    EMBED_MODEL,
    LLM_MODEL,
    BYPASS_MODEL,
    RERANK_MODEL,
    MAX_CHUNKS_PER_DOC,
    PENDING_TTL_SECONDS,
    MIN_RELEVANCE_SCORE,
    MAX_CONTEXT_TOKENS,
    MAX_CHUNKS_PER_FILE,
    RRF_K,
    MIN_BM25_SCORE,
    BM25_DUMMY_DIST,
    ENABLE_SELF_CHECK,
    ENABLE_QUERY_EXPANSION
)
from .text_processor import count_tokens, chunk_text, extract_text
from .state import RAGState

logger = logging.getLogger(__name__)
TOKEN_SPLIT_RE = re.compile(r"\W+")

class RAGService:
    def __init__(self):
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            logger.error("NVIDIA_API_KEY is not set! API calls will likely fail.")

        self._openai = AsyncOpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=api_key or "placeholder",
        )

        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=NVIDIA_BASE_URL,
            api_key=api_key or "placeholder",
            temperature=0.2,
            max_tokens=3000,
        )

        # Bypass LLM для прямого общения с моделью
        self._bypass_llm = ChatOpenAI(
            model=BYPASS_MODEL,
            base_url=NVIDIA_BASE_URL,
            api_key=api_key or "placeholder",
            temperature=0.7,
            max_tokens=4000,
        )
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

        self.chroma = chromadb.PersistentClient(
            path="./chroma_data",
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

        self._pending:      dict[str, dict]         = {}
        self._doc_registry: dict[str, dict]         = {}
        self._index_locks:  dict[str, asyncio.Lock] = {}
        self._chunks_count: int = self.collection.count()

        self.bm25: Optional[BM25Okapi] = None
        self.bm25_ids: List[str] =[]
        
        self._embed_sem = asyncio.Semaphore(5)

        self._build_bm25_sync()
        self._restore_doc_registry()
        self._graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(RAGState)
        
        # Интеграция узла переписывания запроса (Query Expansion)
        if ENABLE_QUERY_EXPANSION:
            g.add_node("rewrite", self._rewrite_node)
            g.set_entry_point("rewrite")
            g.add_edge("rewrite", "retrieve")
        else:
            g.set_entry_point("retrieve")
            
        g.add_node("retrieve",   self._retrieve_node)
        g.add_node("rerank",     self._rerank_node)
        g.add_node("generate",   self._generate_node)
        
        if ENABLE_SELF_CHECK:
            g.add_node("self_check", self._self_check_node)

        g.add_edge("retrieve", "rerank")

        def check_if_chunks_exist(state: RAGState) -> str:
            return "has_chunks" if state.get("chunks") else "empty"

        g.add_conditional_edges("rerank", check_if_chunks_exist, {"empty": END, "has_chunks": "generate"})

        if ENABLE_SELF_CHECK:
            g.add_edge("generate", "self_check")
            def check_hallucinations_route(state: RAGState) -> str:
                if not state.get("is_hallucinating"): return "ok"
                if state.get("attempts", 0) >= 2: return "max_retries"
                return "retry"

            g.add_conditional_edges("self_check", check_hallucinations_route, {
                "ok": END, "max_retries": END, "retry": "generate"
            })
        else:
            g.add_edge("generate", END)

        return g.compile()

    # Узел 1: Multi-Query Expansion
    async def _rewrite_node(self, state: RAGState) -> dict:
        question = state["question"]
        if state.get("attempts", 0) > 0:
            return {"queries": [question]}
            
        sys_msg = SystemMessage(content=(
            "Ты эксперт по семантическому поиску. Твоя задача — сгенерировать 2 альтернативные/синонимичные "
            "формулировки запроса пользователя, чтобы повысить вероятность нахождения нужного документа.\n"
            "Выведи ровно 2 варианта, каждый с новой строки, без лишнего текста и нумерации."
        ))
        usr_msg = HumanMessage(content=f"Оригинальный запрос: {question}")
        
        try:
            res = await self._llm.ainvoke([sys_msg, usr_msg])
            alts =[q.strip() for q in res.content.split("\n") if q.strip()]
            queries = [question] + alts[:2]
            
            tokens = res.usage_metadata.get("total_tokens", 0) if res.usage_metadata else 0
            logger.info(f"[rewrite] Expanded queries: {queries}")
            return {"queries": queries, "tokens_used": (state.get("tokens_used") or 0) + tokens}
        except Exception as e:
            logger.warning(f"[rewrite] Fallback, failed to expand query: {e}")
            return {"queries": [question]}

    # Узел 2: Пакетный поиск + Global RRF
    async def _retrieve_node(self, state: RAGState) -> dict:
        queries = state.get("queries", [state["question"]])
        top_k   = state["top_k"]
        k_pool  = max(top_k * 3, 20)

        if self._chunks_count == 0:
            return {"chunks":[]}

        queries_emb = await self._embed(queries, input_type="query")
        
        vec_res = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=queries_emb,
            n_results=min(k_pool, self._chunks_count),
            include=["documents", "metadatas", "distances"],
        )

        chunk_map = {}
        q_vec_ranks = { i: {} for i in range(len(queries)) }
        q_bm25_ranks = { i: {} for i in range(len(queries)) }
        
        if vec_res and vec_res["ids"]:
            for i, q_ids in enumerate(vec_res["ids"]):
                if not q_ids: continue
                for rank, (cid, doc, meta, dist) in enumerate(zip(
                    q_ids, vec_res["documents"][i], vec_res["metadatas"][i], vec_res["distances"][i]
                )):
                    q_vec_ranks[i][cid] = rank + 1
                    if cid not in chunk_map:
                        chunk_map[cid] =[doc, meta, dist, "vector", 0.0]
                    else:
                        chunk_map[cid][2] = min(chunk_map[cid][2], dist)

        bm25_missing_ids = set()
        if self.bm25:
            for i, q in enumerate(queries):
                tokenized_q = self._tokenize(q)
                scores = self.bm25.get_scores(tokenized_q)
                top_indices = self._top_k_indices(scores, k_pool)
                
                for rank, idx in enumerate(top_indices):
                    score = scores[idx]
                    if score > 0.0:
                        cid = self.bm25_ids[idx]
                        q_bm25_ranks[i][cid] = rank + 1
                        if cid not in chunk_map:
                            bm25_missing_ids.add(cid)
                            chunk_map[cid] =["", {}, BM25_DUMMY_DIST, "bm25", float(score)]
                        else:
                            chunk_map[cid][4] = max(chunk_map[cid][4], float(score))

        if bm25_missing_ids:
            bm25_docs = await asyncio.to_thread(
                self.collection.get, ids=list(bm25_missing_ids), include=["documents", "metadatas"]
            )
            if bm25_docs and bm25_docs["ids"]:
                for cid, doc, meta in zip(bm25_docs["ids"], bm25_docs["documents"], bm25_docs["metadatas"]):
                    if cid in chunk_map:
                        chunk_map[cid][0] = doc
                        chunk_map[cid][1] = meta

        global_rrf = {}
        for i in range(len(queries)):
            all_ids = set(q_vec_ranks[i]) | set(q_bm25_ranks[i])
            for cid in all_ids:
                v_rank = q_vec_ranks[i].get(cid, 1000)
                b_rank = q_bm25_ranks[i].get(cid, 1000)
                score = (1.0 / (RRF_K + v_rank)) + (1.0 / (RRF_K + b_rank))
                global_rrf[cid] = global_rrf.get(cid, 0.0) + score

        sorted_ids = sorted(global_rrf, key=lambda c: global_rrf[c], reverse=True)
        chunks =[tuple(chunk_map[cid]) for cid in sorted_ids]

        logger.info(f"[retrieve] Aggregated {len(chunks)} chunks using {len(queries)} queries.")
        return {"chunks": chunks}

    @retry(
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=15), stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING), reraise=True,
    )
    async def _call_reranker(self, query: str, passages: list) -> dict:
        """Вспомогательный метод для обращения к NVIDIA NIM Reranking API"""
        api_key = os.environ.get("NVIDIA_API_KEY", "placeholder")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # Точный Payload из документации NVIDIA
        payload = {
            "model": RERANK_MODEL,
            "query": {"text": query},
            "passages": [{"text": p[0]} for p in passages]
        }
        
        # Облачный API NVIDIA требует внедрение имени модели в URL
        # При этом точки заменяются на нижние подчеркивания (например "llama-3.2" -> "llama-3_2")
        model_path = RERANK_MODEL.replace(".", "_")
        
        candidate_urls =[
            # 1. Официальный URL из доков (например: https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2.../reranking)
            f"https://ai.api.nvidia.com/v1/retrieval/{model_path}/reranking",
            
            # 2. On-premise NIM локально (обычный путь)
            f"{NVIDIA_BASE_URL.rstrip('/')}/ranking",                         
            
            # 3. Универсальный fallback NVIDIA (для старых версий API)
            "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"         
        ]

        for i, url in enumerate(candidate_urls):
            resp = await self._http_client.post(url, headers=headers, json=payload)
            
            if resp.status_code == 404 and i < len(candidate_urls) - 1:
                logger.debug(f"[rerank] URL {url} returned 404, trying fallback...")
                continue
            
            resp.raise_for_status()
            return resp.json()

    # Узел 3: Оценка контекста через API-модель Reranker'а
    async def _rerank_node(self, state: RAGState) -> dict:
        filtered =[
            (doc, meta, dist, source, bm25_score)
            for doc, meta, dist, source, bm25_score in state["chunks"]
            if (source == "vector" and (1 - dist) >= MIN_RELEVANCE_SCORE)
            or (source == "bm25"   and bm25_score >= MIN_BM25_SCORE)
        ]

        if not filtered:
            logger.info("[rerank] Fast-failing, no chunks passed basic thresholds.")
            return {
                "chunks":[],
                "answer": "К сожалению, в загруженных документах нет релевантной информации.",
                "confidence": 0.0, "sources":[], "retrieval_type": None
            }

        try:
            logger.info(f"[rerank] Running NVIDIA Reranker {RERANK_MODEL} on {len(filtered)} candidates...")
            rerank_res = await self._call_reranker(state["question"], filtered)
            
            rankings = rerank_res.get("rankings",[])
            reranked_chunks =[]
            
            for item in rankings:
                idx = item.get("index")
                logit = item.get("logit", 0.0)
                if idx is not None and idx < len(filtered):
                    if logit < -5.0:
                        continue
                    reranked_chunks.append(filtered[idx])
                    
        except Exception as e:
            logger.warning(f"[rerank] Reranker API failed: {e}. Falling back to default RRF order.")
            reranked_chunks = filtered
            
        file_counts: dict[str, int] = {}
        deduped =[]
        for item in reranked_chunks:
            fid = item[1].get("file_id", "")
            if file_counts.get(fid, 0) < MAX_CHUNKS_PER_FILE:
                deduped.append(item)
                file_counts[fid] = file_counts.get(fid, 0) + 1
                if len(deduped) == state["top_k"]:
                    break

        if not deduped:
            logger.info("[rerank] Fast-failing, no relevant chunks after reranking.")
            return {
                "chunks":[],
                "answer": "К сожалению, в загруженных документах нет релевантной информации.",
                "confidence": 0.0, "sources":[], "retrieval_type": None
            }

        return {"chunks": deduped}

    # Узел 4: Генерация с использованием CoT и XML-разметки
    async def _generate_node(self, state: RAGState) -> dict:
        chunks   = state["chunks"]
        question = state["question"]
        attempts = state.get("attempts", 0) + 1

        context_parts =[]
        total_tokens  = 0
        for doc, meta, dist, source, bm25_score in chunks:
            part = (
                f'<document source="{meta["file_name"]}" chunk="{meta["chunk_index"]+1}/{meta["total_chunks"]}">\n'
                f'{doc}\n'
                f'</document>'
            )
            part_tokens = count_tokens(part)
            if total_tokens + part_tokens > MAX_CONTEXT_TOKENS and len(context_parts) > 0:
                break
            context_parts.append(part)
            total_tokens += part_tokens

        context = "\n".join(context_parts)

        system_instruction = (
            "Ты — точный аналитический ИИ-ассистент.\n"
            "Используй ТОЛЬКО предоставленные ниже документы в тегах <document> для ответа. "
            "Если информации недостаточно, скажи об этом прямо, не фантазируй.\n\n"
            "Формат ответа СТРОГО следующий:\n"
            "**Анализ:** (краткий анализ того, что найдено в документах)\n"
            "**Ответ:** (твой исчерпывающий финальный ответ с цитатами источников)"
        )

        if attempts > 1:
            system_instruction += (
                "\n🚨 ВНИМАНИЕ: Предыдущая попытка содержала факты не из текста. "
                "ОПИРАЙСЯ ИСКЛЮЧИТЕЛЬНО НА ТЕГИ <document>."
            )

        messages =[
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Контекст:\n{context}\n\nВопрос пользователя: {question}"),
        ]

        response = await self._llm.ainvoke(messages)
        answer   = response.content
        
        current_tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        tokens_used    = (state.get("tokens_used") or 0) + current_tokens

        def calibrate(sim: float) -> float:
            return max(0.01, min((sim - 0.15) / 0.25, 1.0))

        vector_dists =[d for _, _, d, src, _ in chunks if src == "vector"]
        bm25_count   = sum(1 for _, _, _, src, _ in chunks if src == "bm25")

        if not vector_dists:
            confidence = 0.40
            retrieval_type = "bm25_only"
        else:
            top3_sims  = sorted([1 - d for d in vector_dists], reverse=True)[:3]
            confidence = round(calibrate(sum(top3_sims) / len(top3_sims)), 2)
            retrieval_type = "hybrid" if bm25_count > 0 else "vector_only"

        sources =[
            {
                "chunk_id":    f"{m['file_id']}_chunk_{m['chunk_index']:04d}",
                "file_name":   m["file_name"],
                "chunk_index": m["chunk_index"],
                "score":       round(calibrate(1 - dist), 2) if src == "vector" else round(min(bm25_score/10, 1.0), 2),
                "source_type": src,
                "preview":     doc[:200] + "…",
            }
            for doc, m, dist, src, bm25_score in chunks[:5]
        ]

        logger.info(f"[generate] Attempt {attempts}: conf={confidence}")
        return {
            "answer": answer, "confidence": confidence, "sources": sources,
            "tokens_used": tokens_used, "retrieval_type": retrieval_type, "attempts": attempts
        }

    async def _self_check_node(self, state: RAGState) -> dict:
        answer   = state["answer"]
        chunks   = state["chunks"]
        attempts = state["attempts"]

        if not chunks: return {"is_hallucinating": False}

        context = "\n\n".join([f"Чанк {m['chunk_index']+1}: {doc}" for doc, m, _, _, _ in chunks[:5]])
        messages =[
            SystemMessage(content=(
                "Ты строгий проверяющий (Validator). Ответь 'YES' если Ответ базируется ТОЛЬКО на Контексте. "
                "Если в Ответе есть внешние факты/числа -> ответь 'NO'."
            )),
            HumanMessage(content=f"Контекст:\n{context}\n\nОтвет на проверку:\n{answer}")
        ]

        response = await self._llm.ainvoke(messages)
        is_hallucinating = "YES" not in response.content.strip().upper()
        
        val_tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        new_total_tokens = (state.get("tokens_used") or 0) + val_tokens

        if is_hallucinating:
            logger.warning(f"[self_check] 🚨 Hallucination detected on attempt {attempts}!")
            
        return {"is_hallucinating": is_hallucinating, "tokens_used": new_total_tokens}

    # ── BM25 ──────────────────────────────────────────────────────────────────
    def _tokenize(self, text: str) -> list[str]:
        return[w for w in TOKEN_SPLIT_RE.split(text.lower()) if w]

    @staticmethod
    def _top_k_indices(scores, k: int) -> list[int]:
        if k <= 0:
            return []
        arr = np.asarray(scores)
        if arr.size == 0:
            return []

        k = min(k, arr.size)
        if k == arr.size:
            return np.argsort(arr)[::-1].tolist()

        partition_idx = np.argpartition(arr, -k)[-k:]
        sorted_top_idx = partition_idx[np.argsort(arr[partition_idx])[::-1]]
        return sorted_top_idx.tolist()

    def _build_bm25_sync(self):
        self._chunks_count = self.collection.count()
        if self._chunks_count == 0:
            self.bm25 = None; self.bm25_ids =[]
            return
        docs = self.collection.get(include=["documents"])
        self.bm25_ids = docs["ids"]
        tokenized =[self._tokenize(d) for d in docs["documents"]]
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)

    async def rebuild_bm25(self):
        await asyncio.to_thread(self._build_bm25_sync)

    # ── Doc registry / Helpers ────────────────────────────────────────────────
    def _restore_doc_registry(self):
        if self._chunks_count == 0: return
        results = self.collection.get(include=["metadatas"])
        for meta in results["metadatas"]:
            fid = meta.get("file_id", "")
            if fid and fid not in self._doc_registry:
                self._doc_registry[fid] = {
                    "file_id": fid, "file_name": meta.get("file_name", ""),
                    "file_type": meta.get("file_type", ""), "indexed_at": meta.get("indexed_at", ""),
                    "chunks": 0,
                }
            if fid: self._doc_registry[fid]["chunks"] += 1

    def _cleanup_expired_pending(self):
        now = time.time()
        expired =[f for f, info in self._pending.items() if now - info["uploaded_ts"] > PENDING_TTL_SECONDS]
        for f in expired: del self._pending[f]

    def extract_text(self, content: bytes, ext: str, filename: str) -> str:
        return extract_text(content, ext, filename)

    def store_pending(self, file_id: str, filename: str, ext: str, text: str):
        self._cleanup_expired_pending()
        self._pending[file_id] = {
            "file_id": file_id, "file_name": filename, "file_type": ext,
            "text": text, "uploaded_at": datetime.now(timezone.utc).isoformat(), "uploaded_ts": time.time(),
        }

    def total_chunks(self) -> int: return self._chunks_count
    def list_documents(self) -> list[dict]: return list(self._doc_registry.values())

    async def delete_document(self, file_id: str):
        results = await asyncio.to_thread(self.collection.get, where={"file_id": file_id}, include=[])
        if results["ids"]:
            await asyncio.to_thread(self.collection.delete, ids=results["ids"])
            self._chunks_count = max(0, self._chunks_count - len(results["ids"]))
            self._doc_registry.pop(file_id, None)
            await self.rebuild_bm25()

    # ── Indexing ──────────────────────────────────────────────────────────────
    def _get_index_lock(self, file_id: str) -> asyncio.Lock:
        if file_id not in self._index_locks: self._index_locks[file_id] = asyncio.Lock()
        return self._index_locks[file_id]

    async def index_file(self, file_id: str) -> dict:
        async with self._get_index_lock(file_id):
            if file_id not in self._pending:
                raise ValueError(f"File {file_id} not found")

            info = self._pending[file_id]
            old = await asyncio.to_thread(self.collection.get, where={"file_id": file_id}, include=[])
            if old["ids"]: await asyncio.to_thread(self.collection.delete, ids=old["ids"])

            chunks = await asyncio.to_thread(chunk_text, info["text"])
            if not chunks: raise ValueError("No chunks produced")
            
            if len(chunks) > MAX_CHUNKS_PER_DOC:
                chunks = chunks[:MAX_CHUNKS_PER_DOC]

            indexed_at = datetime.now(timezone.utc).isoformat()
            
            async def get_embedding_batch(batch_chunks: list[str]) -> list[list[float]]:
                async with self._embed_sem:
                    return await self._embed(batch_chunks, input_type="passage")

            tasks = [get_embedding_batch(chunks[i : i + 32]) for i in range(0, len(chunks), 32)]
            batch_results = await asyncio.gather(*tasks)
            all_embeddings =[emb for batch in batch_results for emb in batch]

            ids =[f"{file_id}_chunk_{i:04d}" for i in range(len(chunks))]
            metadatas =[{
                "file_id": file_id, "file_name": info["file_name"], "file_type": info["file_type"],
                "chunk_index": i, "total_chunks": len(chunks), "indexed_at": indexed_at,
            } for i in range(len(chunks))]

            await asyncio.to_thread(
                self.collection.add, ids=ids, embeddings=all_embeddings, documents=chunks, metadatas=metadatas,
            )
            self._chunks_count = max(0, self._chunks_count - len(old["ids"])) + len(chunks)

            self._doc_registry[file_id] = {
                "file_id": file_id, "file_name": info["file_name"], "file_type": info["file_type"],
                "indexed_at": indexed_at, "chunks": len(chunks),
            }

            del self._pending[file_id]
            self._index_locks.pop(file_id, None)

            return { "file_id": file_id, "file_name": info["file_name"], "chunks": len(chunks), "status": "indexed" }

    async def aclose(self):
        await self._http_client.aclose()

    async def query(self, question: str, top_k: int = 7) -> dict:
        initial_state: RAGState = {
            "question": question, "queries":[], "top_k": top_k, "chunks":[],
            "answer": "", "confidence": 0.0, "sources":[], "tokens_used": 0,
            "retrieval_type": None, "attempts": 0, "is_hallucinating": False
        }
        result = await self._graph.ainvoke(initial_state)
        
        if result.get("is_hallucinating"):
            result["answer"] = "К сожалению, в загруженных документах недостаточно информации для точного ответа на этот вопрос."
            result["confidence"] = 0.0
            
        return {
            "answer": result["answer"], "confidence": result["confidence"], "sources": result["sources"],
            "tokens_used": result["tokens_used"], "retrieval_type": result.get("retrieval_type"),
        }

    @retry(
        retry=retry_if_exception_type((APITimeoutError, APIStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING), reraise=True,
    )
    async def _embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        response = await self._openai.embeddings.create(
            model=EMBED_MODEL, input=texts, extra_body={"input_type": input_type, "truncate": "END"},
        )
        return[item.embedding for item in response.data]

    def _sanitize_model_answer(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"```(?:thinking|reasoning)?[\s\S]*?```", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        return cleaned or "Не удалось сформировать ответ."

    def _build_bypass_messages(self, message: str, history: Optional[list[dict]] = None):
        system_msg = SystemMessage(content=(
            "Ты — полезный ИИ-ассистент. Отвечай кратко и по делу. "
            "Не показывай внутренние рассуждения, цепочки мыслей или скрытые шаги решения. "
            "Если вопрос касается документов, предупреди, что у тебя нет доступа к RAG-контексту в bypass-режиме."
        ))
        
        messages =[system_msg]
        for msg in (history or [])[-10:]:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        
        messages.append(HumanMessage(content=message))
        return messages
        
    async def bypass_chat(self, message: str, history: Optional[list[dict]] = None) -> dict:
        messages = self._build_bypass_messages(message, history)

        response = await self._bypass_llm.ainvoke(messages)
        tokens_used = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        answer = self._sanitize_model_answer(response.content)
        
        logger.info(f"[bypass_chat] Response generated, tokens: {tokens_used}")
        
        return {
            "answer": answer,
            "tokens_used": tokens_used,
            "model": BYPASS_MODEL,
        }
