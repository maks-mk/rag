import os
import time
import heapq
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import chromadb
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
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from .config import (
    NVIDIA_BASE_URL,
    EMBED_MODEL,
    LLM_MODEL,
    MAX_CHUNKS_PER_DOC,
    PENDING_TTL_SECONDS,
    MIN_RELEVANCE_SCORE,
    MAX_CONTEXT_TOKENS,
    MAX_CHUNKS_PER_FILE,
    RRF_K,
    MIN_BM25_SCORE,
    BM25_DUMMY_DIST,
    ENABLE_SELF_CHECK,
)
from .text_processor import (
    count_tokens,
    chunk_text,
    extract_text,
)
from .state import RAGState

logger = logging.getLogger(__name__)

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

        self.bm25: Optional[BM25Okapi] = None
        self.bm25_ids: List[str] = []
        
        self._embed_sem = asyncio.Semaphore(5)

        self._build_bm25_sync()
        self._restore_doc_registry()
        self._graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("retrieve",   self._retrieve_node)
        g.add_node("grade",      self._grade_node)
        g.add_node("generate",   self._generate_node)
        
        if ENABLE_SELF_CHECK:
            g.add_node("self_check", self._self_check_node)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "grade")

        # Fast-fail маршрутизация
        def check_if_chunks_exist(state: RAGState) -> str:
            return "has_chunks" if state.get("chunks") else "empty"

        g.add_conditional_edges(
            "grade",
            check_if_chunks_exist,
            {
                "empty": END,
                "has_chunks": "generate"
            }
        )

        # Маршрутизация с учетом флага ENABLE_SELF_CHECK
        if ENABLE_SELF_CHECK:
            g.add_edge("generate", "self_check")
            
            def check_hallucinations_route(state: RAGState) -> str:
                if not state.get("is_hallucinating"):
                    return "ok"
                # Даем только 1 шанс на исправление, чтобы не жечь токены впустую
                if state.get("attempts", 0) >= 2: 
                    return "max_retries"
                return "retry"

            g.add_conditional_edges(
                "self_check",
                check_hallucinations_route,
                {
                    "ok": END,
                    "max_retries": END,
                    "retry": "generate"
                }
            )
        else:
            g.add_edge("generate", END)

        return g.compile()

    async def _retrieve_node(self, state: RAGState) -> dict:
        question = state["question"]
        top_k    = state["top_k"]
        k_pool   = max(top_k * 3, 20)

        if self.collection.count() == 0:
            return {"chunks": []}

        vector_ranks:  dict[str, int]   = {}
        bm25_ranks:    dict[str, int]   = {}
        bm25_scores:   dict[str, float] = {}
        chunk_map: dict[str, tuple[str, dict, float, str, float]] = {}

        async def do_vector_search():
            q_emb = await self._embed([question], input_type="query")
            return await asyncio.to_thread(
                self.collection.query,
                query_embeddings=q_emb,
                n_results=min(k_pool, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )

        def do_bm25_search():
            if not self.bm25: return []
            tokenized_q = self._tokenize(question)
            scores = self.bm25.get_scores(tokenized_q)
            top_indices = heapq.nlargest(k_pool, range(len(scores)), key=lambda i: scores[i])
            return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0.0]

        vec, bm25_res = await asyncio.gather(
            do_vector_search(),
            asyncio.to_thread(do_bm25_search)
        )

        if vec and vec["ids"] and vec["ids"][0]:
            for rank, (cid, doc, meta, dist) in enumerate(zip(
                vec["ids"][0], vec["documents"][0],
                vec["metadatas"][0], vec["distances"][0],
            )):
                vector_ranks[cid] = rank + 1
                chunk_map[cid]    = (doc, meta, dist, "vector", 0.0)

        bm25_missing_ids = []
        for rank, (idx, score) in enumerate(bm25_res):
            cid = self.bm25_ids[idx]
            bm25_ranks[cid]  = rank + 1
            bm25_scores[cid] = float(score)
            if cid not in chunk_map:
                bm25_missing_ids.append(cid)

        if bm25_missing_ids:
            bm25_docs = await asyncio.to_thread(
                self.collection.get,
                ids=bm25_missing_ids,
                include=["documents", "metadatas"],
            )
            for cid, doc, meta in zip(bm25_docs["ids"], bm25_docs["documents"], bm25_docs["metadatas"]):
                chunk_map[cid] = (doc, meta, BM25_DUMMY_DIST, "bm25", bm25_scores[cid])

        all_ids = set(vector_ranks) | set(bm25_ranks)
        rrf = {
            cid: (1.0 / (RRF_K + vector_ranks.get(cid, 1000))) + (1.0 / (RRF_K + bm25_ranks.get(cid, 1000)))
            for cid in all_ids
        }
        
        sorted_ids = sorted(rrf, key=lambda c: rrf[c], reverse=True)
        chunks = [chunk_map[cid] for cid in sorted_ids]

        logger.info(f"[retrieve] question='{question}' → retrieved {len(chunks)} raw chunks")
        return {"chunks": chunks}

    async def _grade_node(self, state: RAGState) -> dict:
        filtered = [
            (doc, meta, dist, source, bm25_score)
            for doc, meta, dist, source, bm25_score in state["chunks"]
            # BM25_DUMMY_DIST безопасно игнорируется, так как проверяется только source == "vector"
            if (source == "vector" and (1 - dist) >= MIN_RELEVANCE_SCORE)
            or (source == "bm25"   and bm25_score >= MIN_BM25_SCORE)
        ]
        
        file_counts: dict[str, int] = {}
        deduped = []
        for item in filtered:
            fid = item[1].get("file_id", "")
            if file_counts.get(fid, 0) < MAX_CHUNKS_PER_FILE:
                deduped.append(item)
                file_counts[fid] = file_counts.get(fid, 0) + 1
                if len(deduped) == state["top_k"]:
                    break

        if not deduped:
            logger.info("[grade] All chunks filtered out. Fast-failing.")
            return {
                "chunks": [],
                "answer": "К сожалению, в загруженных документах нет релевантной информации для ответа на этот вопрос.",
                "confidence": 0.0,
                "sources": [],
                "retrieval_type": None
            }

        logger.info(f"[grade] final sorted chunks: {len(deduped)}")
        return {"chunks": deduped}

    async def _generate_node(self, state: RAGState) -> dict:
        chunks   = state["chunks"]
        question = state["question"]
        attempts = state.get("attempts", 0) + 1

        context_parts = []
        total_tokens  = 0
        for doc, meta, dist, source, bm25_score in chunks:
            header       = f"[Источник: {meta['file_name']}, чанк {meta['chunk_index']+1}/{meta['total_chunks']}]\n"
            part         = header + doc
            part_tokens  = count_tokens(part)
            
            if total_tokens + part_tokens > MAX_CONTEXT_TOKENS and len(context_parts) > 0:
                # ВОЗВРАЩЕНО: Важная метрика обрезки контекста
                logger.info(
                    f"[generate] context limit reached at {total_tokens} tokens "
                    f"({len(context_parts)}/{len(chunks)} chunks used)"
                )
                break
                
            context_parts.append(part)
            total_tokens += part_tokens

        context = "\n\n---\n\n".join(context_parts)

        system_instruction = (
            "Ты — точный аналитический ассистент. "
            "Отвечай строго на основе предоставленного контекста. "
            "Не придумывай факты, которых нет в контексте. "
            "Если информации недостаточно — скажи об этом прямо. "
            "Отвечай на том же языке, на котором задан вопрос. "
            "Структурируй ответ: краткий вывод, затем детали при необходимости."
        )

        if attempts > 1:
            system_instruction += (
                "\n\n🚨 ВНИМАНИЕ: Твоя предыдущая попытка содержала галлюцинации или факты "
                "за пределами контекста. СЕЙЧАС ТВОЯ ЗАДАЧА — ОПИРАТЬСЯ ИСКЛЮЧИТЕЛЬНО НА ТЕКСТ. "
                "Если точного ответа нет в тексте, просто скажи: 'Информации нет'."
            )
            
            # --- ВЫВОД НОВОГО ПРОМПТА В КОНСОЛЬ ---
            logger.info(
                f"\n[generate] 🔁 Повторная генерация (попытка {attempts}). Отправляем обновленный промпт:\n"
                f"--------------------------------------------------\n"
                f"SYSTEM MESSAGE:\n{system_instruction}\n\n"
                f"HUMAN MESSAGE:\nКонтекст:\n{context}\n\n---\nВопрос: {question}\n"
                f"--------------------------------------------------"
            )

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Контекст:\n\n{context}\n\n---\n\nВопрос: {question}"),
        ]

        response    = await self._llm.ainvoke(messages)
        answer      = response.content
        
        # Аккумулируем токены
        current_tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        tokens_used    = (state.get("tokens_used") or 0) + current_tokens

        def calibrate(sim: float) -> float:
            return max(0.01, min((sim - 0.15) / 0.25, 1.0))

        vector_dists = [d for _, _, d, src, _ in chunks if src == "vector"]
        bm25_count   = sum(1 for _, _, _, src, _ in chunks if src == "bm25")

        if not vector_dists:
            confidence     = 0.40
            retrieval_type = "bm25_only"
        else:
            top3_sims  = sorted([1 - d for d in vector_dists], reverse=True)[:3]
            avg_top3   = sum(top3_sims) / len(top3_sims)
            confidence     = round(calibrate(avg_top3), 2)
            retrieval_type = "hybrid" if bm25_count > 0 else "vector_only"

        sources = [
            {
                "chunk_id":    f"{m['file_id']}_chunk_{m['chunk_index']:04d}",
                "file_name":   m["file_name"],
                "chunk_index": m["chunk_index"],
                "score":       round(calibrate(1 - dist), 2) if src == "vector" else round(min(bm25_score / 10, 1.0), 2),
                "source_type": src,
                "preview":     doc[:200] + ("…" if len(doc) > 200 else ""),
            }
            for doc, m, dist, src, bm25_score in chunks[:5]
        ]

        logger.info(f"[generate] Attempt {attempts}: type={retrieval_type}, conf={confidence}")
        return {
            "answer":         answer,
            "confidence":     confidence,
            "sources":        sources,
            "tokens_used":    tokens_used,
            "retrieval_type": retrieval_type,
            "attempts":       attempts
        }

    async def _self_check_node(self, state: RAGState) -> dict:
        answer   = state["answer"]
        chunks   = state["chunks"]
        attempts = state["attempts"]

        if not chunks:
            return {"is_hallucinating": False}

        context_parts = [f"Чанк {m['chunk_index']+1}: {doc}" for doc, m, _, _, _ in chunks[:5]]
        context = "\n\n".join(context_parts)

        messages = [
            SystemMessage(content=(
                "Ты — строгий проверяющий (Validator). Твоя задача — проверить, основан ли Ответ СТРОГО на предоставленном Контексте.\n"
                "Если Ответ содержит факты, даты, числа или утверждения, которых НЕТ в Контексте -> отвечай 'NO'.\n"
                "Если Ответ полностью подтверждается Контекстом (или корректно говорит, что информации нет) -> отвечай 'YES'.\n"
                "Твой ответ должен состоять только из одного слова: YES или NO."
            )),
            HumanMessage(content=f"Контекст:\n{context}\n\nОтвет на проверку:\n{answer}")
        ]

        response = await self._llm.ainvoke(messages)
        check_result = response.content.strip().upper()
        
        # Считаем и аккумулируем токены самопроверки
        val_tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        new_total_tokens = (state.get("tokens_used") or 0) + val_tokens

        is_hallucinating = "YES" not in check_result

        if is_hallucinating:
            logger.warning(f"[self_check] 🚨 Hallucination detected on attempt {attempts}! Validator output: {check_result}")
        else:
            logger.info(f"[self_check] ✅ Answer passed hallucination check on attempt {attempts}.")

        return {
            "is_hallucinating": is_hallucinating,
            "tokens_used": new_total_tokens
        }

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        import re
        return [w for w in re.split(r"\W+", text.lower()) if w]

    def _build_bm25_sync(self):
        if self.collection.count() == 0:
            self.bm25 = None
            self.bm25_ids = []
            return
        
        docs = self.collection.get(include=["documents"])
        self.bm25_ids = docs["ids"]
        tokenized = [self._tokenize(d) for d in docs["documents"]]
        
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)
            logger.info(f"BM25 built: {len(self.bm25_ids)} chunks")

    async def rebuild_bm25(self):
        await asyncio.to_thread(self._build_bm25_sync)

    # ── Doc registry ──────────────────────────────────────────────────────────

    def _restore_doc_registry(self):
        if self.collection.count() == 0:
            return
        results = self.collection.get(include=["metadatas"])
        for meta in results["metadatas"]:
            fid = meta.get("file_id", "")
            if fid and fid not in self._doc_registry:
                self._doc_registry[fid] = {
                    "file_id":    fid,
                    "file_name":  meta.get("file_name", ""),
                    "file_type":  meta.get("file_type", ""),
                    "indexed_at": meta.get("indexed_at", ""),
                    "chunks":     0,
                }
            if fid:
                self._doc_registry[fid]["chunks"] += 1
        logger.info(f"Doc registry restored: {len(self._doc_registry)} docs")

    def _cleanup_expired_pending(self):
        now     = time.time()
        expired = [
            fid for fid, info in self._pending.items()
            if now - info["uploaded_ts"] > PENDING_TTL_SECONDS
        ]
        for fid in expired:
            logger.warning(f"Pending expired: {self._pending[fid]['file_name']} ({fid})")
            del self._pending[fid]

    # ── Public helpers ────────────────────────────────────────────────────────

    def extract_text(self, content: bytes, ext: str, filename: str) -> str:
        # Delegate to text_processor
        return extract_text(content, ext, filename)

    def store_pending(self, file_id: str, filename: str, ext: str, text: str):
        self._cleanup_expired_pending()
        self._pending[file_id] = {
            "file_id":     file_id,
            "file_name":   filename,
            "file_type":   ext,
            "text":        text,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "uploaded_ts": time.time(),
        }

    def total_chunks(self) -> int:
        return self.collection.count()

    def list_documents(self) -> list[dict]:
        return list(self._doc_registry.values())

    async def delete_document(self, file_id: str):
        results = await asyncio.to_thread(self.collection.get, where={"file_id": file_id}, include=[])
        if results["ids"]:
            await asyncio.to_thread(self.collection.delete, ids=results["ids"])
            self._doc_registry.pop(file_id, None)
            await self.rebuild_bm25()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _get_index_lock(self, file_id: str) -> asyncio.Lock:
        if file_id not in self._index_locks:
            self._index_locks[file_id] = asyncio.Lock()
        return self._index_locks[file_id]

    async def index_file(self, file_id: str) -> dict:
        async with self._get_index_lock(file_id):
            if file_id not in self._pending:
                raise ValueError(f"File {file_id} not found in pending queue")

            info     = self._pending[file_id]
            text     = info["text"]
            filename = info["file_name"]

            old = await asyncio.to_thread(self.collection.get, where={"file_id": file_id}, include=[])
            if old["ids"]:
                await asyncio.to_thread(self.collection.delete, ids=old["ids"])

            chunks = await asyncio.to_thread(chunk_text, text)
            if not chunks:
                raise ValueError("No chunks produced from document")
            
            if len(chunks) > MAX_CHUNKS_PER_DOC:
                pct = 100 - round(MAX_CHUNKS_PER_DOC / len(chunks) * 100)
                logger.warning(
                    f"'{filename}': {len(chunks)} chunks → truncated to {MAX_CHUNKS_PER_DOC} "
                    f"(~{pct}% of content ignored)"
                )
                chunks = chunks[:MAX_CHUNKS_PER_DOC]

            indexed_at = datetime.now(timezone.utc).isoformat()
            
            async def get_embedding_batch(batch_chunks: list[str]) -> list[list[float]]:
                async with self._embed_sem:
                    return await self._embed(batch_chunks, input_type="passage")

            tasks = [
                get_embedding_batch(chunks[i : i + 32])
                for i in range(0, len(chunks), 32)
            ]
            
            batch_results = await asyncio.gather(*tasks)
            all_embeddings = [emb for batch in batch_results for emb in batch]

            ids       = [f"{file_id}_chunk_{i:04d}" for i in range(len(chunks))]
            metadatas = [
                {
                    "file_id":      file_id,
                    "file_name":    filename,
                    "file_type":    info["file_type"],
                    "chunk_index":  i,
                    "total_chunks": len(chunks),
                    "indexed_at":   indexed_at,
                }
                for i in range(len(chunks))
            ]

            await asyncio.to_thread(
                self.collection.add,
                ids=ids, embeddings=all_embeddings,
                documents=chunks, metadatas=metadatas,
            )

            self._doc_registry[file_id] = {
                "file_id":    file_id,
                "file_name":  filename,
                "file_type":  info["file_type"],
                "indexed_at": indexed_at,
                "chunks":     len(chunks),
            }

            del self._pending[file_id]
            self._index_locks.pop(file_id, None)

            logger.info(f"Indexed '{filename}': {len(chunks)} chunks")

            return {
                "file_id":   file_id,
                "file_name": filename,
                "chunks":    len(chunks),
                "status":    "indexed",
            }

    async def query(self, question: str, top_k: int = 7) -> dict:
        initial_state: RAGState = {
            "question":         question,
            "top_k":            top_k,
            "chunks": [],
            "answer":           "",
            "confidence":       0.0,
            "sources": [],
            "tokens_used":      0,
            "retrieval_type":   None,
            "attempts":         0,
            "is_hallucinating": False
        }
        result = await self._graph.ainvoke(initial_state)
        
        # Честный Fallback. Если галлюцинация осталась после всех попыток — признаемся.
        if result.get("is_hallucinating"):
            logger.warning(f"Final answer for '{question}' still hallucinating after {result['attempts']} attempts. Triggering fallback.")
            result["answer"] = "К сожалению, в загруженных документах недостаточно информации для точного ответа на этот вопрос."
            result["confidence"] = 0.0
            
        return {
            "answer":         result["answer"],
            "confidence":     result["confidence"],
            "sources":        result["sources"],
            "tokens_used":    result["tokens_used"],
            "retrieval_type": result.get("retrieval_type"),
        }

    @retry(
        retry=retry_if_exception_type((APITimeoutError, APIStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        response = await self._openai.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        return [item.embedding for item in response.data]
