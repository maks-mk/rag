# Advanced Agentic Document Intelligence RAG

Локальный сервис для поиска по документам и диалога с LLM: загрузка файлов, индексация в ChromaDB, гибридный retrieval (vector + BM25), rerank через NVIDIA, генерация ответа с источниками.

## Что реализовано сейчас
- Backend: FastAPI + Uvicorn.
- Pipeline (LangGraph): `rewrite -> retrieve -> rerank -> generate -> (optional) self_check`.
- Multi-query expansion (опционально): генерация 2 альтернативных формулировок запроса.
- Hybrid retrieval:
  - vector search (embeddings)
  - BM25 lexical search
  - слияние через Global RRF.
- Rerank кандидатов через NVIDIA API с fallback URL.
- Ограничение контекста по токенам и возврат top sources.
- Bypass-режим: прямой чат с моделью без RAG-контекста.
- Хранение индекса: `chroma_data/`.
- Frontend: vanilla HTML/CSS/JS (desktop + mobile), drag-and-drop загрузка, toasts, quick prompts, sticky input, режимный индикатор, сохранение UI-состояния в `localStorage`.

## Актуальная архитектура
### API endpoints
- `GET /health` — статус сервиса и число проиндексированных чанков.
- `POST /upload` — загрузка файла (`.txt/.md/.pdf/.docx`), извлечение текста, помещение в pending.
- `POST /index` — индексация списка `file_ids` (chunking + embeddings + запись в ChromaDB + rebuild BM25).
- `POST /query` — RAG-запрос с `question` и `top_k`.
- `POST /bypass` — прямой чат с `message` и `history`.
- `GET /documents` — список документов в реестре.
- `DELETE /documents/{file_id}` — удаление документа и его чанков.

### Внутренний граф
1. `rewrite` (если `ENABLE_QUERY_EXPANSION=true`): расширение запроса.
2. `retrieve`: пакетный vector + BM25 поиск, Global RRF.
3. `rerank`: фильтрация + внешняя rerank-модель.
4. `generate`: ответ строго по контексту `<document ...>`.
5. `self_check` (если `ENABLE_SELF_CHECK=true`): проверка на галлюцинации с retry генерации.

## Производительность и стабильность (текущая реализация)
- Переиспользуемый `httpx.AsyncClient` для rerank-вызовов.
- Кэш счётчика чанков (`_chunks_count`) для быстрых health/query-path проверок.
- Ускоренный BM25 top-k через `numpy.argpartition`.
- Кэширование splitters в `text_processor`.
- Retry/backoff для embeddings/rerank API (tenacity).

## Форматы документов
Поддерживаются:
- `.txt`
- `.md`
- `.pdf`
- `.docx`

Ограничение размера файла: до 50 MB.

## Быстрый старт
### 1) Установка
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2) Настройка `.env`
```env
NVIDIA_API_KEY=nvapi-...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# Models
LLM_MODEL=meta/llama-3.1-8b-instruct
EMBED_MODEL=nvidia/nv-embedqa-e5-v5
BYPASS_MODEL=meta/llama-3.1-8b-instruct
RERANK_MODEL=nvidia/llama-3.2-nv-rerankqa-1b-v2

# Feature flags
ENABLE_QUERY_EXPANSION=true
ENABLE_SELF_CHECK=true
```

### 3) Запуск
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Открыть: `http://localhost:8000`

## Структура проекта
```text
v0.5/
├── app/
│   ├── main.py
│   ├── rag.py
│   └── rag_engine/
│       ├── config.py
│       ├── service.py
│       ├── state.py
│       └── text_processor.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── chroma_data/
├── requirements.txt
└── run.py
```

## UI/UX (текущая версия)
- Переключение режимов `RAG/BYPASS` в сайдбаре.
- Quick prompts на пустом экране чата.
- Кнопка прокрутки к новым сообщениям.
- Мобильный overlay для сайдбара + закрытие по backdrop/Escape.
- Безопасный рендер сообщений и источников (`escapeHtml`, fallback в markdown/hljs).
- Поддержка `prefers-reduced-motion` без потери видимости сообщений.

## Важно знать
- BM25 индекс не хранится на диске: пересобирается из Chroma при старте/изменениях.
- В bypass-режиме модель не использует загруженные документы.
- Для качества ответов лучше загружать текстовые/структурированные документы.
