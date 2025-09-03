# Local AI Agent — FastAPI × React × MySQL × RAG

> A polished, ready project that demonstrates a production‑grade chat experience with authentication, per‑chat modes (offline / web / auto), document ingestion + retrieval, streaming tokens via SSE, source citations, evaluation hooks, and optional tracing.

---

## ✨ Highlights

- **Modern full‑stack**: FastAPI (Python) backend + React/TypeScript (Vite) frontend
- **Local LLM via Ollama**: default model `aya-expanse:8b` (configurable), streamed via `/api/generate`
- **Accounts & sessions**: Email/password with Argon2, secure cookie sessions
- **Per‑chat modes**: `offline` (local RAG), `web` (internet search), `auto` (routing)
- **RAG pipeline**: stage → ingest → retrieve → cite sources
- **Live streaming**: Server‑Sent Events (SSE) for token‑wise generation
- **Evidence UI**: Sources bar, tool log, optional trace id (Langfuse)
- **Attachments UX**: Drag‑and‑drop, shelf preview, PDF export of a conversation
- **MySQL 8 + SQLAlchemy**: Async engine, clean models & migrations‑ready layout

---

## 🧭 Table of contents

- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Backend (FastAPI)](#backend-fastapi)
- [Frontend (React/TS)](#frontend-reactts)
- [Environment & configuration](#environment--configuration)
- [Running locally](#running-locally)
- [API reference (selected)](#api-reference-selected)
- [Retrieval‑Augmented Generation](#retrievalaugmented-generation)
- [Tracing & evaluation](#tracing--evaluation)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## Architecture

```text
┌────────────┐        SSE /chat/stream        ┌────────────────────────┐
│  Frontend  │  ───────────────────────────▶  │ FastAPI app            │
│  React/TS  │  ◀──── tokens / tool events ─  │ • Auth (cookies)       │
│            │                                │ • Chats & messages     │
│  Chat UI   │   uploads / REST              │ • Files: stage/ingest  │
│  Sources   │  ───────────────────────────▶  │ • RAG retriever        │
│  Tools log │                                │ • Tools: web_search,   │
│  PDF export│    citations / trace id       │   py_eval (sandboxed)   │
└────────────┘  ◀───────────────────────────  │ • SSE streaming        │
                                              │ • Tracing (Langfuse)   │
                                              └─────────┬──────────────┘
                                                        │
                                           ┌────────────▼─────────────┐
                                           │ Storage                  │
                                           │ • MySQL (users/chats)    │
                                           │ • Chroma / local docs    │
                                           │ • uploads_staged/var     │
                                           └──────────────────────────┘
```

**Modes**
- **offline** — answers only from ingested local files (RAG)
- **web** — answers backed by the internet search tool
- **auto** — smart routing between offline and web

---

## Project structure

> Essential folders & files (trimmed):

```text
backend/
  auth.py           # register/login/logout, cookie sessions
  chats.py          # chat CRUD + list messages + auto-title
  db.py             # async engine, .env loading
  main.py           # app factory, SSE /chat/stream, file ingest endpoints
  memory.py         # lightweight conversation memory helpers
  models.py         # SQLAlchemy models (User, Session, Chat, ChatMessage, ...)
  requirements.txt  # backend deps
  tracing.py        # optional Langfuse + local JSONL traces
  rag/ingest.py     # bytes → chunks → vector store
  rag/retriever.py  # semantic retrieval for answers
  tools/py_eval.py  # safe(ish) python evaluation tool
  tools/web_search.py # Tavily‑backed web search tool
  uploads_staged/   # temporary upload staging
  var/, logs/, chroma_db/

frontend/
  src/features/chat/      # Chat experience (Messages, Sources, Tools, ExportPDF)
  src/features/layout/    # Header, Sidebar, Account Section
  src/lib/                # api.ts, sse.ts, chats.ts, auth, modes, markdown utils
  src/routes/             # Chat, Landing, Login/Register, Profile, Reset
  src/components/ui/      # Small, reusable UI primitives
```

---

## Backend (FastAPI)

**Core capabilities**
- **Auth & sessions** — email/password with Argon2; sessions as HTTP‑only cookies
- **Chats** — create, rename, delete; list chat messages; auto‑title from user’s first turn and/or file snippets
- **Streaming** — **SSE** endpoint streams tokens, tool events, source citations, and a `trace_id`
- **Files & RAG** — stage uploads, ingest to the vector store, and cite retrieved passages
- **Tools** — `web_search` (Tavily) and a minimal `py_eval` (sandboxed) for code snippets under guardrails
- **Tracing** — lightweight JSONL audit log; optional Langfuse emission if configured

**Selected models**
- `User`, `Session` (sid/expiry, user agent, IP), `Chat`, `ChatMessage`
- `UserProfile` (display name, **avatar** stored as bytes with MIME & updated_at)

**Selected endpoints**
- `POST /auth/register`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`
- `GET /chats`, `POST /chats`, `PATCH /chats/{id}`, `DELETE /chats/{id}`
- `GET /chats/{id}/messages`, `POST /chats/{id}/auto-title`
- `GET /chat/stream` (SSE) — query params: `q`, `mode` (`offline|web|auto`), `scope` (`user|chat`), optional `chat_id`, `cid`

> The SSE stream emits events: `token` (partial text), `tool` (name/args/result), `sources` (citations), `trace` (UUID), `done` (status), `error`.

---

## Frontend (React/TS)

**UX features**
- **Chat first**: token‑streamed reply, automatic scroll, code‑aware formatting
- **Modes UI**: persist per‑chat `offline | web | auto`, sensible defaults for guests
- **Evidence surfaces**: sources bar (filename, URL host, line ranges), tool calls panel, trace id pill
- **Attachments**: drag‑and‑drop shelf, staged preview, one‑click ingest & commit
- **PDF export**: render current conversation (with sources) into a neat A4 PDF
- **Profile**: display name + avatar (stored and served from backend)

**Client plumbing**
- Small `api.ts` wrapper (JSON + credentials)
- `sse.ts` abstraction with handlers: `onToken`, `onSources`, `onTool`, `onTrace`, `onDone`, `onError`
- `reasoningStorage.ts`: saves last tool/sources/trace per chat in `sessionStorage`

---

## Environment & configuration

Create **`backend/.env`** with the following keys (values are examples/placeholders):

```dotenv
# Database & Auth
DB_URL="mysql+aiomysql://root:password@localhost:3306/agent2"
SESSION_TTL_HOURS=168
COOKIE_NAME=sid
FRONTEND_ORIGIN="http://localhost:5173"

# Ollama (local LLM)
MODEL_NAME="aya-expanse:8b"      # any locally installed model, e.g., llama3.1:8b-instruct
OLLAMA_HOST="http://127.0.0.1:11434"
OLLAMA_NUM_CTX=8192              # optional; backend sends options.num_ctx when >0

# Web search (Tavily)
TAVILY_API_KEY="<your tavily api key>"
TAVILY_BASE="https://api.tavily.com"
RESOLVE_REDIRECTS=1   # follow aggregator links to publisher URLs

# (Optional) Tracing with Langfuse
LANGFUSE_HOST="https://cloud.langfuse.com"
LANGFUSE_PUBLIC_KEY="<pk>"
LANGFUSE_SECRET_KEY="<sk>"
```

> The backend loads `.env` automatically on import and **fails fast** if `DB_URL` is missing, so configuration errors are caught early. If `OLLAMA_HOST` points to localhost and the `ollama` binary is discoverable, the server will attempt to **autostart** `ollama serve` and wait ~30s for readiness.

---

## Running locally

### 0) Ollama (Local LLM)

1. **Install & start**
   - **Windows**: Install the Ollama desktop app, then ensure the service is running (you should see the Ollama tray or be able to run `ollama serve`).
   - **macOS / Linux**: Install Ollama via your platform’s standard method and run `ollama serve`.
2. **Pull a model** (match your `MODEL_NAME`):
   ```bash
   ollama pull aya-expanse:8b   # or: ollama pull llama3.1:8b-instruct
   ```
3. **Smoke test**:
   ```bash
   curl http://127.0.0.1:11434/api/version
   ```
   If you point `OLLAMA_HOST` to a **remote** server, skip local install and ensure the remote is reachable.

> The backend will try to **autostart** `ollama serve` on localhost if it’s not up yet and the binary is found, then poll for readiness for up to ~30 seconds.

### 1) Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2) Frontend

```bash
cd frontend
npm install
# If your backend is not on the default http://localhost:8000, set VITE_API_BASE in a .env file
npm run dev  # Vite dev server
```

> Default CORS/credentials are aligned for `localhost:5173` ⇄ `localhost:8000`. Adjust `FRONTEND_ORIGIN` in `.env` if you change ports or host.

---

## API reference (selected)

> **Auth**
- `POST /auth/register` → `{ id, email, display_name }`
- `POST /auth/login` → sets `sid` cookie
- `POST /auth/logout` → clears session
- `GET /auth/me` → current user or `null`

> **Chats**
- `GET /chats` → list of `{ id, title, updated_at, last_preview }`
- `POST /chats` → create new chat (optional `title`)
- `PATCH /chats/{id}` → rename chat
- `DELETE /chats/{id}` → remove chat
- `GET /chats/{id}/messages` → ordered `{ id, role, content, created_at }`
- `POST /chats/{id}/auto-title` → concise title seeded from first user turn and/or attached files

> **Streaming**
- `GET /chat/stream?q=...&mode=offline|web|auto&scope=user|chat[&chat_id=...][&cid=...]`
  - Events: `token`, `sources`, `tool`, `trace`, `done`, `error`

> **Files**
- Stage → ingest → commit workflow; files are kept under `backend/uploads_staged` until committed. Ingested text becomes retrievable context with line‑ranges for citations.

---

## Retrieval‑Augmented Generation

1. **Upload & Stage**: Files are uploaded to a staging area and inspected for text‑like content.
2. **Ingest**: Text is chunked and embedded into the local vector store.
3. **Retrieve**: On a user query in `offline`/`auto`, top passages are fetched by semantic similarity.
4. **Answer with Evidence**: The model answers and the UI displays source **filename**, optional **URL host**, and **line ranges**.

> In `web`/`auto`, the **web_search** tool queries the internet (Tavily) and normalizes publisher URLs. Citations from the tool are merged with local RAG sources.

---

## Tracing & evaluation

- **Tracing**: a tiny `Tracer` writes compact JSONL logs to `backend/logs/traces.jsonl` and, if `LANGFUSE_*` is set, emits to Langfuse. The **trace id** appears in the UI for quick correlation.
- **Evaluation**: a `ragas_eval.py` hook (offline) can score retrieval quality and save summaries under `backend/logs/ragas/`.

---

## Troubleshooting

- **DB_URL missing**: ensure `backend/.env` exists and `DB_URL` is set to a reachable MySQL DSN.
- **CORS / cookie issues**: confirm `FRONTEND_ORIGIN` matches your Vite dev URL; backend sets `SameSite=Lax` and `HttpOnly` cookies for local dev.
- **No streaming**: verify the frontend points to the correct API base (`VITE_API_BASE`) and that `/chat/stream` is reachable. Firewalls/proxies can block SSE.
- **No sources shown**: for `offline`, confirm you ingested files; for `web`, verify `TAVILY_API_KEY`.
- **Ollama connection failed / 11434 not reachable**: start `ollama serve` (or let backend autostart on localhost), verify `OLLAMA_HOST`.
- **`ERROR: Ollama /api/show ...`**: the model is not installed locally — run `ollama pull <MODEL_NAME>`.
- **Context window issues**: if you need a larger context, set `OLLAMA_NUM_CTX` to a supported value for your model; the backend forwards it as `options.num_ctx`.

---

## Roadmap

- Upload progress + multi‑file batching
- Simple migrations scaffold (Alembic)
- Per‑file privileges & redaction
- More tools (tables/plots, structured browsing)
- Dark‑mode PDF theme

---

## License

Private, academic submission. All third‑party services and trademarks belong to their respective owners.

