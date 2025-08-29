import asyncio
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Dict, Tuple
from typing import Optional, List
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI
from fastapi import HTTPException, Depends, Request
from fastapi import UploadFile, File, Form, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete

from auth import get_db, _current_user
from auth import router as auth_router
from db import init_db, Base, SessionLocal
from eval.ragas_eval import run_evaluation, get_last_summary
from memory import memory
from models import Chat, ChatMessage
from models import FileMeta
from models import User
from rag.ingest import ingest_bytes
from rag.retriever import Retriever
from tracing import Tracer


# ---------------------- Minimal .env loader (no extra dependency) ----------------------
def _load_env_file(path: str) -> None:
    """Best-effort .env parser. Does not override existing environment variables."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                os.environ.setdefault(k, v)
    except FileNotFoundError:
        pass


# Load backend/.env if present
_load_env_file(os.path.join(os.path.dirname(__file__), ".env"))

# --------- Settings (env-driven) ---------
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
MODEL_NAME = os.getenv("MODEL_NAME", "aya-expanse:8b")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
HISTORY_CHAR_BUDGET = int(os.getenv("HISTORY_CHAR_BUDGET", "3500"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "0"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR") or (Path(__file__).parent / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
STAGED_DIR = Path(os.getenv("STAGED_DIR") or (Path(__file__).parent / "uploads_staged"))
STAGED_DIR.mkdir(parents=True, exist_ok=True)
_TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".log"}



def _is_local_ollama(url: str) -> bool:
    try:
        u = urlparse(url)
        host = (u.hostname or "").lower()
        return host in {"127.0.0.1", "localhost", "::1"}
    except Exception:
        return False


def _resolve_ollama_bin() -> str | None:
    # Respect explicit path; otherwise try PATH; otherwise common Windows install path
    candidates = [
        os.getenv("OLLAMA_BIN"),
        shutil.which("ollama"),
        r"C:\Program Files\Ollama\ollama.exe" if os.name == "nt" else None,
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def _spawn_ollama_detached(bin_path: str) -> None:
    # Start "ollama serve" fully detached so the API keeps running if uvicorn reloads
    cmd = [bin_path, "serve"]
    kwargs: dict = dict(
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        cwd=str(Path.home()),
    )
    if sys.platform.startswith("win"):
        # CREATE_NEW_PROCESS_GROUP (0x00000200) | DETACHED_PROCESS (0x00000008)
        kwargs["creationflags"] = 0x00000200 | 0x00000008  # type: ignore[attr-defined]
    else:
        # Start in its own session
        kwargs["preexec_fn"] = os.setsid  # type: ignore[attr-defined]
    subprocess.Popen(cmd, **kwargs)


async def _ollama_ready(client: httpx.AsyncClient) -> bool:
    try:
        r = await client.get(f"{OLLAMA_HOST}/api/version", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


async def ensure_ollama_running() -> None:
    # If host is remote, do not attempt to spawn locally
    async with httpx.AsyncClient() as client:
        if await _ollama_ready(client):
            return

        if not _is_local_ollama(OLLAMA_HOST):
            # Remote or non-local host: just proceed without autostart
            return

        bin_path = _resolve_ollama_bin()
        if not bin_path:
            print("[startup] OLLAMA not reachable and 'ollama' binary not found; skipping autostart")
            return

        print("[startup] OLLAMA not reachable; starting 'ollama serve'…")
        _spawn_ollama_detached(bin_path)

        # Poll for readiness (max ~30s)
        deadline = time.time() + 30
        while time.time() < deadline:
            if await _ollama_ready(client):
                print("[startup] OLLAMA is ready")
                return
            await asyncio.sleep(0.8)

        print("[startup] Warning: OLLAMA did not become ready within 30s")


async def current_user_dep(request: Request, db=Depends(get_db)) -> Optional[User]:
    # Delegate to your real helper
    return await _current_user(request, db)


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "")
    cleaned = _SAFE_NAME_RE.sub("_", base).strip("_")
    return (cleaned or "file")[:200]


def _save_user_file(user_id: int, data: bytes, filename: str) -> tuple[Path, str]:
    """
    Persist raw bytes deterministically under uploads/{user}/{sha256}/{safe_name}
    Returns: (path, sha256_hex)
    """
    sha = hashlib.sha256(data).hexdigest()
    folder = UPLOAD_DIR / str(user_id) / sha
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / _safe_filename(filename)
    if not dest.exists():
        with open(dest, "wb") as fh:
            fh.write(data)
    return dest, sha


def _new_draft_id() -> str:
    return uuid.uuid4().hex


def _draft_folder(user_id: int, draft_id: str) -> Path:
    return STAGED_DIR / str(user_id) / draft_id


def _list_staged_items(user_id: int, draft_id: str) -> list[dict]:
    folder = _draft_folder(user_id, draft_id)
    if not folder.exists():
        return []
    items: list[dict] = []
    # Each file is under .../{sha256}/{filename}
    for sha_dir in folder.iterdir():
        try:
            if not sha_dir.is_dir():
                continue
            sha = sha_dir.name
            for f in sha_dir.iterdir():
                if not f.is_file():
                    continue
                items.append({
                    "sha256_hex": sha,
                    "filename": f.name,
                    "size_bytes": f.stat().st_size,
                    "mime": None,
                })
        except Exception:
            # best-effort listing; ignore broken entries
            continue
    # newest first (by file mtime)
    items.sort(key=lambda it: (_draft_folder(user_id, draft_id) / it["sha256_hex"] / it["filename"]).stat().st_mtime,
               reverse=True)
    return items


# Tools
from tools.web_search import WebSearcher
from tools.py_eval import run_python

app = FastAPI()

# --- CORS (allow cookies from frontend) ---
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(auth_router)

from chats import router as chats_router

app.include_router(chats_router)


@app.on_event("startup")
async def _on_startup():
    await ensure_ollama_running()
    await init_db(Base)


tracer = Tracer()

origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# ---------------------- Helpers ----------------------
def is_hebrew(s: str) -> bool:
    return any("\u0590" <= ch <= "\u05FF" for ch in s or "")


def truncate(txt: str, n: int = 700) -> str:
    txt = (txt or "").replace("\n", " ").strip()
    return txt if len(txt) <= n else txt[: n - 1] + "…"


def _host_from_url(u: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(u).netloc or u
    except Exception:
        return u


def build_context(query: str, docs: List[Dict]) -> Dict:
    cites = []
    ctx_lines = []
    for i, d in enumerate(docs, start=1):
        src = d.get("source") or "unknown"
        preview = truncate(d.get("text", ""))
        sline = int(d.get("start_line") or 0)
        eline = int(d.get("end_line") or 0)
        cites.append({
            "id": i,
            "source": src,
            "start_line": sline,
            "end_line": eline,
            "preview": preview,
            "score": round(float(d.get("score", 0.0)), 4),
        })
        range_txt = f" (lines {sline}-{eline})" if sline and eline else ""
        ctx_lines.append(f"[{i}] {src}{range_txt}\n{preview}\n")
    return {"context_text": "\n".join(ctx_lines), "citations": cites}


def build_web_context(query: str, items: List[Dict]) -> Dict:
    cites = []
    ctx_lines = []
    for i, it in enumerate(items, start=1):
        url = it.get("url") or ""
        title = it.get("title") or ""
        snippet = truncate(it.get("snippet") or title)
        host = _host_from_url(url)
        cites.append({
            "id": i,
            "source": host,
            "preview": snippet,
            "score": round(float(it.get("score", 0.0)), 4),
            "url": url,
        })
        ctx_lines.append(f"[{i}] {title or host} — {host}\n{snippet}\n")
    return {"context_text": "\n".join(ctx_lines), "citations": cites}


def sse(event: str, data: str) -> bytes:
    """
    Build a compliant SSE event. If `data` contains newlines, emit one `data:` line per
    payload line, so the browser reconstructs the exact text with newlines preserved.
    """
    if data is None:
        data = ""
    # Normalize newlines and split into lines
    data = data.replace("\r\n", "\n").replace("\r", "\n")
    payload = "\n".join(f"data: {line}" for line in data.split("\n"))
    return f"event: {event}\n{payload}\n\n".encode("utf-8")


# <<< Helper to render DB messages into the same format as MemoryStore.render
def render_history_from_rows(
        rows: List[Dict[str, str]],
        *,
        lang: str,
        max_chars: int = 40000
) -> str:
    if not rows:
        return ""
    label_user = "משתמש" if lang.lower().startswith("hebrew") else "User"
    label_asst = "עוזר" if lang.lower().startswith("hebrew") else "Assistant"
    lines: List[str] = []
    for m in rows:
        role = m["role"]
        content = (m["content"] or "").strip()
        label = label_user if role == "user" else label_asst
        lines.append(f"{label}: {content}")
    joined = "\n---\n".join(lines)
    if len(joined) <= max_chars:
        return joined
    # Trim from end
    acc: List[str] = []
    total = 0
    for line in reversed(lines):
        piece = (line if not acc else f"---\n{line}")
        if total + len(piece) > max_chars:
            break
        acc.append(line if not acc else f"---\n{line}")
        total += len(piece)
    return "".join(reversed(acc))


# --- Helper: one-shot completion (no stream) ---
async def ollama_generate(prompt: str, model: str = MODEL_NAME) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                # keep it tiny to avoid slowness for a title
                "options": {"num_ctx": 512, "temperature": 0.2}
            },
        )
        r.raise_for_status()
        j = r.json()
        # Ollama returns {"response": "...", ...}
        return (j.get("response") or "").strip()


def _title_prompt(first_user_text: str) -> str:
    return (
        "You are naming a chat based on the user's FIRST message below.\n"
        "Rules: <= 6 words, Title Case, no quotes/emojis, specific but short.\n\n"
        f"First message:\n{first_user_text}\n\n"
        "Output ONLY the title."
    )

def _read_text_from_path(path: Path, max_chars: int = 6000) -> Optional[str]:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    try:
        txt = data.decode("utf-8", errors="ignore")
    except Exception:
        return None
    txt = txt.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not txt:
        return None
    if len(txt) > max_chars:
        txt = txt[: max_chars - 1] + "…"
    return txt

async def _fallback_docs_for_chat(db: AsyncSession, user_id: int, chat_id: int,
                                  max_files: int = 3, per_file_chars: int = 6000) -> List[Dict]:
    """
    Load recent text-like files for this chat directly from disk as a last-resort
    context when the vector retriever returns nothing (e.g., empty/generic query).
    """
    rows = (
        await db.execute(
            select(FileMeta).where(
                FileMeta.user_id == user_id,
                FileMeta.chat_id == chat_id
            ).order_by(FileMeta.created_at.desc()).limit(max_files)
        )
    ).scalars().all()
    out: List[Dict] = []
    for r in rows:
        if not r.sha256_hex:
            continue
        path = UPLOAD_DIR / str(user_id) / r.sha256_hex / r.filename
        if not path.exists():
            continue
        ext = path.suffix.lower()
        if ext not in _TEXT_EXTS:
            # Keep it conservative: only plain-text-ish files in fallback.
            continue
        txt = _read_text_from_path(path, max_chars=per_file_chars)
        if not txt:
            continue
        # Provide a doc entry compatible with build_context()
        # Use rough line numbers for simple preview
        lines = txt.count("\n") + 1
        out.append({
            "text": txt,
            "source": r.filename,
            "start_line": 1,
            "end_line": lines,
            "score": 1.0,
        })
    return out


# ---------------------- Ollama stream ----------------------
async def ensure_model_exists(client: httpx.AsyncClient, model: str) -> Optional[str]:
    try:
        r = await client.post(f"{OLLAMA_HOST}/api/show", json={"name": model})
    except Exception as e:
        return f"Ollama connection failed: {e}"
    if r.status_code != 200:
        return f"Ollama /api/show HTTP {r.status_code}: {r.text}"
    try:
        obj = r.json()
    except Exception as e:
        return f"Ollama /api/show invalid JSON: {e}"
    if "error" in obj:
        return f"Ollama error: {obj.get('error')}"
    return None


async def ollama_stream(prompt: str) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=None) as client:
        err = await ensure_model_exists(client, MODEL_NAME)
        if err:
            yield f"ERROR: {err}"
            return
        url = f"{OLLAMA_HOST}/api/generate"
        payload: Dict[str, object] = {"model": MODEL_NAME, "prompt": prompt, "stream": True}
        # <<< NEW: optionally set num_ctx if provided
        if OLLAMA_NUM_CTX and OLLAMA_NUM_CTX > 0:
            payload["options"] = {"num_ctx": int(OLLAMA_NUM_CTX)}
        try:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield f"ERROR: Ollama /api/generate HTTP {resp.status_code}: {body.decode('utf-8', 'ignore')}"
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in obj:
                        yield f"ERROR: {obj.get('error')}"
                        return
                    if obj.get("done"):
                        break
                    token = obj.get("response")
                    if token:
                        yield token.replace("\r", "")
        except Exception as e:
            yield f"ERROR: Ollama stream exception: {e}"


# ---------------------- Auto tools: heuristics & probe ----------------------
def _have_langchain() -> Tuple[bool, Optional[str]]:
    try:
        from langchain_ollama import ChatOllama  # noqa: F401
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


_HEBREW_DOC_HINTS = [
    r"\bמסמך\b", r"\bמסמכים\b", r"\bקובץ\b", r"\bקבצים\b", r"\bדוגמה\b",
    r"\bשהעליתי\b", r"\bשהעלת\b", r"\bשהעלינו\b", r"\bבמקורות\b", r"\bמקור\b", r"\bמקורות\b",
    r"\bהטקסט\b", r"\bהקובץ\b", r"\bהשורה\b", r"\bשורות\b", r"\bשורה\b",
    r"\bמה כתוב\b", r"\bמה מסופר\b",
]
_EN_DOC_HINTS = [
    r"\bdocument\b", r"\bdocuments\b", r"\bfile\b", r"\bfiles\b", r"\bexample\b", r"\buploaded\b",
    r"\bsources?\b", r"\bcontext\b", r"\blines?\b", r"\bcitation\b", r"\brag\b",
]
_HEBREW_WEB_HINTS = [
    r"\bחפש\b", r"\bחפשי\b", r"\bברשת\b", r"\bבדוק\b", r"\bבדקי\b", r"\bבדוקי\b",
    r"\bבגוגל\b", r"\bחדשות\b", r"\bעדכני\b", r"\bעדכניות\b", r"\bלפי\b\s+\bפורבס\b",
]
_EN_WEB_HINTS = [
    r"\bsearch\b", r"\bgoogle\b", r"\bweb\b", r"\bnews\b", r"\baccording to\b\s+forbes\b",
]

_doc_hint_patterns = [re.compile(p, re.IGNORECASE) for p in (_HEBREW_DOC_HINTS + _EN_DOC_HINTS)]
_web_hint_patterns = [re.compile(p, re.IGNORECASE) for p in (_HEBREW_WEB_HINTS + _EN_WEB_HINTS)]


def _heuristic_wants_rag(q: str) -> bool:
    return any(pat.search(q or "") for pat in _doc_hint_patterns)


def _heuristic_wants_web(q: str) -> bool:
    return any(pat.search(q or "") for pat in _web_hint_patterns)


_HE_STOP = {"של", "עם", "על", "לא", "כן", "אם", "או", "זה", "זו", "אלה", "הוא", "היא", "הם", "הן", "אני", "אתה", "את",
            "אתם", "אתן", "אנחנו", "כל", "גם", "אך", "אבל", "כדי", "כי", "כמו", "אלא", "שרק", "רק", "עוד", "כבר",
            "שהוא", "שהיא", "שזה", "שזו", "שאין", "מאוד", "בה", "בו", "בהם", "בהן", "לפי", "מתוך"}
_EN_STOP = {"the", "a", "an", "and", "or", "but", "if", "so", "to", "of", "in", "on", "for", "by", "with", "as", "at",
            "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those", "from", "i",
            "you", "he", "she", "we", "they", "them", "his", "her", "their", "our", "your", "my", "me"}
_tok_pat = re.compile(r"[\w\u0590-\u05FF]+")


def _extract_keywords(s: str, min_len: int = 4) -> List[str]:
    toks = [t.lower() for t in _tok_pat.findall(s or "")]
    out = []
    for t in toks:
        if len(t) < min_len: continue
        if t in _HE_STOP or t in _EN_STOP: continue
        out.append(t)
    return out


def _auto_decision_prompt() -> str:
    return (
        "Decide whether the user's question likely requires retrieving from the user's LOCAL documents (RAG). "
        "Return ONLY minified JSON with keys: use_retrieval (true/false), query_for_retrieval (string), reason (string). "
        "If the user asks about a specific document, example, file, sources, lines, or uploaded content, choose true."
    )


def _extract_json(text: str) -> Optional[dict]:
    start = text.find("{");
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start: return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


async def _classify_with_llm(user_query: str) -> Optional[Dict]:
    ok, err = _have_langchain()
    if not ok:
        return {"error": f"LangChain not available: {err}"}
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_HOST, temperature=0.0)
    msgs = [SystemMessage(_auto_decision_prompt()), HumanMessage(user_query)]
    ai = await llm.ainvoke(msgs)
    raw = getattr(ai, "content", "") or ""
    obj = _extract_json(raw)
    if not obj:
        return {"error": "Classifier JSON parse failed", "raw": raw}
    return obj


async def _probe_retrieval(user_query: str) -> Tuple[bool, List[Dict], str]:
    retriever = Retriever(k_initial=12, k_final=5, use_reranker=True)
    docs = retriever.retrieve(user_query)
    if not docs:
        return False, [], "Probe: no candidates."
    keywords = _extract_keywords(user_query)
    if keywords:
        for d in docs:
            text = (d.get("text") or "").lower()
            matches = sum(1 for k in keywords if k in text)
            if matches >= 1:
                return True, docs, f"Probe: lexical overlap ({matches}) with top candidates."
    d0 = docs[0].get("distance", None)
    if isinstance(d0, (int, float)) and d0 < 0.55:
        return True, docs, f"Probe: vector match (distance={d0:.3f})."
    return False, docs, "Probe: weak lexical/vector signals."


# ---------------------- Diagnostics ----------------------
@app.get("/tools/web/test")
async def web_test(q: str = Query(..., description="Query")):
    ws = WebSearcher()
    items, provider, debug, timed_out = await ws.search(q, max_results=5, timeout=15.0, time_budget_sec=10.0)
    return {
        "provider": provider,
        "count": len(items),
        "debug": debug + [f"timed_out={1 if timed_out else 0}"],
        "items": items,
    }


@app.get("/tools/py/test")
async def py_test():
    res = await run_python("print(sum(i*i for i in range(1, 11)))")
    return res


# ---------------------- RAG Evaluation ----------------------
@app.get("/eval/ragas/run")
async def eval_ragas_run(limit: int = Query(25, ge=1, le=200)):
    """
    Generate ~limit auto Q&A from your indexed corpus, answer with your pipeline,
    and compute RAGAS scores when available. Writes CSV/JSON (and PNG if possible)
    into backend/logs/ragas/.
    """
    try:
        summary = await run_evaluation(limit=limit)
        return JSONResponse(summary)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/eval/ragas/last")
def eval_ragas_last():
    """
    Return the summary JSON of the last evaluation run (backend/logs/ragas/latest.json).
    """
    try:
        return JSONResponse(get_last_summary())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------- Guardrails & Prompt builders (history-aware) ----------------------
def _base_system_prompt(lang: str, strict: bool = False, mode: str = "general",
                        timed_out_note: Optional[str] = None) -> str:
    if strict:
        policy = (
            "Use ONLY the provided context. If it is insufficient or not relevant, say so briefly and do not invent content."
        )
    else:
        policy = "Use the provided context IF relevant; if missing or insufficient, say so briefly. Be concise and avoid hallucinations."
    web_note = ""
    if mode == "web":
        web_note = " Prefer the most recent, credible sources; when dates are available, anchor statements to those dates."
    elif mode in ("dense", "rerank"):
        web_note = " Your answer MUST rely on the local sources provided; if they are irrelevant to the ask, say so."

    tools_rules = (
        " Tool-use policy: If the user includes a Python code block, execute it in the sandbox FIRST (unless it uses blocked modules like os/subprocess/socket or file/network I/O—in that case, explain refusal). "
        "Never request or suggest reading/writing local files, executing shell commands, installing packages, or making network calls. "
        "If a tool fails or returns no results, say so briefly and suggest one small next step."
    )
    timeout_note = f" Note: {timed_out_note}" if timed_out_note else ""
    return f"You are a helpful assistant. Always answer in {lang}. {policy}{web_note}{tools_rules}{timeout_note}"


def build_prompt_single(
        user_query: str,
        context_text: str,
        *,
        lang: str,
        strict: bool,
        mode: str,
        timed_out_note: Optional[str] = None,
        history_text: str = "",
) -> str:
    sys = _base_system_prompt(lang, strict=strict, mode=mode, timed_out_note=timed_out_note)
    convo = f"\n# Conversation so far:\n{history_text}\n" if history_text else ""
    ctx = f"\n# Context:\n{context_text}\n" if context_text else "\n# Context:\n(none)\n"
    return f"{sys}{convo}{ctx}\n# Question:\n{user_query}\n\n# Answer:"


def build_prompt_mixed(
        user_query: str,
        ctx_rag: str,
        ctx_web: str,
        *,
        lang: str,
        history_text: str = "",
) -> str:
    sys = (
        f"You are a helpful assistant. Always answer in {lang}. "
        f"Consider TWO context sections: (A) Local documents and (B) Web articles. "
        f"Decide relevance: prefer (B) for breaking news/current facts; prefer (A) when the ask references an uploaded file, example, lines, or internal docs. "
        f"If both are relevant, synthesize. If neither is relevant, say so briefly. Be concise and avoid hallucinations."
    )
    convo = f"\n# Conversation so far:\n{history_text}\n" if history_text else ""
    return (
        f"{sys}{convo}\n# Context A — Local (may be partial):\n{ctx_rag}\n\n"
        f"# Context B — Web (may be partial):\n{ctx_web}\n\n"
        f"# Question:\n{user_query}\n\n# Answer:"
    )


# ---------------------- Code detection & web-query sanitization ----------------------
_PY_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_PY_LINE_SIG = re.compile(
    r"""(?mx)
    ^\s*(?:from\s+\w[\w\.]*\s+import\s+|import\s+\w[\w\.]*\s*|print\s*\(|def\s+\w+\s*\(|class\s+\w+\s*\(|for\s+\w+\s+in\s+|while\s+|if\s+__name__\s*==\s*['"]__main__['"])
    """
)


def _extract_python_code(text: str) -> Optional[str]:
    if not text:
        return None
    m = _PY_FENCE_RE.search(text)
    if m:
        code = (m.group(1) or "").strip()
        return code or None
    m2 = re.search(r"```([\s\S]*?)```", text)
    if m2:
        code = (m2.group(1) or "").strip()
        return code or None
    if _PY_LINE_SIG.search(text):
        return text.strip()
    return None


def _sanitize_web_query(text: str) -> str:
    if not text:
        return ""
    s = re.sub(r"```[\s\S]*?```", " ", text)
    if _PY_LINE_SIG.search(s):
        return ""
    s = s.replace("`", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:200]


# ---------------------- API ----------------------
@app.post("/rag/upload")
async def rag_upload(
    request: Request,
    # Accept BOTH "files" (multi) and "file" (single)
    files: Optional[List[UploadFile]] = File(None, alias="files"),
    file: Optional[UploadFile] = File(None, alias="file"),
    # Accept scope and chat_id from query OR from form-data
    scope_q: Optional[str] = Query(None),
    scope_f: Optional[str] = Form(None),
    chat_id_q: Optional[int] = Query(None),
    chat_id_f: Optional[int] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload TXT/PDF/DOCX/MD → persist file bytes+metadata AND chunk+embed into Chroma.
    Scope:
      - "user": global docs (chat_id NULL)
      - "chat": docs attached to a specific chat
    """
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Collect uploads
    uploads: List[UploadFile] = []
    if files:
        uploads.extend(files)
    if file:
        uploads.append(file)
    if not uploads:
        raise HTTPException(status_code=400, detail='No files provided. Use "files" (multi) or "file" (single).')

    # Scope resolution
    scope = (scope_q or scope_f or "user").strip().lower()
    if scope not in ("user", "chat"):
        raise HTTPException(status_code=422, detail='Invalid "scope". Use "user" or "chat".')

    chat_id = chat_id_q if chat_id_q is not None else chat_id_f
    chat_obj: Optional[Chat] = None
    if scope == "chat":
        if chat_id is None:
            raise HTTPException(status_code=422, detail='"chat_id" is required when scope="chat".')
        chat_obj = await db.get(Chat, chat_id)
        if not chat_obj or chat_obj.user_id != user.id:
            raise HTTPException(status_code=404, detail="Chat not found")

    uid = int(user.id)
    cid = int(chat_obj.id) if chat_obj else None
    ns_user = str(uid) if scope == "user" else None
    ns_chat = str(cid) if scope == "chat" else None

    total_chunks = 0
    accepted: List[str] = []
    rejected: List[str] = []

    for up in uploads:
        try:
            data = await up.read()
            dest_path, sha_hex = _save_user_file(uid, data, up.filename)
            size_bytes = len(data)

            # PRE-CHECK: if this (user, sha) already exists, update it to point to this chat (if scope=chat).
            stmt = select(FileMeta).where(
                FileMeta.user_id == uid,
                FileMeta.sha256_hex == sha_hex,
            ).limit(1)
            existing = (await db.execute(stmt)).scalar_one_or_none()

            if existing:
                # Align metadata and chat binding (move from global → chat, or keep as-is)
                changed = False
                if existing.filename != dest_path.name:
                    existing.filename = dest_path.name
                    changed = True
                if existing.mime != up.content_type:
                    existing.mime = up.content_type
                    changed = True
                if existing.size_bytes != size_bytes:
                    existing.size_bytes = size_bytes
                    changed = True
                if cid is not None and existing.chat_id != cid:
                    existing.chat_id = cid
                    changed = True
                if changed:
                    await db.commit()
            else:
                # Fresh row
                fm = FileMeta(
                    user_id=uid,
                    chat_id=cid,
                    filename=dest_path.name,
                    mime=up.content_type,
                    size_bytes=size_bytes,
                    sha256_hex=sha_hex,
                )
                db.add(fm)
                await db.commit()

            # Embed into Chroma — accept even if 0 chunks (tiny files)
            try:
                added = ingest_bytes(filename=up.filename, data=data, user_id=ns_user, chat_id=ns_chat)
                if added > 0:
                    total_chunks += added
                accepted.append(up.filename)
            except Exception:
                # Save succeeded, but indexing failed => still accept the file itself
                accepted.append(up.filename)

        except Exception:
            rejected.append(up.filename)

    return {
        "ok": True,
        "scope": scope,
        "chat_id": ns_chat,
        "files_received": len(uploads),
        "files_indexed": len(accepted),
        "files_skipped": rejected,
        "chunks": total_chunks,
    }


@app.get("/chat/stream")
async def chat_stream(
        q: str = Query(..., description="User query"),
        mode: str = Query("auto", pattern="^(auto|none|dense|rerank|web)$"),
        cid: str = Query("default", min_length=1, max_length=64, pattern=r"^[\w\-]+$"),
        chat_id: Optional[int] = Query(None, ge=1),  # persisted chat id (authenticated)
        scope: str = Query("user", pattern="^(user|chat)$"),  # retrieval namespace scope
        request: Request = None,  # to identify current user
):
    """
    Modes:
      - auto   : Probe local RAG; maybe web if heuristics or weak RAG. Mix contexts smartly.
      - none   : no retrieval (pure LLM)
      - dense  : dense retrieval only (STRICT grounding)
      - rerank : dense + reranker (STRICT grounding)
      - web    : ALWAYS run web search (with up to 60s budget).

    Namespacing:
      - scope=user : retrieval filtered to the current user_id (if logged-in)
      - scope=chat : retrieval filtered to this chat_id (must provide chat_id; falls back to user if missing)
    """

    async def gen() -> AsyncGenerator[bytes, None]:
        try:
            lang = "Hebrew" if is_hebrew(q) else "English"
            used_tool = False

            # ---- Determine current user and (optional) DB chat + load history ----
            db_user: Optional[User] = None
            db_chat: Optional[Chat] = None
            db_history_rows: List[Dict[str, str]] = []

            async with SessionLocal() as db:
                db_user = await _current_user(request, db)
                if db_user and chat_id:
                    # verify ownership
                    db_chat = await db.get(Chat, chat_id)
                    if db_chat and db_chat.user_id != db_user.id:
                        db_chat = None
                    if db_chat:
                        # Load previous turns safely (Row -> mapping)
                        res = await db.execute(
                            ChatMessage.__table__
                            .select()
                            .where(ChatMessage.__table__.c.chat_id == chat_id)
                            .order_by(ChatMessage.__table__.c.id.asc())
                        )
                        for row in res.mappings():
                            db_history_rows.append({
                                "role": row["role"],
                                "content": row["content"],
                            })

            # tracing
            trace_id = tracer.start_trace(q, mode)
            yield sse("trace", json.dumps({"id": trace_id}))

            citations_combined: List[Dict] = []
            prompt = ""
            resp_buf: List[str] = []

            # Build history text BEFORE appending this user turn
            if db_user and db_chat:
                history_text = render_history_from_rows(db_history_rows, lang=lang, max_chars=HISTORY_CHAR_BUDGET)
            else:
                history_text = memory.render(cid, lang=lang, max_chars=HISTORY_CHAR_BUDGET)

            # Persist/append the current user turn
            if db_user and db_chat:
                async with SessionLocal() as db:
                    # attach new user message
                    db.add(ChatMessage(chat_id=db_chat.id, role="user", content=q))
                    # bump chat timestamp
                    chat_obj = await db.get(Chat, db_chat.id)
                    if chat_obj:
                        chat_obj.updated_at = dt.datetime.now(dt.timezone.utc)
                    await db.commit()
            else:
                memory.append(cid, "user", q)

            # ------------------- Python sandbox path -------------------
            py_code = _extract_python_code(q)
            if py_code:
                used_tool = True
                res = await run_python(py_code, timeout_sec=5.0)

                tool_payload = {
                    "name": "py_eval",
                    "args": {"language": "python", "timeout_s": 5},
                    "result": res,
                }
                yield sse("tool", json.dumps(tool_payload, ensure_ascii=False))
                tracer.log_tool(trace_id, "py_eval", {"language": "python", "timeout_s": 5}, res)

                ctx_text = (
                    f"[Code]\n{py_code}\n\n"
                    f"[Program output]\n{res.get('stdout') or '(no stdout)'}\n\n"
                    f"[Errors]\n{res.get('stderr') or '(none)'}\n\n"
                    f"[Meta] ok={res.get('ok')} timeout={res.get('timeout')} duration_ms={res.get('duration_ms')}\n"
                )
                prompt = build_prompt_single(
                    "Please present the program output clearly. If there were errors, explain the most likely cause and suggest a minimal fix.",
                    ctx_text,
                    lang=lang,
                    strict=False,
                    mode="general",
                    history_text=history_text,
                )
                async for tok in ollama_stream(prompt):
                    resp_buf.append(tok)
                    yield sse("token", tok)
                    await asyncio.sleep(0)

                assistant_text = "".join(resp_buf)

                if db_user and db_chat:
                    async with SessionLocal() as db:
                        db.add(ChatMessage(chat_id=db_chat.id, role="assistant", content=assistant_text))
                        chat_obj = await db.get(Chat, db_chat.id)
                        if chat_obj:
                            chat_obj.updated_at = dt.datetime.now(dt.timezone.utc)
                        await db.commit()
                else:
                    memory.append(cid, "assistant", assistant_text)

                tracer.end_trace(trace_id, prompt, assistant_text, citations=[], ok=True)
                yield sse("sources", "[]")
                yield sse("done", json.dumps({"ok": True, "used_tool": used_tool}))
                return

            # ------------------- RAG where-filter (namespacing) --------------------
            rag_where: Optional[Dict[str, str]] = None
            if db_user:
                if scope == "chat" and chat_id:
                    rag_where = {"chat_id": str(chat_id)}
                else:
                    rag_where = {"user_id": str(db_user.id)}

            # ------------------- Regular modes ------------------------------------
            if mode == "none":
                ctx_text = ""
                if (scope == "chat") and db_user and chat_id:
                    async with SessionLocal() as db2:
                        fd = await _fallback_docs_for_chat(db2, int(db_user.id), int(chat_id))
                    if fd:
                        ctx_text = build_context(q, fd)["context_text"]
                prompt = build_prompt_single(
                    q, ctx_text, lang=lang, strict=False, mode="general", history_text=history_text
                )

            elif mode in ("dense", "rerank"):
                retriever = Retriever(
                    k_initial=12,
                    k_final=int(os.getenv("RAG_TOP_K", "5")),
                    use_reranker=(mode == "rerank"),
                )
                docs = retriever.retrieve(q, where=rag_where)

                # >>> Fallback: if nothing retrieved and we’re in chat scope, inject recent chat files as context.
                if (not docs) and (scope == "chat") and db_user and chat_id:
                    async with SessionLocal() as db2:
                        fd = await _fallback_docs_for_chat(db2, int(db_user.id), int(chat_id))
                    if fd:
                        docs = fd

                ctx_rag = build_context(q, docs)
                prompt = build_prompt_single(
                    q, ctx_rag["context_text"], lang=lang, strict=True, mode=mode, history_text=history_text
                )
                citations_combined = ctx_rag["citations"]
                used_tool = True

                ev = {
                    "name": "retrieve_knowledge",
                    "args": {"query": q, "mode": mode, "where": rag_where},
                    "result": ctx_rag["citations"],
                }
                yield sse("tool", json.dumps(ev, ensure_ascii=False))
                tracer.log_tool(
                    trace_id,
                    "retrieve_knowledge",
                    {"query": q, "mode": mode, "where": rag_where},
                    ctx_rag["citations"],
                )

            elif mode == "web":
                ws = WebSearcher()
                sanitized = _sanitize_web_query(q)
                items: List[Dict] = []
                provider = "none"
                debug: List[str] = []
                timed_out_note = None
                timed_out = False

                if sanitized:
                    items, provider, debug, timed_out = await ws.search(
                        sanitized, max_results=5, timeout=15.0, time_budget_sec=60.0
                    )
                    if timed_out:
                        timed_out_note = (
                            "No web results were found within a 60-second search budget. "
                            "Inform the user and suggest refining the query."
                        )
                ev = {
                    "name": "web_search",
                    "args": {"query": sanitized or "(skipped due to code-like text)"},
                    "result": {"provider": provider, "count": len(items), "debug": debug, "items": items},
                }
                yield sse("tool", json.dumps(ev, ensure_ascii=False))
                tracer.log_tool(
                    trace_id,
                    "web_search",
                    {"query": sanitized or "(skipped)"},
                    {"provider": provider, "count": len(items), "debug": debug, "items": items},
                )
                used_tool = True

                ctx_web = build_web_context(q, items)
                citations_combined = ctx_web["citations"]
                if items:
                    prompt = build_prompt_single(
                        q, ctx_web["context_text"], lang=lang, strict=False, mode="web", history_text=history_text
                    )
                else:
                    prompt = build_prompt_single(
                        q, "", lang=lang, strict=False, mode="web", timed_out_note=timed_out_note,
                        history_text=history_text
                    )

            else:  # auto
                retriever = Retriever(
                    k_initial=12,
                    k_final=int(os.getenv("RAG_TOP_K", "5")),
                    use_reranker=True,
                )
                rag_docs = retriever.retrieve(q, where=rag_where)

                # >>> Fallback: if nothing retrieved and we’re in chat scope, inject recent chat files as context.
                if (not rag_docs) and (scope == "chat") and db_user and chat_id:
                    async with SessionLocal() as db2:
                        fd = await _fallback_docs_for_chat(db2, int(db_user.id), int(chat_id))
                    if fd:
                        rag_docs = fd

                ctx_rag = build_context(q, rag_docs)
                ev_probe = {
                    "name": "retrieve_knowledge_probe",
                    "args": {"query": q, "where": rag_where},
                    "result": ctx_rag["citations"],
                }
                yield sse("tool", json.dumps(ev_probe, ensure_ascii=False))
                tracer.log_tool(
                    trace_id, "retrieve_knowledge_probe", {"query": q, "where": rag_where}, ctx_rag["citations"]
                )

                rag_used = len(rag_docs) > 0

                run_web = _heuristic_wants_web(q)
                rag_keywords = _extract_keywords(q)
                rag_overlap = 0
                if rag_keywords and rag_docs:
                    text0 = (rag_docs[0].get("text") or "").lower()
                    rag_overlap = sum(1 for k in rag_keywords if k in text0)
                rag_strong = (rag_overlap >= 1) or (
                        rag_docs
                        and isinstance(rag_docs[0].get("distance", 1.0), (int, float))
                        and float(rag_docs[0]["distance"]) < 0.55
                )

                ctx_web = {"context_text": "", "citations": []}
                if run_web or not rag_strong:
                    # (web search branch unchanged) ...
                    ...
                prompt = build_prompt_mixed(
                    q, ctx_rag["context_text"], ctx_web["context_text"], lang=lang, history_text=history_text
                )
                citations_combined = ctx_rag["citations"] + ctx_web["citations"]
                used_tool = rag_used or (len(ctx_web["citations"]) > 0)

            # ---- Stream assistant answer ----
            async for tok in ollama_stream(prompt):
                resp_buf.append(tok)
                yield sse("token", tok)
                await asyncio.sleep(0)

            # Push sources bar
            yield sse("sources", json.dumps(citations_combined, ensure_ascii=False))

            assistant_text = "".join(resp_buf)

            if db_user and db_chat:
                async with SessionLocal() as db:
                    db.add(ChatMessage(chat_id=db_chat.id, role="assistant", content=assistant_text))
                    chat_obj = await db.get(Chat, db_chat.id)
                    if chat_obj:
                        chat_obj.updated_at = dt.datetime.now(dt.timezone.utc)
                    await db.commit()
            else:
                memory.append(cid, "assistant", assistant_text)

            tracer.end_trace(trace_id, prompt, assistant_text, citations_combined, ok=True)
            yield sse("done", json.dumps({"ok": True, "used_tool": used_tool}))
        except Exception as e:
            yield sse("token", f"ERROR: backend exception: {e}")
            yield sse("sources", "[]")
            try:
                assistant_text = "".join(resp_buf) if 'resp_buf' in locals() else ""
                tracer.end_trace(trace_id, prompt if 'prompt' in locals() else "", assistant_text, [], ok=False)
            except Exception:
                pass
            yield sse("done", json.dumps({"ok": False}))

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


@app.post("/chats/{chat_id}/auto-title")
async def auto_title(chat_id: int, request: Request, db=Depends(get_db)):
    """
    Generates a concise title for the given chat based on its FIRST user message,
    saves it to the DB, and returns { id, title }.
    Auth: current session cookie (same as other endpoints).
    """
    # Deps/typing (keep local to avoid file-wide import churn)
    from sqlalchemy.ext.asyncio import AsyncSession

    assert isinstance(db, AsyncSession)

    # Identify current user (same helper used elsewhere in main.py)
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Load chat and validate ownership
    chat = await db.get(Chat, chat_id)
    if not chat or chat.user_id != user.id:
        raise HTTPException(status_code=404, detail="Chat not found")

    # First user message (oldest) for this chat
    q = (
        ChatMessage.__table__
        .select()
        .where(
            ChatMessage.__table__.c.chat_id == chat_id,
            ChatMessage.__table__.c.role == "user",
        )
        .order_by(ChatMessage.__table__.c.created_at.asc())
        .limit(1)
    )
    res = await db.execute(q)
    first_row = res.mappings().first()
    if not first_row:
        raise HTTPException(status_code=400, detail="No user message found for this chat")

    first_user_text = (first_row.get("content") or "").strip()
    raw = await ollama_generate(_title_prompt(first_user_text))
    title = (raw or "").strip().strip('"').strip()
    if not title:
        title = "Chat"
    title = title[:80]  # conservative cap

    chat.title = title
    await db.commit()

    return {"id": chat.id, "title": chat.title}


@app.get("/files")
async def list_files(
        request: Request,
        scope: str = Query("user", pattern="^(user|chat)$"),
        chat_id: Optional[int] = Query(None, ge=1),
        db: AsyncSession = Depends(get_db),
):
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if scope == "chat":
        if chat_id is None:
            raise HTTPException(status_code=422, detail='"chat_id" is required when scope="chat".')
        # Ensure ownership
        chat = await db.get(Chat, chat_id)
        if not chat or chat.user_id != user.id:
            raise HTTPException(status_code=404, detail="Chat not found")
        stmt = select(FileMeta).where(FileMeta.user_id == user.id, FileMeta.chat_id == chat_id).order_by(
            FileMeta.created_at.desc())
    else:
        # user-scope = global docs (no chat)
        stmt = select(FileMeta).where(FileMeta.user_id == user.id, FileMeta.chat_id.is_(None)).order_by(
            FileMeta.created_at.desc())

    rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "mime": r.mime,
            "size_bytes": r.size_bytes,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@app.delete("/files/{file_id}")
async def delete_file(file_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Load file and verify ownership
    fm = await db.get(FileMeta, file_id)
    if not fm or fm.user_id != user.id:
        raise HTTPException(status_code=404, detail="File not found")

    # Capture path and sha to possibly clean up folder if unused
    path = UPLOAD_DIR / str(user.id) / (fm.sha256_hex or "") / fm.filename
    sha_hex = fm.sha256_hex

    await db.execute(delete(FileMeta).where(FileMeta.id == file_id, FileMeta.user_id == user.id))
    await db.commit()

    # If no remaining rows use this sha for this user, delete the folder safely
    if sha_hex:
        still = await db.execute(
            select(FileMeta.id).where(FileMeta.user_id == user.id, FileMeta.sha256_hex == sha_hex).limit(1)
        )
        if still.first() is None:
            try:
                # remove file and its sha folder if empty
                if path.exists():
                    try:
                        path.unlink()
                    except IsADirectoryError:
                        pass
                folder = path.parent
                if folder.exists() and not any(folder.iterdir()):
                    folder.rmdir()
            except Exception:
                # best-effort cleanup only
                pass

    return {"ok": True}

@app.post("/files/stage")
async def stage_files(
    request: Request,
    files: Optional[List[UploadFile]] = File(None, alias="files"),
    draft_id_f: Optional[str] = Form(None, alias="draft_id"),
    db: AsyncSession = Depends(get_db),
):
    """
    Stage files BEFORE a chat exists. Files are written to disk only, no DB rows.
    Returns a draft_id to reuse for more uploads/list/deletes.
    """
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not files:
        raise HTTPException(status_code=400, detail='No files provided. Use "files".')

    draft_id = (draft_id_f or "").strip() or _new_draft_id()
    base = _draft_folder(user.id, draft_id)
    base.mkdir(parents=True, exist_ok=True)

    for up in files:
        data = await up.read()
        sha = hashlib.sha256(data).hexdigest()
        dest_dir = base / sha
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / _safe_filename(up.filename)
        if not dest.exists():
            with open(dest, "wb") as fh:
                fh.write(data)

    return {"ok": True, "draft_id": draft_id, "items": _list_staged_items(user.id, draft_id)}


@app.get("/files/stage")
async def list_staged(
    request: Request,
    draft_id: str = Query(..., min_length=6),
    db: AsyncSession = Depends(get_db),
):
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return _list_staged_items(user.id, draft_id)


@app.delete("/files/stage/{sha256_hex}")
async def delete_staged(
    sha256_hex: str,
    request: Request,
    draft_id: str = Query(..., min_length=6),
    db: AsyncSession = Depends(get_db),
):
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    folder = _draft_folder(user.id, draft_id) / sha256_hex
    if not folder.exists():
        return {"ok": True}
    try:
        for p in folder.iterdir():
            try:
                p.unlink()
            except Exception:
                pass
        folder.rmdir()
    except Exception:
        pass
    return {"ok": True}



# --- commit staged files into a specific chat ---
@app.post("/files/commit")
async def commit_staged(
    request: Request,
    payload: dict = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Commit staged files to a given chat_id:
      - Creates/updates FileMeta rows (one per (user, sha))
      - Indexes into Chroma under the chat namespace
      - Deletes staged copies
    Body: { "draft_id": string, "chat_id": int }
    """
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    draft_id = str(payload.get("draft_id") or "").strip()
    chat_id = payload.get("chat_id")
    if not draft_id:
        raise HTTPException(status_code=422, detail='"draft_id" is required')
    if not isinstance(chat_id, int):
        raise HTTPException(status_code=422, detail='"chat_id" must be an integer')

    chat_obj = await db.get(Chat, chat_id)
    if not chat_obj or chat_obj.user_id != user.id:
        raise HTTPException(status_code=404, detail="Chat not found")

    staged = _list_staged_items(user.id, draft_id)
    if not staged:
        return {"ok": True, "count": 0}

    ns_chat = str(chat_id)
    committed = 0
    folder = _draft_folder(user.id, draft_id)

    # IMPORTANT: capture plain ids BEFORE any commit/rollback to avoid
    # attribute refreshes that can trigger async I/O inside a sync loader.
    uid = int(user.id)
    cid = int(chat_obj.id)

    for item in staged:
        fpath = None
        try:
            sha = item["sha256_hex"]
            fn = item["filename"]
            fpath = folder / sha / fn
            if not fpath.exists():
                continue
            data = fpath.read_bytes()

            # Persist the bytes (idempotent per (user, sha))
            dest_path, sha_hex = _save_user_file(uid, data, fn)
            size_bytes = len(data)

            fm = FileMeta(
                user_id=uid,
                chat_id=cid,
                filename=dest_path.name,
                mime=None,
                size_bytes=size_bytes,
                sha256_hex=sha_hex,
            )

            try:
                db.add(fm)
                await db.commit()
            except IntegrityError:
                # Row already exists for (user, sha). Align chat_id.
                await db.rollback()
                stmt = select(FileMeta).where(
                    FileMeta.user_id == uid, FileMeta.sha256_hex == sha_hex
                ).limit(1)
                exists = (await db.execute(stmt)).scalar_one_or_none()
                if exists:
                    if exists.chat_id != cid:
                        exists.chat_id = cid
                        await db.commit()
                else:
                    # very unlikely race; retry once
                    db.add(fm)
                    await db.commit()

            # Index under chat namespace (non-fatal if it fails)
            try:
                added = ingest_bytes(filename=fn, data=data, user_id=None, chat_id=ns_chat)
                # treat >=0 as success for commit counting; indexing may produce 0 chunks for tiny text
                if added >= 0:
                    committed += 1
            except Exception:
                pass

        finally:
            # cleanup staged file/folder
            if fpath is not None:
                try:
                    try:
                        fpath.unlink()
                    except Exception:
                        pass
                    sha_dir = (folder / item["sha256_hex"])
                    if sha_dir.exists() and not any(sha_dir.iterdir()):
                        sha_dir.rmdir()
                except Exception:
                    pass

    # delete draft folder if empty
    try:
        if folder.exists() and not any(folder.iterdir()):
            folder.rmdir()
    except Exception:
        pass

    return {"ok": True, "count": committed}
