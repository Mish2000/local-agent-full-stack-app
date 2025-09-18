import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from collections import deque
from typing import AsyncGenerator, Dict, Tuple
from typing import List
from typing import Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
from argon2.exceptions import VerifyMismatchError
from fastapi import FastAPI
from fastapi import Form, Query, Body
from fastapi import UploadFile, File, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_db, _current_user
from auth import ph  # reuse the Argon2 hasher from auth.py
from auth import router as auth_router
from db import init_db, Base, SessionLocal
from eval.ragas_eval import run_evaluation, get_last_summary
from memory import memory
from models import Chat, ChatMessage
from models import FileMeta
from models import User
from models import UserProfile
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
IL_TZ = ZoneInfo("Asia/Jerusalem")

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
_DOCX_EXTS = {".docx"}
_PDF_EXTS = {".pdf"}
AVATAR_DIR = UPLOAD_DIR / "avatars"
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
_ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/webp"}

# --- AUTO routing helpers (drop under your other helpers in main.py) ---
_RECENCY_TERMS_EN = (
    "current", "today", "now", "this week", "latest", "live", "breaking",
    "price", "rate", "exchange rate", "forecast", "weather", "score", "result"
)
_RECENCY_TERMS_HE = (
    "עכשיו", "היום", "עדכני", "מעודכן", "חיים", "מחיר", "שער", "שער החליפין",
    "תחזית", "מזג אוויר", "תוצאה", "תוצאות", "חדשות", "אחרונות"
)
_CURRENCY_LIKE = (
    "usd", "eur", "ils", "$", "€", "₪",
    "dollar", "euro", "shekel", "exchange rate", "fx",
    "דולר", "אירו", "שקל", "שער הדולר", "שער האירו", "שער השקל"
)
_CRYPTO_LIKE = (
    "bitcoin", "btc", "ether", "eth", "קריפטו", "ביטקוין", "אתריום",
    "altcoin", "token", "coin", "מחיר ביטקוין", "מחיר אתריום"
)
_WEATHER_LIKE = ("weather", "forecast", "מזג אוויר", "תחזית")
_SPORTS_LIKE = ("score", "scores", "result", "nba", "nfl", "mlb", "nhl",
                "תוצאה", "ליגה", "מונדיאל", "תוצאות")


def _needs_recency(q: str) -> bool:
    qn = (q or "").lower()
    return any(t in qn for t in _RECENCY_TERMS_EN) or any(t in q for t in _RECENCY_TERMS_HE)


def _classify_web_strategy(q: str) -> dict:
    """
    Decide topic + recency window for WebSearcher. Returns kwargs for WebSearcher.search.
    """
    qn = (q or "").lower()
    topic = None
    days = None

    if any(t in qn for t in _CURRENCY_LIKE):
        topic, days = "fx", 1
    elif any(t in qn for t in _CRYPTO_LIKE):
        topic, days = "crypto", 1
    elif any(t in qn for t in _WEATHER_LIKE):
        topic, days = "weather", 3
    elif any(t in qn for t in _SPORTS_LIKE):
        topic, days = "sports", 3
    elif _needs_recency(q):
        topic, days = "news", 3

    # Conservative default if we couldn't classify but still want recency
    if days is None and _needs_recency(q):
        days = 7

    return {
        "topic": topic,  # WebSearcher understands topic; safe to be None
        "days": days,  # Prefer tight windows when recognized
        "auto_parameters": True,  # Let provider layer tighten freshness/quality
    }


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


logger = logging.getLogger(__name__)


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


def _read_docx_path(path: Path) -> Optional[str]:
    try:
        import docx2txt  # lightweight; used in backend/rag/ingest.py as well
        txt = docx2txt.process(str(path)) or ""
        return txt.strip() or None
    except Exception:
        return None


def _read_pdf_path(path: Path) -> Optional[str]:
    try:
        from pypdf import PdfReader  # lightweight; used in backend/rag/ingest.py as well
        reader = PdfReader(str(path))
        txt = "\n".join(page.extract_text() or "" for page in reader.pages)
        return (txt or "").strip() or None
    except Exception:
        return None


async def _ensure_user_profile_columns(db: AsyncSession) -> None:
    # avatar_kind
    q1 = await db.execute(text("""
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'user_profiles'
          AND COLUMN_NAME = 'avatar_kind'
    """))
    has_kind = (q1.scalar() or 0) > 0

    # avatar_value
    q2 = await db.execute(text("""
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'user_profiles'
          AND COLUMN_NAME = 'avatar_value'
    """))
    has_value = (q2.scalar() or 0) > 0

    # NEW: avatar_blob
    q3 = await db.execute(text("""
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'user_profiles'
          AND COLUMN_NAME = 'avatar_blob'
    """))
    has_blob = (q3.scalar() or 0) > 0

    # NEW: avatar_mime
    q4 = await db.execute(text("""
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'user_profiles'
          AND COLUMN_NAME = 'avatar_mime'
    """))
    has_mime = (q4.scalar() or 0) > 0

    # NEW: avatar_updated_at
    q5 = await db.execute(text("""
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'user_profiles'
          AND COLUMN_NAME = 'avatar_updated_at'
    """))
    has_updated = (q5.scalar() or 0) > 0

    if not has_kind:
        await db.execute(text("ALTER TABLE user_profiles ADD COLUMN avatar_kind VARCHAR(16) NOT NULL DEFAULT ''"))
    if not has_value:
        await db.execute(text("ALTER TABLE user_profiles ADD COLUMN avatar_value VARCHAR(255) NOT NULL DEFAULT ''"))
    if not has_blob:
        await db.execute(text("ALTER TABLE user_profiles ADD COLUMN avatar_blob LONGBLOB NULL"))
    if not has_mime:
        await db.execute(text("ALTER TABLE user_profiles ADD COLUMN avatar_mime VARCHAR(32) NOT NULL DEFAULT ''"))
    if not has_updated:
        await db.execute(text("ALTER TABLE user_profiles ADD COLUMN avatar_updated_at DATETIME(6) NULL"))

    if (not has_kind) or (not has_value) or (not has_blob) or (not has_mime) or (not has_updated):
        await db.commit()


class ProfileAccountIn(BaseModel):
    display_name: Optional[str] = Field(default=None, max_length=120)
    current_password: Optional[str] = Field(default=None, min_length=0, max_length=128)
    new_password: Optional[str] = Field(default=None, min_length=8, max_length=128)


class ProfileSettingsIO(BaseModel):
    instruction_enabled: bool = True
    instruction_text: str = Field(default="", max_length=8000)
    avatar_kind: str = Field(default="")  # "", "system", "upload"
    avatar_value: str = Field(default="", max_length=255)


# Tools
from tools.web_search import WebSearcher, _parse_date_heuristic, _apply_relevance_filter, _sort_results, _diversify, \
    _annotate_debug_fields
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


# -------- Intent-aware classification (drop-in) --------
from dataclasses import dataclass
from enum import Enum
import re


class Intent(str, Enum):
    FILE_QA = "file_qa"
    CREATIVE = "creative"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CODING = "coding"
    MATH = "math"
    DYNAMIC_FACT = "dynamic_fact"  # news/markets/weather/sports/time-sensitive
    STATIC_FACT = "static_fact"  # stable encyclopedic facts
    PERSONAL = "personal"  # advice/opinion/chat
    OTHER = "other"


@dataclass
class IntentDecision:
    intent: Intent
    wants_rag: bool
    wants_web: bool
    reason: str


# Signals (HE+EN)
_HE_CREATIVE = [
    r"\bסיפור\b", r"\bשיר\b", r"\bבדיחה\b", r"\bפואמה\b", r"\bהייקו\b", r"\bדיאלוג\b",
    r"\bכת(ו|בי)\b", r"\bתכתבי\b", r"\bתחברי\b", r"\bדמיינ(י)?\b", r"\bדמות\b"
]
_EN_CREATIVE = [
    r"\bstory\b", r"\bpoem\b", r"\bhaiku\b", r"\bjoke\b", r"\bsong\b", r"\blyrics\b",
    r"\brole\s*play\b", r"\broleplay\b", r"\bmonologue\b", r"\bdialogue\b", r"\bwrite\b", r"\bcompose\b"
]

_HE_TRANSLATE = [r"\bתרג(ו|מי|ם|ום)\b", r"\bלתרגם\b", r"\bתרגום\b"]
_EN_TRANSLATE = [r"\btranslate\b", r"\btranslation\b"]

_HE_SUMMARY = [r"\bסכ(ם|מי|מה)\b", r"\bתקציר\b", r"\bלסכם\b"]
_EN_SUMMARY = [r"\bsummariz(e|e it|ation)\b", r"\btl;dr\b"]

_HE_MATH = [r"\bחשב\b", r"\bחישוב\b", r"\bאחוז\b", r"\bשבר\b"]
_EN_MATH = [r"\bcalc(ulate|ulation)?\b", r"\bpercentage\b", r"\bpercent\b"]

# You already have these in code; we reuse them:
# - _heuristic_wants_rag (doc/file hints)【turn3file6†main.py†L10-L12】
# - _heuristic_wants_web (explicit “search/google/news” wording)【turn3file6†main.py†L14-L15】
# - _needs_recency + keyword families (fx/crypto/weather/sports)【turn3file8†main.py†L9-L31】
# - _is_time_sensitive patterns【turn3file3†main.py†L5-L20】

_cre_pat = [re.compile(p, re.IGNORECASE) for p in (_HE_CREATIVE + _EN_CREATIVE)]
_trn_pat = [re.compile(p, re.IGNORECASE) for p in (_HE_TRANSLATE + _EN_TRANSLATE)]
_sum_pat = [re.compile(p, re.IGNORECASE) for p in (_HE_SUMMARY + _EN_SUMMARY)]
_mth_pat = [re.compile(p, re.IGNORECASE) for p in (_HE_MATH + _EN_MATH)]

_Q_WORDS_EN = re.compile(r"\b(what|who|when|where|why|how|define|explain)\b", re.IGNORECASE)
_Q_WORDS_HE = re.compile(r"\b(מה|מי|מתי|איפה|למה|כיצד|הגדרה|הסבר)\b")


def _looks_creative(q: str) -> bool:
    return any(p.search(q or "") for p in _cre_pat)


def _looks_translation(q: str) -> bool:
    return any(p.search(q or "") for p in _trn_pat)


def _looks_summary(q: str) -> bool:
    return any(p.search(q or "") for p in _sum_pat)


def _looks_math(q: str) -> bool:
    return any(p.search(q or "") for p in _mth_pat) or bool(re.search(r"\d\s*[\+\-\*/]\s*\d", q or ""))


def _looks_coding(q: str) -> bool:
    if _extract_python_code(q):  # you already have this helper【turn3file14†backend.txt†L41-L54】
        return True
    return any(
        s in (q or "").lower() for s in ["stack trace", "traceback", "exception", "error:", "undefined", "segfault"])


def classify_intent(q: str, *, rag_strong: bool, doc_hint: bool) -> IntentDecision:
    qn = (q or "").strip()
    if doc_hint:
        return IntentDecision(Intent.FILE_QA, wants_rag=True, wants_web=False, reason="Explicit file/document hint")
    if _looks_creative(qn):
        return IntentDecision(Intent.CREATIVE, wants_rag=False, wants_web=False, reason="Creative generation")
    if _looks_translation(qn):
        return IntentDecision(Intent.TRANSLATION, wants_rag=rag_strong, wants_web=False, reason="Translation task")
    if _looks_summary(qn):
        # If a URL is present, summary likely needs web. Otherwise, prefer RAG if we have it.
        has_url = bool(re.search(r"https?://", qn))
        return IntentDecision(Intent.SUMMARIZATION, wants_rag=rag_strong and not has_url, wants_web=has_url,
                              reason="Summarization")
    if _looks_coding(qn):
        return IntentDecision(Intent.CODING, wants_rag=False, wants_web=False, reason="Coding/debug task")
    if _looks_math(qn):
        return IntentDecision(Intent.MATH, wants_rag=False, wants_web=False, reason="Calculation/math task")
    if _needs_recency(qn) or _is_time_sensitive(qn):
        return IntentDecision(Intent.DYNAMIC_FACT, wants_rag=rag_strong, wants_web=True,
                              reason="Time-sensitive/dynamic facts")
    if _heuristic_wants_web(qn):  # user said “search the web”, etc.
        return IntentDecision(Intent.DYNAMIC_FACT, wants_rag=rag_strong, wants_web=True,
                              reason="User explicitly asked to search")
    # General Q&A: if it looks like a question but not dynamic, treat as static facts.
    if _Q_WORDS_EN.search(qn) or _Q_WORDS_HE.search(qn):
        return IntentDecision(Intent.STATIC_FACT, wants_rag=rag_strong, wants_web=False,
                              reason="Static/encyclopedic query")
    # Advice/opinion/chatty prompts → keep local
    return IntentDecision(Intent.PERSONAL, wants_rag=rag_strong, wants_web=False, reason="Personal/advice/chat")


def build_web_context(query: str, items: List[Dict]) -> Dict:
    """
    Build a lightweight context block from web search items.
    - No extra ranking/filters here (already done in WebSearcher).
    - Ensures a 'published' alias exists if only 'published_date' was provided.
    """
    cites = []
    ctx_lines = []
    for i, it in enumerate(items, start=1):
        url = it.get("url") or ""
        title = (it.get("title") or "").strip()
        # Prefer snippet, fallback to content, then title
        snippet = truncate(it.get("snippet") or it.get("content") or title)
        host = _host_from_url(url)

        # Normalize date alias for the downstream prompt builder
        published = (it.get("published") or it.get("published_date") or "").strip()
        if published and "published" not in it:
            it["published"] = published

        date_tag = f" ({published[:10]})" if published else ""
        cites.append({
            "id": i,
            "source": host,
            "preview": snippet,
            "score": round(float(it.get("score") or 0.0), 4),
            "url": url,
            "published": published,
        })
        # Include host + (YYYY-MM-DD) to nudge the LLM toward dated, sourced statements
        ctx_lines.append(f"[{i}] {title or host}{date_tag} — {host}\n{snippet}\n")

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


async def _fallback_docs_for_chat(
        db: AsyncSession,
        user_id: int,
        chat_id: int,
        max_files: int = 3,
        per_file_chars: int = 6000,
) -> List[Dict]:
    """
    Load recent files for this chat directly from disk as last-resort context
    when the vector retriever returns nothing (e.g., empty/generic query).

    Now supports TXT/MD/CSV/JSON/LOG *and* PDF/DOCX.
    """
    rows = (
        await db.execute(
            select(FileMeta)
            .where(FileMeta.user_id == user_id, FileMeta.chat_id == chat_id)
            .order_by(FileMeta.created_at.desc())
            .limit(max_files)
        )
    ).scalars().all()

    out: List[Dict] = []
    for r in rows:
        if not r.sha256_hex:
            continue
        path = UPLOAD_DIR / str(user_id) / r.sha256_hex / r.filename
        if not path.exists() or not path.is_file():
            continue

        ext = path.suffix.lower()
        txt: Optional[str] = None

        if ext in _TEXT_EXTS:
            txt = _read_text_from_path(path, max_chars=per_file_chars)
        elif ext in _DOCX_EXTS:
            txt = _read_docx_path(path)
            if txt and len(txt) > per_file_chars:
                txt = txt[: per_file_chars - 1] + "…"
        elif ext in _PDF_EXTS:
            txt = _read_pdf_path(path)
            if txt and len(txt) > per_file_chars:
                txt = txt[: per_file_chars - 1] + "…"
        else:
            # Keep future-proof: unknown types skipped here.
            continue

        if not txt:
            continue

        # Provide a doc entry compatible with build_context()
        lines = txt.count("\n") + 1
        out.append(
            {
                "text": txt,
                "source": r.filename,
                "start_line": 1,
                "end_line": lines,
                "score": 1.0,
            }
        )

    return out


# --- Small helper to ensure a UserProfile row exists for the current user ---
async def _get_or_create_profile(db: AsyncSession, user_id: int) -> UserProfile:
    prof = await db.get(UserProfile, user_id)
    if prof is None:
        prof = UserProfile(
            user_id=user_id,
            instruction_enabled=True,
            instruction_text="",
            avatar_kind="",
            avatar_value="",
            avatar_blob=None,
            avatar_mime="",
            avatar_updated_at=None,
        )
        db.add(prof)
        await db.commit()
    return prof


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
async def web_test(
        q: str = Query(..., description="Query"),
        time_range: str | None = Query(None, pattern="^(day|week|month|year|d|w|m|y)$"),
        days: int | None = Query(None, ge=1, le=365),
        topic: str | None = Query(None, pattern="^(news|general)$"),
):
    ws = WebSearcher()
    items, provider, debug, timed_out = await ws.search(
        q,
        max_results=5,
        timeout=15.0,
        time_budget_sec=12.0,
        topic=topic,
        time_range=time_range,
        days=days,
        auto_parameters=True,
    )
    # age stats for quick sanity
    ages = []
    for it in items:
        d = _parse_date_heuristic(it)
        if d:
            ages.append((dt.date.today() - d).days)
    age_stats = {"min": min(ages) if ages else None, "median": (sorted(ages)[len(ages) // 2] if ages else None),
                 "max": max(ages) if ages else None}

    debug["timed_out"] = 1 if timed_out else 0
    return {
        "provider": provider,
        "count": len(items),
        "debug": debug,
        "age_stats": age_stats,
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
def _base_system_prompt(
        lang: str,
        strict: bool = False,
        mode: str = "general",
        timed_out_note: Optional[str] = None,
        personal_extra: str = "",
) -> str:
    """
    Build the base system prompt according to the selected mode.

    Language rule (restores original Hebrew-first behavior for local docs):
      - If the user's question OR any provided context contains Hebrew characters,
        answer in Hebrew.
      - Otherwise answer in {lang}.

    - mode="web": prefer recency and credible sources from web context.
    - mode="offline": never use the web; prefer local/personal context.
    - other/general: neutral guidance.

    When personal_extra is non-empty AND user enabled it in profile settings,
    it is appended as "Personal instructions".
    """
    # Policy strictness
    if strict:
        policy = (
            "Use ONLY the provided context. If it is insufficient or not relevant, say so briefly and do not invent content."
        )
    else:
        policy = (
            "Use the provided context IF relevant; if missing or insufficient, say so briefly. "
            "Be concise and avoid hallucinations."
        )

    # Recency/web mode hint
    web_note = ""
    if mode == "web":
        web_note = (
            " Prefer the most recent, credible sources; when dates are available, anchor statements to those dates."
        )
    elif mode == "offline":
        web_note = (
            " Do not use the web. Prefer local context (personal instruction and conversation files) when available."
        )

    tools_rules = (
        " Tool-use policy: If the user includes a Python code block and explicitly asks to run it, you may execute it "
        "in the provided sandbox. Never attempt network access, shell/OS commands, or arbitrary file I/O. "
        "If a tool fails or returns no results, say so briefly and suggest one small next step."
    )

    timeout_note = f" Note: {timed_out_note}" if timed_out_note else ""
    personal_section = (
        f"\nPersonal instructions from the user (honor when compatible with policy):\n{personal_extra}\n"
        if (personal_extra or "").strip()
        else ""
    )

    # Key language instruction (Hebrew-first when appropriate)
    lang_rule = (
        f"Language: If the question or any provided context contains Hebrew characters, answer in Hebrew. "
        f"Otherwise answer in {lang}."
    )

    anti_tokens = (
        " Output rule: Never include special placeholder tokens such as "
        "<EOS_TOKEN>, <BOS_TOKEN>, <CLS>, <SEP>, <MASK_TOKEN>, <PAD>, or raw tokenizer markers."
    )

    return (
        f"You are a helpful assistant. {lang_rule} {policy}{web_note}{tools_rules}{anti_tokens}{timeout_note}{personal_section}"
    )


def build_prompt_single(
        user_query: str,
        context_text: str,
        *,
        lang: str,
        strict: bool,
        mode: str,
        timed_out_note: Optional[str] = None,
        history_text: str = "",
        personal_extra: str = "",
) -> str:
    sys = _base_system_prompt(
        lang,
        strict=strict,
        mode=mode,
        timed_out_note=timed_out_note,
        personal_extra=personal_extra,
    )
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
        personal_extra: str = "",
) -> str:
    """
    Mixed-context builder. Language rule:
      - If the question OR either context includes Hebrew characters, answer in Hebrew.
      - Otherwise answer in {lang}.
    """
    sys = (
        "You are a helpful assistant. "
        f"Language rule: If the question or any provided context contains Hebrew characters, answer in Hebrew. "
        f"Otherwise answer in {lang}. "
        "Consider TWO context sections: (A) Local documents and (B) Web articles. "
        "Decide relevance: prefer (B) for breaking news/current facts; prefer (A) when the ask references an uploaded file, example, lines, or internal docs. "
        "If both are relevant, synthesize. If neither is relevant, say so briefly. Be concise and avoid hallucinations."
    )
    if (personal_extra or "").strip():
        sys += f"\nPersonal instructions from the user (honor when compatible with policy):\n{personal_extra}\n"

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

_SPECIAL_TOKEN_RE = re.compile(
    r"</?\s*(?:EOS|BOS|CLS|SEP|MASK|PAD)_?TOKEN\s*>|<\|im_end\|>", re.IGNORECASE
)


def _sanitize_stream_token(tok: str) -> str:
    if not tok:
        return tok
    return _SPECIAL_TOKEN_RE.sub("", tok)


def _clean_llm_text(text: str) -> str:
    s = _SPECIAL_TOKEN_RE.sub("", text or "")
    # Remove empty code fences the model sometimes emits when it glitches
    s = re.sub(r"```(?:\w+)?\s*```", "", s)
    # Remove stray zero-width chars that occasionally appear
    return s.replace("\u200b", "").strip()


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
        # Accept three common names: "files" (multi), "file" (single), "attachments" (multi)
        files: Optional[List[UploadFile]] = File(None, alias="files"),
        file: Optional[UploadFile] = File(None, alias="file"),
        attachments: Optional[List[UploadFile]] = File(None, alias="attachments"),
        # Accept scope and chat_id from query OR from form-data
        scope_q: Optional[str] = Query(None),
        scope_f: Optional[str] = Form(None),
        chat_id_q: Optional[int] = Query(None),
        chat_id_f: Optional[int] = Form(None),
        db: AsyncSession = Depends(get_db),
):
    """
    Upload TXT/PDF/DOCX/MD/CSV/JSON/LOG → persist file bytes+metadata AND chunk+embed into Chroma.
    Scope:
      - "user": global docs (chat_id NULL)
      - "chat": docs attached to a specific chat
    """
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Collect uploads from all accepted aliases
    uploads: List[UploadFile] = []
    if files:
        uploads.extend(files)
    if attachments:
        uploads.extend(attachments)
    if file:
        uploads.append(file)

    if not uploads:
        raise HTTPException(
            status_code=400,
            detail='No files provided. Use "files" (multi), "attachments" (multi), or "file" (single).',
        )

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
        filename = up.filename or "file"
        try:
            data = await up.read()
        except Exception:
            rejected.append(filename)
            continue

        try:
            dest_path, sha_hex = _save_user_file(uid, data, filename)
            size_bytes = len(data)
            mime = up.content_type or ""

            # Upsert simple metadata row
            existing = (
                await db.execute(
                    select(FileMeta).where(FileMeta.user_id == uid, FileMeta.sha256_hex == sha_hex).limit(1)
                )
            ).scalar_one_or_none()

            if existing:
                changed = False
                if existing.filename != dest_path.name:
                    existing.filename = dest_path.name
                    changed = True
                if (existing.mime or "") != mime:
                    existing.mime = mime
                    changed = True
                if existing.size_bytes != size_bytes:
                    existing.size_bytes = size_bytes
                    changed = True
                # If the caller wants this bound to a chat, move it there
                if cid is not None and existing.chat_id != cid:
                    existing.chat_id = cid
                    changed = True
                if changed:
                    await db.commit()
            else:
                fm = FileMeta(
                    user_id=uid,
                    chat_id=cid,
                    sha256_hex=sha_hex,
                    filename=dest_path.name,
                    size_bytes=size_bytes,
                    mime=mime,
                )
                db.add(fm)
                await db.commit()

            # Index into Chroma (works for txt/md/csv/json/log + pdf/docx)
            added = ingest_bytes(
                filename=dest_path.name,
                data=data,
                user_id=ns_user,
                chat_id=ns_chat,
            )
            total_chunks += int(added or 0)
            accepted.append(dest_path.name)

        except Exception:
            rejected.append(filename)

    return {
        "ok": True,
        "scope": scope,
        "chat_id": cid,
        "accepted": accepted,
        "rejected": rejected,
        "total_chunks": total_chunks,
    }


@app.get("/chat/stream")
async def chat_stream(
        q: str = Query(..., description="User query"),
        mode: str = Query("auto", pattern="^(auto|offline|web)$"),
        cid: str = Query("default", min_length=1, max_length=64, pattern=r"^[\w\-]+$"),
        chat_id: Optional[int] = Query(None, ge=1),
        scope: str = Query("user", pattern="^(user|chat)$"),
        request: Request = None,
):
    """
    Modes:
      - auto    : Decide per prompt; prefer local context; use web unless there is a STRONG local signal.
      - offline : No web. Use personal instruction and conversation files.
      - web     : Always perform a network search for each prompt.
    """
    # --- NEW: time-sensitivity heuristic (EN+HE)
    _HE_TIME = [
        r"\bהיום\b", r"\bאתמול\b", r"\bמחר\b", r"\bעדכני(?:ות)?\b", r"\bמעודכן\b", r"\bחדש(?:ה)?\b",
        r"\bמי\s+ניצח\b", r"\bתוצאות\b", r"\bמחיר\b", r"\bשער\b", r"\bמזג\s+האוויר\b", r"\bתחזית\b",
        r"\bשוחרר\b", r"\bהושק\b", r"\bהוכרז\b", r"\bמועד\s+אחרון\b", r"\bדדליין\b", r"\bגרסה\b",
    ]
    _EN_TIME = [
        r"\btoday\b", r"\byesterday\b", r"\btomorrow\b", r"\bnow\b", r"\blatest\b", r"\bcurrent\b", r"\brecent\b",
        r"\bwho\s+won\b", r"\bresults?\b", r"\bscore\b", r"\bprice\b", r"\brate\b", r"\bexchange\b",
        r"\bweather\b", r"\bforecast\b", r"\breleased?\b", r"\blaunch(?:ed)?\b", r"\bannounc(?:e|ed|ement)\b",
        r"\bdeadline\b", r"\bdue\s+date\b", r"\brelease\s+date\b", r"\bversion\b", r"\b202[4-9]\b", r"\b20[3-9]\d\b",
    ]
    _time_hint_patterns = [re.compile(p, re.IGNORECASE) for p in (_HE_TIME + _EN_TIME)]

    def _is_time_sensitive(text: str) -> bool:
        return any(p.search(text or "") for p in _time_hint_patterns)

    async def _persist_assistant_message(chat_id_val: int, content: str) -> None:
        # Persist the assistant's full reply into DB for the given chat (best-effort).
        if not content.strip():
            return
        async with SessionLocal() as dbp:
            chat_obj = await dbp.get(Chat, chat_id_val)
            if not chat_obj:
                return
            dbp.add(ChatMessage(chat_id=chat_obj.id, role="assistant", content=content))
            chat_obj.updated_at = dt.datetime.now(IL_TZ)
            await dbp.commit()

    async def gen() -> AsyncGenerator[bytes, None]:
        try:
            lang = "Hebrew" if is_hebrew(q) else "English"
            used_tool = False

            # ---- Determine current user and optional DB chat + load history ---
            db_user: Optional[User] = None
            db_chat: Optional[Chat] = None
            db_history_rows: List[Dict] = []
            if request is not None:
                async with SessionLocal() as db:
                    db_user = await _current_user(request, db)
                    if db_user and chat_id:
                        db_chat = await db.get(Chat, chat_id)
                        if db_chat and db_chat.user_id != db_user.id:
                            db_chat = None
                        if db_chat:
                            rows = (
                                await db.execute(
                                    select(ChatMessage)
                                    .where(ChatMessage.chat_id == db_chat.id)
                                    .order_by(ChatMessage.id.asc())
                                    .limit(200)
                                )
                            ).scalars().all()
                            db_history_rows = [
                                {"role": m.role, "content": m.content or ""} for m in rows
                            ]

            tracer_local = Tracer()
            trace_id = tracer_local.start_trace(q, mode)
            if trace_id:
                yield sse("trace", trace_id)

            # ---- Build conversation history text ----
            if db_user and db_chat:
                history_text = render_history_from_rows(
                    db_history_rows, lang=lang, max_chars=HISTORY_CHAR_BUDGET
                )
            else:
                history_text = memory.render(cid, lang=lang, max_chars=HISTORY_CHAR_BUDGET)

            # --- Personal instructions (global toggle from user profile) ---
            personal_extra = ""
            if db_user:
                async with SessionLocal() as db2:
                    prof = await _get_or_create_profile(db2, int(db_user.id))
                    if prof.instruction_enabled and (prof.instruction_text or "").strip():
                        personal_extra = prof.instruction_text.strip()

            # ---- Persist current user turn (DB or in-memory) ----
            if db_user and db_chat:
                async with SessionLocal() as db:
                    db.add(ChatMessage(chat_id=db_chat.id, role="user", content=q))
                    chat_obj = await db.get(Chat, db_chat.id)
                    if chat_obj:
                        chat_obj.updated_at = dt.datetime.now(dt.timezone.utc)
                    await db.commit()
            else:
                memory.append(cid, "user", q)

            # ------------------- Python sandbox (explicit) -------------------
            py_code = _extract_python_code(q)
            if py_code:
                used_tool = True
                res = await run_python(py_code, timeout_sec=5.0)

                tool_payload = {
                    "name": "python",
                    "args": {"timeout_sec": 5.0},
                    "result": res,
                }
                yield sse("tool", json.dumps(tool_payload, ensure_ascii=False))
                tracer_local.log_tool(trace_id, "python", {"timeout_sec": 5.0}, res)

                prompt = build_prompt_single(
                    q,
                    f"Python execution result:\n{res.get('stdout', '')}\nErrors:\n{res.get('stderr', '')}",
                    lang=lang,
                    strict=False,
                    mode="offline",
                    history_text=history_text,
                    personal_extra=personal_extra,
                )

                # --- STREAM & PERSIST ASSISTANT (python path)
                assistant_chunks: List[str] = []
                had_error = False
                async for token in ollama_stream(prompt):
                    if token.startswith("ERROR:"):
                        had_error = True
                        yield sse("error", json.dumps({"message": token}))
                        break
                    token = _sanitize_stream_token(token)
                    if not token:
                        continue
                    assistant_chunks.append(token)
                    yield sse("token", token)

                full_reply = _clean_llm_text("".join(assistant_chunks))
                if db_user and db_chat and not had_error:
                    await _persist_assistant_message(int(db_chat.id), full_reply)

                yield sse("sources", "[]")
                yield sse("done", json.dumps({"ok": True, "used_tool": used_tool}))
                return

            # ------------------- RAG where-filter (namespacing) -------------------
            rag_where: Optional[Dict[str, str]] = None
            if db_user:
                if scope == "chat" and chat_id:
                    rag_where = {"chat_id": str(chat_id)}
                else:
                    rag_where = {"user_id": str(db_user.id)}

            citations_combined: List[Dict] = []
            prompt: str
            timed_out_note: Optional[str] = None

            # ------------------- Mode: OFFLINE -------------------
            if mode == "offline":
                ctx_text = ""
                if (scope == "chat") and db_user and chat_id:
                    async with SessionLocal() as db2:
                        fd = await _fallback_docs_for_chat(db2, int(db_user.id), int(chat_id))
                    if fd:
                        b = build_context(q, fd)
                        ctx_text = b["context_text"]
                        citations_combined = b["citations"]

                prompt = build_prompt_single(
                    q,
                    ctx_text,
                    lang=lang,
                    strict=False,
                    mode="offline",
                    history_text=history_text,
                    personal_extra=personal_extra,
                )

            # ------------------- Mode: WEB -------------------
            elif mode == "web":
                ws = WebSearcher()
                sanitized = _sanitize_web_query(q) or (q[:200] if q else "")
                ws_args = _classify_web_strategy(q)  # topic + days (same helper you use in AUTO)

                items: List[Dict] = []
                provider = "none"
                debug: Dict = {}
                timed_out = False
                try:
                    # Ask for a bit more and let the deduper + diversifier prune
                    items, provider, debug, timed_out = await ws.search(
                        sanitized,
                        max_results=10,
                        timeout=20.0,
                        time_budget_sec=60.0,
                        **ws_args,
                    )
                except Exception as e:
                    debug = {"error": str(e)}

                ev = {
                    "name": "web_search",
                    "args": {"query": sanitized or "(skipped)", **{k: v for k, v in ws_args.items() if v is not None}},
                    "result": {"provider": provider, "count": len(items)},
                    "debug": debug,
                }
                yield sse("tool", json.dumps(ev, ensure_ascii=False))
                tracer_local.log_tool(
                    trace_id, "web_search",
                    {"query": sanitized or "(skipped)", **{k: v for k, v in ws_args.items() if v is not None}},
                    {"provider": provider, "count": len(items)},
                )
                used_tool = True
                timed_out_note = "Web search timed out; proceeding with partial context." if timed_out else None

                ctx_web = build_web_context(q, items)
                citations_combined = ctx_web["citations"]
                ctx_text = ctx_web["context_text"] if items else ""
                prompt = build_prompt_single(
                    q,
                    ctx_text,
                    lang=lang,
                    strict=False,
                    mode="web",
                    timed_out_note=timed_out_note,
                    history_text=history_text,
                    personal_extra=personal_extra,
                )

            # ------------------- Mode: AUTO -------------------
            else:
                retriever = Retriever(
                    k_initial=12,
                    k_final=int(os.getenv("RAG_TOP_K", "5")),
                    use_reranker=True,
                )

                rag_docs = retriever.retrieve(q, where=rag_where)

                # Fallback: recent chat files as context if nothing retrieved (chat scope)
                if (not rag_docs) and (scope == "chat") and db_user and chat_id:
                    async with SessionLocal() as db2:
                        fd = await _fallback_docs_for_chat(db2, int(db_user.id), int(chat_id))
                    if fd:
                        rag_docs = fd

                # RAG context (always probe + emit trace so user sees why we did/didn't use it)
                ctx_rag = build_context(q, rag_docs)
                yield sse("tool", json.dumps({
                    "name": "retrieve_knowledge_probe",
                    "args": {"query": q, "where": rag_where},
                    "result": ctx_rag["citations"],
                }, ensure_ascii=False))
                tracer_local.log_tool(
                    trace_id, "retrieve_knowledge_probe", {"query": q, "where": rag_where}, ctx_rag["citations"]
                )

                # Strength estimate for local fit
                rag_keywords = _extract_keywords(q)
                overlap = 0
                if rag_keywords and rag_docs:
                    text0 = (rag_docs[0].get("text") or "").lower()
                    overlap = sum(1 for k in rag_keywords if k in text0)

                top_dist = None
                if rag_docs and isinstance(rag_docs[0].get("distance", None), (int, float)):
                    top_dist = float(rag_docs[0]["distance"])

                doc_hint = _heuristic_wants_rag(q)  # mention of “document/file/lines…”
                rag_strong = bool(rag_docs) or bool(
                    doc_hint and (
                            (overlap >= 1) or
                            (top_dist is not None and top_dist < 0.60)
                    )
                )

                # Decide if we ALSO need the web. We’re stricter here:
                # - Always use web for clearly time-sensitive or market questions (even if RAG is strong).
                # - If RAG is weak, use web as well.
                doc_hint = _heuristic_wants_rag(q)  # already present earlier
                decision = classify_intent(q, rag_strong=bool(rag_strong), doc_hint=bool(doc_hint))

                # Optional: emit a trace event so you can see why auto chose its path
                yield sse("tool", json.dumps({
                    "name": "auto_intent",
                    "args": {"query": q},
                    "result": {
                        "intent": decision.intent,
                        "wants_rag": decision.wants_rag,
                        "wants_web": decision.wants_web,
                        "reason": decision.reason,
                    },
                }, ensure_ascii=False))
                tracer_local.log_tool(trace_id, "auto_intent", {"query": q}, {
                    "intent": decision.intent, "wants_rag": decision.wants_rag,
                    "wants_web": decision.wants_web, "reason": decision.reason
                })

                wants_web = bool(decision.wants_web)

                citations_combined: List[Dict] = []
                prompt: str
                timed_out_note = None
                used_tool = False

                if wants_web:
                    # Best-effort decision summary (for trace only)
                    decision = await _classify_with_llm(q)
                    yield sse("tool", json.dumps({
                        "name": "auto_decision",
                        "args": {"query": q},
                        "result": decision,
                    }, ensure_ascii=False))
                    tracer_local.log_tool(trace_id, "auto_decision", {"query": q}, decision)

                    ws = WebSearcher()
                    sanitized = _sanitize_web_query(q) or (q[:200] if q else "")
                    ws_args = _classify_web_strategy(q)  # <-- NEW: topic+days+auto_parameters

                    items: List[Dict] = []
                    provider = "none"
                    debug: Dict = {}
                    timed_out = False
                    try:
                        items, provider, debug, timed_out = await ws.search(
                            sanitized,
                            max_results=6,
                            timeout=20.0,
                            time_budget_sec=60.0,
                            **ws_args,
                        )
                    except Exception as e:
                        debug = {"error": str(e)}

                    yield sse("tool", json.dumps({
                        "name": "web_search",
                        "args": {"query": sanitized or "(skipped)",
                                 **{k: v for k, v in ws_args.items() if v is not None}},
                        "result": {"provider": provider, "count": len(items)},
                        "debug": debug,
                    }, ensure_ascii=False))
                    tracer_local.log_tool(
                        trace_id, "web_search",
                        {"query": sanitized or "(skipped)", **{k: v for k, v in ws_args.items() if v is not None}},
                        {"provider": provider, "count": len(items)},
                    )
                    used_tool = True
                    timed_out_note = "Web search timed out; proceeding with partial context." if timed_out else None

                    ctx_web = build_web_context(q, items)

                    # Combine: attached file(s) / RAG first, then fresh web (if any)
                    citations_combined = (ctx_rag["citations"] or []) + (ctx_web["citations"] or [])
                    prompt = build_prompt_mixed(
                        q,
                        ctx_rag["context_text"],
                        ctx_web["context_text"],
                        lang=lang,
                        history_text=history_text,
                        personal_extra=personal_extra,
                    )
                else:
                    # Purely local (RAG/offline)
                    citations_combined = ctx_rag["citations"]
                    prompt = build_prompt_single(
                        q,
                        ctx_rag["context_text"],
                        lang=lang,
                        strict=False,
                        mode="offline",
                        history_text=history_text,
                        personal_extra=personal_extra,
                    )
            # ------------------- STREAM & PERSIST ASSISTANT (all non-python paths) -------------------
            assistant_chunks: List[str] = []
            had_error = False
            async for token in ollama_stream(prompt):
                if token.startswith("ERROR:"):
                    had_error = True
                    yield sse("error", json.dumps({"message": token}))
                    break
                token = _sanitize_stream_token(token)
                if not token:
                    continue
                assistant_chunks.append(token)
                yield sse("token", token)

            full_reply = _clean_llm_text("".join(assistant_chunks))
            if db_user and db_chat and not had_error:
                await _persist_assistant_message(int(db_chat.id), full_reply)

            # Sources (citations)
            try:
                yield sse("sources", json.dumps(citations_combined, ensure_ascii=False))
            except Exception:
                yield sse("sources", "[]")

            yield sse("done", json.dumps({"ok": True, "used_tool": used_tool}))
            return

        except Exception as e:
            # Always end the stream cleanly on unexpected errors
            yield sse("error", json.dumps({"message": str(e)}))
            yield sse("done", json.dumps({"ok": False, "used_tool": False}))
            return

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/chats/{chat_id}/auto-title")
async def auto_title(chat_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    """
    Generate a concise title for a chat.

    Behavior:
      - Prefer the earliest 'user' message from DB.
      - If missing/empty, try to use client-provided {"first_user_text": "..."}.
      - If still empty, build a seed from the *attached files' text content* (recent text-like files in this chat).
      - If everything is empty, fall back to a safe default ("Chat").
    """
    # Identify current user
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Load chat and validate ownership
    chat = await db.get(Chat, chat_id)
    if not chat or chat.user_id != user.id:
        raise HTTPException(status_code=404, detail="Chat not found")

    # 1) Earliest 'user' role message
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.chat_id == chat_id, ChatMessage.role == "user")
        .order_by(ChatMessage.id.asc())
        .limit(1)
    )
    row = (await db.execute(stmt)).scalars().first()
    typed_text = ((row.content or "").strip() if row else "")

    # 2) Optional client-provided fallback
    fallback_text = ""
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            fallback_text = str(payload.get("first_user_text") or "").strip()
    except Exception:
        # tolerate empty/invalid bodies
        pass

    # Start building a seed
    seed_text = typed_text or fallback_text

    # 3) If we still don't have anything useful, derive a seed from attached files' content
    if not seed_text:
        docs = await _fallback_docs_for_chat(
            db,
            user_id=int(user.id),
            chat_id=int(chat_id),
            max_files=2,
            per_file_chars=2000,
        )
        parts: List[str] = []
        for d in docs:
            src = str(d.get("source") or "file")
            txt = str(d.get("text") or "")
            if not txt:
                continue
            # keep it short — enough for a good title, cheap for the model
            snippet = txt[:800]
            parts.append(f"{src}:\n{snippet}")
        if parts:
            seed_text = "\n---\n".join(parts)

    # 4) Final fallback: safe default if we couldn't build a seed
    if not seed_text:
        chat.title = "Chat"
        await db.commit()
        return {"id": chat.id, "title": chat.title}

    # 5) Ask the model for a short title
    raw = await ollama_generate(_title_prompt(seed_text))
    title = (raw or "").strip().strip('"')[:80] or "Chat"

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


@app.get("/modes")
async def get_modes():
    """
    Catalog of agent modes (used by the frontend’s Mode Picker).
    """
    return [
        {"id": "offline", "label": "Offline", "desc": "Local files & personal instruction. No web."},
        {"id": "web", "label": "Network search", "desc": "Always search the web for every prompt."},
        {"id": "auto", "label": "Automatic", "desc": "Decides between local context and web per prompt."},
    ]


# --- GET profile settings ---
@app.get("/profile/settings")
async def get_profile_settings(request: Request, db: AsyncSession = Depends(get_db)):
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    await _ensure_user_profile_columns(db)
    prof = await _get_or_create_profile(db, int(user.id))

    return {
        "instruction_enabled": prof.instruction_enabled,
        "instruction_text": prof.instruction_text or "",
        "avatar_kind": prof.avatar_kind or "",
        "avatar_value": prof.avatar_value or "",
        "avatar_has_blob": bool(prof.avatar_blob),
        "display_name": user.display_name or "",
    }


# --- PUT profile settings ---
@app.put("/profile/settings")
async def put_profile_settings(payload: ProfileSettingsIO, request: Request, db: AsyncSession = Depends(get_db)):
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    await _ensure_user_profile_columns(db)
    prof = await _get_or_create_profile(db, int(user.id))

    prof.instruction_enabled = bool(payload.instruction_enabled)
    prof.instruction_text = payload.instruction_text or ""

    # Only update avatar_kind/value here (bytes handled by upload endpoint)
    kind = (payload.avatar_kind or "").strip()
    if kind not in ("", "system", "upload"):
        raise HTTPException(status_code=422, detail="Invalid avatar_kind")
    prof.avatar_kind = kind
    prof.avatar_value = (payload.avatar_value or "").strip()

    await db.commit()
    return {"ok": True}


# --- POST avatar upload (stores bytes in DB) ---
@app.post("/profile/avatar/upload")
async def upload_avatar(
        file: UploadFile = File(...),
        request: Request = None,
        db: AsyncSession = Depends(get_db),
):
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    await _ensure_user_profile_columns(db)
    prof = await _get_or_create_profile(db, int(user.id))

    if file.content_type not in _ALLOWED_IMAGE_MIME:
        raise HTTPException(status_code=415, detail="Only PNG, JPEG or WEBP are allowed.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Max size is 5MB.")

    prof.avatar_blob = data
    prof.avatar_mime = file.content_type
    prof.avatar_kind = "upload"
    prof.avatar_value = ""
    prof.avatar_updated_at = dt.datetime.now(IL_TZ)

    await db.commit()
    return {"ok": True}


# --- GET avatar image (upload wins; fallback to system SVG; else 404) ---

def _system_avatar_svg(code: str, label: str) -> bytes:
    # deterministic colors by code (sys-1..sys-8). Keep it simple, circle + initial.
    palette = [
        ("#2563eb", "#60a5fa"),
        ("#7c3aed", "#a78bfa"),
        ("#db2777", "#f472b6"),
        ("#059669", "#34d399"),
        ("#ea580c", "#fdba74"),
        ("#0ea5e9", "#67e8f9"),
        ("#16a34a", "#86efac"),
        ("#9333ea", "#d8b4fe"),
    ]
    idx = 0
    try:
        n = int(code.replace("sys-", "").strip()) - 1
        idx = n % len(palette)
    except Exception:
        idx = 0
    c1, c2 = palette[idx]
    initial = (label.strip()[:1] or "U").upper()
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{c1}"/>
      <stop offset="100%" stop-color="{c2}"/>
    </linearGradient>
  </defs>
  <circle cx="64" cy="64" r="64" fill="url(#g)"/>
  <text x="50%" y="54%" dominant-baseline="middle" text-anchor="middle"
        font-family="Inter, Arial, sans-serif" font-weight="700" font-size="56"
        fill="white">{initial}</text>
</svg>"""
    return svg.encode("utf-8")


@app.get("/profile/avatar")
async def get_avatar(request: Request, db: AsyncSession = Depends(get_db)):
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    await _ensure_user_profile_columns(db)
    prof = await _get_or_create_profile(db, int(user.id))

    # If user uploaded a custom avatar, serve it.
    if prof.avatar_kind == "upload" and prof.avatar_blob:
        mime = prof.avatar_mime or "application/octet-stream"
        headers = {}
        if prof.avatar_updated_at:
            headers["Last-Modified"] = prof.avatar_updated_at.isoformat()
        return Response(content=prof.avatar_blob, media_type=mime, headers=headers)

    # If user selected a system avatar, synthesize SVG for it
    if prof.avatar_kind == "system" and prof.avatar_value:
        svg = _system_avatar_svg(prof.avatar_value, user.display_name or "U")
        return Response(content=svg, media_type="image/svg+xml")

    # No avatar at all
    raise HTTPException(status_code=404, detail="No avatar configured")


@app.put("/profile/account")
async def update_profile_account(payload: ProfileAccountIn, request: Request, db: AsyncSession = Depends(get_db)):
    user = await _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    touched = False

    # Update display_name (optional)
    if payload.display_name is not None:
        dn = (payload.display_name or "").strip()
        user.display_name = dn or None
        touched = True

    # Change password (optional, requires current_password)
    if payload.new_password:
        if not payload.current_password:
            raise HTTPException(status_code=422, detail="Current password is required to change password.")
        try:
            ph.verify(user.password_hash, payload.current_password)
        except VerifyMismatchError:
            raise HTTPException(status_code=401, detail="Current password is incorrect.")
        user.password_hash = ph.hash(payload.new_password)
        touched = True

    if touched:
        await db.commit()

    return {"ok": True}
