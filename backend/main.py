import os
import json
import re
import asyncio
from typing import AsyncGenerator, List, Dict, Optional, Tuple

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag.retriever import Retriever
from tracing import Tracer

# --------- Settings ---------
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
MODEL_NAME = os.getenv("MODEL_NAME", "aya-expanse:8b")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

import httpx

# Tools
from tools.web_search import WebSearcher
from tools.py_eval import run_python

app = FastAPI()
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
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")

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
        payload = {"model": MODEL_NAME, "prompt": prompt, "stream": True}
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

_HE_STOP = {"של","עם","על","לא","כן","אם","או","זה","זו","אלה","הוא","היא","הם","הן","אני","אתה","את","אתם","אתן","אנחנו","כל","גם","אך","אבל","כדי","כי","כמו","אלא","שרק","רק","עוד","כבר","שהוא","שהיא","שזה","שזו","שאין","מאוד","בה","בו","בהם","בהן","לפי","מתוך"}
_EN_STOP = {"the","a","an","and","or","but","if","so","to","of","in","on","for","by","with","as","at","is","are","was","were","be","been","being","it","this","that","these","those","from","i","you","he","she","we","they","them","his","her","their","our","your","my","me"}
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
    start = text.find("{"); end = text.rfind("}")
    if start == -1 or end == -1 or end <= start: return None
    try:
        return json.loads(text[start:end+1])
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

# Quick py_eval test
@app.get("/tools/py/test")
async def py_test():
    res = await run_python("print(sum(i*i for i in range(1, 11)))")
    return res

# ---------------------- Guardrails in system prompt ----------------------
def _base_system_prompt(lang: str, strict: bool = False, mode: str = "general", timed_out_note: Optional[str] = None) -> str:
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

def build_prompt_single(user_query: str, context_text: str, *, lang: str, strict: bool, mode: str, timed_out_note: Optional[str] = None) -> str:
    sys = _base_system_prompt(lang, strict=strict, mode=mode, timed_out_note=timed_out_note)
    return f"{sys}\n\n# Context:\n{context_text}\n\n# Question:\n{user_query}\n\n# Answer:"

def build_prompt_mixed(user_query: str, ctx_rag: str, ctx_web: str, *, lang: str) -> str:
    sys = (
        f"You are a helpful assistant. Always answer in {lang}. "
        f"Consider TWO context sections: (A) Local documents and (B) Web articles. "
        f"Decide relevance: prefer (B) for breaking news/current facts; prefer (A) when the ask references an uploaded file, example, lines, or internal docs. "
        f"If both are relevant, synthesize. If neither is relevant, say so briefly. Be concise and avoid hallucinations."
    )
    return (
        f"{sys}\n\n# Context A — Local (may be partial):\n{ctx_rag}\n\n"
        f"# Context B — Web (may be partial):\n{ctx_web}\n\n"
        f"# Question:\n{user_query}\n\n# Answer:"
    )

# ---------------------- Python code fence detector (robust) ----------------------
_CODE_PATTERNS = [
    re.compile(r"```(?:python|py)\s*\r?\n(.*?)\r?\n```", re.IGNORECASE | re.DOTALL),
    re.compile(r"```(?:python|py)\s*\r?\n([\s\S]*)\Z", re.IGNORECASE),
    re.compile(r"```\s*\r?\n(.*?)\r?\n```", re.DOTALL),
    re.compile(r"```\s*\r?\n([\s\S]*)\Z"),
    re.compile(r"```(.*?)```", re.DOTALL),
]

_PY_LINE_SIG = re.compile(
    r"(?m)^\s*(?:"
    r"import\s+\w+|"
    r"from\s+\w[\w\.]*\s+import\s+\w+|"
    r"def\s+\w+\s*\(|"
    r"class\s+\w+\s*:|"
    r"print\s*\(|"
    r"for\s+\w+\s+in\s+|"
    r"while\s+|"
    r"if\s+.*:|"
    r"with\s+|"
    r"async\s+def\s+\w+\s*\(|"
    r"lambda\s+|"
    r"try\s*:|"
    r"except\s+|"
    r"@\w+"
    r")"
)

# ------------ Code detection helpers ------------
_PY_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_PY_LINE_SIG = re.compile(
    r"""(?mx)
    ^\s*(?:from\s+\w[\w\.]*\s+import\s+|import\s+\w[\w\.]*\s*|print\s*\(|def\s+\w+\s*\(|class\s+\w+\s*\(|for\s+\w+\s+in\s+|while\s+|if\s+__name__\s*==\s*['"]__main__['"])
    """
)

def _extract_python_code(text: str) -> Optional[str]:
    """
    Returns Python code if the user pasted code (with or without fences), else None.
    Preference order:
      1) fenced ```python ... ```
      2) any fenced ``` ... ```
      3) unfenced, but looks like code by common line signatures
    """
    if not text:
        return None
    m = _PY_FENCE_RE.search(text)
    if m:
        code = (m.group(1) or "").strip()
        return code or None

    # Any generic fenced region (drop the fence markers)
    m2 = re.search(r"```([\s\S]*?)```", text)
    if m2:
        code = (m2.group(1) or "").strip()
        return code or None

    # Unfenced but starts/contains typical Python lines
    if _PY_LINE_SIG.search(text):
        return text.strip()

    return None


# ---------------------- Sanitization for web tool ----------------------
def _sanitize_web_query(text: str) -> str:
    """Remove code fences and obvious code; normalize spacing; cap length. Empty means 'don't search'."""
    if not text:
        return ""
    # Drop fenced code completely
    s = re.sub(r"```[\s\S]*?```", " ", text)
    # If the remaining string still looks like Python (no fences), treat as code => skip web
    if _PY_LINE_SIG.search(s):
        return ""
    # Remove stray backticks and compress spaces
    s = s.replace("`", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:200]


# ---------------------- API ----------------------
@app.get("/chat/stream")
async def chat_stream(
    q: str = Query(..., description="User query"),
    mode: str = Query("auto", pattern="^(auto|none|dense|rerank|web)$")
):
    """
    Modes:
      - auto   : Always probe local RAG; maybe web if heuristics or if RAG looks weak. Mix contexts smartly.
      - none   : no retrieval (pure LLM)
      - dense  : dense retrieval only (STRICT grounding)
      - rerank : dense + reranker (STRICT grounding)
      - web    : ALWAYS run web search (with up to 60s budget). If none found, say so.
    """
    async def gen() -> AsyncGenerator[bytes, None]:
        try:
            lang = "Hebrew" if is_hebrew(q) else "English"
            used_tool = False

            # ---- tracing (always) ----
            trace_id = tracer.start_trace(q, mode)
            yield sse("trace", json.dumps({"id": trace_id}))

            citations_combined: List[Dict] = []
            strict = False
            prompt = ""
            resp_buf: List[str] = []  # accumulate streamed tokens for trace

            # ---- Run Python sandbox first when code is present (ALL modes) ----
            py_code = _extract_python_code(q)
            if py_code:
                used_tool = True
                res = await run_python(py_code, timeout_sec=5.0)

                # Tool event (UI) + tracing
                tool_payload = {
                    "name": "py_eval",
                    "args": {"language": "python", "timeout_s": 5},
                    "result": res,
                }
                yield sse("tool", json.dumps(tool_payload, ensure_ascii=False))
                tracer.log_tool(trace_id, "py_eval", {"language": "python", "timeout_s": 5}, res)

                # Build assistant summary prompt over the program output
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
                )
                async for tok in ollama_stream(prompt):
                    resp_buf.append(tok)
                    yield sse("token", tok)
                    await asyncio.sleep(0)

                assistant_text = "".join(resp_buf)
                tracer.end_trace(trace_id, prompt, assistant_text, citations=[], ok=True)

                yield sse("sources", "[]")
                yield sse("done", json.dumps({"ok": True, "used_tool": used_tool}))
                return

            # ---------------------- Regular path (no code) ----------------------
            # --- NONE ---
            if mode == "none":
                prompt = build_prompt_single(q, "", lang=lang, strict=False, mode="general")

            # --- DENSE / RERANK (STRICT) ---
            elif mode in ("dense", "rerank"):
                retriever = Retriever(k_initial=12, k_final=RAG_TOP_K, use_reranker=(mode == "rerank"))
                docs = retriever.retrieve(q)
                ctx_rag = build_context(q, docs)
                strict = True
                prompt = build_prompt_single(q, ctx_rag["context_text"], lang=lang, strict=True, mode=mode)
                citations_combined = ctx_rag["citations"]
                used_tool = True

                # UI + trace tool event
                ev = {
                    "name": "retrieve_knowledge",
                    "args": {"query": q, "mode": mode},
                    "result": ctx_rag["citations"],
                }
                yield sse("tool", json.dumps(ev, ensure_ascii=False))
                tracer.log_tool(trace_id, "retrieve_knowledge", {"query": q, "mode": mode}, ctx_rag["citations"])

            # --- WEB (always search; up to 60s) ---
            elif mode == "web":
                ws = WebSearcher()
                sanitized = _sanitize_web_query(q)
                items = []
                provider = "none"
                debug = []
                timed_out_note = None

                if sanitized:
                    items, provider, debug, timed_out = await ws.search(sanitized, max_results=5, timeout=15.0, time_budget_sec=60.0)
                    if timed_out:
                        timed_out_note = "No web results were found within a 60-second search budget. Inform the user and suggest refining the query."
                # UI + trace tool event
                ev = {
                    "name": "web_search",
                    "args": {"query": sanitized or "(skipped due to code-like text)"},
                    "result": {"provider": provider, "count": len(items), "debug": debug, "items": items},
                }
                yield sse("tool", json.dumps(ev, ensure_ascii=False))
                tracer.log_tool(trace_id, "web_search", {"query": sanitized or "(skipped)"}, {
                    "provider": provider, "count": len(items), "debug": debug, "items": items
                })
                used_tool = True

                ctx_web = build_web_context(q, items)
                citations_combined = ctx_web["citations"]

                if items:
                    prompt = build_prompt_single(q, ctx_web["context_text"], lang=lang, strict=False, mode="web")
                else:
                    prompt = build_prompt_single(q, "", lang=lang, strict=False, mode="web", timed_out_note=timed_out_note)

            # --- AUTO (probe RAG always; web if heuristics/weak RAG) ---
            else:
                # 1) RAG probe
                retriever = Retriever(k_initial=12, k_final=RAG_TOP_K, use_reranker=True)
                rag_docs = retriever.retrieve(q)
                ctx_rag = build_context(q, rag_docs)

                # UI + trace probe event
                ev_probe = {
                    "name": "retrieve_knowledge_probe",
                    "args": {"query": q},
                    "result": ctx_rag["citations"],
                }
                yield sse("tool", json.dumps(ev_probe, ensure_ascii=False))
                tracer.log_tool(trace_id, "retrieve_knowledge_probe", {"query": q}, ctx_rag["citations"])

                rag_used = len(rag_docs) > 0

                # 2) Decide if we should also run web
                run_web = _heuristic_wants_web(q)
                rag_keywords = _extract_keywords(q)
                rag_overlap = 0
                if rag_keywords and rag_docs:
                    text0 = (rag_docs[0].get("text") or "").lower()
                    rag_overlap = sum(1 for k in rag_keywords if k in text0)
                rag_strong = (rag_overlap >= 1) or (rag_docs and isinstance(rag_docs[0].get("distance", 1.0), (int, float)) and float(rag_docs[0]["distance"]) < 0.55)

                ctx_web = {"context_text": "", "citations": []}
                if run_web or not rag_strong:
                    ws = WebSearcher()
                    sanitized = _sanitize_web_query(q)
                    items = []
                    provider = "none"
                    debug = []
                    timed_out = False
                    if sanitized:
                        items, provider, debug, timed_out = await ws.search(sanitized, max_results=5, timeout=15.0, time_budget_sec=20.0)
                    # UI + trace
                    ev_web = {
                        "name": "web_search",
                        "args": {"query": sanitized or "(skipped due to code-like text)"},
                        "result": {"provider": provider, "count": len(items), "debug": debug + [f"timed_out={1 if timed_out else 0}"], "items": items},
                    }
                    yield sse("tool", json.dumps(ev_web, ensure_ascii=False))
                    tracer.log_tool(trace_id, "web_search", {"query": sanitized or "(skipped)"}, {
                        "provider": provider, "count": len(items), "debug": debug + [f"timed_out={1 if timed_out else 0}"], "items": items
                    })
                    ctx_web = build_web_context(q, items)

                # 3) Build mixed prompt
                prompt = build_prompt_mixed(q, ctx_rag["context_text"], ctx_web["context_text"], lang=lang)
                citations_combined = ctx_rag["citations"] + ctx_web["citations"]
                used_tool = rag_used or (len(ctx_web["citations"]) > 0)

            # ---- Stream assistant answer ----
            async for tok in ollama_stream(prompt):
                resp_buf.append(tok)
                yield sse("token", tok)
                await asyncio.sleep(0)

            # Push sources bar
            yield sse("sources", json.dumps(citations_combined, ensure_ascii=False))

            # End trace
            assistant_text = "".join(resp_buf)
            tracer.end_trace(trace_id, prompt, assistant_text, citations_combined, ok=True)

            yield sse("done", json.dumps({"ok": True, "used_tool": used_tool}))
        except Exception as e:
            # Stream error to user
            yield sse("token", f"ERROR: backend exception: {e}")
            yield sse("sources", "[]")
            # Try to close trace as failed
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

