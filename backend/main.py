import os
import json
import re
import asyncio
from typing import AsyncGenerator, List, Dict, Optional, Tuple

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag.retriever import Retriever

# --------- Settings ---------
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
MODEL_NAME = os.getenv("MODEL_NAME", "aya-expanse:8b")  # your chosen model
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

import httpx

app = FastAPI()

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
    return any("\u0590" <= ch <= "\u05FF" for ch in s)

def truncate(txt: str, n: int = 700) -> str:
    txt = txt.replace("\n", " ").strip()
    return txt if len(txt) <= n else txt[: n - 1] + "…"

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

def build_prompt(user_query: str, context_text: str) -> str:
    lang = "Hebrew" if is_hebrew(user_query) else "English"
    sys = (
        f"You are a helpful assistant. Always answer in {lang}. "
        f"Use the provided context IF relevant; if missing or insufficient, say so briefly. "
        f"Be concise and avoid hallucinations."
    )
    return f"{sys}\n\n# Context (may be partial):\n{context_text}\n\n# Question:\n{user_query}\n\n# Answer:"

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

# ---------------------- Day 6: Auto RAG decision ----------------------
def _have_langchain() -> Tuple[bool, Optional[str]]:
    try:
        from langchain_ollama import ChatOllama  # noqa: F401
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)

# Heuristic: explicit document mentions
_HEBREW_DOC_HINTS = [
    r"\bמסמך\b", r"\bמסמכים\b", r"\bקובץ\b", r"\bקבצים\b", r"\bדוגמה\b",
    r"\bשהעליתי\b", r"\bשהעלת\b", r"\bשהעלינו\b", r"\bבמקורות\b", r"\bמקור\b", r"\bמקורות\b",
    r"\bהטקסט\b", r"\bהקובץ\b", r"\bהשורה\b", r"\bשורות\b", r"\bשורה\b",
    r"\bמה כתוב\b", r"\bמה מסופר\b",
]
_EN_DOC_HINTS = [
    r"\bdocument\b", r"\bfile\b", r"\bexample\b", r"\buploaded\b",
    r"\bsources?\b", r"\bcontext\b", r"\blines?\b", r"\bcitation\b", r"\brag\b",
]
_doc_hint_patterns = [re.compile(p, re.IGNORECASE) for p in (_HEBREW_DOC_HINTS + _EN_DOC_HINTS)]

def _heuristic_wants_rag(q: str) -> bool:
    return any(pat.search(q) for pat in _doc_hint_patterns)

# Simple keyword extraction (for probe overlap)
_HE_STOP = {"של","עם","על","לא","כן","אם","או","זה","זו","אלה","הוא","היא","הם","הן","אני","אתה","את","אתם","אתן","אנחנו","כל","גם","אך","אבל","כדי","כי","כמו","אלא","שרק","רק","עוד","כבר","שהוא","שהיא","שזה","שזו","שאין","מאוד","בה","בו","בהם","בהן","לפי","מתוך"}
_EN_STOP = {"the","a","an","and","or","but","if","so","to","of","in","on","for","by","with","as","at","is","are","was","were","be","been","being","it","this","that","these","those","from","i","you","he","she","we","they","them","his","her","their","our","your","my","me"}

_tok_pat = re.compile(r"[\w\u0590-\u05FF]+")

def _extract_keywords(s: str, min_len: int = 4) -> List[str]:
    toks = [t.lower() for t in _tok_pat.findall(s)]
    out = []
    for t in toks:
        if len(t) < min_len:
            continue
        if t in _HE_STOP or t in _EN_STOP:
            continue
        out.append(t)
    return out

def _auto_decision_prompt() -> str:
    return (
        "Decide whether the user's question likely requires retrieving from the user's LOCAL documents (RAG). "
        "Return ONLY minified JSON with keys: use_retrieval (true/false), query_for_retrieval (string), reason (string). "
        "If the user asks about a specific document, example, file, sources, lines, or uploaded content, choose true."
    )

def _extract_json(text: str) -> Optional[dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
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
    """
    Run a tiny retrieval + rerank and decide based on lexical overlap and distance.
    Returns (use, docs, reason).
    """
    retriever = Retriever(k_initial=8, k_final=3, use_reranker=True)
    docs = retriever.retrieve(user_query)
    if not docs:
        return False, [], "Probe: no candidates."

    # lexical overlap on previews/text
    keywords = _extract_keywords(user_query)
    if keywords:
        for d in docs:
            text = (d.get("text") or "").lower()
            matches = sum(1 for k in keywords if k in text)
            if matches >= 1:  # at least one meaningful token found
                return True, docs, f"Probe: lexical overlap ({matches}) with top candidates."

    # fallback: vector distance threshold (smaller is better)
    # use the best candidate distance if present
    d0 = docs[0].get("distance", None)
    if isinstance(d0, (int, float)) and d0 < 0.45:
        return True, docs, f"Probe: strong vector match (distance={d0:.3f})."

    return False, docs, "Probe: weak lexical/vector signals."

async def decide_auto_rag(user_query: str) -> Dict:
    """
    Returns:
      {
        "use": bool,
        "reason": str,
        "tool_query": str,
        "method": "heuristic" | "llm" | "probe" | "default",
        "llm_error": Optional[str],
        "pre_docs": Optional[List[Dict]]
      }
    """
    # 1) Heuristic fast-path
    if _heuristic_wants_rag(user_query):
        return {
            "use": True,
            "reason": "Heuristic: query references documents/files/examples/sources.",
            "tool_query": user_query,
            "method": "heuristic",
            "llm_error": None,
            "pre_docs": None,
        }

    # 2) LLM JSON classifier
    cls = await _classify_with_llm(user_query)
    if isinstance(cls, dict) and "error" in cls:
        # 3) Probe anyway if classifier failed
        use, docs, why = await _probe_retrieval(user_query)
        return {
            "use": use,
            "reason": f"Classifier error → {why}",
            "tool_query": user_query,
            "method": "probe" if use else "default",
            "llm_error": cls.get("error"),
            "pre_docs": docs if use else None,
        }

    use = bool(cls.get("use_retrieval"))
    tq = (cls.get("query_for_retrieval") or user_query).strip()
    reason = str(cls.get("reason") or "No reason")

    if use:
        return {
            "use": True,
            "reason": reason,
            "tool_query": tq if tq else user_query,
            "method": "llm",
            "llm_error": None,
            "pre_docs": None,
        }

    # 4) If LLM said "no", do a semantic probe to catch cases like RL
    use_p, docs_p, why_p = await _probe_retrieval(user_query)
    return {
        "use": use_p,
        "reason": f"LLM=no → {why_p}",
        "tool_query": tq if tq else user_query,
        "method": "probe" if use_p else "llm",
        "llm_error": None,
        "pre_docs": docs_p if use_p else None,
    }

# ---------------------- API ----------------------
@app.get("/chat/stream")
async def chat_stream(
    q: str = Query(..., description="User query"),
    mode: str = Query("auto", pattern="^(auto|none|dense|rerank)$")
):
    """
    Modes:
      - auto   : Auto decision (heuristics + JSON LLM + semantic probe). Emits 'tool' events.
      - none   : no retrieval (pure LLM)
      - dense  : dense retrieval only
      - rerank : dense + reranker
    """
    async def gen() -> AsyncGenerator[bytes, None]:
        try:
            docs: List[Dict] = []
            ctx = {"context_text": "", "citations": []}
            used_tool = False
            tool_query = q

            if mode == "none":
                pass

            elif mode in ("dense", "rerank"):
                retriever = Retriever(k_initial=12, k_final=RAG_TOP_K, use_reranker=(mode == "rerank"))
                docs = retriever.retrieve(q)
                ctx = build_context(q, docs)

            else:
                # ----- AUTO (heuristics → LLM JSON → semantic probe) -----
                decision = await decide_auto_rag(q)
                # Emit decision
                yield sse("tool", json.dumps({
                    "name": "auto_decision",
                    "args": {"query": q},
                    "result": {
                        "use_retrieval": decision["use"],
                        "reason": decision["reason"],
                        "method": decision["method"],
                        "llm_error": decision["llm_error"],
                        "tool_query": decision["tool_query"],
                    },
                }, ensure_ascii=False))

                if decision["use"]:
                    used_tool = True
                    tool_query = decision["tool_query"]

                    # Use pre_docs from probe if we already have them
                    if decision.get("pre_docs"):
                        docs = decision["pre_docs"]  # type: ignore
                    else:
                        retriever = Retriever(k_initial=12, k_final=RAG_TOP_K, use_reranker=True)
                        docs = retriever.retrieve(tool_query)

                    ctx = build_context(tool_query, docs)

                    yield sse("tool", json.dumps({
                        "name": "retrieve_knowledge",
                        "args": {"query": tool_query},
                        "result": ctx["citations"],
                    }, ensure_ascii=False))

            # Build final prompt and stream the LLM answer
            prompt = build_prompt(q, ctx["context_text"])
            async for tok in ollama_stream(prompt):
                yield sse("token", tok)
                await asyncio.sleep(0)

            # Push sources bar
            yield sse("sources", json.dumps(ctx["citations"], ensure_ascii=False))
            yield sse("done", json.dumps({"ok": True, "used_tool": used_tool}))
        except Exception as e:
            yield sse("token", f"ERROR: backend exception: {e}")
            yield sse("sources", "[]")
            yield sse("done", json.dumps({"ok": False}))

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
