# backend/eval/ragas_eval.py
from __future__ import annotations

import json
import math
import os
import random
import string
import time
from typing import Any, Dict, List, Tuple, Optional

import chromadb
import httpx
import pandas as pd
from chromadb.config import Settings
from langchain_core.embeddings.embeddings import Embeddings as LCEmb
# LangChain core
from langchain_core.language_models.llms import LLM
# Ragas (>= 0.2) – evaluate, typed dataset, class-based metrics, and LC wrappers
from ragas import evaluate, EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.evaluation import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall, ResponseRelevancy
# Sentence-Transformers for embeddings
from sentence_transformers import SentenceTransformer

# Project retriever
from rag.retriever import Retriever

# Docs for this API style: evaluate + EvaluationDataset + class metrics + wrappers.  # noqa
# (See: Ragas Evaluate docs and “Evaluate Using Metrics” tutorial)
# https://docs.ragas.io/en/stable/references/evaluate/
# https://docs.ragas.io/en/v0.2.3/getstarted/rag_evaluation/

# --------- Config via env ---------
QAS_PER_SNIPPET = int(os.getenv("EVAL_QAS_PER_SNIPPET", "3"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
MODEL_NAME = os.getenv("MODEL_NAME", "aya-expanse:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# --------- Utils ---------
def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def _slug(n: int = 6) -> str:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _truncate(txt: str, n: int = 500) -> str:
    txt = (txt or "").replace("\n", " ").strip()
    return txt if len(txt) <= n else txt[: n - 1] + "…"

def _json_safe(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, list):
        return [_json_safe(i) for i in x]
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    return x

# --- Minimal LangChain LLM that hits Ollama /api/generate (non-chat) ---
class OllamaGenerateLLM(LLM):
    base_url: str = OLLAMA_HOST
    model: str = MODEL_NAME
    http_timeout: float = 60.0
    options: Dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        return "ollama_generate_httpx"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> str:
        payload: Dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False}
        opts = dict(self.options or {})
        if stop:
            opts["stop"] = stop
        if opts:
            payload["options"] = opts
        with httpx.Client(timeout=self.http_timeout) as client:
            r = client.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return str(data.get("response") or "")

# --- Embeddings shim via sentence-transformers (LangChain interface) ---
class _LCEmb(LCEmb):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self._st = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._st.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._st.encode([text], normalize_embeddings=True)[0].tolist()

# --------- Ollama helpers ---------
async def _ollama_complete(prompt: str, options: Optional[dict] = None, timeout: float = 120.0) -> str:
    payload: Dict[str, Any] = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
        return str(data.get("response") or "")

# --------- Chroma sampling ---------
def _fetch_chroma_samples(limit_docs: int = 40) -> List[Dict[str, Any]]:
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection("docs")
    data = coll.get(include=["metadatas", "documents"])
    ids = data.get("ids") or []
    metas = data.get("metadatas") or []
    texts = data.get("documents") or []

    docs: List[Dict[str, Any]] = []
    for i, did in enumerate(ids[:limit_docs]):
        meta = metas[i] or {}
        text = (texts[i] or "").strip()
        if text:
            docs.append({
                "id": did,
                "source": meta.get("source") or "unknown",
                "start_line": int(meta.get("start_line") or 0),
                "end_line": int(meta.get("end_line") or 0),
                "text": text,
            })

    # Dedup by (source, truncated text)
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for d in docs:
        k = (d["source"], _truncate(d["text"], 120))
        if k not in seen:
            seen.add(k)
            uniq.append(d)
    random.shuffle(uniq)
    return uniq[:limit_docs]

# --------- Q/A generation from a snippet ---------
_QA_PROMPT = """You are an expert data annotator. Given the SOURCE SNIPPET below, write {n} factual Q&A pairs that can be answered using ONLY the snippet.

Rules:
- Keep questions short and unambiguous (1 sentence).
- Answers must be concise and directly supported by the snippet. Do NOT invent.
- Return STRICT JSON with this shape: {{ "items": [ {{"question": "...", "answer": "...", "source_lines": [START, END]}} ] }}

SOURCE ({source}:{start_line}-{end_line}):
\"\"\"
{snippet}
\"\"\""""

async def _generate_qas_from_snippet(snippet: Dict[str, Any], n: int = 1) -> List[Dict[str, Any]]:
    prompt = _QA_PROMPT.format(
        n=max(1, n),
        source=snippet.get("source") or "unknown",
        start_line=snippet.get("start_line") or 0,
        end_line=snippet.get("end_line") or 0,
        snippet=snippet.get("text") or "",
    )
    raw = await _ollama_complete(prompt)
    obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    items: List[Dict[str, Any]] = []
    for it in obj["items"]:
        q = (it["question"] or "").strip()
        a = (it["answer"] or "").strip()
        if q and a:
            items.append({
                "question": q,
                "ground_truth": a,
                "ref_source": snippet.get("source"),
                "ref_range": [snippet.get("start_line"), snippet.get("end_line")]
            })
    return items[:n]

# --------- RAG answerer ---------
async def _answer_with_rag(question: str) -> Tuple[str, List[str]]:
    retr = Retriever(k_initial=12, k_final=RAG_TOP_K, use_reranker=True)
    docs = retr.retrieve(question)
    contexts = [d.get("text") or "" for d in docs]
    ctx_text = "\n\n".join(f"[{i + 1}] {_truncate(t, 1200)}" for i, t in enumerate(contexts))
    sys = ("You are a concise assistant. Use ONLY the provided context if it answers the question. "
           "If insufficient, say briefly what is missing. Provide 1–3 short sentences.")
    full = f"{sys}\n\n# Context:\n{ctx_text}\n\n# Question:\n{question}\n\n# Answer:"
    answer = await _ollama_complete(full, options=None, timeout=120.0)
    return answer.strip(), contexts

# --------- Public API ---------
async def run_evaluation(limit: int = 25) -> Dict[str, Any]:
    out_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "..", "logs", "ragas"))
    stamp = _now_ts() + "-" + _slug()
    base = os.path.join(out_dir, f"eval-{stamp}")
    csv_path = base + ".csv"
    json_path = base + ".json"

    # 1) Sample & synthesize QA
    samples = _fetch_chroma_samples(limit_docs=100)
    qa_items: List[Dict[str, Any]] = []
    needed = limit
    for snip in samples:
        need = min(QAS_PER_SNIPPET, needed)
        if need <= 0:
            break
        qa_items.extend(await _generate_qas_from_snippet(snip, n=need))
        needed = limit - len(qa_items)
        if needed <= 0:
            break
    qa_items = qa_items[:limit]

    # 2) RAG answers
    rows: List[Dict[str, Any]] = []
    for idx, qa in enumerate(qa_items, start=1):
        ans, ctxs = await _answer_with_rag(qa["question"])
        rows.append({
            "id": idx,
            "question": qa["question"],
            "ground_truth": qa["ground_truth"],
            "answer": ans,
            "contexts": ctxs,
            "ref_source": qa.get("ref_source", ""),
            "ref_range": qa.get("ref_range", []),
        })

    # 3) Persist CSV via pandas
    df = pd.DataFrame([{
        "id": r["id"],
        "question": r["question"],
        "ground_truth": r["ground_truth"],
        "answer": r["answer"],
        "contexts_n": len(r["contexts"]),
        "ref_source": r["ref_source"],
        "ref_range": str(r["ref_range"]),
    } for r in rows])
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # 4) RAGAS evaluation (strict API, no fallbacks)
    # Wrap LC LLM/Embeddings for Ragas
    eval_llm = LangchainLLMWrapper(
        OllamaGenerateLLM(base_url=OLLAMA_HOST, model=MODEL_NAME, http_timeout=60.0, options={"temperature": 0.0})
    )
    eval_emb = LangchainEmbeddingsWrapper(_LCEmb(os.getenv("EMBED_MODEL", "BAAI/bge-m3")))

    eval_dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            reference=r["ground_truth"],
            retrieved_contexts=r["contexts"],
        )
        for r in rows
    ])

    metrics = [
        Faithfulness(llm=eval_llm),
        ContextPrecision(llm=eval_llm),
        ContextRecall(llm=eval_llm),
        ResponseRelevancy(llm=eval_llm),  # modern name for “answer relevance”
    ]  # See: Response Relevancy docs. https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/

    result = evaluate(dataset=eval_dataset, metrics=metrics, show_progress=False, raise_exceptions=False)
    df_r = result.to_pandas()

    # 5) Summaries
    scores: Dict[str, Optional[float]] = {}
    for col in df_r.columns:
        try:
            m = float(df_r[col].mean())
        except Exception:
            m = float("nan")
        scores[col] = m if math.isfinite(m) else None

    summary = {
        "stamp": stamp,
        "csv_path": os.path.abspath(csv_path),
        "png_path": None,
        "n": len(rows),
        "scores": _json_safe(scores),
        "top_failures": [],  # can be populated with lowest faithfulness, etc.
        "ragas_available": True,
        "ragas_error": None,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)

    latest_path = os.path.join(out_dir, "latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)

    return summary

def get_last_summary() -> Dict[str, Any]:
    out_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "..", "logs", "ragas"))
    latest_path = os.path.join(out_dir, "latest.json")
    if not os.path.exists(latest_path):
        return {"error": "no_previous_runs"}
    with open(latest_path, "r", encoding="utf-8") as f:
        return json.load(f)
