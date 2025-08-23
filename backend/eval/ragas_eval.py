# backend/eval/ragas_eval.py
from __future__ import annotations
import inspect
import math
import os
import json
import time
import random
import string
import pathlib
from typing import List, Dict, Any, Tuple, Optional

# External deps used optionally:
# - chromadb (already used by project)
# - httpx (already used)
# - pandas, matplotlib (for CSV + charts; optional but recommended)
# - ragas (optional; if missing we produce a fallback report)
import httpx

# --- Minimal LangChain LLM that returns proper LLMResult for Ragas ---
from langchain_core.language_models.llms import LLM
from typing import Optional

class OllamaGenerateLLM(LLM):
    """
    LangChain LLM subclass that calls Ollama's /api/generate (non-chat) endpoint.
    Returning text from _call() lets BaseLLM build an LLMResult with .generations,
    which is exactly what ragas' metrics need.
    """
    base_url: str = os.getenv("OLLAMA_HOST")
    model: str = os.getenv("MODEL_NAME")
    http_timeout: float = 60.0
    options: Dict[str, Any] = {}  # e.g., {"temperature": 0.0}

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
            # Ollama supports stop via options
            opts["stop"] = stop
        if opts:
            payload["options"] = opts

        with httpx.Client(timeout=self.http_timeout) as client:
            r = client.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return str(data.get("response") or "")

import chromadb
from chromadb.config import Settings

# Reuse embedder via retriever for consistency
from rag.retriever import Retriever

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
    """Recursively replace NaN/Inf with None so Starlette/FastAPI JSON won't error."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, list):
        return [_json_safe(i) for i in x]
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    return x

def _nan_to_none(x):
    try:
        import math
        return None if (isinstance(x, float) and math.isnan(x)) else x
    except Exception:
        return x

# --------- Minimal Ollama call (non-stream) ---------

async def _ollama_complete(prompt: str, options: Optional[dict] = None, timeout: float = 120.0) -> str:
    payload: Dict[str, Any] = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
        return str(data.get("response") or "")


# --------- Minimal Ollama call (sync) for Ragas' sync path ---------

def _ollama_complete_sync(prompt: str, options: Optional[dict] = None, timeout: float = 120.0) -> str:
    payload: Dict[str, Any] = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
        return str(data.get("response") or "")


# --------- Step 1: Sample corpus snippets from Chroma ---------

def _fetch_chroma_samples(limit_docs: int = 40) -> List[Dict[str, Any]]:
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection("docs")

    # Try to get a bunch of items; newer chromadb supports limit
    try:
        data = coll.get(include=["metadatas", "documents"], limit=limit_docs)
    except TypeError:
        # Older versions don't support limit; get all then truncate
        data = coll.get(include=["metadatas", "documents"])

    docs = []
    ids = data.get("ids") or []
    metas = data.get("metadatas") or []
    texts = data.get("documents") or []

    for i, did in enumerate(ids):
        meta = metas[i] or {}
        text = texts[i] or ""
        if text.strip():
            docs.append({
                "id": did,
                "source": meta.get("source") or "unknown",
                "start_line": int(meta.get("start_line") or 0),
                "end_line": int(meta.get("end_line") or 0),
                "text": text.strip(),
            })

    # Deduplicate by identical text to avoid many near-duplicates
    seen = set()
    uniq = []
    for d in docs:
        k = (d["source"], _truncate(d["text"], 120))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(d)

    random.shuffle(uniq)
    return uniq[:limit_docs]


# --------- Step 2: Ask LLM to create Q/A pairs from snippets ---------

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

    # Try to parse JSON
    import json as _json
    items: List[Dict[str, Any]] = []
    try:
        obj = _json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        for it in obj.get("items", []):
            q = (it.get("question") or "").strip()
            a = (it.get("answer") or "").strip()
            if q and a:
                items.append({"question": q, "ground_truth": a, "ref_source": snippet.get("source"),
                              "ref_range": [snippet.get("start_line"), snippet.get("end_line")]})
    except Exception:
        pass

    # Fallback: naive split if JSON failed (best-effort, still useful)
    if not items:
        lines = [ln.strip("-• ") for ln in raw.splitlines() if ln.strip()]
        for ln in lines[:n]:
            if ":" in ln:
                q, a = ln.split(":", 1)
                q, a = q.strip(), a.strip()
                if q and a:
                    items.append({"question": q, "ground_truth": a, "ref_source": snippet.get("source"),
                                  "ref_range": [snippet.get("start_line"), snippet.get("end_line")]})

    return items[:n]


# --------- Step 3: Run our pipeline to retrieve contexts + answer ---------

async def _answer_with_rag(question: str) -> Tuple[str, List[str]]:
    retr = Retriever(k_initial=12, k_final=RAG_TOP_K, use_reranker=True)
    docs = retr.retrieve(question)
    contexts = [d.get("text") or "" for d in docs]

    # Build a minimal grounded prompt:
    ctx_text = "\n\n".join(f"[{i + 1}] {_truncate(t, 1200)}" for i, t in enumerate(contexts))
    sys = ("You are a concise assistant. Use ONLY the provided context if it answers the question. "
           "If insufficient, say briefly what is missing. Provide 1–3 short sentences.")
    full = f"{sys}\n\n# Context:\n{ctx_text}\n\n# Question:\n{question}\n\n# Answer:"

    answer = await _ollama_complete(full, options=None, timeout=120.0)
    return answer.strip(), contexts


# --------- Step 4: Optional: RAGAS metrics ---------

def _try_import_ragas():
    """
    Load ragas evaluate API and build LangChain-native LLM & Embeddings.
    Let ragas auto-wrap these (per docs), avoiding direct wrapper/version churn.
    """
    try:
        from ragas import evaluate  # main entrypoint
        # Metrics: prefer new names, fall back gracefully
        try:
            from ragas.metrics import (
                faithfulness,
                context_precision,
                context_recall,
                answer_relevance,  # current canonical name
            )
            _answer_rel = answer_relevance
        except Exception:
            # Older export name on some installs
            from ragas.metrics import (
                faithfulness,
                context_precision,
                context_recall,
                answer_relevancy,
            )
            _answer_rel = answer_relevancy

        # LangChain embeddings shim using sentence-transformers
        from sentence_transformers import SentenceTransformer
        from langchain.embeddings.base import Embeddings as LCEmb

        emb_model = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
        st = SentenceTransformer(emb_model)

        class _LCEmb(LCEmb):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return st.encode(texts, normalize_embeddings=True).tolist()
            def embed_query(self, text: str) -> List[float]:
                return st.encode([text], normalize_embeddings=True)[0].tolist()

        # Build our non-chat LLM that produces LLMResult via LangChain base class
        lc_llm = OllamaGenerateLLM(
            base_url=OLLAMA_HOST,
            model=MODEL_NAME,
            http_timeout=60.0,
            options={"temperature": 0.0},  # safe default; goes under 'options', not as kwarg
        )

        return {
            "evaluate": evaluate,
            "metrics": [faithfulness, context_precision, context_recall, _answer_rel],
            "llm": lc_llm,          # pass raw LangChain LLM; ragas will wrap it
            "embeddings": _LCEmb(), # pass raw LangChain Embeddings; ragas will wrap it
        }

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

# --------- Public API ---------

async def run_evaluation(limit: int = 25) -> Dict[str, Any]:
    """
    Returns a summary dict and writes:
      - CSV with per-sample results
      - PNG chart (if matplotlib available)
      - JSON summary
    """
    import inspect
    import math

    # Local JSON sanitizer to avoid NaN/Inf serialization failures
    def _json_safe(val):
        if isinstance(val, float):
            return None if not math.isfinite(val) else val
        if isinstance(val, dict):
            return {k: _json_safe(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_json_safe(v) for v in val]
        return val

    out_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "..", "logs", "ragas"))
    stamp = _now_ts() + "-" + _slug()
    base = os.path.join(out_dir, f"eval-{stamp}")
    csv_path = base + ".csv"
    json_path = base + ".json"
    png_path = base + ".png"

    # Step 1: sample snippets and generate QAs
    samples = _fetch_chroma_samples(limit_docs=100)
    qa_items: List[Dict[str, Any]] = []
    needed = limit
    for snip in samples:
        need = min(QAS_PER_SNIPPET, needed)
        if need <= 0:
            break
        qas = await _generate_qas_from_snippet(snip, n=need)
        qa_items.extend(qas)
        needed = limit - len(qa_items)
        if needed <= 0:
            break
    # If we couldn't generate enough QAs, trim
    qa_items = qa_items[:limit]

    # Step 2: for each QA, retrieve + answer
    rows: List[Dict[str, Any]] = []
    for idx, qa in enumerate(qa_items, start=1):
        q = qa["question"]
        gt = qa["ground_truth"]
        try:
            ans, ctxs = await _answer_with_rag(q)
        except Exception as e:
            ans, ctxs = f"(error during answering: {e})", []
        rows.append({
            "id": idx,
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "contexts": ctxs,
            "ref_source": qa.get("ref_source", ""),
            "ref_range": qa.get("ref_range", []),
        })

    # Step 3: write CSV (if pandas available)
    pd = None
    try:
        import pandas as _pd
        pd = _pd
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
    except Exception:
        # Fallback: write minimal CSV manually
        import csv
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id","question","ground_truth","answer","contexts_n","ref_source","ref_range"])
            for r in rows:
                w.writerow([r["id"], r["question"], r["ground_truth"], r["answer"], len(r["contexts"]), r["ref_source"], r["ref_range"]])

    # Step 4: try RAGAS (robust across versions; JSON-safe)
    ragas = _try_import_ragas()
    scores: Dict[str, Optional[float]] = {}
    per_sample: List[Dict[str, Any]] = []

    if ragas and "error" not in ragas:
        try:
            # Prefer typed dataset in 0.2+, else HF Dataset for 0.1.x
            ds_obj = None
            try:
                # New typed API
                from ragas.evaluation import EvaluationDataset, SingleTurnSample
                typed_samples = []
                for r in rows:
                    typed_samples.append(
                        SingleTurnSample(
                            user_input=r["question"],
                            response=r["answer"],
                            reference=r["ground_truth"],
                            retrieved_contexts=r["contexts"],  # list[str]
                        )
                    )
                ds_obj = EvaluationDataset(samples=typed_samples)
            except Exception:
                # Older API: HuggingFace Dataset
                from datasets import Dataset as HFDataset
                ds_dict = {
                    "question": [r["question"] for r in rows],
                    "answer": [r["answer"] for r in rows],
                    "contexts": [r["contexts"] for r in rows],
                    "ground_truth": [r["ground_truth"] for r in rows],
                }
                ds_obj = HFDataset.from_dict(ds_dict)

            # Build kwargs by inspecting the installed ragas.evaluate signature
            eval_sig = inspect.signature(ragas["evaluate"])
            eval_kwargs: Dict[str, Any] = {}

            # Dataset arg name can be 'dataset' (new) or 'data' (old)
            if "dataset" in eval_sig.parameters:
                eval_kwargs["dataset"] = ds_obj
            elif "data" in eval_sig.parameters:
                eval_kwargs["data"] = ds_obj

            if "metrics" in eval_sig.parameters:
                eval_kwargs["metrics"] = ragas["metrics"]
            if "llm" in eval_sig.parameters:
                eval_kwargs["llm"] = ragas["llm"]  # LangChain LLM; ragas wraps internally
            if "embeddings" in eval_sig.parameters:
                eval_kwargs["embeddings"] = ragas["embeddings"]  # LangChain Embeddings; ragas wraps internally
            if "raise_exceptions" in eval_sig.parameters:
                eval_kwargs["raise_exceptions"] = False
            if "show_progress" in eval_sig.parameters:
                eval_kwargs["show_progress"] = False

            # Final call (uses only args actually supported by your installed ragas)
            result = ragas["evaluate"](**eval_kwargs)

            # Normalize to pandas-like table
            try:
                df_r = result.to_pandas() if hasattr(result, "to_pandas") else result
            except Exception:
                df_r = result

            # Collect available metric columns safely
            cols: List[str] = []
            if hasattr(df_r, "columns"):
                # Try canonical names from metric objects
                for m in ragas["metrics"]:
                    name = getattr(m, "name", None) or getattr(m, "__name__", None)
                    if name and (name in getattr(df_r, "columns", [])):
                        cols.append(name)
                # Also tolerate common historical aliases
                for alias in ("answer_relevancy", "response_relevancy", "answer_relevance"):
                    if alias in getattr(df_r, "columns", []) and alias not in cols:
                        cols.append(alias)

                # Means (guard NaN/Inf)
                for c in cols:
                    try:
                        mean_val = float(df_r[c].mean())
                        scores[c] = mean_val if math.isfinite(mean_val) else None
                    except Exception:
                        scores[c] = None

                # Per-sample (align by row index)
                for i, base in enumerate(rows):
                    item = {
                        "id": base["id"],
                        "question": base["question"],
                        "ground_truth": base["ground_truth"],
                        "answer": base["answer"],
                    }
                    for c in cols:
                        try:
                            v = float(df_r.iloc[i][c])
                            item[c] = v if math.isfinite(v) else None
                        except Exception:
                            item[c] = None
                    per_sample.append(item)

        except Exception as e:
            ragas = {"error": f"{type(e).__name__}: {e}"}

    # Step 5: write JSON summary
    summary = {
        "stamp": stamp,
        "csv_path": os.path.abspath(csv_path),
        "png_path": os.path.abspath(png_path) if os.path.exists(png_path) else None,
        "n": len(rows),
        "scores": scores,
        "top_failures": sorted(
            per_sample,
            key=lambda x: ((x.get("faithfulness") or 1.0) + (
                x.get("answer_relevancy") or x.get("response_relevancy") or x.get("answer_relevance") or 1.0)),
        )[:5] if per_sample else [],
        "ragas_available": "error" not in ragas if ragas else False,
        "ragas_error": ragas.get("error") if ragas and "error" in ragas else None,
    }

    # Ensure the payload is JSON-safe (no NaN/Inf)
    summary = _json_safe(summary)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Also write a "latest" marker
    latest_path = os.path.join(out_dir, "latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

def get_last_summary() -> Dict[str, Any]:
    out_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "..", "logs", "ragas"))
    latest_path = os.path.join(out_dir, "latest.json")
    if os.path.exists(latest_path):
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"error": "no_previous_runs"}