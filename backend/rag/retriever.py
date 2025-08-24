# backend/rag/retriever.py
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

# Silence Chroma telemetry noise
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
for name in (
    "chromadb.telemetry",
    "chromadb.telemetry.product",
    "chromadb.telemetry.product.posthog",
):
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).disabled = True

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION = os.getenv("CHROMA_COLLECTION", "docs")
RERANKER_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# Minimal bilingual stopwords for light query normalization
_HE_STOP = {
    "של","עם","על","לא","כן","אם","או","זה","זו","אלה","הוא","היא","הם","הן",
    "אני","אתה","את","אתם","אתן","אנחנו","וכו","כל","גם","אך","אך־","אבל","כדי",
    "כי","כמו","אלא","שרק","רק","עוד","כבר","שהוא","שהיא","שזה","שזו","שאין","אין",
    "היה","הייתה","היו","יש","מאוד","בה","בו","בהם","בהן","לפי","מתוך",
}
_EN_STOP = {
    "the","a","an","and","or","but","if","so","to","of","in","on","for","by","with","as","at",
    "is","are","was","were","be","been","being","that","this","these","those","it","its","from",
}

def _normalize_query(q: str) -> str:
    if not q:
        return q
    q = q.strip()
    # Remove punctuation except word chars, whitespace and Hebrew range
    q2 = re.sub(r"[^\w\u0590-\u05FF\s]+", " ", q)
    toks = [t for t in q2.split() if t.lower() not in _EN_STOP and t not in _HE_STOP]
    return " ".join(toks) if toks else q

class Retriever:
    """
    Two-stage retriever:
      1) Dense retrieve (BGE-M3) from Chroma
      2) Optional CrossEncoder rerank (bge-reranker-base) for multilingual precision
    """

    def __init__(self, k_initial: int = 12, k_final: int = 5, use_reranker: bool = True):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
        self.coll = self.client.get_or_create_collection(COLLECTION)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_NAME) if use_reranker else None
        self.k_initial = k_initial
        self.k_final = k_final
        self.use_reranker = use_reranker

    def retrieve(self, query: str, where: Dict[str, str] | None = None) -> List[Dict[str, Any]]:
        q_norm = _normalize_query(query) or query

        qvec = self.embedder.encode([q_norm], normalize_embeddings=True)[0].tolist()

        where_clean = None
        if where:
            # Chroma expects scalar values; coerce to str
            where_clean = {k: (str(v) if v is not None else v) for k, v in where.items()}

        res = self.coll.query(
            query_embeddings=[qvec],
            n_results=self.k_initial,
            include=["documents", "metadatas", "distances"],
            where=where_clean,
        )

        candidates: List[Dict[str, Any]] = []
        if res and res.get("documents"):
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0]
            for i in range(len(docs)):
                meta = metas[i] or {}
                candidates.append({
                    "text": docs[i],
                    "source": meta.get("source"),
                    "chunk": meta.get("chunk"),
                    "start_line": int(meta.get("start_line") or 0),
                    "end_line": int(meta.get("end_line") or 0),
                    "distance": float(dists[i]),
                })

        if not candidates:
            return []

        if self.use_reranker and self.reranker is not None:
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c["score"] = float(s)
            candidates.sort(key=lambda x: x["score"], reverse=True)
        else:
            # smaller distance is better
            candidates.sort(key=lambda x: x["distance"])

        return candidates[: self.k_final]
