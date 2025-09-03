# backend/rag/common.py
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

# --- Silence Chroma telemetry ---
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
from sentence_transformers import SentenceTransformer

# --- Env / paths (single source of truth) ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
if not os.path.isabs(CHROMA_DIR):
    # anchor to backend/ root (avoid accidental DB at repo root)
    CHROMA_DIR = str((Path(__file__).parent.parent / CHROMA_DIR).resolve())

COLLECTION = os.getenv("CHROMA_COLLECTION", "docs")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANKER_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# --- Factories (cached singletons) ---
@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))

@lru_cache(maxsize=16)
def get_collection(name: Optional[str] = None):
    return get_chroma_client().get_or_create_collection(name or COLLECTION)

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    # normalize=True call sites can still pass normalize_embeddings=True per encode
    return SentenceTransformer(EMBED_MODEL_NAME)

# optional import to avoid heavy dependency when not needed
def get_reranker_or_none():
    try:
        from sentence_transformers import CrossEncoder  # local import
        return CrossEncoder(RERANKER_NAME)
    except Exception:
        return None
