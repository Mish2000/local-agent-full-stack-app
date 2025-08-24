# backend/rag/ingest.py
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Silence Chroma telemetry
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Optional readers (installable in dev)
import docx2txt  # type: ignore
from pypdf import PdfReader  # type: ignore

# ---- Config via env (keeps consistent with retriever.py) ----
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "docs")

TEXT_EXTS = {".txt", ".md"}
DOCX_EXTS = {".docx"}
PDF_EXTS = {".pdf"}

# ---------------- File readers ----------------
def read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def read_docx_file(p: Path) -> str:
    try:
        return docx2txt.process(str(p)) or ""
    except Exception:
        return ""

def read_pdf_file(p: Path) -> str:
    try:
        reader = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""

# ---------------- Chunking ----------------
def _normalize_whitespace(s: str) -> str:
    # normalize hard newlines but keep structure
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # strip trailing spaces on lines
    lines = [ln.rstrip() for ln in s.split("\n")]
    return "\n".join(lines).strip()

def chunk_text(text: str, *, max_chars: int = 1200, min_chars: int = 400) -> List[Tuple[str, int, int]]:
    """
    Split text on line boundaries into pieces around max_chars.
    Returns list of (chunk_text, start_line_1based, end_line_1based).
    """
    text = _normalize_whitespace(text)
    if not text:
        return []

    lines = text.splitlines()
    n = len(lines)
    chunks: List[Tuple[str, int, int]] = []
    i = 0
    while i < n:
        j = i
        total = 0
        while j < n:
            ln = lines[j]
            add = len(ln) + (1 if total > 0 else 0)  # +1 for newline if not first
            # Stop if we would exceed max and we already have some content
            if total + add > max_chars and j > i:
                break
            total += add
            j += 1

        # If the chunk is tiny (< min_chars) and there is more text, greedily add one more line
        if (j - i) >= 1 and total < min_chars and j < n:
            total += len(lines[j]) + 1
            j += 1

        chunk = "\n".join(lines[i:j]).strip()
        if chunk:
            chunks.append((chunk, i + 1, j))
        i = max(j, i + 1)

    return chunks

# ---------------- Helpers ----------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def load_file(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in TEXT_EXTS:
        return read_text_file(p)
    if ext in DOCX_EXTS:
        return read_docx_file(p)
    if ext in PDF_EXTS:
        return read_pdf_file(p)
    return ""

# ---------------- Main ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder or file path to index")
    ap.add_argument("--user-id", type=str, default=None, help="Namespace: user id for retrieval filters")
    ap.add_argument("--chat-id", type=str, default=None, help="Optional narrower namespace: chat id")
    ap.add_argument("--collection", type=str, default=CHROMA_COLLECTION, help="Chroma collection name (default: docs)")
    args = ap.parse_args()

    src_path = Path(args.src).resolve()
    if not src_path.exists():
        print(f"[ingest] Source not found: {src_path}")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(args.collection)

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    total_added = 0

    paths: List[Path]
    if src_path.is_dir():
        paths = [p for p in src_path.rglob("*") if p.suffix.lower() in (TEXT_EXTS | DOCX_EXTS | PDF_EXTS)]
    else:
        paths = [src_path]

    for p in paths:
        text = load_file(p)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        # Build batch
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []

        for idx, (ck, start_line, end_line) in enumerate(chunks, start=1):
            doc_id = sha1(f"{p}:{start_line}-{end_line}:{ck[:48]}")
            ids.append(doc_id)
            docs.append(ck)
            meta: Dict[str, Any] = {
                "source": str(p),
                "chunk": idx,
                "start_line": start_line,
                "end_line": end_line,
            }
            if args.user_id:
                meta["user_id"] = str(args.user_id)
            if args.chat_id:
                meta["chat_id"] = str(args.chat_id)
            metas.append(meta)

        # Embed once per file
        vectors = embedder.encode(docs, normalize_embeddings=True).tolist()

        # Upsert
        if hasattr(coll, "upsert"):
            coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
        else:
            coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)

        total_added += len(docs)
        print(f"[ingest] {p.name}: {len(docs)} chunks")

    print(f"[ingest] Done. Added/updated {total_added} chunks to '{args.collection}' in {CHROMA_DIR}")

if __name__ == "__main__":
    main()
