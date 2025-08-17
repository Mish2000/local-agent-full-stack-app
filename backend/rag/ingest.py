import argparse
import os
import hashlib
from pathlib import Path
from typing import List, Tuple

# Silence Chroma telemetry
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import docx2txt
from pypdf import PdfReader

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION = "docs"

TEXT_EXTS = {".txt", ".md"}
DOCX_EXTS = {".docx"}
PDF_EXTS = {".pdf"}

def read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def read_docx_file(p: Path) -> str:
    return docx2txt.process(str(p)) or ""

def read_pdf_file(p: Path) -> str:
    text_parts: List[str] = []
    reader = PdfReader(str(p))
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def load_text(p: Path) -> str:
    suf = p.suffix.lower()
    if suf in TEXT_EXTS:
        return read_text_file(p)
    if suf in DOCX_EXTS:
        return read_docx_file(p)
    if suf in PDF_EXTS:
        return read_pdf_file(p)
    return ""

def iter_files(root: Path) -> List[Path]:
    out = []
    for ext in (TEXT_EXTS | DOCX_EXTS | PDF_EXTS):
        out.extend(root.rglob(f"*{ext}"))
    return sorted(set(out))

def make_line_chunks(text: str, max_chars: int = 1200, overlap_lines: int = 2) -> List[Tuple[str, int, int]]:
    """
    Chunk by lines so we can compute exact line ranges.
    Returns list of (chunk_text, start_line_1based, end_line_1based).
    """
    lines = text.splitlines()
    n = len(lines)
    chunks: List[Tuple[str, int, int]] = []
    i = 0
    while i < n:
        j = i
        total = 0
        while j < n:
            # +1 for the implicit newline that will be re-joined
            next_len = len(lines[j]) + (1 if total > 0 else 0)
            if total + next_len > max_chars and j > i:
                break
            total += next_len
            j += 1
        if j == i:  # very long single line; force include
            j = i + 1
        # Build chunk text
        chunk_lines = lines[i:j]
        chunk_text = "\n".join(chunk_lines).strip()
        if chunk_text:
            start_line = i + 1
            end_line = j
            chunks.append((chunk_text, start_line, end_line))
        # Move window with overlap
        i = max(i + 1, j - overlap_lines if overlap_lines > 0 else j)
    return chunks

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder to index (recursively)")
    args = ap.parse_args()

    src_root = Path(args.src).resolve()
    if not src_root.exists():
        print(f"[ingest] Source folder not found: {src_root}")
        return

    print(f"[ingest] Loading embedder: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"[ingest] Connecting to Chroma at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(COLLECTION)

    files = iter_files(src_root)
    if not files:
        print(f"[ingest] No files found under: {src_root}")
        return

    added = 0
    for p in files:
        text = load_text(p)
        if not text.strip():
            continue
        chunks = make_line_chunks(text, max_chars=1200, overlap_lines=2)
        print(f"[ingest] {p} -> {len(chunks)} chunks")

        # Prepare batches (use upsert so re-runs refresh)
        ids, docs, metas, embs = [], [], [], []
        for idx, (chunk_text, start_line, end_line) in enumerate(chunks):
            uid = f"{p.as_posix()}::lines:{start_line}-{end_line}::sha1:{sha1(chunk_text)[:8]}"
            ids.append(uid)
            docs.append(chunk_text)
            metas.append({
                "source": str(p),
                "chunk": idx,
                "start_line": start_line,
                "end_line": end_line,
            })

        # Compute embeddings once for the batch
        vectors = embedder.encode(docs, normalize_embeddings=True)
        embs.extend(vectors)

        # Upsert (falls back to add if new)
        if hasattr(coll, "upsert"):
            coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        else:
            coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

        added += len(docs)

    print(f"[ingest] Done. Added/updated {added} chunks to '{COLLECTION}' in {CHROMA_DIR}")

if __name__ == "__main__":
    main()
