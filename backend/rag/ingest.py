import os
# Disable Chroma telemetry and mute telemetry logs before imports
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import logging
for name in (
    "chromadb.telemetry",
    "chromadb.telemetry.product",
    "chromadb.telemetry.product.posthog",
):
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).disabled = True

import argparse
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx2txt

EMBED_MODEL_NAME = "BAAI/bge-m3"
CHROMA_DIR = "chroma_db"
COLLECTION = "docs"
ALLOWED_EXTS = {".pdf", ".txt", ".md", ".csv", ".log", ".docx"}

def load_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join([(page.extract_text() or "") for page in reader.pages])
    if ext in {".txt", ".md", ".csv", ".log"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".docx":
        return docx2txt.process(str(path)) or ""
    return ""

def iter_documents(src_dir: Path):
    for p in src_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            text = load_text_from_file(p).strip()
            if text:
                yield str(p), text

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def seed_sample_if_empty(src_dir: Path) -> None:
    if any(src_dir.rglob("*")):
        return
    sample = src_dir / "sample_seed_he_en.txt"
    sample.write_text(
        "זהו קובץ דוגמה קצר בעברית המדגים אינדקס מסמכים למערכת RAG.\n"
        "This is a short English sample file demonstrating document indexing for RAG.\n"
        "למידת חיזוק (Reinforcement Learning) היא פרדיגמה בלמידת מכונה שבה סוכן לומד על ידי אינטראקציה עם סביבה.\n",
        encoding="utf-8",
    )

def main(src: str):
    src_dir = Path(src).resolve()
    assert src_dir.exists(), f"Source dir not found: {src_dir}"
    seed_sample_if_empty(src_dir)

    print(f"[ingest] Loading embedder: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"[ingest] Connecting to Chroma at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    coll = client.get_or_create_collection(COLLECTION)

    total = 0
    found_any = False
    for path, text in iter_documents(src_dir):
        found_any = True
        chunks = chunk_text(text)
        if not chunks:
            continue

        embeddings = embedder.encode(
            chunks, normalize_embeddings=True, show_progress_bar=True
        )

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": path, "chunk": i} for i, _ in enumerate(chunks)]

        coll.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        total += len(chunks)
        print(f"[ingest] {path} -> {len(chunks)} chunks")

    if not found_any:
        print(f"[ingest] No supported files in {src_dir}. Supported: {sorted(ALLOWED_EXTS)}")
    print(f"[ingest] Done. Added {total} chunks to '{COLLECTION}' in {CHROMA_DIR}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data", help="Folder with documents to index")
    args = ap.parse_args()
    main(args.src)
