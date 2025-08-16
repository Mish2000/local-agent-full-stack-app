import os
# Disable telemetry and mute logs
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import logging
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

EMBED_MODEL_NAME = "BAAI/bge-m3"
CHROMA_DIR = "chroma_db"
COLLECTION = "docs"

class SimpleRetriever:
    def __init__(self, k: int = 5):
        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.coll = self.client.get_or_create_collection(COLLECTION)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.k = k

    def query(self, q: str):
        qvec = self.embedder.encode([q], normalize_embeddings=True)[0].tolist()
        res = self.coll.query(
            query_embeddings=[qvec],
            n_results=self.k,
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        if res and res.get("documents"):
            for i, doc in enumerate(res["documents"][0]):
                meta = res["metadatas"][0][i]
                dist = float(res["distances"][0][i])
                docs.append({
                    "text": doc,
                    "source": meta.get("source"),
                    "chunk": meta.get("chunk"),
                    "distance": dist,
                })
        return docs

if __name__ == "__main__":
    r = SimpleRetriever(k=3)
    for d in r.query("הסבר קצר על למידת חיזוק"):
        print(d["distance"], d["source"], "…", d["text"][:120].replace("\n", " "))
