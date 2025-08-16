# Local AI Agent (Ollama + LangChain + RAG)

Single-repo project with a FastAPI backend and a React (Vite + TS) frontend.  
Models run **locally** via **Ollama**; retrieval uses **ChromaDB** with **BAAI/bge-m3** embeddings.

## Quickstart (dev)

**Requirements**
- Python 3.11 
- Node.js 20+ 
- Ollama running locally

**1) Backend**
```powershell
conda activate local-agent-311
cd app\backend
uvicorn main:app --reload --port 8000
