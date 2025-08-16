from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

import re

app = FastAPI(title="Local AI Agent — Backend", version="0.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Backend is running"}

def is_hebrew(text: str) -> bool:
    return bool(re.search(r"[\u0590-\u05FF]", text))

@app.get("/chat/stream")
async def chat_stream(request: Request, q: str):
    # Pick target language based on user's last message
    heb = is_hebrew(q)
    lang_instruction = (
        "ענה בעברית." if heb else "Respond in English."
    )

    system_prompt = (
        f"{lang_instruction} "
        "When there is code or formulas, write them in English and keep exact formatting. "
        "Be concise unless explicitly asked to elaborate."
    )

    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=q)]

    llm = ChatOllama(
        model="aya-expanse:8b",
        temperature=0.2,
        streaming=True,
        num_ctx=8192,
    )

    async def event_publisher():
        try:
            async for chunk in llm.astream(msgs):
                if await request.is_disconnected():
                    break
                delta = chunk.content or ""
                if delta:
                    yield {"event": "token", "data": delta}
            yield {"event": "done", "data": "[DONE]"}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_publisher())
