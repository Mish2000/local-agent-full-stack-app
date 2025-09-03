# backend/chats.py
from __future__ import annotations
import datetime as dt
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select, func, desc, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_db, _current_user  # reuse existing helpers
from models import Chat, ChatMessage, User, ilnow

router = APIRouter(prefix="/chats", tags=["chats"])

# ---------- Schemas ----------

class ChatOut(BaseModel):
    id: int
    title: str
    updated_at: dt.datetime
    last_preview: Optional[str] = None

class ChatCreateIn(BaseModel):
    title: Optional[str] = Field(default="New chat", max_length=255)

class ChatRenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=255)

class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: dt.datetime

# ---------- Utilities ----------

async def _require_user(req: Request, db: AsyncSession) -> User:
    user = await _current_user(req, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

async def _require_chat(db: AsyncSession, user_id: int, chat_id: int) -> Chat:
    q = select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id).limit(1)
    r = await db.execute(q)
    chat = r.scalars().first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

# ---------- Routes ----------

@router.get("", response_model=List[ChatOut])
async def list_chats(request: Request, db: AsyncSession = Depends(get_db)):
    user = await _require_user(request, db)

    # Get chats with last message preview
    # Subquery for latest message per chat
    sub = (
        select(
            ChatMessage.chat_id,
            func.max(ChatMessage.id).label("max_id")
        )
        .join(Chat, Chat.id == ChatMessage.chat_id)
        .where(Chat.user_id == user.id)
        .group_by(ChatMessage.chat_id)
        .subquery()
    )

    # Join back for preview text
    q = (
        select(
            Chat.id, Chat.title, Chat.updated_at,
            ChatMessage.content
        )
        .join_from(Chat, sub, Chat.id == sub.c.chat_id, isouter=True)
        .join(ChatMessage, ChatMessage.id == sub.c.max_id, isouter=True)
        .where(Chat.user_id == user.id)
        .order_by(desc(Chat.updated_at))
    )
    rows = (await db.execute(q)).all()
    out: List[ChatOut] = []
    for cid, title, upd, preview in rows:
        prev = (preview or "").strip()
        if len(prev) > 160:
            prev = prev[:159] + "…"
        out.append(ChatOut(id=cid, title=title, updated_at=upd, last_preview=prev or None))
    return out

@router.post("", response_model=ChatOut)
async def create_chat(body: ChatCreateIn, request: Request, db: AsyncSession = Depends(get_db)):
    user = await _require_user(request, db)
    now = ilnow()
    chat = Chat(user_id=user.id, title=body.title or "New chat", created_at=now, updated_at=now)
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return ChatOut(id=chat.id, title=chat.title, updated_at=chat.updated_at, last_preview=None)

@router.patch("/{chat_id}", response_model=ChatOut)
async def rename_chat(chat_id: int, body: ChatRenameIn, request: Request, db: AsyncSession = Depends(get_db)):
    user = await _require_user(request, db)
    chat = await _require_chat(db, user.id, chat_id)
    chat.title = body.title
    chat.updated_at = ilnow()
    await db.commit()
    await db.refresh(chat)
    return ChatOut(id=chat.id, title=chat.title, updated_at=chat.updated_at, last_preview=None)

@router.delete("/{chat_id}")
async def delete_chat(chat_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    user = await _require_user(request, db)
    # Ensure ownership
    await _require_chat(db, user.id, chat_id)
    await db.execute(delete(Chat).where(Chat.id == chat_id, Chat.user_id == user.id))
    await db.commit()
    return {"ok": True}

@router.get("/{chat_id}/messages", response_model=List[MessageOut])
async def list_messages(chat_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    user = await _require_user(request, db)
    await _require_chat(db, user.id, chat_id)
    q = (
        select(ChatMessage.id, ChatMessage.role, ChatMessage.content, ChatMessage.created_at)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.id.asc())
    )
    rows = (await db.execute(q)).all()
    return [MessageOut(id=i, role=r, content=c, created_at=t) for (i, r, c, t) in rows]
