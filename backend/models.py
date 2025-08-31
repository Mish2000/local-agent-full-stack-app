# backend/models.py
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import Boolean, Text, ForeignKey, BigInteger, LargeBinary
from sqlalchemy import String, DateTime, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# --- Users & Sessions ---

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(120))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    sessions: Mapped[list["Session"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    sid: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(255))
    ip: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user: Mapped["User"] = relationship(back_populates="sessions")


# --- Chats ---

class Chat(Base):
    __tablename__ = "chats"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), default="New chat", nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    messages: Mapped[list["ChatMessage"]] = relationship(backref="chat", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# --- Files ---

class FileMeta(Base):
    __tablename__ = "files"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    chat_id: Mapped[Optional[int]] = mapped_column(ForeignKey("chats.id", ondelete="SET NULL"), nullable=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime: Mapped[Optional[str]] = mapped_column(String(100))
    size_bytes: Mapped[Optional[int]]
    sha256_hex: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "sha256_hex", name="uq_user_filehash"),
    )


# --- Password reset ---

class PasswordReset(Base):
    __tablename__ = "password_resets"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


# --- User profile & personalization ---

class UserProfile(Base):
    __tablename__ = "user_profiles"

    # one row per user
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    instruction_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    instruction_text: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # avatar settings:
    # avatar_kind: "", "system", or "upload"
    avatar_kind: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    # avatar_value: for system => preset id (e.g., "sys-1"), for upload => ignored (kept for compatibility)
    avatar_value: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    # NEW: persist upload bytes so the avatar survives restarts / logouts
    avatar_blob: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    avatar_mime: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    avatar_updated_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
