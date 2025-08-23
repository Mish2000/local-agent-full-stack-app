# backend/auth.py
from __future__ import annotations
import os, secrets, datetime as dt
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Response, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from db import SessionLocal
from models import User, Session

from sqlalchemy import update
from models import PasswordReset

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_NAME = os.getenv("COOKIE_NAME", "sid")
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "168"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
ALLOW_DEV_PASSWORD_RESET = os.getenv("ALLOW_DEV_PASSWORD_RESET", "1") == "1"

ph = PasswordHasher()

# --------- Schemas ---------

class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: Optional[str] = Field(default=None, max_length=120)

class AuthOut(BaseModel):
    id: int
    email: EmailStr
    display_name: Optional[str]

# --------- DB dep ---------

async def get_db():
    async with SessionLocal() as s:
        yield s

# --------- Helpers ---------

def _cookie_settings():
    # for local dev: secure=False; same-site Lax works across ports
    return dict(
        key=COOKIE_NAME,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=SESSION_TTL_HOURS * 3600,
        path="/",
    )

async def _create_session(resp: Response, s: AsyncSession, user: User, req: Request):
    sid = secrets.token_urlsafe(48)
    now = dt.datetime.now(dt.timezone.utc)
    expires = now + dt.timedelta(hours=SESSION_TTL_HOURS)
    sess = Session(
        user_id=user.id, sid=sid,
        user_agent=req.headers.get("user-agent"),
        ip=req.client.host if req.client else None,
        created_at=now, expires_at=expires,
    )
    s.add(sess)
    await s.commit()
    resp.set_cookie(value=sid, **_cookie_settings())

async def _current_user(req: Request, s: AsyncSession) -> Optional[User]:
    sid = req.cookies.get(COOKIE_NAME)
    if not sid:
        return None
    now = dt.datetime.now(dt.timezone.utc)
    q = (
        select(User)
        .join(Session, Session.user_id == User.id)
        .where(Session.sid == sid, Session.expires_at > now)
        .limit(1)
    )
    r = await s.execute(q)
    return r.scalars().first()

# --------- Routes ---------

@router.post("/register", response_model=AuthOut)
async def register(inp: RegisterIn, response: Response, request: Request, db: AsyncSession = Depends(get_db)):
    exists = await db.execute(select(User.id).where(User.email == inp.email).limit(1))
    if exists.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=inp.email,
        password_hash=ph.hash(inp.password),
        display_name=inp.display_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # IMPORTANT: do NOT auto-create a session. User must log in after register.
    return AuthOut(id=user.id, email=user.email, display_name=user.display_name)


class LoginIn(BaseModel):
    email: EmailStr
    password: str

@router.post("/login", response_model=AuthOut)
async def login(inp: LoginIn, response: Response, request: Request, db: AsyncSession = Depends(get_db)):
    row = await db.execute(select(User).where(User.email == inp.email).limit(1))
    user = row.scalars().first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    try:
        ph.verify(user.password_hash, inp.password)
    except VerifyMismatchError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    await _create_session(response, db, user, request)
    return AuthOut(id=user.id, email=user.email, display_name=user.display_name)

@router.get("/me", response_model=Optional[AuthOut])
async def me(request: Request, db: AsyncSession = Depends(get_db)):
    user = await _current_user(request, db)
    if not user:
        return None
    return AuthOut(id=user.id, email=user.email, display_name=user.display_name)

@router.post("/logout")
async def logout(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    sid = request.cookies.get(COOKIE_NAME)
    if sid:
        await db.execute(delete(Session).where(Session.sid == sid))
        await db.commit()
    # clear cookie
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return {"ok": True}

class ForgotIn(BaseModel):
    email: EmailStr

@router.post("/forgot")
async def forgot_password(inp: ForgotIn, request: Request, db: AsyncSession = Depends(get_db)):
    # Always return ok (don’t disclose account existence)
    row = await db.execute(select(User).where(User.email == inp.email).limit(1))
    user = row.scalars().first()
    reset_url = None
    if user:
        token = secrets.token_urlsafe(32)
        now = dt.datetime.now(dt.timezone.utc)
        expires = now + dt.timedelta(hours=1)
        pr = PasswordReset(user_id=user.id, token=token, created_at=now, expires_at=expires)
        db.add(pr)
        await db.commit()
        if ALLOW_DEV_PASSWORD_RESET:
            base = FRONTEND_ORIGIN.rstrip("/")
            reset_url = f"{base}/reset?token={token}"
            print(f"[DEV] Password reset link for {user.email}: {reset_url}")
    return {"ok": True, "reset_url": reset_url}

class ResetIn(BaseModel):
    token: str = Field(min_length=16, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)

@router.post("/reset")
async def reset_password(inp: ResetIn, db: AsyncSession = Depends(get_db)):
    now = dt.datetime.now(dt.timezone.utc)
    q = select(PasswordReset).where(
        PasswordReset.token == inp.token,
        PasswordReset.used_at.is_(None),
        PasswordReset.expires_at > now
    ).limit(1)
    row = await db.execute(q)
    pr = row.scalars().first()
    if not pr:
        raise HTTPException(status_code=400, detail="Invalid or expired token")

    # Change password
    user_row = await db.execute(select(User).where(User.id == pr.user_id).limit(1))
    user = user_row.scalars().first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid token")

    user.password_hash = ph.hash(inp.new_password)
    pr.used_at = now

    # (Optional) Invalidate existing sessions for this user
    await db.execute(delete(Session).where(Session.user_id == user.id))
    await db.commit()
    return {"ok": True}

