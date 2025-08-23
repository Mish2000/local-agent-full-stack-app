# backend/db.py
from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv  # pip install python-dotenv
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# --- Load .env from the backend folder (same dir as this file) ---
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)  # ensures DB_URL is available during import

DB_URL = os.environ.get("DB_URL")  # use environ to ensure we read post-load
if not DB_URL:
    raise RuntimeError(
        f"Missing DB_URL environment variable. Expected in {ENV_PATH}. "
        "Set DB_URL or create backend/.env with DB_URL=..."
    )

class Base(DeclarativeBase):
    pass

engine = create_async_engine(DB_URL, pool_pre_ping=True, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def init_db(BaseModel: type[Base]) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(BaseModel.metadata.create_all)
