# backend/tracing.py
from __future__ import annotations

import os
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional dependency: Langfuse client. If not installed or not configured, we noop gracefully.
try:
    from langfuse import Langfuse  # type: ignore
except Exception:  # pragma: no cover - safe import guard
    Langfuse = None  # type: ignore


class Tracer:
    """
    Lightweight tracing utility for the chat pipeline.

    Behavior:
      - If LANGFUSE_* env vars are present *and* the 'langfuse' package is installed,
        it emits trace/event/generation records to Langfuse.
      - It always writes a compact JSONL audit log to 'backend/logs/traces.jsonl'.
      - Exposes a UUID `trace_id` to surface in the UI via SSE 'trace' events.

    Public API (used in main.py):
      - start_trace(user_query: str, mode: str) -> str
      - log_tool(trace_id: str, name: str, args: Dict[str, Any], result: Any = None, error: Optional[str] = None) -> None
      - end_trace(trace_id: str, prompt: str, response_text: str, citations: List[Dict[str, Any]], ok: bool = True) -> None
    """

    def __init__(self) -> None:
        # Prepare local JSONL sink
        logs_dir = Path(__file__).resolve().parent / "logs"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort; if fails we still continue without local file
            pass
        self.file_path: Path = logs_dir / "traces.jsonl"

        # Langfuse config
        host = os.getenv("LANGFUSE_HOST") or ""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or ""
        secret_key = os.getenv("LANGFUSE_SECRET_KEY") or ""

        self.enabled: bool = False
        self.client: Any = None

        if host and public_key and secret_key and Langfuse is not None:
            try:
                # Lazily initialize client; this is cheap and thread-safe for our usage pattern
                self.client = Langfuse(
                    host=host,
                    public_key=public_key,
                    secret_key=secret_key,
                )
                self.enabled = True
            except Exception:
                # If Langfuse client fails to construct, fall back silently to file-only logging
                self.client = None
                self.enabled = False

    # ----------------- internal helpers -----------------
    def _write(self, obj: Dict[str, Any]) -> None:
        """Append a single JSON object to the local JSONL file."""
        try:
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            # Never raise from tracing
            pass

    # ----------------- public API -----------------
    def start_trace(self, user_query: str, mode: str) -> str:
        """Start a new trace and return its ID."""
        trace_id = str(uuid.uuid4())
        ts_ms = int(time.time() * 1000)

        if self.enabled and self.client:
            try:
                # Minimal fields to make the trace useful in Langfuse UI
                self.client.trace(
                    id=trace_id,
                    name="chat",
                    input={"query": user_query, "mode": mode},
                    metadata={"mode": mode},
                    timestamp=ts_ms,
                )
            except Exception:
                # Continue silently
                pass

        self._write({
            "type": "trace.start",
            "trace_id": trace_id,
            "ts": ts_ms,
            "query": user_query,
            "mode": mode,
        })
        return trace_id

    def log_tool(
        self,
        trace_id: str,
        name: str,
        args: Dict[str, Any],
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """Record a tool call (e.g., web search, RAG probe)."""
        ts_ms = int(time.time() * 1000)
        ok = error is None

        if self.enabled and self.client:
            try:
                self.client.event(
                    trace_id=trace_id,
                    name=f"tool:{name}",
                    input=args,
                    output=result,
                    metadata={"ok": ok},
                    timestamp=ts_ms,
                )
            except Exception:
                pass

        self._write({
            "type": "tool",
            "trace_id": trace_id,
            "ts": ts_ms,
            "name": name,
            "args": args,
            "result": result,
            "error": error,
        })

    def end_trace(
        self,
        trace_id: str,
        prompt: str,
        response_text: str,
        citations: List[Dict[str, Any]],
        ok: bool = True,
    ) -> None:
        """Record the final assistant message as a 'generation' and close out the local log."""
        ts_ms = int(time.time() * 1000)

        if self.enabled and self.client:
            try:
                self.client.generation(
                    trace_id=trace_id,
                    name="assistant",
                    model=os.getenv("MODEL_NAME", "unknown"),
                    input=prompt,
                    output=response_text,
                    metadata={"citations": citations, "ok": ok},
                    timestamp=ts_ms,
                )
            except Exception:
                pass

        self._write({
            "type": "trace.end",
            "trace_id": trace_id,
            "ts": ts_ms,
            "ok": ok,
            "citations": citations,
        })
