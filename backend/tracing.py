import os
import json
import time
import uuid
from typing import Any, Dict, List, Optional


class Tracer:
    """
    Thin wrapper:
      - If LANGFUSE_* env vars exist and 'langfuse' is installed, sends traces/spans.
      - Always writes a minimal JSONL log to backend/logs/traces.jsonl.
      - Exposes a UUID trace_id you can show in the UI.
    """
    def __init__(self) -> None:
        self.host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.enabled = bool(self.public_key and self.secret_key)
        self.client = None
        if self.enabled:
            try:
                from langfuse import Langfuse  # type: ignore
                self.client = Langfuse(public_key=self.public_key, secret_key=self.secret_key, host=self.host)
            except Exception:
                # If SDK missing or misconfigured, silently disable remote sending
                self.enabled = False

        # Local JSONL fallback log
        logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
        os.makedirs(logs_dir, exist_ok=True)
        self.file_path = os.path.join(logs_dir, "traces.jsonl")

    def _write(self, obj: Dict[str, Any]) -> None:
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def start_trace(self, user_query: str, mode: str) -> str:
        trace_id = str(uuid.uuid4())
        ts_ms = int(time.time() * 1000)
        if self.enabled and self.client:
            try:
                self.client.trace(
                    id=trace_id,
                    name="chat",
                    input={"query": user_query, "mode": mode},
                    metadata={"mode": mode},
                    timestamp=ts_ms,
                )
            except Exception:
                pass
        self._write({"type": "trace.start", "trace_id": trace_id, "ts": ts_ms, "query": user_query, "mode": mode})
        return trace_id

    def log_tool(self, trace_id: str, name: str, args: Dict[str, Any], result: Any = None, error: Optional[str] = None):
        ts_ms = int(time.time() * 1000)
        payload = {"ok": error is None}
        if self.enabled and self.client:
            try:
                # Use generic event for tools (simple and robust)
                self.client.event(
                    trace_id=trace_id,
                    name=f"tool:{name}",
                    input=args,
                    output=result if error is None else {"error": error},
                    metadata=payload,
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

    def end_trace(self, trace_id: str, prompt: str, response_text: str, citations: List[Dict[str, Any]], ok: bool = True):
        ts_ms = int(time.time() * 1000)
        if self.enabled and self.client:
            try:
                # Log a single 'generation' for the final assistant message
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
        self._write({"type": "trace.end", "trace_id": trace_id, "ts": ts_ms, "ok": ok, "citations": citations})
