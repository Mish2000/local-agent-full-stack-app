# backend/memory.py
from __future__ import annotations
from typing import List, Dict, Literal

Role = Literal["user", "assistant"]
Message = Dict[str, str]  # {"role": "user"|"assistant", "content": "..."}

class MemoryStore:
    """
    Ephemeral per-process conversation memory.
    Keyed by 'cid' (conversation id) coming from the frontend.
    """
    def __init__(self) -> None:
        self._store: Dict[str, List[Message]] = {}

    def get(self, cid: str) -> List[Message]:
        return list(self._store.get(cid, []))

    def clear(self, cid: str) -> None:
        self._store.pop(cid, None)

    def append(self, cid: str, role: Role, content: str) -> None:
        if not cid:
            cid = "default"
        arr = self._store.setdefault(cid, [])
        arr.append({"role": role, "content": content or ""})

    def append_pair(self, cid: str, user_text: str, assistant_text: str) -> None:
        self.append(cid, "user", user_text or "")
        self.append(cid, "assistant", assistant_text or "")

    def render(self, cid: str, lang: str = "English", max_chars: int = 3500) -> str:
        """
        Render most recent turns within a character budget for prompt injection.
        Newest last. Clips from the back until within budget.
        """
        msgs = self._store.get(cid, [])
        if not msgs:
            return ""

        # Build full transcript first
        label_user = "משתמש" if lang.lower().startswith("hebrew") else "User"
        label_asst = "עוזר" if lang.lower().startswith("hebrew") else "Assistant"

        lines: List[str] = []
        for m in msgs:
            label = label_user if m["role"] == "user" else label_asst
            lines.append(f"{label}: {m['content'].strip()}")

        # Trim to budget from the end (most recent)
        joined = "\n---\n".join(lines)
        if len(joined) <= max_chars:
            return joined

        # Take as many recent lines as fit
        acc: List[str] = []
        total = 0
        for line in reversed(lines):
            piece = (line if not acc else f"---\n{line}")
            if total + len(piece) > max_chars:
                break
            acc.append(line if not acc else f"---\n{line}")
            total += len(piece)
        # Reverse back to chronological order
        clipped = "".join(reversed(acc))
        return clipped

memory = MemoryStore()
