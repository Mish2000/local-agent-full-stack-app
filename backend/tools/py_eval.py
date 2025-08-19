# backend/tools/py_eval.py
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import subprocess
from typing import Dict

# Whitelisted modules (pandas/numpy are optional; if not installed, import will fail harmlessly)
ALLOWED_MODULES = os.getenv("PY_EVAL_ALLOWED", "math,statistics,random,numpy,pandas").split(",")

# Very quick static guard to reject obvious dangerous usage early
_BLOCKED_SNIPPETS = [
    "import os", "from os", "import sys", "from sys",
    "import subprocess", "from subprocess",
    "socket.", "import socket", "from socket",
    "shutil", "pathlib", "ctypes",
    "open(", "__import__", "eval(", "exec(", "compile(", "globals(", "locals(", "vars(",
]


def _find_blocked(s: str) -> str | None:
    low = (s or "").lower()
    for pat in _BLOCKED_SNIPPETS:
        if pat in low:
            return pat
    return None


# Minimal runner that (1) restricts builtins, (2) overrides __import__ to allow only whitelisted modules
_RUNNER_SOURCE = r"""
import sys, builtins

blocked = {
    'open','exec','eval','compile','input','help','dir','vars','locals','globals',
    '__loader__','__spec__','__build_class__'
}

allowed = set((sys.argv[1] or '').split(',')) if len(sys.argv) > 1 else set()

_real_import = __import__
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split('.')[0]
    if root in allowed:
        return _real_import(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of '{name}' is blocked.")

# Build a safe builtins dict
safe_builtins = {}
for k in dir(builtins):
    if k in blocked:
        continue
    safe_builtins[k] = getattr(builtins, k)
safe_builtins['__import__'] = safe_import

ns = {'__builtins__': safe_builtins}

code = sys.stdin.read()
exec(compile(code, '<py_eval>', 'exec'), ns, ns)
"""


async def run_python(code: str, timeout_sec: float = 5.0) -> Dict:
    """
    Execute user Python code in a very restricted subprocess.

    Returns dict: {ok, stdout, stderr, timeout, duration_ms}
    """
    bad = _find_blocked(code or "")
    if bad:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Blocked pattern detected: {bad}",
            "timeout": False,
            "duration_ms": 0,
        }

    with tempfile.TemporaryDirectory() as td:
        runner_path = os.path.join(td, "runner.py")
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(_RUNNER_SOURCE)

        args = [
            sys.executable, "-I", "-S", "-B",
            runner_path, ",".join(ALLOWED_MODULES)
        ]

        start = time.perf_counter()

        def _run():
            return subprocess.run(
                args,
                input=(code or "").encode("utf-8", "ignore"),
                capture_output=True,
                cwd=td,
                timeout=timeout_sec,
            )

        try:
            proc = await asyncio.to_thread(_run)
            duration_ms = int((time.perf_counter() - start) * 1000)
            out = (proc.stdout or b"").decode("utf-8", "ignore")
            err = (proc.stderr or b"").decode("utf-8", "ignore")
            if len(out) > 8000:
                out = out[:8000] + "... [truncated]"
            if len(err) > 4000:
                err = err[:4000] + "... [truncated]"
            ok = (proc.returncode == 0) and not err.strip()
            return {"ok": ok, "stdout": out, "stderr": err, "timeout": False, "duration_ms": duration_ms}

        except subprocess.TimeoutExpired:
            return {
                "ok": False, "stdout": "", "stderr": "Execution timed out.",
                "timeout": True, "duration_ms": int((time.perf_counter() - start) * 1000),
            }
