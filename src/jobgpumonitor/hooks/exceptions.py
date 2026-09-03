"""Uncaught exception capture (main thread and other threads), chained to previous hooks."""

from __future__ import annotations

import re
import sys
import threading
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .._log import dbg
from ..context import SECRET_RE

if TYPE_CHECKING:  # pragma: no cover
    from ..runtime import Run

MAX_TRACEBACK_CHARS = 32_000
MAX_FRAMES = 12


def _frames(tb: Any, capture_locals: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        entries = traceback.extract_tb(tb)
    except Exception:
        return out
    frames_objs: List[Any] = []
    if capture_locals:
        t = tb
        while t is not None:
            frames_objs.append(t.tb_frame)
            t = t.tb_next
    selected = list(enumerate(entries))[-MAX_FRAMES:]
    for idx, fs in selected:
        fr: Dict[str, Any] = {"file": fs.filename, "line": fs.lineno, "func": fs.name, "code": (fs.line or "")[:300]}
        if capture_locals and idx < len(frames_objs):
            loc: Dict[str, str] = {}
            try:
                for k, v in list(frames_objs[idx].f_locals.items())[:40]:
                    if k.startswith("__"):
                        continue
                    if SECRET_RE.search(k):
                        loc[k] = "***"
                        continue
                    try:
                        r = repr(v)
                    except Exception:
                        r = "<repr failed>"
                    loc[k] = r[:200]
            except Exception:
                pass
            fr["locals"] = loc
        out.append(fr)
    return out


def format_exception(exc_type: Any, exc: Any, tb: Any, capture_locals: bool = False) -> Dict[str, Any]:
    try:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
    except Exception:
        text = f"{getattr(exc_type, '__name__', exc_type)}: {exc!r}"
    if len(text) > MAX_TRACEBACK_CHARS:
        text = text[:MAX_TRACEBACK_CHARS // 2] + "\n… [truncated] …\n" + text[-MAX_TRACEBACK_CHARS // 2 :]
    try:
        msg = str(exc)
    except Exception:
        msg = "<str failed>"
    msg = re.sub(r"\s+", " ", msg).strip()[:2000]
    return {
        "type": getattr(exc_type, "__name__", str(exc_type)),
        "module": getattr(exc_type, "__module__", None),
        "message": msg,
        "traceback": text,
        "frames": _frames(tb, capture_locals),
        "is_oom": _looks_like_oom(exc_type, msg),
    }


def _looks_like_oom(exc_type: Any, msg: str) -> bool:
    name = getattr(exc_type, "__name__", "")
    if name in ("OutOfMemoryError", "MemoryError", "ResourceExhaustedError"):
        return True
    m = msg.lower()
    return "out of memory" in m or "cuda error: out of memory" in m or "cublas_status_alloc_failed" in m


def install(run: Run) -> None:
    prev_hook = sys.excepthook
    prev_thread_hook = getattr(threading, "excepthook", None)

    def excepthook(exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if issubclass(exc_type, KeyboardInterrupt):
                run.note_uncaught("interrupted", exc_type, exc, tb)
            else:
                run.note_uncaught("error", exc_type, exc, tb)
        except Exception as e:
            dbg(f"excepthook failed: {e}")
        try:
            prev_hook(exc_type, exc, tb)
        except Exception:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = excepthook

    if prev_thread_hook is not None:

        def thread_hook(args: Any) -> None:
            try:
                if args.exc_type is not SystemExit:
                    run.note_thread_exception(args)
            except Exception as e:
                dbg(f"threading.excepthook failed: {e}")
            prev_thread_hook(args)

        threading.excepthook = thread_hook  # type: ignore[assignment]


def install_exit_tracking(run: Run) -> None:
    """Record the code passed to ``sys.exit`` so ``run.end`` can report it.

    ``SystemExit`` never reaches ``sys.excepthook`` and ``atexit`` cannot see the exit
    status, so wrapping ``sys.exit`` is the only in-process way to know it.
    """
    orig_exit = sys.exit

    def exit_(code: Optional[Any] = None) -> None:
        try:
            run.note_exit_code(code)
        except Exception:
            pass
        orig_exit(code)

    exit_.__doc__ = orig_exit.__doc__
    sys.exit = exit_  # type: ignore[assignment]
