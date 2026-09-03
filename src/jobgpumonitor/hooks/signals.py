"""Signal observation, chained to whatever handler was installed before us.

Slurm sends SIGTERM (then SIGKILL after KillWait) at time limit / scancel, and
``--signal=USR1@60``-style warnings. OAR uses SIGUSR2 for checkpoint requests. We emit
``signal.received`` and then reproduce the previous behaviour exactly:

* previous handler callable  -> call it;
* ``SIG_DFL`` for a terminating signal -> emit ``run.end`` (status ``killed``), flush,
  restore the default disposition and re-raise so the exit status is the real one;
* ``SIG_IGN`` -> nothing.
"""

from __future__ import annotations

import os
import signal
import threading
from typing import TYPE_CHECKING, Any, Dict

from .._log import dbg

if TYPE_CHECKING:  # pragma: no cover
    from ..runtime import Run

_WATCH = ("SIGTERM", "SIGINT", "SIGHUP", "SIGUSR1", "SIGUSR2", "SIGXCPU", "SIGQUIT")
_TERMINATES_BY_DEFAULT = {"SIGTERM", "SIGHUP", "SIGUSR1", "SIGUSR2", "SIGXCPU", "SIGQUIT", "SIGINT"}


def install(run: Run) -> Dict[str, Any]:
    """Install chained handlers. Returns the map of previous handlers (for tests)."""
    if threading.current_thread() is not threading.main_thread():
        dbg("not in main thread; signal hooks skipped")
        return {}
    previous: Dict[int, Any] = {}
    for name in _WATCH:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            prev = signal.getsignal(sig)
        except (ValueError, OSError):
            continue
        if prev is None:  # handler set from C, not overridable safely
            continue
        previous[int(sig)] = prev

        def handler(signum: int, frame: Any, _name: str = name) -> None:
            _handle(run, previous, signum, frame, _name)

        try:
            signal.signal(sig, handler)
        except (ValueError, OSError) as e:
            dbg(f"cannot hook {name}: {e}")
            previous.pop(int(sig), None)
    return {signal.Signals(k).name: v for k, v in previous.items()}


def _handle(run: Run, previous: Dict[int, Any], signum: int, frame: Any, name: str) -> None:
    prev = previous.get(signum, signal.SIG_DFL)
    fatal = prev == signal.SIG_DFL and name in _TERMINATES_BY_DEFAULT
    try:
        run.emit("signal.received", {
            "signal": name,
            "signum": signum,
            "previous_handler": _describe(prev),
            "will_terminate": bool(fatal),
            "deadline_remaining_s": run.deadline_remaining_s(),
        })
    except Exception as e:
        dbg(f"signal event failed: {e}")
    if callable(prev):
        prev(signum, frame)  # user's handler (or Python's default SIGINT -> KeyboardInterrupt)
        return
    if fatal:
        try:
            run.finish(status="killed", signal=name, exit_code=128 + signum)
        except Exception:
            pass
        try:
            signal.signal(signum, signal.SIG_DFL)
        except Exception:
            pass
        os.kill(os.getpid(), signum)
    # SIG_IGN: swallow, as before.


def _describe(h: Any) -> str:
    if h == signal.SIG_DFL:
        return "default"
    if h == signal.SIG_IGN:
        return "ignore"
    if h is signal.default_int_handler:
        return "python_default_int"
    return getattr(h, "__qualname__", repr(h))[:100]
