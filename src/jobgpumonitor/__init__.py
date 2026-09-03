"""jobgpumonitor: watch a running job and stream structured events, with one line of code.

    import jobgpumonitor.auto          # zero configuration
    # or
    import jobgpumonitor; jobgpumonitor.watch()
    jobgpumonitor.log(loss=0.12, epoch=3)
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from .config import Config
from .runtime import Run, in_child_process

__version__ = "0.1.0"
__all__ = ["watch", "log", "emit", "finish", "current_run", "Run", "Config", "__version__"]

_run: Optional[Run] = None
_lock = threading.Lock()


def watch(**overrides: Any) -> Run:
    """Start monitoring this process (idempotent). Keyword arguments override ``JGM_*`` env vars.

    Returns the :class:`Run`. In a ``multiprocessing`` child, or when ``JGM_DISABLED=1``,
    the returned run is disabled and every call is a no-op.
    """
    global _run
    with _lock:
        if _run is not None:
            return _run
        cfg = Config.from_env(overrides)
        if in_child_process():
            cfg.enabled = False
        run = Run(cfg)
        _run = run
    try:
        run.start()
    except Exception as e:  # never break the host program
        from ._log import dbg

        dbg(f"start failed: {e}")
        run.enabled = False
    return run


def current_run() -> Optional[Run]:
    return _run


def log(metrics: Optional[Dict[str, Any]] = None, step: Optional[int] = None, epoch: Optional[int] = None, **kw: Any) -> bool:
    """Record named scalars (``metric.log``). Starts monitoring if needed."""
    return watch().log(metrics, step=step, epoch=epoch, **kw)


def emit(event_type: str, data: Optional[Dict[str, Any]] = None, **kw: Any) -> bool:
    """Emit a custom event; undotted names are namespaced as ``custom.<name>``."""
    return watch().emit(event_type, data, **kw)


def finish(status: Optional[str] = None, exit_code: Optional[int] = None, **extra: Any) -> None:
    """Emit ``run.end`` now (useful in notebooks); later calls are ignored."""
    r = _run
    if r is not None:
        r.finish(status=status, exit_code=exit_code, **extra)
