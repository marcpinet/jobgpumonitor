"""Internal debug logging. Never uses the ``logging`` module (we hook it) and never raises."""

from __future__ import annotations

import sys

_enabled = False


def set_debug(flag: bool) -> None:
    global _enabled
    _enabled = bool(flag)


def dbg(msg: str) -> None:
    if not _enabled:
        return
    try:
        sys.__stderr__.write("[jgm] " + msg + "\n")
        sys.__stderr__.flush()
    except Exception:
        pass


def warn(msg: str) -> None:
    """User-facing one-liner on stderr (setup problems only, never per event)."""
    try:
        sys.__stderr__.write("[jgm] " + msg + "\n")
        sys.__stderr__.flush()
    except Exception:
        pass
