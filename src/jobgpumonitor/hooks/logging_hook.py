"""Forward ``logging`` records at or above a level as ``log.line`` events."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from ..runtime import Run


class _Handler(logging.Handler):
    def __init__(self, run: Run, level: int) -> None:
        super().__init__(level)
        self._run = run
        self._local = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._local, "busy", False):
            return
        self._local.busy = True
        try:
            try:
                msg = record.getMessage()
            except Exception:
                msg = str(record.msg)
            data = {
                "logger": record.name,
                "level": record.levelname,
                "levelno": record.levelno,
                "message": msg[:4000],
                "file": record.pathname,
                "line": record.lineno,
                "func": record.funcName,
            }
            if record.exc_info and record.exc_info[0] is not None:
                try:
                    data["exception"] = self.format(record)[-8000:]
                except Exception:
                    pass
            self._run.emit("log.line", data)
        except Exception:
            pass
        finally:
            self._local.busy = False


def install(run: Run, level_name: str) -> Optional[logging.Handler]:
    if not level_name:
        return None
    level = logging.getLevelName(level_name.upper())
    if not isinstance(level, int):
        level = logging.WARNING
    h = _Handler(run, level)
    h.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(h)
    return h
