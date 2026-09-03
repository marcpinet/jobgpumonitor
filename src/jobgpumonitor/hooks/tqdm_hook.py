"""Turn tqdm refreshes into throttled ``progress.update`` events.

We patch ``tqdm.std.tqdm.display`` (called on every visual refresh, already rate-limited
by ``mininterval``) and ``close``. Subclasses (``tqdm.auto``, ``tqdm.notebook``) inherit
the patch. Bars with ``disable=True`` are ignored.
"""

from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from .._log import dbg
from .importhook import on_import

if TYPE_CHECKING:  # pragma: no cover
    from ..runtime import Run

_bar_ids = itertools.count(1)


def _bar_payload(bar: Any, done: bool) -> Optional[Dict[str, Any]]:
    try:
        n = bar.n
        total = bar.total
        fd = bar.format_dict
        rate = fd.get("rate")
        elapsed = fd.get("elapsed")
        if rate is None and elapsed:
            rate = n / elapsed if elapsed > 0 else None
        eta = None
        if total is not None and rate:
            eta = max(0.0, (total - n) / rate)
        return {
            "bar_id": bar._jgm_id,
            "desc": (bar.desc or "")[:200],
            "n": n,
            "total": total,
            "unit": fd.get("unit"),
            "rate": round(rate, 4) if rate else None,
            "elapsed_s": round(elapsed, 3) if elapsed is not None else None,
            "eta_s": round(eta, 1) if eta is not None else None,
            "done": done,
        }
    except Exception as e:
        dbg(f"tqdm payload failed: {e}")
        return None


def install(run: Run) -> None:
    def patch(module: Any) -> None:
        try:
            cls = module.tqdm
        except AttributeError:
            return
        if getattr(cls, "_jgm_patched", False):
            return
        orig_display = cls.display
        orig_close = cls.close
        min_interval = run.config.progress_s

        def display(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = orig_display(self, *args, **kwargs)
            try:
                if not getattr(self, "disable", False):
                    now = time.monotonic()
                    last = getattr(self, "_jgm_last", 0.0)
                    if now - last >= min_interval:
                        if not hasattr(self, "_jgm_id"):
                            self._jgm_id = next(_bar_ids)
                        self._jgm_last = now
                        payload = _bar_payload(self, done=False)
                        if payload:
                            run.progress(payload)
            except Exception as e:
                dbg(f"tqdm display hook failed: {e}")
            return result

        def close(self: Any, *args: Any, **kwargs: Any) -> Any:
            emit = False
            try:
                emit = not getattr(self, "disable", False) and hasattr(self, "_jgm_id") and not getattr(self, "_jgm_closed", False)
            except Exception:
                pass
            result = orig_close(self, *args, **kwargs)
            if emit:
                try:
                    self._jgm_closed = True
                    payload = _bar_payload(self, done=True)
                    if payload:
                        run.progress(payload)
                except Exception as e:
                    dbg(f"tqdm close hook failed: {e}")
            return result

        cls.display = display
        cls.close = close
        cls._jgm_patched = True
        dbg("tqdm patched")

    on_import("tqdm.std", patch)
