"""Tail the stdout/stderr files of the jobs the probe follows and emit ``log.chunk`` events.

Runs on the login node, which sees the shared filesystem. The scheduler probe learns the
real ``.out`` / ``.err`` paths from ``scontrol``; this tailer reads what is new in them at
every poll and emits it as ``log.chunk`` events, which ``jgm forward`` ships like the rest.
A consumer concatenates the chunks of a stream to rebuild the file as it grows.

Limits: at most ``chunk_bytes`` per file per poll, ``max_bytes`` per file in total (then a
final chunk flagged ``truncated`` and the file is dropped), tailing continues ``grace_s``
after the job ended so the last lines flushed by Slurm are not missed.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

from .._log import dbg

Emit = Callable[[str, str, Dict[str, Any]], Any]  # (run_id, etype, data)


class LogTailer:
    def __init__(
        self,
        state: Dict[str, Any],
        chunk_bytes: int = 64 * 1024,
        max_bytes: int = 8 * 1024 * 1024,
        grace_s: float = 180.0,
        now: Callable[[], float] = time.time,
    ) -> None:
        #: persisted with the probe state: {run_id: {stream: {path, offset, inode, sent, done}}}
        self.state = state
        self.chunk_bytes = chunk_bytes
        self.max_bytes = max_bytes
        self.grace_s = grace_s
        self.now = now
        self.active: Dict[str, Dict[str, Any]] = {}   # run_id -> {"stdout": path, "stderr": path, "ended_ts": float|None}

    # ------------------------------------------------------------------ bookkeeping

    def track(self, run_id: str, stdout: Optional[str], stderr: Optional[str], ended_ts: Optional[float]) -> None:
        streams: Dict[str, Any] = {"ended_ts": ended_ts}
        if stdout:
            streams["stdout"] = stdout
        if stderr and stderr != stdout:
            streams["stderr"] = stderr
        if len(streams) > 1:
            self.active[run_id] = streams

    def forget(self, run_id: str) -> None:
        self.active.pop(run_id, None)
        self.state.pop(run_id, None)

    # ------------------------------------------------------------------ polling

    def poll(self, emit: Emit) -> int:
        """Read what is new in every tracked file; returns the number of chunks emitted."""
        emitted = 0
        now = self.now()
        for run_id in list(self.active):
            info = self.active[run_id]
            ended = info.get("ended_ts")
            for stream in ("stdout", "stderr"):
                path = info.get(stream)
                if not path:
                    continue
                try:
                    emitted += self._read(run_id, stream, path, emit, final=bool(ended))
                except Exception as e:  # never let one file break the loop
                    dbg(f"tail {path}: {e}")
            if ended and now - ended > self.grace_s:
                self.active.pop(run_id, None)
        return emitted

    def _read(self, run_id: str, stream: str, path: str, emit: Emit, final: bool) -> int:
        rec = self.state.setdefault(run_id, {}).setdefault(stream, {"path": path, "offset": 0, "inode": None, "sent": 0, "done": False})
        if rec.get("done"):
            return 0
        if rec.get("path") != path:  # requeue with a new file, or path learned late
            rec.update(path=path, offset=0, inode=None)
        try:
            st = os.stat(path)
        except FileNotFoundError:
            return 0
        if rec.get("inode") not in (None, st.st_ino) or st.st_size < rec["offset"]:
            rec["offset"] = 0  # file replaced or truncated: start over
        rec["inode"] = st.st_ino
        if st.st_size == rec["offset"]:
            return 0
        budget = self.max_bytes - int(rec.get("sent", 0))
        want = min(self.chunk_bytes, st.st_size - rec["offset"], max(budget, 0))
        if want <= 0:
            emit(run_id, "log.chunk", {"stream": stream, "path": path, "offset": rec["offset"], "text": "",
                                       "size": st.st_size, "truncated": True, "eof": final})
            rec["done"] = True
            return 1
        with open(path, "rb") as f:
            f.seek(rec["offset"])
            data = f.read(want)
        if not data:
            return 0
        # keep lines whole: cut on the last line end (\n, or \r for tqdm-style progress).
        # At the current end of the file a line still being written waits for its end,
        # unless the job is over or this is the last of the per-file budget.
        remaining = st.st_size - rec["offset"] - len(data)
        nl = max(data.rfind(b"\n"), data.rfind(b"\r"))
        if remaining > 0:
            if nl > 0:
                data = data[: nl + 1]
        elif not final and budget > len(data) and nl < len(data) - 1 and len(data) - (nl + 1) < 4096:
            data = data[: nl + 1]
        if not data:
            return 0
        text = data.decode("utf-8", "replace")
        emit(run_id, "log.chunk", {"stream": stream, "path": path, "offset": rec["offset"], "text": text,
                                   "size": st.st_size, "truncated": False, "eof": final and rec["offset"] + len(data) >= st.st_size})
        rec["offset"] += len(data)
        rec["sent"] = int(rec.get("sent", 0)) + len(data)
        return 1
