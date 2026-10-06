"""Tail the stdout/stderr files of the jobs the probe follows and emit ``log.chunk`` events.

Runs on the login node, which sees the shared filesystem. The scheduler probe learns the
real ``.out`` / ``.err`` paths from ``scontrol``; this tailer reads what is new in them at
every poll and emits it as ``log.chunk`` events, which ``jgm forward`` ships like the rest.
A consumer concatenates the chunks of a stream to rebuild the file as it grows.

Carriage-return redraws (tqdm) are folded to the last state of each line before they are
sent, so a progress bar does not eat the budget. Limits: at most ``chunk_bytes`` per file
per poll and ``max_bytes`` sent per file while the job runs. Past that budget nothing is
sent until the job ends; then the last ``tail_bytes`` are sent (``skipped`` tells how many
bytes were jumped over), because how a job ended matters more than how it started. Tailing
continues ``grace_s`` after the job ended so the last lines flushed by Slurm are not missed.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

from .._log import dbg

Emit = Callable[[str, str, Dict[str, Any]], Any]  # (run_id, etype, data)


def fold_redraws(data: bytes) -> bytes:
    """Keep the last state of each line redrawn with carriage returns (tqdm and friends):
    ``b"a 10%\\ra 20%\\n"`` -> ``b"a 20%\\n"``, which is what a terminal ends up showing.

    A last line still being drawn keeps its trailing ``\\r``: the next chunk, appended
    after it, then overwrites it, on screen as in a consumer that folds again."""
    if b"\r" not in data:
        return data
    lines = data.split(b"\n")
    last = len(lines) - 1
    for i, line in enumerate(lines):
        if b"\r" in line:
            parts = [p for p in line.split(b"\r") if p]
            folded = parts[-1] if parts else b""
            if i == last and line.endswith(b"\r"):
                folded += b"\r"
            lines[i] = folded
    return b"\n".join(lines)


class LogTailer:
    def __init__(
        self,
        state: Dict[str, Any],
        chunk_bytes: int = 64 * 1024,
        max_bytes: int = 8 * 1024 * 1024,
        tail_bytes: int = 64 * 1024,
        grace_s: float = 180.0,
        now: Callable[[], float] = time.time,
    ) -> None:
        #: persisted with the probe state: {run_id: {stream: {path, offset, inode, sent}}}
        self.state = state
        self.chunk_bytes = chunk_bytes
        self.max_bytes = max_bytes
        self.tail_bytes = tail_bytes
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
        rec = self.state.setdefault(run_id, {}).setdefault(stream, {"path": path, "offset": 0, "inode": None, "sent": 0})
        if rec.get("path") != path:  # requeue with a new file, or path learned late
            rec.update(path=path, offset=0, inode=None)
        try:
            st = os.stat(path)
        except FileNotFoundError:
            return 0
        if rec.get("inode") not in (None, st.st_ino) or st.st_size < rec["offset"]:
            rec["offset"] = 0  # file replaced or truncated: start over
        rec["inode"] = st.st_ino
        offset, size = rec["offset"], st.st_size
        if size == offset:
            return 0
        budget = self.max_bytes - int(rec.get("sent", 0))
        over = budget <= 0
        if final and size - offset > self.tail_bytes:
            # the job is over and we are far behind (budget spent, or a job writing faster
            # than we read): what matters now is how it ended, jump to the end of the file
            start, want = size - self.tail_bytes, self.tail_bytes
        elif over and not final:
            return 0  # budget spent: wait for the end of the job, then send the tail
        else:
            start = offset
            want = min(self.chunk_bytes, size - offset) if over else min(self.chunk_bytes, size - offset, budget)
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(want)
        if start > offset:  # resume on a whole line
            nl = data.find(b"\n")
            if 0 <= nl < len(data) - 1:
                data = data[nl + 1:]
                start += nl + 1
        if not data:
            return 0
        # keep lines whole: cut on the last line end (\n, or \r for tqdm-style progress).
        # At the current end of the file a line still being written waits for its end,
        # unless the job is over or this read was cut short by the budget.
        remaining = size - start - len(data)
        nl = max(data.rfind(b"\n"), data.rfind(b"\r"))
        if remaining > 0:
            if nl > 0:
                data = data[: nl + 1]
        elif not final and want < budget and nl < len(data) - 1 and len(data) - (nl + 1) < 4096:
            data = data[: nl + 1]
        if not data:
            return 0
        text = fold_redraws(data)
        chunk: Dict[str, Any] = {
            "stream": stream, "path": path, "offset": start, "text": text.decode("utf-8", "replace"),
            "size": size, "truncated": over or start > offset, "eof": final and start + len(data) >= size,
        }
        if start > offset:
            chunk["skipped"] = start - offset
        emit(run_id, "log.chunk", chunk)
        rec["offset"] = start + len(data)
        rec["sent"] = int(rec.get("sent", 0)) + len(text)
        return 1
