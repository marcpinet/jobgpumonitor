"""Sinks: where serialised events go. All sinks fail silently and count what they drop."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, List, Optional, Tuple

from ._log import dbg, warn


class Sink:
    name = "base"

    def __init__(self) -> None:
        self.dropped = 0

    def write(self, line: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return self.name


class StderrSink(Sink):
    name = "stderr"

    def write(self, line: str) -> None:
        try:
            sys.__stderr__.write(line + "\n")
            sys.__stderr__.flush()
        except Exception:
            self.dropped += 1


class ListSink(Sink):
    """In-memory sink, for tests and for the ``jgm doctor`` dry run."""

    name = "list"

    def __init__(self) -> None:
        super().__init__()
        self.lines: List[str] = []

    def write(self, line: str) -> None:
        self.lines.append(line)


class FileSink(Sink):
    """Append-only JSONL file, one event per line, flushed (and fsynced) on every write.

    Every write is durable before we return, so a job killed by SIGKILL loses at most the
    event being written. On NFS, ``fsync`` is also what makes the data visible to other
    nodes (close-to-open consistency).
    """

    name = "file"

    def __init__(self, path: str, fsync: bool = True, reopen_after_s: float = 30.0) -> None:
        super().__init__()
        self.path = path
        self.fsync = fsync
        self._fh: Optional[Any] = None
        self._failed_at: Optional[float] = None
        self._reopen_after = reopen_after_s
        self._warned = False

    def _open(self) -> bool:
        if self._fh is not None:
            return True
        if self._failed_at is not None and time.monotonic() - self._failed_at < self._reopen_after:
            return False
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
            self._failed_at = None
            return True
        except OSError as e:
            self._failed_at = time.monotonic()
            if not self._warned:
                warn(f"cannot open event file {self.path}: {e} (events dropped until it works)")
                self._warned = True
            return False

    def write(self, line: str) -> None:
        if not self._open():
            self.dropped += 1
            return
        try:
            assert self._fh is not None
            self._fh.write(line + "\n")
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())
        except OSError as e:
            self.dropped += 1
            dbg(f"write failed on {self.path}: {e}")
            try:
                self._fh.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._fh = None
            self._failed_at = time.monotonic()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                if self.fsync:
                    os.fsync(self._fh.fileno())
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def describe(self) -> str:
        return f"file:{self.path}"


def _writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, f".jgm-write-test-{os.getpid()}")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


def resolve_base_dir(configured: Optional[str]) -> Tuple[Optional[str], str]:
    """Pick the event directory: ``JGM_DIR`` > ``~/.jobgpumonitor`` > ``./.jgm``.

    Inside a container ``$HOME`` is often not mounted, hence the cwd fallback.
    Returns ``(path or None, human-readable note)``.
    """
    candidates: List[Tuple[str, str]] = []
    if configured:
        candidates.append((os.path.expanduser(configured), "JGM_DIR"))
    home = os.path.expanduser("~")
    if home and home != "~":
        candidates.append((os.path.join(home, ".jobgpumonitor"), "home"))
    try:
        candidates.append((os.path.join(os.getcwd(), ".jgm"), "cwd"))
    except OSError:
        pass
    for path, origin in candidates:
        if _writable_dir(path):
            return path, origin
    return None, "no writable directory (tried: " + ", ".join(p for p, _ in candidates) + ")"


def build_sinks(names: Tuple[str, ...], base_dir: Optional[str], run_id: str, emitter: str, fsync: bool) -> List[Sink]:
    from .context import run_dir_parts

    sinks: List[Sink] = []
    for n in names:
        if n == "file":
            if base_dir is None:
                warn("file sink requested but no writable directory found; set JGM_DIR")
                continue
            run_dir = os.path.join(base_dir, "runs", *run_dir_parts(run_id))
            sinks.append(FileSink(os.path.join(run_dir, emitter + ".jsonl"), fsync=fsync))
        elif n == "stderr":
            sinks.append(StderrSink())
        elif n == "list":
            sinks.append(ListSink())
        elif n == "http":
            warn("http sink is not available yet (phase 2); ignoring")
        else:
            warn(f"unknown sink {n!r}; ignoring")
    return sinks


def run_dir_for(base_dir: str, run_id: str) -> str:
    from .context import run_dir_parts

    return os.path.join(base_dir, "runs", *run_dir_parts(run_id))
