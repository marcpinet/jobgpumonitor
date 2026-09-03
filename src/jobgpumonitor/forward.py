"""``jgm forward``: ship the JSONL event files to a jobgpumonitor-server over HTTPS.

Runs on a login node (the only place with both the shared filesystem and an outbound
proxy). The JSONL files *are* the durable queue: the forwarder only remembers, per file,
how many bytes were already acknowledged by the server, so an unreachable server simply
delays delivery and a restart never loses or duplicates anything the server already has
(the server de-duplicates on emitter/pid/seq anyway).

stdlib only; ``urllib`` honours ``https_proxy`` / ``no_proxy``.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._log import dbg, warn

Poster = Callable[[bytes, Dict[str, str]], Tuple[int, str]]


def http_post(url: str, body: bytes, headers: Dict[str, str], timeout: float = 30.0) -> Tuple[int, str]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read(2048).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read(2048).decode("utf-8", "replace")


class Forwarder:
    def __init__(
        self,
        base_dir: str,
        url: str,
        token: str,
        batch_events: int = 500,
        batch_bytes: int = 4 * 1024 * 1024,
        max_line: int = 4 * 1024 * 1024,
        post: Poster = None,  # type: ignore[assignment]
        host: Optional[str] = None,
    ) -> None:
        self.base_dir = base_dir
        self.url = url
        self.token = token
        self.batch_events = batch_events
        self.batch_bytes = batch_bytes
        self.max_line = max_line
        self.post = post or (lambda body, headers: http_post(self.url, body, headers))
        self.host = host or socket.gethostname().split(".")[0]
        self.state_path = os.path.join(base_dir, "forward", "state.json")
        self.offsets: Dict[str, Dict[str, Any]] = {}
        self.sent = 0
        self.failures = 0
        self.last_ok_ts: Optional[float] = None
        self._load()

    # ------------------------------------------------------------------ state

    def _load(self) -> None:
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("offsets"), dict):
                self.offsets = data["offsets"]
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"offsets": self.offsets, "sent": self.sent, "last_ok_ts": self.last_ok_ts, "host": self.host}, f)
            os.replace(tmp, self.state_path)
        except OSError as e:
            dbg(f"state save failed: {e}")

    # ------------------------------------------------------------------ reading

    def files(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.base_dir, "runs", "*", "*", "*", "*.jsonl")))

    def _read_new(self, path: str) -> Tuple[List[Dict[str, Any]], int, int]:
        """New complete lines of one file -> (events, consumed_bytes, start_offset)."""
        try:
            st = os.stat(path)
        except OSError:
            return [], 0, 0
        rec = self.offsets.get(path) or {}
        offset = int(rec.get("offset", 0))
        if rec.get("inode") not in (None, st.st_ino) or st.st_size < offset:
            offset = 0
        if st.st_size == offset:
            return [], 0, offset
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                chunk = f.read(min(st.st_size - offset, self.batch_bytes))
        except OSError:
            return [], 0, offset
        lines = chunk.split(b"\n")
        tail = lines.pop()
        consumed = len(chunk) - len(tail)
        if tail and len(tail) >= self.max_line:
            consumed = len(chunk)
        events: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line.decode("utf-8", "replace"))
            except ValueError:
                continue
            if isinstance(ev, dict) and ev.get("run_id") and ev.get("emitter") and ev.get("type"):
                events.append(ev)
        self.offsets[path] = {"offset": offset, "inode": st.st_ino, "size": st.st_size}
        return events, consumed, offset

    # ------------------------------------------------------------------ sending

    def _send(self, events: List[Dict[str, Any]]) -> bool:
        body = gzip.compress(json.dumps(events, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "Authorization": "Bearer " + self.token,
            "User-Agent": "jgm-forward",
            "X-JGM-Host": self.host,
        }
        try:
            status, text = self.post(body, headers)
        except Exception as e:  # network / proxy / TLS
            self.failures += 1
            warn(f"forward: {type(e).__name__}: {e}")
            return False
        if status != 200:
            self.failures += 1
            warn(f"forward: server answered {status}: {text[:200]}")
            return False
        self.failures = 0
        self.last_ok_ts = time.time()
        return True

    def cycle(self) -> Dict[str, int]:
        """Ship what is new. Returns counters. Stops at the first failed batch (offsets untouched)."""
        shipped = 0
        batches = 0
        for path in self.files():
            while True:
                events, consumed, offset = self._read_new(path)
                if not events and consumed == 0:
                    break
                if events:
                    # send in slices so one huge file never makes a single giant request
                    for i in range(0, len(events), self.batch_events):
                        part = events[i : i + self.batch_events]
                        if not self._send(part):
                            self._save()
                            return {"shipped": shipped, "batches": batches, "ok": 0}
                        shipped += len(part)
                        batches += 1
                self.offsets[path]["offset"] = offset + consumed
                self.sent += len(events)
                if consumed < self.batch_bytes:
                    break  # file fully read for now
        self._save()
        return {"shipped": shipped, "batches": batches, "ok": 1}

    def run_forever(self, interval_s: float, stop=None) -> None:  # type: ignore[no-untyped-def]
        import threading

        stop = stop or threading.Event()
        backoff = interval_s
        while not stop.is_set():
            t = time.monotonic()
            try:
                r = self.cycle()
            except Exception as e:
                warn(f"forward cycle failed: {e}")
                r = {"ok": 0, "shipped": 0}
            if r.get("ok"):
                backoff = interval_s
                if r.get("shipped"):
                    dbg(f"shipped {r['shipped']} events")
            else:
                backoff = min(backoff * 2, 600.0)
                warn(f"forward: retrying in {backoff:.0f}s")
            stop.wait(max(0.5, backoff - (time.monotonic() - t)))
