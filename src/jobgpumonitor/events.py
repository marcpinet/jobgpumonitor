"""Event envelope helpers: ULIDs, timestamps, JSON serialisation, value sanitising."""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import time
from typing import Any, Dict, Optional

PROTOCOL_VERSION = 1

#: Event types emitted by this library (the protocol is open: consumers ignore unknown types).
KNOWN_TYPES = (
    "run.start",
    "run.heartbeat",
    "run.exception",
    "run.end",
    "resource.sample",
    "progress.update",
    "metric.log",
    "log.line",
    "signal.received",
    "checkpoint.saved",
    "stack.dump",
    "scheduler.state",
)
_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid(ts_ms: Optional[int] = None) -> str:
    """26-char ULID: 48-bit ms timestamp + 80 random bits, Crockford base32, sortable."""
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    value = (ts_ms << 80) | int.from_bytes(os.urandom(10), "big")
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(out))


def utc_iso(t: Optional[float] = None) -> str:
    t = time.time() if t is None else t
    d = _dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc)
    return d.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def normalize_type(name: str) -> str:
    """Custom event types are namespaced under ``custom.`` unless already dotted."""
    name = str(name).strip().lower()
    if "." not in name:
        name = "custom." + re.sub(r"[^a-z0-9_]", "_", name)
    if not _TYPE_RE.match(name):
        name = "custom." + re.sub(r"[^a-z0-9_]", "_", name.replace(".", "_"))
    return name


def sanitize(value: Any, depth: int = 0, max_str: int = 4096) -> Any:
    """Make a value JSON-safe without importing numpy/torch: scalars via ``.item()``,
    NaN/inf -> None, unknown objects -> ``repr``."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, str):
        return value if len(value) <= max_str else value[:max_str] + "…"
    if isinstance(value, bytes):
        return repr(value[:256])
    if depth > 6:
        return repr(value)[:max_str]
    # numpy / torch / jax 0-d arrays and scalars expose .item() (torch CUDA tensors sync here).
    item = getattr(value, "item", None)
    shape = getattr(value, "shape", None)
    if callable(item) and (shape == () or shape is None or (hasattr(shape, "__len__") and len(shape) == 0)):
        try:
            return sanitize(item(), depth + 1, max_str)
        except Exception:
            pass
    if hasattr(value, "tolist") and shape is not None:
        try:
            size = 1
            for s in shape:
                size *= int(s)
            if size <= 64:
                return sanitize(value.tolist(), depth + 1, max_str)
            return {"__array__": True, "shape": list(shape), "dtype": str(getattr(value, "dtype", "?"))}
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): sanitize(v, depth + 1, max_str) for k, v in list(value.items())[:256]}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize(v, depth + 1, max_str) for v in list(value)[:256]]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    try:
        return repr(value)[:max_str]
    except Exception:
        return "<unrepresentable>"


def to_json(obj: Any) -> str:
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return json.dumps(sanitize(obj), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def make_envelope(
    *,
    run_id: str,
    emitter: str,
    pid: int,
    source: str,
    rank: Optional[int],
    seq: int,
    etype: str,
    data: Dict[str, Any],
    mono: float,
    ts: Optional[float] = None,
) -> Dict[str, Any]:
    ts = time.time() if ts is None else ts
    return {
        "v": PROTOCOL_VERSION,
        "id": ulid(int(ts * 1000)),
        "seq": seq,
        "ts": utc_iso(ts),
        "mono": round(mono, 3),
        "run_id": run_id,
        "emitter": emitter,
        "pid": pid,
        "source": source,
        "rank": rank,
        "type": etype,
        "data": data,
    }
