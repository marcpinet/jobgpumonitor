"""Where to ship events: the server URL and write token.

Resolved, in order, from ``JGM_FORWARD_URL`` / ``JGM_FORWARD_TOKEN`` in the environment,
then from ``<event dir>/remote.json`` written once by ``jgm setup``. The event directory
is shared (``$JGM_DIR``, mounted into containers), so one ``jgm setup`` on the login node
configures every job, every node, every process: nothing else to install or keep alive.
"""

from __future__ import annotations

import json
import os
import stat
from typing import Any, Dict, Optional, Tuple

from ._log import dbg

FILENAME = "remote.json"


def path_for(base_dir: str) -> str:
    return os.path.join(base_dir, FILENAME)


def load(base_dir: Optional[str], env: Optional[Dict[str, str]] = None) -> Optional[Tuple[str, str]]:
    """``(url, token)`` or ``None`` when no server is configured."""
    env = os.environ if env is None else env
    url = (env.get("JGM_FORWARD_URL") or "").strip()
    token = (env.get("JGM_FORWARD_TOKEN") or "").strip()
    if url and token:
        return url.rstrip("/"), token
    if not base_dir:
        return None
    try:
        with open(path_for(base_dir), encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        url = str(data.get("url") or "").strip()
        token = str(data.get("token") or "").strip()
        if url and token:
            return url.rstrip("/"), token
    except (OSError, ValueError) as e:
        dbg(f"remote config unreadable: {e}")
    return None


def save(base_dir: str, url: str, token: str, extra: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(base_dir, exist_ok=True)
    p = path_for(base_dir)
    data: Dict[str, Any] = {"url": url.rstrip("/"), "token": token}
    if extra:
        data.update(extra)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(tmp, p)
    return p


def check(url: str, token: str, timeout: float = 20.0) -> Tuple[bool, str]:
    """Send an empty batch to the ingest endpoint: proves URL, proxy, TLS and token at once."""
    from .forward import http_post

    try:
        status, text = http_post(url, b"[]", {"Content-Type": "application/json", "Authorization": "Bearer " + token}, timeout=timeout)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    if status == 200:
        return True, "ok"
    if status == 401:
        return False, "server reachable but the token is refused (401)"
    if status == 503:
        return False, "server reachable but ingest is disabled there (no ingest_token in its config)"
    return False, f"HTTP {status}: {text[:120]}"
