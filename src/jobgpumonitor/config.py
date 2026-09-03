"""Configuration: explicit overrides > ``JGM_*`` environment variables > defaults.

The library must work with zero configuration, so every knob has a sane default and
nothing here can raise on a malformed value (bad values fall back to the default).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Mapping, Optional, Tuple

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}


def env_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return default


def env_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_tuple(value: Optional[str], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return default
    return tuple(s.strip() for s in value.split(",") if s.strip())


@dataclass
class Config:
    """All knobs of the emitter. Field names double as ``JGM_<UPPER_NAME>`` env vars."""

    enabled: bool = True
    #: Base directory for the file sink. ``None`` = ``~/.jobgpumonitor`` then ``./.jgm``.
    dir: Optional[str] = None
    #: Sinks, in order. ``file`` (JSONL on disk), ``stderr`` (one line per event).
    sinks: Tuple[str, ...] = ("file",)
    #: Optional cluster label used in ``run_id``; defaults to the scheduler's cluster name.
    cluster: Optional[str] = None
    heartbeat_s: float = 15.0
    sample_s: float = 10.0
    sample_slow_s: float = 60.0
    sample_slow_after_s: float = 3600.0
    #: Minimum interval between two ``progress.update`` events for the same bar.
    progress_s: float = 1.0
    #: ``rank0``: only global rank 0 emits everything, other ranks emit start/heartbeat/exception/end.
    #: ``all``: every rank emits everything. ``off``: ranks other than 0 emit nothing.
    rank_mode: str = "rank0"
    capture_locals: bool = False
    tqdm: bool = True
    #: Minimum ``logging`` level forwarded as ``log.line``; empty string disables.
    logging_level: str = "WARNING"
    signals: bool = True
    faulthandler: bool = True
    fsync: bool = True
    #: Extra environment variable prefixes to include in ``run.start`` (values still masked).
    env_include: Tuple[str, ...] = ()
    #: Extra regex fragments that mark an env var name as secret.
    env_secret: Tuple[str, ...] = ()
    packages: Tuple[str, ...] = (
        "torch", "numpy", "transformers", "lightning", "pytorch-lightning", "jax",
        "tensorflow", "scikit-learn", "accelerate", "datasets", "wandb", "tqdm",
    )
    debug: bool = False
    #: Internal: free-form overrides that are not config fields (kept for forward compat).
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls, overrides: Optional[Mapping[str, Any]] = None, env: Optional[Mapping[str, str]] = None
    ) -> Config:
        env = os.environ if env is None else env
        cfg = cls()
        for f in fields(cls):
            if f.name == "extra":
                continue
            raw = env.get("JGM_" + f.name.upper())
            if raw is None:
                continue
            cur = getattr(cfg, f.name)
            if isinstance(cur, bool):
                setattr(cfg, f.name, env_bool(raw, cur))
            elif isinstance(cur, float):
                setattr(cfg, f.name, env_float(raw, cur))
            elif isinstance(cur, tuple):
                setattr(cfg, f.name, env_tuple(raw, cur))
            else:
                setattr(cfg, f.name, raw.strip() or None if cur is None else raw.strip())
        # JGM_DISABLED=1 is the memorable way to switch off, JGM_ENABLED=0 works too.
        if env_bool(env.get("JGM_DISABLED"), False):
            cfg.enabled = False
        if overrides:
            for k, v in overrides.items():
                if k in {f.name for f in fields(cls)} and k != "extra":
                    if isinstance(v, list):
                        v = tuple(v)
                    setattr(cfg, k, v)
                else:
                    cfg.extra[k] = v
        cfg.heartbeat_s = max(1.0, cfg.heartbeat_s)
        cfg.sample_s = max(1.0, cfg.sample_s)
        cfg.sample_slow_s = max(cfg.sample_s, cfg.sample_slow_s)
        cfg.progress_s = max(0.1, cfg.progress_s)
        if cfg.rank_mode not in ("rank0", "all", "off"):
            cfg.rank_mode = "rank0"
        return cfg

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out
