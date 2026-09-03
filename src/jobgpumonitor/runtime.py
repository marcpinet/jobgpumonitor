"""The ``Run``: one emitter per process. Wires hooks, probes, a writer thread and sinks.

Invariants:
* nothing here ever raises into user code;
* every emit is asynchronous (a queue drained by one daemon thread) so slow network
  filesystems never stall the training loop;
* after ``run.end`` nothing is emitted anymore;
* forked children are silent.
"""

from __future__ import annotations

import atexit
import itertools
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from . import events as ev
from ._log import dbg, set_debug, warn
from .config import Config
from .context import build_context
from .probes import CgroupProbe, GpuProbe, ProcessProbe, disk_usage, load_average
from .sinks import ListSink, Sink, build_sinks, resolve_base_dir, run_dir_for

#: Event types that ranks other than 0 do not emit in ``rank0`` mode.
_LIGHT_SUPPRESSED = {"resource.sample", "progress.update", "metric.log", "log.line", "checkpoint.saved"}


class _Writer(threading.Thread):
    def __init__(self, sinks: List[Sink]) -> None:
        super().__init__(name="jgm-writer", daemon=True)
        self.sinks = sinks
        self.q: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self.written = 0

    def put(self, line: str) -> None:
        self.q.put(line)

    def run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            if isinstance(item, threading.Event):
                item.set()
                continue
            for s in self.sinks:
                try:
                    s.write(item)
                except Exception as e:  # sinks are supposed to be silent already
                    dbg(f"sink {s.name} raised: {e}")
            self.written += 1

    def flush(self, timeout: float) -> bool:
        if not self.is_alive():
            return False
        done = threading.Event()
        self.q.put(done)
        return done.wait(timeout)

    def stop(self, timeout: float) -> None:
        if self.is_alive():
            self.q.put(None)
            self.join(timeout)
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                pass


class _Stats:
    """Running aggregates for the ``run.end`` summary."""

    def __init__(self) -> None:
        self.samples = 0
        self.gpu_util_sum: Dict[str, float] = {}
        self.gpu_util_n: Dict[str, int] = {}
        self.gpu_mem_max: Dict[str, int] = {}
        self.gpu_idle_samples: Dict[str, int] = {}
        self.rss_max = 0
        self.cgroup_max = 0
        self.cpu_sum = 0.0
        self.cpu_n = 0

    def add(self, sample: Dict[str, Any]) -> None:
        self.samples += 1
        for g in sample.get("gpus") or []:
            key = str(g.get("uuid") or g.get("index"))
            if g.get("util") is not None:
                self.gpu_util_sum[key] = self.gpu_util_sum.get(key, 0.0) + float(g["util"])
                self.gpu_util_n[key] = self.gpu_util_n.get(key, 0) + 1
                if g["util"] < 5:
                    self.gpu_idle_samples[key] = self.gpu_idle_samples.get(key, 0) + 1
            if g.get("mem_used") is not None:
                self.gpu_mem_max[key] = max(self.gpu_mem_max.get(key, 0), int(g["mem_used"]))
        proc = sample.get("proc") or {}
        if proc.get("rss"):
            self.rss_max = max(self.rss_max, int(proc["rss"]) + int(proc.get("children_rss") or 0))
        if proc.get("cpu_pct") is not None:
            self.cpu_sum += float(proc["cpu_pct"])
            self.cpu_n += 1
        cg = sample.get("cgroup") or {}
        if cg.get("mem_used"):
            self.cgroup_max = max(self.cgroup_max, int(cg["mem_used"]))

    def summary(self) -> Dict[str, Any]:
        gpus = []
        for key in sorted(set(self.gpu_util_n) | set(self.gpu_mem_max)):
            n = self.gpu_util_n.get(key, 0)
            gpus.append({
                "gpu": key,
                "util_mean": round(self.gpu_util_sum[key] / n, 1) if n else None,
                "idle_fraction": round(self.gpu_idle_samples.get(key, 0) / n, 3) if n else None,
                "mem_used_max": self.gpu_mem_max.get(key),
            })
        return {
            "samples": self.samples,
            "gpus": gpus,
            "rss_max": self.rss_max or None,
            "cgroup_mem_max": self.cgroup_max or None,
            "cpu_pct_mean": round(self.cpu_sum / self.cpu_n, 1) if self.cpu_n else None,
        }


class Run:
    def __init__(
        self,
        config: Optional[Config] = None,
        context: Optional[Dict[str, Any]] = None,
        source: str = "process",
        hooks: bool = True,
        monitor: bool = True,
        probe_pid: Optional[int] = None,
    ) -> None:
        self.config = config or Config.from_env()
        set_debug(self.config.debug)
        self.source = source
        self.ctx = context or build_context(self.config, source=source)
        self.run_id: str = self.ctx["run_id"]
        self.emitter: str = self.ctx["emitter"]
        self.pid: int = self.ctx["pid"]
        self.rank: Optional[int] = self.ctx["rank"]["rank"] if self.ctx.get("rank") else None
        self.enabled = bool(self.config.enabled)
        self.light = False
        if self.rank not in (None, 0):
            if self.config.rank_mode == "off":
                self.enabled = False
            elif self.config.rank_mode == "rank0":
                self.light = True
        self._hooks = hooks
        self._monitor_enabled = monitor
        self._probe_pid = probe_pid
        self._seq = itertools.count()
        self._t0 = time.monotonic()
        self.start_ts: float = self.ctx.get("start_ts") or time.time()
        self.started = False
        self.ended = False
        self._ending = False
        self._child = False
        self._lock = threading.RLock()
        self.latest_progress: Dict[Any, Dict[str, Any]] = {}
        self.latest_metrics: Dict[str, Any] = {}
        self._uncaught: Optional[Dict[str, Any]] = None
        self._exit_code: Optional[int] = None
        self._stats = _Stats()
        self.base_dir: Optional[str] = None
        self.run_dir: Optional[str] = None
        self.sinks: List[Sink] = []
        self._writer: Optional[_Writer] = None
        self._monitor: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.gpu: Optional[GpuProbe] = None
        self.proc: Optional[ProcessProbe] = None
        self.cgroup: Optional[CgroupProbe] = None
        self.installed_hooks: List[str] = []

    # ------------------------------------------------------------------ lifecycle

    def start(self, emit_start: bool = True) -> Run:
        if self.started or not self.enabled:
            self.started = True
            return self
        self.started = True
        self.base_dir, origin = resolve_base_dir(self.config.dir)
        if self.base_dir:
            self.run_dir = run_dir_for(self.base_dir, self.run_id)
        dbg(f"base_dir={self.base_dir} ({origin})")
        self.sinks = build_sinks(self.config.sinks, self.base_dir, self.run_id, self.emitter, self.config.fsync)
        if not self.sinks:
            warn("no usable sink; monitoring disabled for this process")
            self.enabled = False
            return self
        self._writer = _Writer(self.sinks)
        self._writer.start()

        pid = self._probe_pid or self.pid
        try:
            self.gpu = GpuProbe()
        except Exception as e:
            dbg(f"gpu probe init failed: {e}")
        try:
            self.proc = ProcessProbe(pid)
            self.cgroup = CgroupProbe(pid)
        except Exception as e:
            dbg(f"system probes init failed: {e}")

        if self._hooks:
            self._install_hooks()
        try:
            atexit.register(self._atexit)
        except Exception:
            pass
        if hasattr(os, "register_at_fork"):
            try:
                os.register_at_fork(after_in_child=self._after_fork_child)
            except Exception:
                pass

        if emit_start:
            self.emit("run.start", self._start_payload())
        if self._monitor_enabled and not self.light:
            self._monitor = threading.Thread(target=self._monitor_loop, name="jgm-monitor", daemon=True)
            self._monitor.start()
        elif self._monitor_enabled:
            self._monitor = threading.Thread(target=self._monitor_loop, name="jgm-monitor", daemon=True, kwargs={"heartbeat_only": True})
            self._monitor.start()
        return self

    def _install_hooks(self) -> None:
        from .hooks import exceptions as _exc

        try:
            _exc.install(self)
            _exc.install_exit_tracking(self)
            self.installed_hooks.append("excepthook")
        except Exception as e:
            dbg(f"excepthook install failed: {e}")
        if self.config.signals:
            try:
                from .hooks import signals as _sig

                if _sig.install(self):
                    self.installed_hooks.append("signals")
            except Exception as e:
                dbg(f"signal hooks failed: {e}")
        if self.config.tqdm and not self.light:
            try:
                from .hooks import tqdm_hook

                tqdm_hook.install(self)
                self.installed_hooks.append("tqdm")
            except Exception as e:
                dbg(f"tqdm hook failed: {e}")
        if self.config.logging_level and not self.light:
            try:
                from .hooks import logging_hook

                if logging_hook.install(self, self.config.logging_level):
                    self.installed_hooks.append("logging")
            except Exception as e:
                dbg(f"logging hook failed: {e}")
        if self.config.faulthandler:
            try:
                import faulthandler

                if not faulthandler.is_enabled():
                    faulthandler.enable()
                    self.installed_hooks.append("faulthandler")
            except Exception as e:
                dbg(f"faulthandler failed: {e}")

    def _start_payload(self) -> Dict[str, Any]:
        c = self.ctx
        static: Dict[str, Any] = {
            "cpu_count": c.get("cpu_count"),
            "mem_total": c.get("mem_total"),
            "gpu": self.gpu.static() if self.gpu else None,
            "cgroup": self.cgroup.limits() if self.cgroup else None,
            "disk": disk_usage(c.get("cwd")),
        }
        payload = {
            "scheduler": c["job"],
            "rank": c.get("rank"),
            "host": c["host"],
            "user": c.get("user"),
            "cwd": c.get("cwd"),
            "argv": c.get("argv"),
            "main_file": c.get("main_file"),
            "executable": c.get("executable"),
            "python": c.get("python"),
            "platform": c.get("platform"),
            "container": c.get("container"),
            "mounts": c.get("mounts"),
            "git": c.get("git"),
            "packages": c.get("packages"),
            "env": c.get("env"),
            "stdio": c.get("stdio"),
            "interactive": c.get("interactive"),
            "wrapped": c.get("wrapped"),
            "deadline": c.get("deadline"),
            "start_ts": self.start_ts,
            "resources": static,
            "emitter_config": {
                "version": _version(),
                "heartbeat_s": self.config.heartbeat_s,
                "sample_s": self.config.sample_s,
                "sinks": [s.describe() for s in self.sinks],
                "light": self.light,
                "hooks": list(self.installed_hooks),
            },
        }
        if self.config.extra:
            payload["extra"] = ev.sanitize(self.config.extra)
        return payload

    # ------------------------------------------------------------------ emit API

    def emit(self, etype: str, data: Optional[Dict[str, Any]] = None, **kw: Any) -> bool:
        if not self.enabled or self._child or self.ended:
            return False
        etype = ev.normalize_type(etype)
        if self.light and etype in _LIGHT_SUPPRESSED:
            return False
        if self._writer is None:
            return False
        payload: Dict[str, Any] = dict(data or {})
        if kw:
            payload.update(kw)
        try:
            # The lock makes "seq assignment + enqueue" atomic, so ``run.end`` (emitted
            # while ``_ending`` is set) is guaranteed to be the last line of the file.
            with self._lock:
                if self._ending and etype != "run.end":
                    return False
                env = ev.make_envelope(
                    run_id=self.run_id, emitter=self.emitter, pid=self.pid, source=self.source,
                    rank=self.rank, seq=next(self._seq), etype=etype,
                    data=ev.sanitize(payload) if etype.startswith("custom.") or etype == "metric.log" else payload,
                    mono=time.monotonic() - self._t0,
                )
                self._writer.put(ev.to_json(env))
            return True
        except Exception as e:
            dbg(f"emit {etype} failed: {e}")
            return False

    def log(self, metrics: Optional[Dict[str, Any]] = None, step: Optional[int] = None, epoch: Optional[int] = None, **kw: Any) -> bool:
        m: Dict[str, Any] = dict(metrics or {})
        m.update(kw)
        if not m:
            return False
        clean = ev.sanitize(m)
        with self._lock:
            self.latest_metrics.update(clean)
            if len(self.latest_metrics) > 64:
                for k in list(self.latest_metrics)[:-64]:
                    del self.latest_metrics[k]
        return self.emit("metric.log", {"metrics": clean, "step": step, "epoch": epoch})

    def progress(self, payload: Dict[str, Any]) -> bool:
        rem = self.deadline_remaining_s()
        if rem is not None and payload.get("eta_s") is not None:
            payload["deadline_remaining_s"] = round(rem, 1)
            payload["eta_vs_deadline_s"] = round(rem - payload["eta_s"], 1)
        with self._lock:
            key = payload.get("bar_id")
            if payload.get("done"):
                self.latest_progress.pop(key, None)
            else:
                self.latest_progress[key] = payload
                if len(self.latest_progress) > 8:
                    self.latest_progress.pop(next(iter(self.latest_progress)))
        return self.emit("progress.update", payload)

    def finish(self, status: Optional[str] = None, exit_code: Optional[int] = None, **extra: Any) -> None:
        with self._lock:
            if self.ended or self._ending or not self.enabled or self._child:
                return
            self._ending = True
        self._stop.set()
        if status is None:
            status, derived_code = self._derive_status()
            if exit_code is None:
                exit_code = derived_code
        data: Dict[str, Any] = {
            "status": status,
            "exit_code": exit_code,
            "duration_s": round(time.monotonic() - self._t0, 3),
            "end_ts": time.time(),
            "exception": self._uncaught,
            "metrics": dict(self.latest_metrics),
            "progress": list(self.latest_progress.values()),
            "summary": self._stats.summary(),
            "dropped": self._dropped(),
        }
        if extra:
            data.update(ev.sanitize(extra))
        with self._lock:
            self.emit("run.end", data)
            self.ended = True
        self.flush(5.0)
        if self.gpu:
            try:
                self.gpu.close()
            except Exception:
                pass

    def flush(self, timeout: float = 5.0) -> bool:
        if self._writer is None:
            return True
        return self._writer.flush(timeout)

    def close(self, timeout: float = 5.0) -> None:
        if self._writer is not None:
            self._writer.stop(timeout)

    # ------------------------------------------------------------------ hook callbacks

    def note_uncaught(self, kind: str, exc_type: Any, exc: Any, tb: Any) -> None:
        from .hooks.exceptions import format_exception

        info = format_exception(exc_type, exc, tb, self.config.capture_locals)
        with self._lock:
            self._uncaught = {"kind": kind, "type": info["type"], "message": info["message"], "is_oom": info["is_oom"]}
        info.update(fatal=True, thread=threading.current_thread().name, kind=kind)
        self.emit("run.exception", info)

    def note_thread_exception(self, args: Any) -> None:
        from .hooks.exceptions import format_exception

        info = format_exception(args.exc_type, args.exc_value, args.exc_traceback, self.config.capture_locals)
        thread = getattr(args.thread, "name", None) if args.thread is not None else None
        info.update(fatal=False, thread=thread, kind="thread")
        self.emit("run.exception", info)

    def note_exit_code(self, code: Any) -> None:
        if code is None:
            c = 0
        elif isinstance(code, bool):
            c = int(code)
        elif isinstance(code, int):
            c = code
        else:
            c = 1  # Python prints the object and exits 1
        with self._lock:
            self._exit_code = c

    def _derive_status(self) -> tuple[str, Optional[int]]:
        if self._uncaught:
            if self._uncaught["kind"] == "interrupted":
                return "interrupted", 130
            return "error", 1
        if self._exit_code not in (None, 0):
            return "error", self._exit_code
        return "ok", 0 if self._exit_code is None else self._exit_code

    def deadline_remaining_s(self) -> Optional[float]:
        d = self.ctx.get("deadline")
        if not d or not d.get("end_ts"):
            return None
        return float(d["end_ts"]) - time.time()

    # ------------------------------------------------------------------ threads

    def _dropped(self) -> int:
        return sum(getattr(s, "dropped", 0) for s in self.sinks)

    def _sample(self) -> Dict[str, Any]:
        s: Dict[str, Any] = {}
        try:
            if self.gpu and self.gpu.available:
                s["gpus"] = self.gpu.sample()
        except Exception:
            pass
        try:
            if self.proc:
                s["proc"] = self.proc.sample()
        except Exception:
            pass
        try:
            if self.cgroup and self.cgroup.available:
                s["cgroup"] = self.cgroup.sample()
        except Exception:
            pass
        s["disk"] = disk_usage(self.ctx.get("cwd"))
        s["load"] = load_average()
        return s

    def _heartbeat_payload(self) -> Dict[str, Any]:
        with self._lock:
            progress = list(self.latest_progress.values())
            metrics = dict(self.latest_metrics)
        return {
            "uptime_s": round(time.monotonic() - self._t0, 1),
            "deadline_remaining_s": _round(self.deadline_remaining_s()),
            "progress": progress,
            "metrics": metrics,
            "samples": self._stats.samples,
            "dropped": self._dropped(),
            "threads": threading.active_count(),
        }

    def _monitor_loop(self, heartbeat_only: bool = False) -> None:
        cfg = self.config
        next_sample = time.monotonic() + (0.0 if not heartbeat_only else 1e12)
        next_hb = time.monotonic() + cfg.heartbeat_s
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_sample:
                try:
                    sample = self._sample()
                    self._stats.add(sample)
                    self.emit("resource.sample", sample)
                except Exception as e:
                    dbg(f"sample failed: {e}")
                uptime = now - self._t0
                interval = cfg.sample_s if uptime < cfg.sample_slow_after_s else cfg.sample_slow_s
                next_sample = now + interval
            if now >= next_hb:
                self.emit("run.heartbeat", self._heartbeat_payload())
                next_hb = now + cfg.heartbeat_s
            wait = max(0.2, min(next_sample, next_hb) - time.monotonic())
            self._stop.wait(wait)

    def _after_fork_child(self) -> None:
        self._child = True

    def _atexit(self) -> None:
        try:
            if not self.ended:
                self.finish()
            self.close(3.0)
        except Exception as e:
            dbg(f"atexit failed: {e}")

    # ------------------------------------------------------------------ misc

    def __repr__(self) -> str:
        return f"<jobgpumonitor.Run {self.run_id} emitter={self.emitter} enabled={self.enabled} light={self.light}>"

    def list_sink_lines(self) -> List[str]:
        for s in self.sinks:
            if isinstance(s, ListSink):
                return s.lines
        return []


def _round(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 1)


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:
        return "?"


def in_child_process() -> bool:
    """True inside a ``multiprocessing`` child (fork or spawn), where we stay silent."""
    # Spawned children re-import the main module *before* ``parent_process()`` is set,
    # but ``current_process().name`` and ``--multiprocessing-fork`` in argv are already there.
    if "--multiprocessing-fork" in sys.argv:
        return True
    try:
        import multiprocessing

        if multiprocessing.parent_process() is not None:  # type: ignore[attr-defined]
            return True
        if multiprocessing.current_process().name != "MainProcess":
            return True
    except Exception:
        pass
    return bool(getattr(sys, "_jgm_child", False))
