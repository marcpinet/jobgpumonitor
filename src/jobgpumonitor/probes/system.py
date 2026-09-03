"""Process, cgroup and disk probes. ``psutil`` is used when present, ``/proc`` otherwise."""

from __future__ import annotations

import os
import shutil
import time
from typing import Any, Dict, Optional

from .._log import dbg

try:  # optional
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


class ProcessProbe:
    """CPU %, RSS, thread count of one process (and RSS of its children with psutil)."""

    def __init__(self, pid: Optional[int] = None) -> None:
        self.pid = pid or os.getpid()
        self._ps: Any = None
        self._last_cpu_ticks: Optional[int] = None
        self._last_wall: Optional[float] = None
        self._tick = None
        try:
            self._tick = os.sysconf("SC_CLK_TCK")
        except (ValueError, OSError, AttributeError):
            self._tick = 100
        if psutil is not None:
            try:
                self._ps = psutil.Process(self.pid)
                self._ps.cpu_percent(interval=None)  # prime
            except Exception:
                self._ps = None

    def sample(self) -> Optional[Dict[str, Any]]:
        try:
            if self._ps is not None:
                return self._sample_psutil()
            return self._sample_proc()
        except Exception as e:
            dbg(f"process sample failed: {e}")
            return None

    def _sample_psutil(self) -> Dict[str, Any]:
        p = self._ps
        with p.oneshot():
            mem = p.memory_info()
            out: Dict[str, Any] = {
                "pid": self.pid,
                "cpu_pct": round(p.cpu_percent(interval=None), 1),
                "rss": int(mem.rss),
                "threads": p.num_threads(),
            }
        try:
            children = p.children(recursive=True)[:64]
            if children:
                rss = 0
                for c in children:
                    try:
                        rss += c.memory_info().rss
                    except Exception:
                        continue
                out["children"] = len(children)
                out["children_rss"] = int(rss)
        except Exception:
            pass
        try:
            out["open_fds"] = p.num_fds()
        except Exception:
            pass
        return out

    def _sample_proc(self) -> Optional[Dict[str, Any]]:
        base = f"/proc/{self.pid}"
        out: Dict[str, Any] = {"pid": self.pid}
        try:
            with open(f"{base}/stat", encoding="ascii", errors="replace") as f:
                stat = f.read()
            # comm may contain spaces; fields after the last ')'
            rest = stat[stat.rindex(")") + 2 :].split()
            utime, stime = int(rest[11]), int(rest[12])
            now = time.monotonic()
            ticks = utime + stime
            if self._last_cpu_ticks is not None and self._last_wall is not None and now > self._last_wall:
                out["cpu_pct"] = round(100.0 * (ticks - self._last_cpu_ticks) / self._tick / (now - self._last_wall), 1)
            self._last_cpu_ticks, self._last_wall = ticks, now
            out["threads"] = int(rest[17])
        except (OSError, ValueError, IndexError):
            return None
        try:
            with open(f"{base}/status", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        out["rss"] = int(line.split()[1]) * 1024
                    elif line.startswith("VmHWM:"):
                        out["rss_peak"] = int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return out


class CgroupProbe:
    """Memory usage vs. the cgroup limit (Slurm enforces ``--mem`` through cgroups).

    Handles cgroup v2 (``memory.current`` / ``memory.max``) and v1
    (``memory.usage_in_bytes`` / ``memory.limit_in_bytes``). Inside a container the
    process may only see the cgroup root, which still reflects the job's limit.
    """

    def __init__(self, pid: Optional[int] = None) -> None:
        self.pid = pid or os.getpid()
        self.version: Optional[int] = None
        self.dir: Optional[str] = None
        self._detect()

    def _detect(self) -> None:
        try:
            with open(f"/proc/{self.pid}/cgroup", encoding="utf-8", errors="replace") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except OSError:
            return
        v2_path = None
        v1_mem_path = None
        for ln in lines:
            parts = ln.split(":", 2)
            if len(parts) != 3:
                continue
            hid, ctrls, path = parts
            if hid == "0" and ctrls == "":
                v2_path = path
            elif "memory" in ctrls.split(","):
                v1_mem_path = path
        if os.path.exists("/sys/fs/cgroup/cgroup.controllers") and v2_path is not None:
            for cand in (os.path.join("/sys/fs/cgroup", v2_path.lstrip("/")), "/sys/fs/cgroup"):
                if os.path.exists(os.path.join(cand, "memory.current")):
                    self.version, self.dir = 2, cand
                    return
        if v1_mem_path is not None:
            for cand in (os.path.join("/sys/fs/cgroup/memory", v1_mem_path.lstrip("/")), "/sys/fs/cgroup/memory"):
                if os.path.exists(os.path.join(cand, "memory.usage_in_bytes")):
                    self.version, self.dir = 1, cand
                    return

    @property
    def available(self) -> bool:
        return self.dir is not None

    def _read(self, name: str) -> Optional[str]:
        assert self.dir is not None
        try:
            with open(os.path.join(self.dir, name), encoding="ascii", errors="replace") as f:
                return f.read().strip()
        except OSError:
            return None

    def _read_int(self, name: str) -> Optional[int]:
        v = self._read(name)
        if v is None or v == "max":
            return None
        try:
            i = int(v)
        except ValueError:
            return None
        if i >= 2**60:  # v1 "unlimited"
            return None
        return i

    def limits(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False}
        out: Dict[str, Any] = {"available": True, "version": self.version, "path": self.dir}
        if self.version == 2:
            out["mem_limit"] = self._read_int("memory.max")
            out["swap_limit"] = self._read_int("memory.swap.max")
            cpu = self._read("cpu.max")
            if cpu and not cpu.startswith("max"):
                try:
                    q, p = cpu.split()
                    out["cpu_limit"] = round(int(q) / int(p), 2)
                except ValueError:
                    pass
        else:
            out["mem_limit"] = self._read_int("memory.limit_in_bytes")
        return out

    def sample(self) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        try:
            if self.version == 2:
                out: Dict[str, Any] = {
                    "mem_used": self._read_int("memory.current"),
                    "mem_limit": self._read_int("memory.max"),
                    "mem_peak": self._read_int("memory.peak"),
                    "swap_used": self._read_int("memory.swap.current"),
                }
                ev = self._read("memory.events")
                if ev:
                    for ln in ev.splitlines():
                        k, _, v = ln.partition(" ")
                        if k in ("oom", "oom_kill", "max", "high"):
                            try:
                                out["events_" + k] = int(v)
                            except ValueError:
                                pass
            else:
                out = {
                    "mem_used": self._read_int("memory.usage_in_bytes"),
                    "mem_limit": self._read_int("memory.limit_in_bytes"),
                    "mem_peak": self._read_int("memory.max_usage_in_bytes"),
                }
                oc = self._read("memory.oom_control")
                if oc:
                    for ln in oc.splitlines():
                        if ln.startswith("oom_kill "):
                            try:
                                out["events_oom_kill"] = int(ln.split()[1])
                            except (ValueError, IndexError):
                                pass
            if out.get("mem_used") is not None and out.get("mem_limit"):
                out["mem_pct"] = round(100.0 * out["mem_used"] / out["mem_limit"], 1)
            return out
        except Exception as e:
            dbg(f"cgroup sample failed: {e}")
            return None


def disk_usage(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        u = shutil.disk_usage(path)
        return {"path": path, "total": int(u.total), "free": int(u.free)}
    except OSError:
        return None


def load_average() -> Optional[list]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):
        return None
