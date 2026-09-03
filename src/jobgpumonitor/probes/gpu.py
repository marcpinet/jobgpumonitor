"""GPU probe: NVML through ``nvidia-ml-py`` when installed, otherwise ``nvidia-smi``.

Never touches CUDA (no ``torch.cuda`` calls, hence no context creation and no sync).
Respects ``CUDA_VISIBLE_DEVICES`` (indices, ``GPU-<uuid>`` or ``MIG-<uuid>`` entries).
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any, Dict, List, Optional

from .._log import dbg


def _visible_filter(env_value: Optional[str]) -> Optional[List[str]]:
    if env_value is None:
        return None
    entries = [e.strip() for e in env_value.split(",") if e.strip()]
    return entries  # may be [] -> no GPU visible


def _matches(entry: str, index: int, uuid: Optional[str]) -> bool:
    if entry.isdigit():
        return int(entry) == index
    if uuid and (entry == uuid or uuid.startswith(entry) or entry.startswith(uuid)):
        return True
    return False


class GpuProbe:
    def __init__(self, env: Optional[Dict[str, str]] = None) -> None:
        env = os.environ if env is None else env
        self.backend: Optional[str] = None
        self._visible = _visible_filter(env.get("CUDA_VISIBLE_DEVICES"))
        self._nvml: Any = None
        self._devices: List[Dict[str, Any]] = []  # [{index, uuid, name, mem_total, handle}]
        self._failures = 0
        self._disabled_until = 0.0
        self._init()

    # ------------------------------------------------------------------ init

    def _init(self) -> None:
        if self._visible == []:
            self.backend = None
            return
        if self._init_nvml():
            self.backend = "nvml"
            return
        if self._init_smi():
            self.backend = "nvidia-smi"

    def _init_nvml(self) -> bool:
        try:
            import pynvml  # type: ignore
        except Exception:
            return False
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
        except Exception as e:
            dbg(f"nvml init failed: {e}")
            return False
        self._nvml = pynvml
        for i in range(count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                uuid = _s(pynvml.nvmlDeviceGetUUID(h))
                name = _s(pynvml.nvmlDeviceGetName(h))
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                dev = {"index": i, "uuid": uuid, "name": name, "mem_total": int(mem.total), "handle": h, "mig": False}
                migs = self._mig_children(h, i)
                if migs:
                    for m in migs:
                        if self._visible is None or any(_matches(e, m["index"], m["uuid"]) or _matches(e, i, uuid) for e in self._visible):
                            self._devices.append(m)
                    continue
                if self._visible is None or any(_matches(e, i, uuid) for e in self._visible):
                    self._devices.append(dev)
            except Exception as e:
                dbg(f"nvml device {i}: {e}")
        return True

    def _mig_children(self, handle: Any, parent_index: int) -> List[Dict[str, Any]]:
        nv = self._nvml
        try:
            current, _pending = nv.nvmlDeviceGetMigMode(handle)
            if current != nv.NVML_DEVICE_MIG_ENABLE:
                return []
            out = []
            for j in range(nv.nvmlDeviceGetMaxMigDeviceCount(handle)):
                try:
                    mh = nv.nvmlDeviceGetMigDeviceHandleByIndex(handle, j)
                except Exception:
                    continue
                mem = nv.nvmlDeviceGetMemoryInfo(mh)
                out.append({
                    "index": parent_index, "mig_index": j, "uuid": _s(nv.nvmlDeviceGetUUID(mh)),
                    "name": _s(nv.nvmlDeviceGetName(mh)), "mem_total": int(mem.total), "handle": mh, "mig": True,
                })
            return out
        except Exception:
            return []

    _SMI_FIELDS = "index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw"

    def _run_smi(self) -> Optional[List[List[str]]]:
        try:
            p = subprocess.run(
                ["nvidia-smi", f"--query-gpu={self._SMI_FIELDS}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8,
            )
        except (OSError, subprocess.SubprocessError) as e:
            dbg(f"nvidia-smi failed: {e}")
            return None
        if p.returncode != 0:
            dbg(f"nvidia-smi rc={p.returncode}: {p.stderr.strip()[:200]}")
            return None
        rows = []
        for line in p.stdout.splitlines():
            parts = [c.strip() for c in line.split(",")]
            if len(parts) >= 9:
                rows.append(parts)
        return rows

    def _init_smi(self) -> bool:
        rows = self._run_smi()
        if not rows:
            return False
        for r in rows:
            try:
                i = int(r[0])
            except ValueError:
                continue
            if self._visible is None or any(_matches(e, i, r[1]) for e in self._visible):
                self._devices.append({"index": i, "uuid": r[1], "name": r[2], "mem_total": _mib(r[3]), "mig": False})
        return True

    # ------------------------------------------------------------------ public

    @property
    def available(self) -> bool:
        return self.backend is not None and bool(self._devices)

    def static(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "visible": self._visible,
            "devices": [
                {k: v for k, v in d.items() if k != "handle"} for d in self._devices
            ],
        }

    def sample(self) -> Optional[List[Dict[str, Any]]]:
        if not self.available:
            return None
        if time.monotonic() < self._disabled_until:
            return None
        try:
            if self.backend == "nvml":
                out = self._sample_nvml()
            else:
                out = self._sample_smi()
        except Exception as e:
            dbg(f"gpu sample failed: {e}")
            out = None
        if out is None:
            self._failures += 1
            if self._failures >= 3:
                self._disabled_until = time.monotonic() + 300
                self._failures = 0
        else:
            self._failures = 0
        return out

    def _sample_nvml(self) -> List[Dict[str, Any]]:
        nv = self._nvml
        out = []
        for d in self._devices:
            h = d["handle"]
            s: Dict[str, Any] = {"index": d["index"], "uuid": d["uuid"]}
            if d.get("mig"):
                s["mig_index"] = d["mig_index"]
            try:
                u = nv.nvmlDeviceGetUtilizationRates(h)
                s["util"] = int(u.gpu)
                s["mem_util"] = int(u.memory)
            except Exception:
                s["util"] = None
            try:
                m = nv.nvmlDeviceGetMemoryInfo(h)
                s["mem_used"] = int(m.used)
                s["mem_total"] = int(m.total)
            except Exception:
                pass
            try:
                s["temp_c"] = int(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            try:
                s["power_w"] = round(nv.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            except Exception:
                pass
            try:
                s["sm_clock_mhz"] = int(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM))
            except Exception:
                pass
            out.append(s)
        return out

    def _sample_smi(self) -> Optional[List[Dict[str, Any]]]:
        rows = self._run_smi()
        if rows is None:
            return None
        wanted = {d["index"] for d in self._devices}
        out = []
        for r in rows:
            try:
                i = int(r[0])
            except ValueError:
                continue
            if i not in wanted:
                continue
            out.append({
                "index": i, "uuid": r[1], "mem_total": _mib(r[3]), "mem_used": _mib(r[4]),
                "util": _int(r[5]), "mem_util": _int(r[6]), "temp_c": _int(r[7]), "power_w": _f(r[8]),
            })
        return out

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None


def _s(v: Any) -> str:
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def _int(v: str) -> Optional[int]:
    try:
        return int(float(v))
    except ValueError:
        return None


def _f(v: str) -> Optional[float]:
    try:
        return float(v)
    except ValueError:
        return None


def _mib(v: str) -> Optional[int]:
    i = _int(v)
    return None if i is None else i * 1024 * 1024
