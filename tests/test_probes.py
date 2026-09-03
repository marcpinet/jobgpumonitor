from __future__ import annotations

import os

from jobgpumonitor.probes import gpu as G
from jobgpumonitor.probes.system import CgroupProbe, ProcessProbe, disk_usage


def test_process_probe_samples_self():
    p = ProcessProbe()
    s = p.sample()
    assert s is not None and s["pid"] == os.getpid()
    if "rss" in s:
        assert s["rss"] > 0


def test_process_probe_proc_fallback(monkeypatch):
    monkeypatch.setattr("jobgpumonitor.probes.system.psutil", None)
    p = ProcessProbe()
    s = p.sample()
    if not os.path.exists("/proc/self/stat"):
        assert s is None
    else:
        assert s["threads"] >= 1


def test_disk_usage():
    d = disk_usage(os.getcwd())
    assert d and d["total"] > 0


def test_cgroup_v2_parsing(tmp_path, monkeypatch):
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.current").write_text("1073741824\n")
    (cg / "memory.max").write_text("2147483648\n")
    (cg / "memory.peak").write_text("1200000000\n")
    (cg / "memory.events").write_text("low 0\nhigh 3\nmax 0\noom 1\noom_kill 1\n")
    (cg / "cpu.max").write_text("300000 100000\n")
    probe = CgroupProbe.__new__(CgroupProbe)
    probe.pid = os.getpid()
    probe.fallback_limit = None
    probe.version, probe.dir = 2, str(cg)
    lim = probe.limits()
    assert lim["mem_limit"] == 2147483648 and lim["cpu_limit"] == 3.0
    s = probe.sample()
    assert s["mem_pct"] == 50.0 and s["events_oom_kill"] == 1 and s["mem_peak"] == 1200000000


def test_cgroup_v1_unlimited(tmp_path):
    cg = tmp_path / "cg1"
    cg.mkdir()
    (cg / "memory.usage_in_bytes").write_text("10\n")
    (cg / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    probe = CgroupProbe.__new__(CgroupProbe)
    probe.pid = os.getpid()
    probe.fallback_limit = None
    probe.version, probe.dir = 1, str(cg)
    s = probe.sample()
    assert s["mem_limit"] is None and "mem_pct" not in s
    probe.fallback_limit = 20
    s = probe.sample()
    assert s["mem_limit"] == 20 and s["mem_limit_source"] == "scheduler" and s["mem_pct"] == 50.0


def test_gpu_probe_without_gpu_is_quiet(monkeypatch):
    monkeypatch.setattr(G.GpuProbe, "_init_nvml", lambda self: False)
    monkeypatch.setattr(G.GpuProbe, "_run_smi", lambda self: None)
    p = G.GpuProbe(env={})
    assert not p.available and p.sample() is None
    assert p.static()["devices"] == []


def test_gpu_probe_smi_parsing_and_visible_filter(monkeypatch):
    rows = [
        ["0", "GPU-aaaa", "NVIDIA A100", "81920", "1024", "37", "12", "45", "120.5"],
        ["1", "GPU-bbbb", "NVIDIA A100", "81920", "40960", "99", "80", "70", "300.0"],
    ]
    monkeypatch.setattr(G.GpuProbe, "_init_nvml", lambda self: False)
    monkeypatch.setattr(G.GpuProbe, "_run_smi", lambda self: rows)
    p = G.GpuProbe(env={"CUDA_VISIBLE_DEVICES": "1"})
    assert p.backend == "nvidia-smi" and p.available
    assert [d["index"] for d in p.static()["devices"]] == [1]
    s = p.sample()
    assert len(s) == 1 and s[0]["util"] == 99 and s[0]["mem_used"] == 40960 * 1024 * 1024 and s[0]["power_w"] == 300.0
    p2 = G.GpuProbe(env={"CUDA_VISIBLE_DEVICES": "GPU-aaaa"})
    assert [d["index"] for d in p2.static()["devices"]] == [0]
    p3 = G.GpuProbe(env={"CUDA_VISIBLE_DEVICES": ""})
    assert not p3.available


def test_gpu_probe_backs_off_after_failures(monkeypatch):
    rows = [["0", "GPU-aaaa", "X", "1", "1", "1", "1", "1", "1"]]
    calls = {"n": 0}

    def smi(self):
        calls["n"] += 1
        return rows if calls["n"] == 1 else None

    monkeypatch.setattr(G.GpuProbe, "_init_nvml", lambda self: False)
    monkeypatch.setattr(G.GpuProbe, "_run_smi", smi)
    p = G.GpuProbe(env={})
    for _ in range(3):
        assert p.sample() is None
    before = calls["n"]
    assert p.sample() is None  # disabled for a while: no new call
    assert calls["n"] == before
