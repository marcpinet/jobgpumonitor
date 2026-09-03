from __future__ import annotations

import glob
import json
import time
from typing import Dict, List, Optional

from jobgpumonitor.scheduler import base as B
from jobgpumonitor.scheduler.probe import SchedulerProbe
from jobgpumonitor.scheduler.slurm import SlurmAdapter

# --------------------------------------------------------------------------- fixtures

SQUEUE_PENDING = "8224458|8224458|N/A|jgm-smoke|PENDING|Priority|short|normal|0:00|5:00|N/A|N/A|2026-09-03T11:54:58||1|2|8G|gres/gpu:1|kxxl2403|others|1234\n"
SQUEUE_RUNNING = "8224458|8224458|N/A|jgm-smoke|RUNNING|None|short|normal|0:10|5:00|2026-09-03T11:55:05|2026-09-03T12:00:05|2026-09-03T11:54:58|r-marcel-c3-abs20-04|1|2|8G|gres/gpu:1|kxxl2403|others|1234\n"
SQUEUE_ARRAY = (
    "9000_[3-5]|9000|N/A|arr|PENDING|Resources|All|normal|0:00|1:00:00|N/A|N/A|2026-09-03T11:00:00||1|1|4G|N/A|kxxl2403|others|10\n"
    "9000_1|9000|1|arr|RUNNING|None|All|normal|1:00|1:00:00|2026-09-03T11:01:00|2026-09-03T12:01:00|2026-09-03T11:00:00|n1|1|1|4G|N/A|kxxl2403|others|10\n"
)
SCONTROL = (
    "JobId=8224458 JobName=jgm-smoke UserId=kxxl2403(11249) GroupId=others(1501) Priority=1234 Account=others QOS=normal "
    "JobState=RUNNING Reason=None Dependency=(null) Requeue=1 Restarts=0 BatchFlag=1 ExitCode=0:0 RunTime=00:00:10 TimeLimit=00:05:00 "
    "SubmitTime=2026-09-03T11:54:58 StartTime=2026-09-03T11:55:05 EndTime=2026-09-03T12:00:05 Partition=short NodeList=r-marcel-c3-abs20-04 "
    "BatchHost=r-marcel-c3-abs20-04 NumNodes=1 NumCPUs=2 ReqTRES=cpu=2,mem=8G,node=1,gres/gpu=1 MinMemoryNode=8G "
    "Command=/home/k/jobgpumonitor/examples/slurm_smoke.sbatch WorkDir=/home/k StdErr=/home/k/jgm-smoke_8224458.err StdIn=/dev/null "
    "StdOut=/home/k/jgm-smoke_8224458.out TresPerNode=gres/gpu:1\n"
)
SACCT_OOM = (
    "8224458|jgm-smoke|OUT_OF_MEMORY|0:125|0:0|2026-09-03T11:55:05|2026-09-03T11:57:00|115|r-marcel-c3-abs20-04||8G|5|short|normal|others|2026-09-03T11:54:58|/home/k|None|cpu=2,mem=8G,node=1,gres/gpu=1|kxxl2403|1|2\n"
    "8224458.batch|batch|OUT_OF_MEMORY|0:125|0:0|2026-09-03T11:55:05|2026-09-03T11:57:00|115|r-marcel-c3-abs20-04|7812000K|8G|5|short|normal|others|2026-09-03T11:54:58|/home/k|None|cpu=2,mem=8G,node=1|kxxl2403|1|2\n"
    "8224458.0|bash|CANCELLED by 11249|0:9|0:0|2026-09-03T11:55:06|2026-09-03T11:57:00|114|r-marcel-c3-abs20-04|123456K|8G|5|short|normal|others|2026-09-03T11:54:58|/home/k|None|cpu=2,mem=8G,node=1|kxxl2403|1|2\n"
)


class FakeSlurm:
    """Scripted command runner: ``squeue`` output changes with each call."""

    def __init__(self, squeue_outputs: List[str], sacct: Optional[str] = None, scontrol: str = SCONTROL) -> None:
        self.squeue_outputs = list(squeue_outputs)
        self.sacct = sacct
        self.scontrol = scontrol
        self.calls: List[List[str]] = []

    def __call__(self, argv: List[str]) -> Optional[str]:
        self.calls.append(argv)
        if argv[0] == "squeue":
            return self.squeue_outputs.pop(0) if len(self.squeue_outputs) > 1 else self.squeue_outputs[0]
        if argv[0] == "scontrol" and argv[1:3] == ["show", "config"]:
            return "ClusterName               = marcel-c3\nSlurmctldHost = x\n"
        if argv[0] == "scontrol":
            return self.scontrol
        if argv[0] == "sacct":
            return self.sacct
        return None


def read_events(base: str) -> List[Dict]:
    out = []
    for f in sorted(glob.glob(f"{base}/runs/**/scheduler-*.jsonl", recursive=True)):
        with open(f) as fh:
            out += [json.loads(ln) for ln in fh if ln.strip()]
    return out


# --------------------------------------------------------------------------- parsing


def test_duration_and_datetime_parsing():
    assert B.parse_duration("1-02:03:04") == 93784
    assert B.parse_duration("5:00") == 300
    assert B.parse_duration("12:34:56") == 45296
    assert B.parse_duration("UNLIMITED") is None and B.parse_duration("N/A") is None
    ts = B.parse_datetime("2026-09-03T11:55:05")
    assert ts and time.localtime(ts).tm_hour == 11 and time.localtime(ts).tm_min == 55
    assert B.parse_datetime("N/A") is None and B.parse_datetime("Unknown") is None
    assert B.parse_exit_code("0:125") == (0, 125) and B.parse_exit_code("1:0") == (1, None)
    assert B.parse_mem_kb_suffix("7812000K") == 7812000 * 1024 and B.parse_mem_kb_suffix("1.5G") == int(1.5 * 1024**3)
    assert B.normalise_state("CANCELLED by 11249") == "CANCELLED" and B.normalise_state("OUT_OF_ME+") == "OUT_OF_MEMORY"


def test_squeue_parsing_and_array_forms():
    a = SlurmAdapter(FakeSlurm([SQUEUE_ARRAY]))
    jobs = a.list_jobs("kxxl2403")
    assert jobs is not None and len(jobs) == 2
    agg, task = jobs
    assert agg.job_key == "9000" and agg.array_pending_tasks == "[3-5]" and agg.array_job_id == "9000" and agg.state == "PENDING"
    assert agg.extra["reason_hint"].startswith("Requested resources")
    assert task.job_key == "9000_1" and task.array_task_id == "1" and task.state == "RUNNING" and task.gres is None


def test_scontrol_enrich_and_sacct_terminal():
    fake = FakeSlurm([SQUEUE_RUNNING], sacct=SACCT_OOM)
    a = SlurmAdapter(fake)
    job = a.list_jobs(None)[0]
    assert job.gres == "gres/gpu:1" and job.time_limit_s == 300 and job.priority == 1234
    job = a.enrich(job)
    assert job.stdout == "/home/k/jgm-smoke_8224458.out" and job.restarts == 0 and job.requeue is True
    assert job.command.endswith("slurm_smoke.sbatch")
    assert job.dependency is None and job.extra["batch_host"] == "r-marcel-c3-abs20-04"
    fin = a.finished(["8224458"], time.time() - 100, "kxxl2403")
    j = fin["8224458"]
    assert j.state == "OUT_OF_MEMORY" and j.terminal and j.exit_signal == 125 and j.elapsed_s == 115
    assert j.max_rss == 7812000 * 1024 and j.gres == "gpu:1" and j.time_limit_s == 300


# --------------------------------------------------------------------------- probe


def test_probe_lifecycle_pending_running_oom(tmp_path):
    fake = FakeSlurm([SQUEUE_PENDING, SQUEUE_RUNNING, SQUEUE_RUNNING, ""], sacct=SACCT_OOM)
    clock = [1_000_000.0]
    probe = SchedulerProbe(SlurmAdapter(fake), str(tmp_path), user="kxxl2403", interval_s=30, host="login01", now=lambda: clock[0])
    assert probe.cluster == "marcel-c3"

    e1 = probe.poll()  # pending
    assert [e["data"]["change"] for e in e1] == ["first_seen"] and e1[0]["data"]["state"] == "PENDING"
    assert e1[0]["run_id"] == "marcel-c3/8224458/0" and e1[0]["source"] == "scheduler" and e1[0]["emitter"] == "scheduler-login01"
    assert e1[0]["data"]["reason_hint" if "reason_hint" in e1[0]["data"] else "extra"]["reason_hint"].startswith("Waiting behind")

    clock[0] += 30
    e2 = probe.poll()  # running
    assert len(e2) == 1 and e2[0]["data"]["change"] == "state" and e2[0]["data"]["state"] == "RUNNING"
    assert e2[0]["data"]["stdout"] == "/home/k/jgm-smoke_8224458.out" and e2[0]["data"]["active"] is True

    clock[0] += 30
    assert probe.poll() == []  # no change, no event

    clock[0] += 30
    e4 = probe.poll()  # gone from the queue -> sacct says OOM
    assert len(e4) == 1 and e4[0]["data"]["change"] == "ended"
    d = e4[0]["data"]
    assert d["state"] == "OUT_OF_MEMORY" and d["terminal"] is True and d["max_rss"] == 7812000 * 1024
    assert d["stdout"] == "/home/k/jgm-smoke_8224458.out"  # remembered from scontrol while alive
    assert d["job_name"] == "jgm-smoke"

    clock[0] += 30
    assert probe.poll() == []  # terminal event is not repeated
    probe.close()

    evs = read_events(str(tmp_path))
    assert [e["seq"] for e in evs] == [0, 1, 2]
    state = json.loads((tmp_path / "scheduler" / "state-marcel-c3.json").read_text())
    assert state["jobs"]["8224458"]["ended"] is True
    hb = json.loads(next(iter((tmp_path / "scheduler").glob("probe-*.json"))).read_text())
    assert hb["ok"] is True and hb["polls"] == 5


def test_probe_restart_safe_and_refresh(tmp_path):
    fake = FakeSlurm([SQUEUE_RUNNING])
    clock = [2_000_000.0]
    p1 = SchedulerProbe(SlurmAdapter(fake), str(tmp_path), user=None, interval_s=30, refresh_s=100, host="l", now=lambda: clock[0])
    assert len(p1.poll()) == 1
    p1.close()
    # a new probe process with the same state file: nothing new to say
    p2 = SchedulerProbe(SlurmAdapter(fake), str(tmp_path), user=None, interval_s=30, refresh_s=100, host="l", now=lambda: clock[0])
    assert p2.poll() == []
    clock[0] += 120
    e = p2.poll()  # periodic refresh for active jobs
    assert len(e) == 1 and e[0]["data"]["change"] == "refresh"


def test_probe_unknown_end_without_sacct(tmp_path):
    fake = FakeSlurm([SQUEUE_RUNNING, "", "", ""], sacct=None)
    clock = [3_000_000.0]
    probe = SchedulerProbe(SlurmAdapter(fake), str(tmp_path), user=None, interval_s=30, host="l", now=lambda: clock[0])
    probe.poll()
    clock[0] += 30
    assert probe.poll() == []  # grace period: accounting may lag
    clock[0] += 60
    e = probe.poll()
    assert len(e) == 1 and e[0]["data"]["state"] == "UNKNOWN_ENDED" and e[0]["data"]["terminal"] is True


def test_probe_requeue_makes_new_run_dir(tmp_path):
    scontrol_r1 = SCONTROL.replace("Restarts=0", "Restarts=1").replace("JobState=RUNNING", "JobState=PENDING")
    fake = FakeSlurm([SQUEUE_RUNNING, SQUEUE_PENDING.replace("Priority", "BeginTime")], scontrol=SCONTROL)
    clock = [4_000_000.0]
    probe = SchedulerProbe(SlurmAdapter(fake), str(tmp_path), user=None, interval_s=30, host="l", now=lambda: clock[0])
    probe.poll()
    fake.scontrol = scontrol_r1
    clock[0] += 30
    e = probe.poll()
    assert len(e) == 1 and e[0]["run_id"] == "marcel-c3/8224458/1" and e[0]["data"]["restarts"] == 1
