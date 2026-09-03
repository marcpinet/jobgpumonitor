"""Slurm adapter: ``squeue`` for the queue, ``scontrol`` for details, ``sacct`` for the end.

Only plain-text outputs with explicit formats are used, so it works on Slurm releases
older than the ``--json`` support (21.08) as well as on recent ones.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

from .._log import dbg
from .base import (
    JobInfo,
    SchedulerAdapter,
    normalise_state,
    parse_datetime,
    parse_duration,
    parse_exit_code,
    parse_int,
    parse_mem_kb_suffix,
)

# %i jobid, %F array job id, %K array task id, %j name, %T state, %r reason, %P partition,
# %q qos, %M time used, %l time limit, %S start, %e end, %V submit, %N nodelist, %D nodes,
# %C cpus, %m min mem, %b tres per node, %u user, %a account, %Q priority
_SQUEUE_FMT = "%i|%F|%K|%j|%T|%r|%P|%q|%M|%l|%S|%e|%V|%N|%D|%C|%m|%b|%u|%a|%Q"
_SQUEUE_FIELDS = (
    "job_id", "array_job_id", "array_task_id", "job_name", "state", "reason", "partition", "qos",
    "time_used", "time_limit", "start", "end", "submit", "nodes", "num_nodes", "num_cpus", "mem",
    "gres", "user", "account", "priority",
)

_SACCT_FIELDS = (
    "JobID", "JobName", "State", "ExitCode", "DerivedExitCode", "Start", "End", "ElapsedRaw",
    "NodeList", "MaxRSS", "ReqMem", "TimelimitRaw", "Partition", "QOS", "Account", "Submit",
    "WorkDir", "Reason", "AllocTRES", "User", "NNodes", "NCPUS",
)

#: Human hints for the most common pending reasons (a consumer may show them verbatim).
REASON_HINTS = {
    "Priority": "Waiting behind higher-priority jobs; nothing wrong with the request.",
    "Resources": "Requested resources are all busy; the job is next in line for them.",
    "Dependency": "Waiting for another job (see dependency).",
    "BeginTime": "A --begin time was requested and has not come yet.",
    "ReqNodeNotAvail": "Required nodes are unavailable, often a scheduled maintenance reservation. Check `sinfo -R`.",
    "PartitionTimeLimit": "The --time asked exceeds the partition limit; the job will never start as is.",
    "PartitionNodeLimit": "More nodes requested than the partition allows.",
    "BadConstraints": "The --constraint / GPU type cannot be satisfied by any node in the partition.",
    "QOSMaxGRESPerUser": "You already use the maximum number of GPUs allowed per user by this QOS.",
    "QOSMaxJobsPerUserLimit": "You already run the maximum number of jobs allowed by this QOS.",
    "QOSMaxCpuPerUserLimit": "Per-user CPU limit of the QOS reached.",
    "QOSMaxMemoryPerUser": "Per-user memory limit of the QOS reached.",
    "QOSGrpGRES": "The QOS as a whole has no GPU left.",
    "AssocGrpGRES": "Your account's GPU quota is exhausted.",
    "AssocMaxJobsLimit": "Your account's concurrent-job limit is reached.",
    "JobHeldUser": "Held by you (scontrol release <id>).",
    "JobHeldAdmin": "Held by an administrator.",
    "Licenses": "Waiting for a software license.",
    "NodeDown": "A node the job needs is down.",
    "Reservation": "Waiting for a reservation to begin.",
    "InvalidQOS": "The QOS does not exist or you cannot use it.",
    "PartitionDown": "The partition is down.",
}


def _split_scontrol(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in re.split(r"\s+(?=[A-Za-z][\w/:.]*=)", line.strip()):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


class SlurmAdapter(SchedulerAdapter):
    name = "slurm"

    def __init__(self, run=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(run) if run is not None else super().__init__()
        self._cluster: Optional[str] = None
        self._sacct_ok: Optional[bool] = None

    # ------------------------------------------------------------------ cluster

    def cluster_name(self) -> Optional[str]:
        if self._cluster is not None:
            return self._cluster or None
        out = self.run(["scontrol", "show", "config"])
        name = ""
        if out:
            m = re.search(r"^ClusterName\s*=\s*(\S+)", out, re.M)
            if m:
                name = m.group(1)
        self._cluster = name
        return name or None

    # ------------------------------------------------------------------ queue

    def list_jobs(self, user: Optional[str]) -> Optional[List[JobInfo]]:
        argv = ["squeue", "-h", "-o", _SQUEUE_FMT]
        if user:
            argv += ["-u", user]
        out = self.run(argv)
        if out is None:
            return None
        jobs: List[JobInfo] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < len(_SQUEUE_FIELDS):
                dbg(f"squeue line ignored: {line[:120]}")
                continue
            r = dict(zip(_SQUEUE_FIELDS, parts))
            jobs.append(self._from_squeue(r))
        return jobs

    def _from_squeue(self, r: Dict[str, str]) -> JobInfo:
        job_id = r["job_id"].strip()
        array_job = r["array_job_id"].strip()
        task = r["array_task_id"].strip()
        pending_tasks = None
        array_job_id: Optional[str] = None
        array_task_id: Optional[str] = None
        m = re.match(r"^(\d+)_\[(.+)\]$", job_id)
        if m:  # pending, still aggregated: "1234_[1-100]"
            array_job_id, pending_tasks = m.group(1), "[" + m.group(2) + "]"
            key = array_job_id
        elif "_" in job_id:
            array_job_id, array_task_id = job_id.split("_", 1)
            key = job_id
        elif task not in ("", "N/A") and array_job and array_job != job_id:
            array_job_id, array_task_id = array_job, task
            key = f"{array_job}_{task}"
        else:
            key = job_id
        state = normalise_state(r["state"])
        reason = r["reason"].strip()
        if reason in ("None", ""):
            reason = None
        return JobInfo(
            scheduler="slurm",
            job_id=job_id,
            job_key=key,
            state=state,
            job_name=r["job_name"].strip() or None,
            user=r["user"].strip() or None,
            account=r["account"].strip() or None,
            array_job_id=array_job_id,
            array_task_id=array_task_id,
            array_pending_tasks=pending_tasks,
            state_reason=reason,
            partition=r["partition"].strip() or None,
            qos=r["qos"].strip() or None,
            nodes=r["nodes"].strip() or None,
            num_nodes=parse_int(r["num_nodes"]),
            num_cpus=parse_int(r["num_cpus"]),
            mem=r["mem"].strip() or None,
            gres=r["gres"].strip() if r["gres"].strip() not in ("N/A", "") else None,
            submit_ts=parse_datetime(r["submit"]),
            start_ts=parse_datetime(r["start"]),
            end_ts=parse_datetime(r["end"]),
            time_limit_s=parse_duration(r["time_limit"]),
            time_used_s=parse_duration(r["time_used"]),
            priority=parse_int(r["priority"]),
            extra={"reason_hint": REASON_HINTS.get(reason or "", None)} if reason else {},
        )

    # ------------------------------------------------------------------ details

    def enrich(self, job: JobInfo) -> JobInfo:
        out = self.run(["scontrol", "show", "job", "-o", job.job_id])
        if not out:
            return job
        line = out.strip().splitlines()[0] if out.strip() else ""
        kv = _split_scontrol(line)
        if not kv:
            return job
        job.restarts = parse_int(kv.get("Restarts")) or 0
        job.requeue = kv.get("Requeue") == "1" if "Requeue" in kv else None
        job.stdout = kv.get("StdOut") or job.stdout
        job.stderr = kv.get("StdErr") or job.stderr
        job.workdir = kv.get("WorkDir") or job.workdir
        job.command = kv.get("Command") or job.command
        job.dependency = kv.get("Dependency") if kv.get("Dependency") not in (None, "(null)") else None
        code, sig = parse_exit_code(kv.get("ExitCode"))
        if job.terminal:
            job.exit_code, job.exit_signal = code, sig
        if job.time_limit_s is None:
            job.time_limit_s = parse_duration(kv.get("TimeLimit"))
        if job.gres is None and kv.get("TresPerNode"):
            job.gres = kv["TresPerNode"]
        if job.mem is None and kv.get("MinMemoryNode"):
            job.mem = kv["MinMemoryNode"]
        job.extra["batch_host"] = kv.get("BatchHost")
        job.extra["alloc_tres"] = kv.get("AllocTRES") or kv.get("ReqTRES")
        return job

    # ------------------------------------------------------------------ accounting

    def finished(self, job_ids: List[str], since_ts: float, user: Optional[str]) -> Dict[str, JobInfo]:
        if not job_ids or self._sacct_ok is False:
            return {}
        since = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(since_ts - 3600))
        argv = ["sacct", "-n", "-P", "-S", since, "-j", ",".join(job_ids), "-o", ",".join(_SACCT_FIELDS)]
        if user:
            argv += ["-u", user]
        out = self.run(argv)
        if out is None:
            if self._sacct_ok is None:
                dbg("sacct unavailable; terminal states will be UNKNOWN_ENDED")
            self._sacct_ok = False
            return {}
        self._sacct_ok = True
        rows: Dict[str, Dict[str, str]] = {}
        steps: Dict[str, List[Dict[str, str]]] = {}
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) < len(_SACCT_FIELDS):
                continue
            r = dict(zip(_SACCT_FIELDS, parts))
            jid = r["JobID"]
            if "." in jid:
                steps.setdefault(jid.split(".", 1)[0], []).append(r)
            else:
                rows[jid] = r
        result: Dict[str, JobInfo] = {}
        for jid, r in rows.items():
            job = self._from_sacct(r, steps.get(jid, []))
            result[job.job_id] = job
        return result

    def _from_sacct(self, r: Dict[str, str], steps: List[Dict[str, str]]) -> JobInfo:
        job_id = r["JobID"].strip()
        array_job_id = array_task_id = None
        key = job_id
        if "_" in job_id and "[" not in job_id:
            array_job_id, array_task_id = job_id.split("_", 1)
        state = normalise_state(r["State"])
        code, sig = parse_exit_code(r["ExitCode"])
        dcode, _ = parse_exit_code(r["DerivedExitCode"])
        max_rss = None
        for s in steps:
            v = parse_mem_kb_suffix(s.get("MaxRSS"))
            if v is not None:
                max_rss = max(max_rss or 0, v)
        tl = parse_int(r["TimelimitRaw"])
        gres = None
        m = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", r.get("AllocTRES", ""))
        if m:
            gres = f"gpu:{m.group(1)}"
        reason = r["Reason"].strip()
        return JobInfo(
            scheduler="slurm",
            job_id=job_id,
            job_key=key,
            state=state,
            job_name=r["JobName"].strip() or None,
            user=r["User"].strip() or None,
            account=r["Account"].strip() or None,
            array_job_id=array_job_id,
            array_task_id=array_task_id,
            state_reason=reason if reason not in ("None", "") else None,
            partition=r["Partition"].strip() or None,
            qos=r["QOS"].strip() or None,
            nodes=r["NodeList"].strip() if r["NodeList"].strip() not in ("None assigned", "") else None,
            num_nodes=parse_int(r["NNodes"]),
            num_cpus=parse_int(r["NCPUS"]),
            mem=r["ReqMem"].strip() or None,
            gres=gres,
            submit_ts=parse_datetime(r["Submit"]),
            start_ts=parse_datetime(r["Start"]),
            end_ts=parse_datetime(r["End"]),
            time_limit_s=tl * 60 if tl is not None else None,
            elapsed_s=parse_int(r["ElapsedRaw"]),
            exit_code=code,
            exit_signal=sig,
            derived_exit_code=dcode,
            workdir=r["WorkDir"].strip() or None,
            max_rss=max_rss,
            extra={"cancelled_by": r["State"].strip()} if r["State"].strip().startswith("CANCELLED by") else {},
        )
