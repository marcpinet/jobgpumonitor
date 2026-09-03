"""Scheduler adapters: a common ``JobInfo`` and the parsing helpers they share.

Adapters run *scheduler commands* (squeue, sacct, oarstat...) and therefore only ever run
on a login node, never inside the job. They are used by ``jgm scheduler`` (the probe).
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .._log import dbg

#: States after which a job will never run again (the union of Slurm and OAR vocabularies).
TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
    "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT", "UNKNOWN_ENDED", "TERMINATED", "ERROR",
}
#: States that mean the job is currently consuming an allocation.
ACTIVE_STATES = {"RUNNING", "COMPLETING", "STAGE_OUT", "SUSPENDED", "SIGNALING", "LAUNCHING", "FINISHING"}

CommandRunner = Callable[[List[str]], Optional[str]]


def run_command(argv: List[str], timeout: float = 30.0) -> Optional[str]:
    """Run a scheduler command; ``None`` on any failure (missing binary, timeout, rc != 0)."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        dbg(f"{argv[0]} failed: {e}")
        return None
    if p.returncode != 0:
        dbg(f"{argv[0]} rc={p.returncode}: {p.stderr.strip()[:300]}")
        return None
    return p.stdout


@dataclass
class JobInfo:
    scheduler: str
    job_id: str                      # allocation id as the scheduler prints it ("1234", "1234_7")
    job_key: str                     # run_id component: "1234" or "<array>_<task>"
    state: str                       # normalised upper-case state
    job_name: Optional[str] = None
    user: Optional[str] = None
    account: Optional[str] = None
    array_job_id: Optional[str] = None
    array_task_id: Optional[str] = None
    array_pending_tasks: Optional[str] = None   # "[1-100]" while the array is still aggregated
    state_reason: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    nodes: Optional[str] = None
    num_nodes: Optional[int] = None
    num_cpus: Optional[int] = None
    mem: Optional[str] = None
    gres: Optional[str] = None
    submit_ts: Optional[float] = None
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None            # actual end for terminal jobs, expected end otherwise
    time_limit_s: Optional[int] = None
    time_used_s: Optional[int] = None
    restarts: int = 0
    requeue: Optional[bool] = None
    exit_code: Optional[int] = None
    exit_signal: Optional[int] = None
    derived_exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    workdir: Optional[str] = None
    command: Optional[str] = None
    priority: Optional[int] = None
    dependency: Optional[str] = None
    max_rss: Optional[int] = None
    elapsed_s: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def signature(self) -> str:
        """What must change for a new event to be worth emitting."""
        return "|".join(str(x) for x in (
            self.state, self.state_reason, self.nodes, self.start_ts, self.restarts, self.end_ts,
            self.exit_code, self.array_pending_tasks, self.priority,
        ))

    def to_data(self) -> Dict[str, Any]:
        d = asdict(self)
        d["terminal"] = self.terminal
        d["active"] = self.active
        return d


class SchedulerAdapter:
    name = "base"

    def __init__(self, run: CommandRunner = run_command) -> None:
        self.run = run

    def cluster_name(self) -> Optional[str]:  # pragma: no cover - interface
        return None

    def list_jobs(self, user: Optional[str]) -> Optional[List[JobInfo]]:
        """Jobs currently known to the scheduler (queued or running). ``None`` if the command failed."""
        raise NotImplementedError

    def enrich(self, job: JobInfo) -> JobInfo:
        """Add details that the listing does not carry (stdout paths, restarts...)."""
        return job

    def finished(self, job_ids: List[str], since_ts: float, user: Optional[str]) -> Dict[str, JobInfo]:
        """Final state of jobs that left the queue, keyed by job_id. Empty if unsupported."""
        return {}


# --------------------------------------------------------------------------- parsing helpers

_NA = {"", "N/A", "NONE", "(null)", "Unknown", "UNLIMITED", "INVALID", "NOT_SET", "None", "n/a"}


def parse_duration(v: Optional[str]) -> Optional[int]:
    """``D-HH:MM:SS``, ``HH:MM:SS``, ``MM:SS``, ``MM`` -> seconds. Unlimited/unknown -> None."""
    if v is None or v.strip() in _NA:
        return None
    v = v.strip()
    days = 0
    if "-" in v:
        d, v = v.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            return None
    parts = v.split(":")
    try:
        nums = [int(float(p)) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, s = nums
    elif len(nums) == 2:
        h, (m, s) = 0, nums
    elif len(nums) == 1:
        h, m, s = 0, nums[0], 0
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + s


def parse_datetime(v: Optional[str]) -> Optional[float]:
    """Scheduler local-time stamps like ``2026-09-03T11:55:05`` -> epoch seconds."""
    if v is None or v.strip() in _NA:
        return None
    v = v.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return time.mktime(time.strptime(v, fmt))
        except ValueError:
            continue
    if re.fullmatch(r"\d{9,11}", v):
        return float(v)
    return None


def parse_int(v: Optional[str]) -> Optional[int]:
    if v is None or v.strip() in _NA:
        return None
    try:
        return int(float(v.strip()))
    except ValueError:
        return None


def parse_exit_code(v: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """Slurm ``exit:signal`` -> (exit_code, signal)."""
    if v is None or v.strip() in _NA:
        return None, None
    parts = v.strip().split(":")
    try:
        code = int(parts[0])
        sig = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None, None
    return code, (sig or None)


def parse_mem_kb_suffix(v: Optional[str]) -> Optional[int]:
    """sacct MaxRSS style ``123456K`` / ``1.5G`` -> bytes."""
    if v is None or v.strip() in _NA:
        return None
    s = v.strip().upper()
    mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    try:
        if s and s[-1] in mult:
            return int(float(s[:-1]) * mult[s[-1]])
        return int(float(s))
    except ValueError:
        return None


def normalise_state(v: Optional[str]) -> str:
    """``CANCELLED by 123`` -> ``CANCELLED``; ``OUT_OF_ME+`` -> ``OUT_OF_MEMORY``."""
    if not v:
        return "UNKNOWN"
    s = v.strip().upper().split()[0].rstrip("+")
    aliases = {"OUT_OF_ME": "OUT_OF_MEMORY", "CANCELLE": "CANCELLED", "COMPLETIN": "COMPLETING", "PD": "PENDING",
               "R": "RUNNING", "CG": "COMPLETING", "CD": "COMPLETED", "F": "FAILED", "TO": "TIMEOUT",
               "OOM": "OUT_OF_MEMORY", "NF": "NODE_FAIL", "PR": "PREEMPTED", "CA": "CANCELLED", "S": "SUSPENDED",
               "RQ": "REQUEUED", "BF": "BOOT_FAIL", "DL": "DEADLINE", "RV": "REVOKED"}
    return aliases.get(s, s)
