"""OAR adapter (best effort, not validated on a real OAR cluster).

Uses ``oarstat -J`` (JSON) for the queue and ``oarstat -fj <id> -J`` for details. OAR keeps
terminated jobs visible in ``oarstat -J`` for a while and exposes them through
``oarstat -j <id> -J`` afterwards, which is what ``finished`` relies on.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .._log import dbg
from .base import JobInfo, SchedulerAdapter, normalise_state, parse_int

_STATE_MAP = {
    "WAITING": "PENDING", "TOLAUNCH": "PENDING", "TOACKLAUNCH": "PENDING", "HOLD": "PENDING",
    "LAUNCHING": "LAUNCHING", "RUNNING": "RUNNING", "FINISHING": "COMPLETING",
    "TERMINATED": "COMPLETED", "ERROR": "FAILED", "SUSPENDED": "SUSPENDED", "RESUMING": "RUNNING",
}


class OarAdapter(SchedulerAdapter):
    name = "oar"

    def cluster_name(self) -> Optional[str]:
        return None

    def _query(self, argv: List[str]) -> Optional[Dict[str, Any]]:
        out = self.run(argv)
        if not out:
            return None
        try:
            data = json.loads(out)
        except ValueError as e:
            dbg(f"oarstat json parse failed: {e}")
            return None
        return data if isinstance(data, dict) else None

    def list_jobs(self, user: Optional[str]) -> Optional[List[JobInfo]]:
        argv = ["oarstat", "-J"]
        if user:
            argv += ["-u", user]
        data = self._query(argv)
        if data is None:
            return None
        return [self._from_json(jid, j) for jid, j in data.items() if isinstance(j, dict)]

    def _from_json(self, jid: str, j: Dict[str, Any]) -> JobInfo:
        raw_state = str(j.get("state", "")).upper()
        state = _STATE_MAP.get(raw_state, normalise_state(raw_state))
        if state == "COMPLETED" and parse_int(str(j.get("exit_code", ""))) not in (None, 0):
            state = "FAILED"
        exit_raw = parse_int(str(j.get("exit_code", "")))
        # OAR encodes the exit code like the shell: code << 8 | signal
        exit_code = (exit_raw >> 8) if exit_raw is not None else None
        exit_signal = (exit_raw & 0x7F) or None if exit_raw is not None else None
        nodes = j.get("assigned_network_address") or j.get("nodes") or []
        walltime = parse_int(str(j.get("walltime", ""))) if str(j.get("walltime", "")).isdigit() else None
        start = j.get("startTime") or j.get("start_time")
        stop = j.get("stopTime") or j.get("stop_time")
        start_ts = float(start) if isinstance(start, (int, float)) and start else None
        end_ts = float(stop) if isinstance(stop, (int, float)) and stop else None
        if end_ts is None and start_ts and walltime:
            end_ts = start_ts + walltime
        array_id = str(j.get("array_id") or "") or None
        array_index = str(j.get("array_index") or "") or None
        key = f"{array_id}_{array_index}" if array_id and array_index and array_id != jid else str(jid)
        props = j.get("properties") or ""
        return JobInfo(
            scheduler="oar",
            job_id=str(jid),
            job_key=key,
            state=state,
            job_name=j.get("name") or None,
            user=j.get("owner") or None,
            array_job_id=array_id if array_id != str(jid) else None,
            array_task_id=array_index,
            state_reason=j.get("reason") or None,
            partition=j.get("queue") or None,
            nodes=",".join(nodes) if isinstance(nodes, list) else str(nodes) or None,
            num_nodes=len(nodes) if isinstance(nodes, list) and nodes else None,
            submit_ts=float(j["submissionTime"]) if isinstance(j.get("submissionTime"), (int, float)) else None,
            start_ts=start_ts,
            end_ts=end_ts,
            time_limit_s=walltime,
            exit_code=exit_code if state in ("COMPLETED", "FAILED") else None,
            exit_signal=exit_signal if state in ("COMPLETED", "FAILED") else None,
            stdout=j.get("stdout_file") or None,
            stderr=j.get("stderr_file") or None,
            workdir=j.get("launchingDirectory") or None,
            command=j.get("command") or None,
            extra={"types": j.get("types"), "properties": props, "raw_state": raw_state},
        )

    def enrich(self, job: JobInfo) -> JobInfo:
        data = self._query(["oarstat", "-fj", job.job_id, "-J"])
        if not data:
            return job
        j = data.get(job.job_id) or next(iter(data.values()), None)
        if isinstance(j, dict):
            full = self._from_json(job.job_id, j)
            for f in ("stdout", "stderr", "workdir", "command", "time_limit_s", "submit_ts"):
                if getattr(job, f) is None:
                    setattr(job, f, getattr(full, f))
        return job

    def finished(self, job_ids: List[str], since_ts: float, user: Optional[str]) -> Dict[str, JobInfo]:
        out: Dict[str, JobInfo] = {}
        for jid in job_ids:
            data = self._query(["oarstat", "-j", jid, "-J"])
            if not data:
                continue
            j = data.get(jid) or next(iter(data.values()), None)
            if isinstance(j, dict):
                info = self._from_json(jid, j)
                if info.terminal:
                    out[jid] = info
        return out
