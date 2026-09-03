"""The probe loop: poll the scheduler, emit ``scheduler.state`` on change, remember what was seen.

Events go to the same run directories as the in-process emitters
(``$JGM_DIR/runs/<cluster>/<job key>/<restart>/scheduler-<host>.jsonl``), so a consumer joins
both sources on ``run_id`` without any lookup table. A small JSON state file makes the probe
restart-safe: no duplicate terminal events, stdout paths kept after ``scontrol`` forgets the job.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import threading
import time
from typing import Any, Dict, List, Optional

from .. import events as ev
from .._log import dbg, warn
from ..sinks import FileSink, run_dir_for
from .base import JobInfo, SchedulerAdapter, run_command

_SANITISE = str.maketrans({c: "_" for c in "/\\ \t\n"})


def detect_adapter(prefer: Optional[str] = None, run=run_command) -> Optional[SchedulerAdapter]:  # type: ignore[no-untyped-def]
    from .oar import OarAdapter
    from .slurm import SlurmAdapter

    candidates = {"slurm": (SlurmAdapter, "squeue"), "oar": (OarAdapter, "oarstat")}
    order = [prefer] if prefer in candidates else list(candidates)
    for name in order:
        cls, binary = candidates[name]
        if shutil.which(binary) or prefer == name:
            return cls(run)
    return None


class SchedulerProbe:
    def __init__(
        self,
        adapter: SchedulerAdapter,
        base_dir: str,
        user: Optional[str],
        cluster: Optional[str] = None,
        interval_s: float = 30.0,
        refresh_s: float = 600.0,
        host: Optional[str] = None,
        now=time.time,  # type: ignore[no-untyped-def]
    ) -> None:
        self.adapter = adapter
        self.base_dir = base_dir
        self.user = user
        self.interval_s = max(5.0, interval_s)
        self.refresh_s = refresh_s
        self.now = now
        self.host = (host or socket.gethostname().split(".")[0]).translate(_SANITISE)
        self.cluster = (cluster or adapter.cluster_name() or adapter.name).translate(_SANITISE)
        self.emitter = f"scheduler-{self.host}"
        self.pid = os.getpid()
        self.state_path = os.path.join(base_dir, "scheduler", f"state-{self.cluster}.json")
        self.heartbeat_path = os.path.join(base_dir, "scheduler", f"probe-{self.cluster}-{self.host}.json")
        self.state: Dict[str, Any] = {"jobs": {}, "version": 1}
        self._seq: Dict[str, int] = {}
        self._sinks: Dict[str, FileSink] = {}
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self.polls = 0
        self.emitted = 0
        self._load_state()

    # ------------------------------------------------------------------ state file

    def _load_state(self) -> None:
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("jobs"), dict):
                self.state = data
        except (OSError, ValueError):
            pass

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.state, f)
            os.replace(tmp, self.state_path)
        except OSError as e:
            dbg(f"state save failed: {e}")

    def _write_heartbeat(self, extra: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self.heartbeat_path), exist_ok=True)
            tmp = self.heartbeat_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({
                    "ts": ev.utc_iso(self.now()), "host": self.host, "pid": self.pid, "cluster": self.cluster,
                    "scheduler": self.adapter.name, "user": self.user, "interval_s": self.interval_s,
                    "polls": self.polls, "emitted": self.emitted, "uptime_s": round(time.monotonic() - self._t0, 1),
                    **extra,
                }, f)
            os.replace(tmp, self.heartbeat_path)
        except OSError as e:
            dbg(f"heartbeat write failed: {e}")

    # ------------------------------------------------------------------ emission

    def run_id_for(self, job: JobInfo) -> str:
        return f"{self.cluster}/{job.job_key.translate(_SANITISE)}/{job.restarts}"

    def _sink(self, run_id: str) -> FileSink:
        s = self._sinks.get(run_id)
        if s is None:
            if len(self._sinks) > 256:
                for k in list(self._sinks)[:64]:
                    self._sinks.pop(k).close()
            path = os.path.join(run_dir_for(self.base_dir, run_id), self.emitter + ".jsonl")
            s = FileSink(path, fsync=True)
            self._sinks[run_id] = s
        return s

    def emit(self, job: JobInfo, change: str) -> Dict[str, Any]:
        run_id = self.run_id_for(job)
        seq = self._seq.get(run_id, 0)
        self._seq[run_id] = seq + 1
        data = job.to_data()
        data["change"] = change  # first_seen | state | detail | refresh | ended
        data["probe"] = {"host": self.host, "poll_ts": self.now(), "scheduler": self.adapter.name}
        env = ev.make_envelope(
            run_id=run_id, emitter=self.emitter, pid=self.pid, source="scheduler", rank=None,
            seq=seq, etype="scheduler.state", data=data, mono=time.monotonic() - self._t0, ts=self.now(),
        )
        self._sink(run_id).write(ev.to_json(env))
        self.emitted += 1
        return env

    # ------------------------------------------------------------------ polling

    def poll(self) -> List[Dict[str, Any]]:
        """One pass. Returns the events emitted (for tests and ``--once``)."""
        self.polls += 1
        now = self.now()
        emitted: List[Dict[str, Any]] = []
        jobs = self.adapter.list_jobs(self.user)
        if jobs is None:
            warn(f"{self.adapter.name}: queue listing failed; will retry")
            self._write_heartbeat({"ok": False})
            return emitted
        known: Dict[str, Any] = self.state["jobs"]
        seen_ids = set()
        for job in jobs:
            seen_ids.add(job.job_id)
            rec = known.get(job.job_id)
            # keep details that only scontrol has and that we captured earlier
            if rec:
                job.restarts = rec.get("restarts", job.restarts) if job.restarts == 0 else job.restarts
                for f in ("stdout", "stderr", "workdir", "command"):
                    if getattr(job, f) is None and rec.get(f):
                        setattr(job, f, rec[f])
            need_detail = rec is None or rec.get("state") != job.state or not rec.get("stdout")
            if need_detail:
                try:
                    job = self.adapter.enrich(job)
                except Exception as e:  # never let one job break the loop
                    dbg(f"enrich {job.job_id} failed: {e}")
            sig = job.signature()
            if rec is None:
                change = "first_seen"
            elif rec.get("sig") != sig:
                change = "state" if rec.get("state") != job.state or rec.get("restarts") != job.restarts else "detail"
            elif now - rec.get("last_emit_ts", 0) >= self.refresh_s and job.active:
                change = "refresh"
            else:
                change = ""
            if change:
                emitted.append(self.emit(job, change))
                last_emit = now
            else:
                last_emit = rec.get("last_emit_ts", now)
            known[job.job_id] = {
                "sig": sig, "state": job.state, "restarts": job.restarts, "job_key": job.job_key,
                "stdout": job.stdout, "stderr": job.stderr, "workdir": job.workdir, "command": job.command,
                "job_name": job.job_name, "first_seen_ts": (rec or {}).get("first_seen_ts", now),
                "last_seen_ts": now, "last_emit_ts": last_emit, "ended": False,
            }
        # jobs that left the queue
        gone = [jid for jid, rec in known.items() if jid not in seen_ids and not rec.get("ended")]
        if gone:
            since = min(known[j].get("first_seen_ts", now) for j in gone)
            try:
                final = self.adapter.finished(gone, since, self.user)
            except Exception as e:
                dbg(f"finished lookup failed: {e}")
                final = {}
            for jid in gone:
                rec = known[jid]
                job = final.get(jid)
                if job is None:
                    if now - rec.get("last_seen_ts", now) < 2 * self.interval_s and self.polls > 1:
                        continue  # accounting may lag a little behind the queue
                    job = JobInfo(scheduler=self.adapter.name, job_id=jid, job_key=rec.get("job_key", jid),
                                  state="UNKNOWN_ENDED", job_name=rec.get("job_name"), restarts=rec.get("restarts", 0),
                                  extra={"note": "left the queue; accounting unavailable"})
                job.restarts = rec.get("restarts", 0) if not job.restarts else job.restarts
                job.job_key = rec.get("job_key") or job.job_key
                for f in ("stdout", "stderr", "workdir", "command", "job_name"):
                    if getattr(job, f) is None and rec.get(f):
                        setattr(job, f, rec[f])
                emitted.append(self.emit(job, "ended"))
                rec.update(ended=True, state=job.state, last_emit_ts=now, ended_ts=now)
        # forget finished jobs after a day
        for jid in [j for j, r in known.items() if r.get("ended") and now - r.get("ended_ts", now) > 86400]:
            del known[jid]
        self._save_state()
        self._write_heartbeat({"ok": True, "queued": len(jobs), "emitted_this_poll": len(emitted)})
        return emitted

    def run_forever(self) -> None:
        def stop(signum, frame):  # type: ignore[no-untyped-def]
            self._stop.set()

        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                signal.signal(s, stop)
            except (ValueError, OSError):
                pass
        while not self._stop.is_set():
            t = time.monotonic()
            try:
                self.poll()
            except Exception as e:
                warn(f"poll failed: {e}")
            self._stop.wait(max(1.0, self.interval_s - (time.monotonic() - t)))
        self.close()

    def close(self) -> None:
        for s in self._sinks.values():
            s.close()
        self._sinks.clear()
