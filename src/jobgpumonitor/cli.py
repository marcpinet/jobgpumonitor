"""``jgm`` command line: run (wrapper), emit (from shell), doctor, ls, version."""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional

from . import __version__
from ._log import dbg
from .config import Config
from .context import build_context
from .runtime import Run
from .sinks import resolve_base_dir

_FORWARD = ("SIGTERM", "SIGHUP", "SIGUSR1", "SIGUSR2", "SIGQUIT", "SIGXCPU")


# --------------------------------------------------------------------------- jgm run


def cmd_run(args: argparse.Namespace) -> int:
    cmd: List[str] = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("usage: jgm run [--name NAME] -- <command> [args...]", file=sys.stderr)
        return 2
    cfg = Config.from_env()
    ctx = build_context(cfg, source="wrapper")
    ctx["argv"] = cmd
    if args.name:
        ctx["job"]["job_name"] = ctx["job"].get("job_name") or args.name
    env = dict(os.environ)
    env["JGM_WRAPPED"] = "1"
    hook_dir: Optional[str] = None
    if not args.no_auto:
        # Every Python the command starts monitors itself (tqdm, metrics, traceback...),
        # without pip install nor a line of code: JGM_AUTO + a start-up hook on PYTHONPATH.
        env.setdefault("JGM_AUTO", "1")
        hook_dir = _install_auto_hook(env)

    if not args.keep_buffering:
        env.setdefault("PYTHONUNBUFFERED", "1")  # otherwise stdout of a batch job shows up hours late

    tail: Deque[str] = collections.deque(maxlen=args.tail_lines)
    try:
        child = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=args.cwd or None)
    except OSError as e:
        print(f"jgm run: cannot start {cmd[0]!r}: {e}", file=sys.stderr)
        _remove_auto_hook(hook_dir)
        return 127

    run = Run(cfg, ctx, source="wrapper", hooks=False, monitor=True, probe_pid=child.pid)
    run.ctx["child_pid"] = child.pid
    run.start(emit_start=False)
    payload = run._start_payload()
    payload["command"] = cmd
    payload["child_pid"] = child.pid
    payload["name"] = args.name
    run.emit("run.start", payload)

    chunks = _ChunkEmitter(run)

    def tee(stream: str, src: Any, dst: Any) -> None:
        """Pass the child's output through unchanged, as it arrives (tqdm redraws lines with
        ``\\r`` and no newline: a line-based read would hold them back). stderr also feeds a
        tail of whole lines, carriage returns folded to the last state of the line; both
        streams are emitted as ``log.chunk`` so the dashboard shows them live."""
        out = getattr(dst, "buffer", None)
        pending = b""
        while True:
            chunk = src.read1(65536) if hasattr(src, "read1") else src.read(1)
            if not chunk:
                break
            try:
                if out is not None:
                    out.write(chunk)
                    out.flush()
                else:
                    dst.write(chunk.decode("utf-8", "replace"))
            except Exception:
                pass
            chunks.feed(stream, chunk)
            if stream == "stderr":
                pending += chunk
                *lines, pending = pending.split(b"\n")
                for line in lines:
                    tail.append(_fold_cr(line))
                if len(pending) > 65536:
                    pending = pending[-65536:]
        if stream == "stderr" and pending.strip():
            tail.append(_fold_cr(pending))
        chunks.close(stream)

    tees = [threading.Thread(target=tee, args=("stdout", child.stdout, sys.stdout), name="jgm-stdout-tee", daemon=True),
            threading.Thread(target=tee, args=("stderr", child.stderr, sys.stderr), name="jgm-stderr-tee", daemon=True)]
    for t in tees:
        t.start()
    pusher = _RunPusher(run) if not args.no_push else None
    if pusher:
        pusher.start()

    forwarded: Dict[int, str] = {}

    def forward(signum: int, frame: Any) -> None:
        name = signal.Signals(signum).name
        forwarded[signum] = name
        run.emit("signal.received", {"signal": name, "signum": signum, "forwarded": not args.no_forward,
                                     "deadline_remaining_s": run.deadline_remaining_s()})
        if not args.no_forward and child.poll() is None:
            try:
                os.kill(child.pid, signum)
            except OSError:
                pass

    for name in _FORWARD:
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, forward)
            except (ValueError, OSError):
                pass
    try:
        signal.signal(signal.SIGINT, lambda s, f: run.emit("signal.received", {"signal": "SIGINT", "signum": int(s), "forwarded": False}))
    except (ValueError, OSError):
        pass

    rc = child.wait()
    for t in tees:
        t.join(5.0)
    chunks.flush(final=True)
    sig_name: Optional[str] = None
    if rc < 0:
        try:
            sig_name = signal.Signals(-rc).name
        except ValueError:
            sig_name = str(-rc)
        status = "killed"
        exit_code = 128 - rc
    else:
        status = "ok" if rc == 0 else "error"
        exit_code = rc
    run.finish(
        status=status,
        exit_code=exit_code,
        signal=sig_name,
        command=cmd,
        stderr_tail="\n".join(tail)[-16000:],
        forwarded_signals=sorted(forwarded.values()),
    )
    run.close(3.0)
    if pusher:
        pusher.stop()  # last shipment: everything this job wrote, children included
    _remove_auto_hook(hook_dir)
    return exit_code


class _ChunkEmitter:
    """Batches the child's stdout/stderr into ``log.chunk`` events (every 2 s or 64 KB)."""

    def __init__(self, run: Run, max_bytes: int = 8 * 1024 * 1024) -> None:
        self.run = run
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._buf: Dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        self._offset: Dict[str, int] = {"stdout": 0, "stderr": 0}
        self._done: Dict[str, bool] = {}
        self._truncated: Dict[str, bool] = {}
        self._last = time.monotonic()

    def feed(self, stream: str, data: bytes) -> None:
        with self._lock:
            if self._truncated.get(stream):
                return
            self._buf[stream] += data
            due = len(self._buf[stream]) >= 65536 or time.monotonic() - self._last >= 2.0
        if due:
            self.flush()

    def close(self, stream: str) -> None:
        with self._lock:
            self._done[stream] = True

    def flush(self, final: bool = False) -> None:
        with self._lock:
            self._last = time.monotonic()
            for stream in ("stdout", "stderr"):
                buf = self._buf[stream]
                if not buf and not (final and self._done.get(stream)):
                    continue
                data = bytes(buf)
                if not final:
                    cut = max(data.rfind(b"\n"), data.rfind(b"\r")) + 1  # whole lines only while running
                    if cut <= 0:
                        continue
                    data = data[:cut]
                del buf[: len(data)]
                offset = self._offset[stream]
                if offset + len(data) > self.max_bytes:
                    data = data[: max(0, self.max_bytes - offset)]
                    self._truncated[stream] = True
                self._offset[stream] += len(data)
                self.run.emit("log.chunk", {
                    "stream": stream, "path": f"<{stream}>", "offset": offset,
                    "text": data.decode("utf-8", "replace"), "size": self._offset[stream],
                    "truncated": bool(self._truncated.get(stream)), "eof": bool(final and self._done.get(stream)),
                })


class _RunPusher(threading.Thread):
    """Ships this run's directory to the configured server every few seconds, and once more at exit."""

    def __init__(self, run: Run, interval_s: float = 10.0) -> None:
        super().__init__(name="jgm-push", daemon=True)
        self.jrun = run
        self.interval_s = interval_s
        self._halt = threading.Event()
        self.fw = None
        remote = _remote_for(run)
        if remote and run.base_dir and run.run_dir:
            from .forward import Forwarder

            self.fw = Forwarder(run.base_dir, remote[0], remote[1], only_dir=run.run_dir, persist=False)

    def run_once(self) -> None:
        if self.fw is None:
            return
        try:
            self.jrun.flush(2.0)
            self.fw.cycle()
        except Exception as e:  # never disturb the job
            dbg(f"push failed: {e}")

    def run(self) -> None:  # noqa: D102
        if self.fw is None:
            return
        while not self._halt.wait(self.interval_s):
            self.run_once()

    def stop(self) -> None:
        self._halt.set()
        self.join(2.0)
        self.run_once()


def _remote_for(run: Run):  # type: ignore[no-untyped-def]
    from . import remote

    return remote.load(run.base_dir)


def _fold_cr(line: bytes) -> str:
    """``b"10%\\r20%\\r30%"`` -> ``"30%"``: what the terminal would show at the end."""
    parts = [p for p in line.split(b"\r") if p]
    text = (parts[-1] if parts else b"").decode("utf-8", "replace")
    return text[:2000]


def _install_auto_hook(env: Dict[str, str]) -> Optional[str]:
    """Make ``import jobgpumonitor`` and the start-up hook reachable by the child.

    Adds to the child's PYTHONPATH: the zipapp or ``src`` checkout we run from (not a
    site-packages dir, which would leak this environment into another interpreter), and
    a temporary directory holding a ``sitecustomize.py`` that runs the hook and then hands
    over to the environment's own sitecustomize, if it has one.
    """
    import tempfile

    entries: List[str] = []
    try:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # zipapp file or src dir
        if os.path.basename(here) not in ("site-packages", "dist-packages"):
            entries.append(here)
    except Exception:
        pass
    hook_dir: Optional[str] = None
    try:
        hook_dir = tempfile.mkdtemp(prefix="jgm-hook-")
        with open(os.path.join(hook_dir, "sitecustomize.py"), "w", encoding="utf-8") as f:
            f.write(_SITECUSTOMIZE)
        entries.append(hook_dir)
    except OSError:
        hook_dir = None
    if entries:
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(entries + ([prev] if prev else []))
    return hook_dir


def _remove_auto_hook(hook_dir: Optional[str]) -> None:
    if hook_dir:
        import shutil

        shutil.rmtree(hook_dir, ignore_errors=True)


_SITECUSTOMIZE = """\
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
try:
    import jobgpumonitor._autohook  # noqa: F401
except Exception:
    pass
try:
    import importlib.machinery, importlib.util
    _others = [p for p in sys.path if os.path.abspath(p) != _here]
    _spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _others)
    if _spec is not None and _spec.origin != __file__ and _spec.loader is not None:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
except Exception:
    pass
"""


# --------------------------------------------------------------------------- jgm emit


def _parse_kv(items: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            out[it] = True
            continue
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def cmd_emit(args: argparse.Namespace) -> int:
    cfg = Config.from_env()
    ctx = build_context(cfg, source="shell")
    ctx["emitter"] = f"shell-{ctx['host']}"
    run = Run(cfg, ctx, source="shell", hooks=False, monitor=False)
    run.start(emit_start=False)
    data = _parse_kv(args.kv)
    if args.message:
        data.setdefault("message", args.message)
    ok = run.emit(args.type, data)
    run.flush(5.0)
    run.close(3.0)
    if not ok:
        print("jgm emit: nothing written (disabled or no sink)", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- jgm doctor


def cmd_doctor(args: argparse.Namespace) -> int:
    from .context import parse_mem_bytes
    from .probes import CgroupProbe, GpuProbe

    cfg = Config.from_env()
    ctx = build_context(cfg, source="process")
    base, origin = resolve_base_dir(cfg.dir)
    gpu = GpuProbe()
    cg = CgroupProbe(fallback_limit=parse_mem_bytes(ctx["job"].get("mem")))
    try:
        import psutil  # type: ignore # noqa: F401

        has_psutil = True
    except Exception:
        has_psutil = False
    try:
        import importlib.util

        has_tqdm = importlib.util.find_spec("tqdm") is not None
        has_nvml = importlib.util.find_spec("pynvml") is not None
    except Exception:
        has_tqdm = has_nvml = False
    report: Dict[str, Any] = {
        "version": __version__,
        "python": ctx["python"],
        "enabled": cfg.enabled,
        "run_id": ctx["run_id"],
        "emitter": ctx["emitter"],
        "scheduler": ctx["job"]["name"],
        "job": {k: v for k, v in ctx["job"].items() if v not in (None, False, 0)},
        "rank": ctx["rank"],
        "container": ctx["container"],
        "interactive": ctx["interactive"],
        "deadline": ctx["deadline"],
        "event_dir": base,
        "event_dir_origin": origin,
        "sinks": list(cfg.sinks),
        "gpu_backend": gpu.backend,
        "gpus": gpu.static()["devices"],
        "cgroup": cg.limits(),
        "psutil": has_psutil,
        "nvidia_ml_py": has_nvml,
        "tqdm": has_tqdm,
        "git": ctx["git"],
        "stdio": ctx["stdio"],
        "hooks": {"signals": cfg.signals, "tqdm": cfg.tqdm, "logging": cfg.logging_level or None, "faulthandler": cfg.faulthandler},
    }
    gpu.close()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    print(f"jobgpumonitor {__version__} on Python {report['python']}")
    print(f"  enabled       : {cfg.enabled}")
    print(f"  scheduler     : {report['scheduler']}  job={report['job'].get('job_id')} name={report['job'].get('job_name')}")
    print(f"  run_id        : {report['run_id']}")
    print(f"  rank          : {report['rank'] or 'not distributed'}")
    print(f"  container     : {report['container'] or 'none'}")
    print(f"  event dir     : {base or 'NONE (set JGM_DIR)'}  [{origin}]")
    print(f"  sinks         : {', '.join(cfg.sinks)}")
    print(f"  gpu backend   : {gpu.backend or 'none'}  ({len(report['gpus'])} visible device(s))")
    for d in report["gpus"]:
        print(f"      [{d['index']}] {d['name']}  {d['mem_total'] // (1024 * 1024) if d.get('mem_total') else '?'} MiB  {d['uuid']}")
    lim = report["cgroup"]
    fb = lim.get("fallback_limit")
    fb_txt = f"; scheduler request {fb // (1024 * 1024)} MiB used as limit" if fb else ""
    if lim.get("available"):
        ml = lim.get("mem_limit")
        vis = f"{ml // (1024 * 1024)} MiB" if ml else "unlimited in the visible cgroup"
        print(f"  cgroup v{lim['version']}     : {vis}{fb_txt}")
    else:
        print(f"  cgroup        : not available{fb_txt}")
    print(f"  psutil        : {has_psutil}   nvidia-ml-py: {has_nvml}   tqdm: {has_tqdm}")
    if report["deadline"]:
        rem = report["deadline"]["end_ts"] - time.time()
        print(f"  deadline      : in {rem / 3600:.1f} h ({report['deadline']['source']})")
    else:
        print("  deadline      : unknown (no SLURM_JOB_END_TIME / OAR walltime in env)")
    g = report["git"]
    if g and g.get("commit"):
        print(f"  git           : {g.get('commit')} {g.get('branch')}{' (dirty)' if g.get('dirty') else ''}")
    elif g:
        print(f"  git           : repo at {g.get('root')} but no usable git binary")
    else:
        print("  git           : not a git checkout")
    print(f"  stdout        : {report['stdio'].get('stdout')}")
    if not base:
        print("\n!! No writable event directory. Set JGM_DIR to a shared, writable path.", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- jgm ls


def _last_line(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 65536)
            f.seek(size - chunk)
            data = f.read(chunk)
        lines = [ln for ln in data.split(b"\n") if ln.strip()]
        if not lines:
            return None
        return lines[-1].decode("utf-8", "replace")
    except OSError:
        return None


def cmd_ls(args: argparse.Namespace) -> int:
    cfg = Config.from_env()
    base = os.path.expanduser(args.dir) if args.dir else resolve_base_dir(cfg.dir)[0]
    if not base:
        print("no event directory", file=sys.stderr)
        return 1
    root = os.path.join(base, "runs")
    rows: List[Dict[str, Any]] = []
    for cluster in _listdir(root):
        for key in _listdir(os.path.join(root, cluster)):
            for restart in _listdir(os.path.join(root, cluster, key)):
                d = os.path.join(root, cluster, key, restart)
                files = [f for f in _listdir(d) if f.endswith(".jsonl")]
                latest: Optional[Dict[str, Any]] = None
                for f in files:
                    ln = _last_line(os.path.join(d, f))
                    if not ln:
                        continue
                    try:
                        e = json.loads(ln)
                    except ValueError:
                        continue
                    if latest is None or e.get("ts", "") > latest.get("ts", ""):
                        latest = e
                rows.append({"run_id": f"{cluster}/{key}/{restart}", "files": len(files), "last": latest, "dir": d})
    rows.sort(key=lambda r: (r["last"] or {}).get("ts", ""), reverse=True)
    if args.json:
        print(json.dumps(rows, default=str))
        return 0
    if not rows:
        print(f"no runs under {root}")
        return 0
    print(f"{'run_id':<40} {'last event':<20} {'type':<18} status/uptime")
    for r in rows[: args.limit]:
        e = r["last"] or {}
        d = e.get("data") or {}
        info = d.get("status") or (f"up {d['uptime_s']:.0f}s" if d.get("uptime_s") is not None else "")
        print(f"{r['run_id']:<40} {e.get('ts', '?')[:19]:<20} {e.get('type', '?'):<18} {info}")
    return 0


def _listdir(path: str) -> List[str]:
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


# --------------------------------------------------------------------------- jgm scheduler


def cmd_scheduler(args: argparse.Namespace) -> int:
    from ._log import set_debug
    from .scheduler import SchedulerProbe, detect_adapter

    cfg = Config.from_env()
    set_debug(cfg.debug or args.verbose)
    adapter = detect_adapter(args.scheduler)
    if adapter is None:
        print("jgm scheduler: no scheduler found (squeue / oarstat not on PATH); use --scheduler to force one", file=sys.stderr)
        return 2
    base, origin = resolve_base_dir(cfg.dir)
    if not base:
        print("jgm scheduler: no writable event directory; set JGM_DIR", file=sys.stderr)
        return 1
    user: Optional[str] = None if args.all_users else (args.user or os.environ.get("USER") or os.environ.get("LOGNAME"))
    probe = SchedulerProbe(adapter, base, user=user, cluster=cfg.cluster, interval_s=args.interval, refresh_s=args.refresh,
                           tail_logs=not args.no_logs, log_max_bytes=int(args.log_max_mb * 1024 * 1024))
    print(f"jgm scheduler: {adapter.name} cluster={probe.cluster} user={user or 'all'} dir={base} [{origin}] "
          f"interval={probe.interval_s:.0f}s", file=sys.stderr)
    if args.once:
        events = probe.poll()
        probe.close()
        for e in events:
            d = e["data"]
            print(f"{e['run_id']:<36} {d['state']:<14} {d['change']:<10} {d.get('job_name') or ''}")
        print(f"{len(events)} event(s) emitted", file=sys.stderr)
        return 0
    probe.run_forever()
    return 0


# --------------------------------------------------------------------------- jgm forward


def cmd_forward(args: argparse.Namespace) -> int:
    from ._log import set_debug
    from .forward import Forwarder

    cfg = Config.from_env()
    set_debug(cfg.debug or args.verbose)
    base, origin = resolve_base_dir(cfg.dir)
    if not base:
        print("jgm forward: no event directory; set JGM_DIR", file=sys.stderr)
        return 1
    url, token = _remote_args(args, base)
    if not url or not token:
        print("jgm forward: no server configured; run `jgm setup --url ... --token ...` (or JGM_FORWARD_URL / JGM_FORWARD_TOKEN)", file=sys.stderr)
        return 2
    fw = Forwarder(base, url.rstrip("/"), token)
    print(f"jgm forward: {base} [{origin}] -> {url}  every {args.interval:.0f}s"
          + (f"  proxy={os.environ.get('https_proxy')}" if os.environ.get("https_proxy") else ""), file=sys.stderr)
    if args.once:
        r = fw.cycle()
        print(f"shipped={r['shipped']} batches={r['batches']} ok={bool(r['ok'])}", file=sys.stderr)
        return 0 if r["ok"] else 1
    fw.run_forever(args.interval)
    return 0


def _remote_args(args: argparse.Namespace, base: str) -> tuple[Optional[str], Optional[str]]:
    from . import remote

    url = getattr(args, "url", None)
    token = getattr(args, "token", None)
    if url and token:
        return url.rstrip("/"), token
    found = remote.load(base)
    if found:
        return found[0] if not url else url.rstrip("/"), found[1] if not token else token
    return url, token


# --------------------------------------------------------------------------- jgm setup / agent


def cmd_setup(args: argparse.Namespace) -> int:
    from . import remote

    cfg = Config.from_env()
    base, origin = resolve_base_dir(cfg.dir)
    if not base:
        print("jgm setup: no writable event directory; set JGM_DIR", file=sys.stderr)
        return 1
    url = args.url.rstrip("/")
    ok, msg = remote.check(url, args.token)
    if not ok and not args.force:
        print(f"jgm setup: cannot reach {url}: {msg}\n  (use --force to save anyway)", file=sys.stderr)
        return 1
    p = remote.save(base, url, args.token, {"cluster": cfg.cluster} if cfg.cluster else None)
    print(f"jgm setup: server {url} {'verified' if ok else 'saved without verification'}; config in {p}")
    print("  every job wrapped with `jgm run` (and every process with JGM_AUTO=1) now ships its events there.")
    print("  optional, for Slurm verdicts/queue state: `jgm agent install` on the login node.")
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    from . import agent
    from ._log import set_debug

    cfg = Config.from_env()
    set_debug(cfg.debug or getattr(args, "verbose", False))
    base, origin = resolve_base_dir(cfg.dir)
    if not base:
        print("jgm agent: no writable event directory; set JGM_DIR", file=sys.stderr)
        return 1
    action = args.action
    if action == "status":
        pid = agent.running_pid(base)
        print(f"jgm agent: {'running, pid ' + str(pid) if pid else 'not running'}  dir={base}  log={agent.log_path(base)}")
        return 0 if pid else 3
    if action == "stop":
        agent.remove_cron()
        print("jgm agent: stopped" if agent.stop(base) else "jgm agent: was not running", file=sys.stderr)
        return 0
    url, token = _remote_args(args, base)
    if not url or not token:
        print("jgm agent: no server configured; run `jgm setup --url ... --token ...` first", file=sys.stderr)
        return 2
    user: Optional[str] = None if args.all_users else (args.user or os.environ.get("USER") or os.environ.get("LOGNAME"))
    extra = ["--interval", str(args.interval)]
    if args.all_users:
        extra.append("--all-users")
    elif args.user:
        extra += ["--user", args.user]
    if args.scheduler:
        extra += ["--scheduler", args.scheduler]
    if args.no_logs:
        extra.append("--no-logs")
    if action == "serve":
        return agent.serve(base, url, token, args.interval, cfg.cluster, user, args.scheduler, tail_logs=not args.no_logs)
    if action in ("install", "keepalive", "start"):
        pid = agent.running_pid(base)
        if pid:
            if action != "keepalive":
                print(f"jgm agent: already running (pid {pid})", file=sys.stderr)
            return 0
        pid = agent.start_detached(base, extra)
        if action == "install":
            cron = agent.install_cron(base, extra)
            print(f"jgm agent: started (pid {pid}), log in {agent.log_path(base)}")
            print("  crontab keepalive installed: restarts it within 5 min if it dies or the node reboots" if cron
                  else "  no crontab here: re-run `jgm agent start` after a reboot")
        elif action == "start":
            print(f"jgm agent: started (pid {pid})", file=sys.stderr)
        return 0
    return 2


# --------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jgm", description="jobgpumonitor emitter tools")
    p.add_argument("--version", action="version", version=f"jobgpumonitor {__version__}")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="run a command under monitoring (any language)")
    r.add_argument("--name", help="human name for the run (defaults to the scheduler job name)")
    r.add_argument("--cwd", help="working directory for the command")
    r.add_argument("--tail-lines", type=int, default=100, help="stderr lines kept for run.end")
    r.add_argument("--no-forward", action="store_true", help="do not forward signals to the child")
    r.add_argument("--no-auto", action="store_true", help="do not auto-instrument the Python processes the command starts")
    r.add_argument("--no-push", action="store_true", help="do not ship events to the configured server from the job")
    r.add_argument("--keep-buffering", action="store_true", help="do not set PYTHONUNBUFFERED=1 for the command")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("emit", help="emit one event from a shell script")
    e.add_argument("type", help="event type, e.g. custom.stage or checkpoint.saved")
    e.add_argument("kv", nargs="*", help="key=value pairs (values parsed as JSON when possible)")
    e.add_argument("-m", "--message")
    e.set_defaults(func=cmd_emit)

    d = sub.add_parser("doctor", help="show what would be detected here")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_doctor)

    ls = sub.add_parser("ls", help="list runs found in the event directory")
    ls.add_argument("--dir")
    ls.add_argument("--json", action="store_true")
    ls.add_argument("-n", "--limit", type=int, default=30)
    ls.set_defaults(func=cmd_ls)

    sc = sub.add_parser("scheduler", help="login-node probe: emit scheduler.state from squeue/sacct (or oarstat)")
    sc.add_argument("--interval", type=float, default=30.0, help="seconds between polls (default 30)")
    sc.add_argument("--refresh", type=float, default=600.0, help="re-emit unchanged running jobs every N seconds (default 600)")
    sc.add_argument("--once", action="store_true", help="poll once and exit (cron friendly)")
    sc.add_argument("--user", help="only this user's jobs (default: $USER)")
    sc.add_argument("--all-users", action="store_true", help="watch every job on the cluster")
    sc.add_argument("--scheduler", choices=["slurm", "oar"], help="force the adapter instead of auto-detecting")
    sc.add_argument("--no-logs", action="store_true", help="do not tail the jobs' stdout/stderr files")
    sc.add_argument("--log-max-mb", type=float, default=8.0, help="stop tailing a file past this size (default 8 MB)")
    sc.add_argument("-v", "--verbose", action="store_true")
    sc.set_defaults(func=cmd_scheduler)

    fw = sub.add_parser("forward", help="login-node relay: ship the JSONL event files to a jobgpumonitor-server")
    fw.add_argument("--url", help="ingest endpoint, e.g. https://host/jgm/ingest (or JGM_FORWARD_URL)")
    fw.add_argument("--token", help="ingest token (or JGM_FORWARD_TOKEN)")
    fw.add_argument("--interval", type=float, default=10.0, help="seconds between scans (default 10)")
    fw.add_argument("--once", action="store_true", help="ship what is new and exit")
    fw.add_argument("-v", "--verbose", action="store_true")
    fw.set_defaults(func=cmd_forward)

    st = sub.add_parser("setup", help="record the server to ship events to (once, in the shared event directory)")
    st.add_argument("--url", required=True, help="ingest endpoint, e.g. https://host/jgm/ingest")
    st.add_argument("--token", required=True, help="the server's ingest token")
    st.add_argument("--force", action="store_true", help="save even if the server cannot be reached now")
    st.set_defaults(func=cmd_setup)

    ag = sub.add_parser("agent", help="login-node companion: scheduler probe + forwarder, detached, kept alive by cron")
    ag.add_argument("action", choices=["install", "start", "stop", "status", "keepalive", "serve"])
    ag.add_argument("--url")
    ag.add_argument("--token")
    ag.add_argument("--interval", type=float, default=30.0, help="seconds between scheduler polls (default 30)")
    ag.add_argument("--user", help="only this user's jobs (default: $USER)")
    ag.add_argument("--all-users", action="store_true")
    ag.add_argument("--scheduler", choices=["slurm", "oar"])
    ag.add_argument("--no-logs", action="store_true", help="do not tail the jobs' stdout/stderr files")
    ag.add_argument("-v", "--verbose", action="store_true")
    ag.set_defaults(func=cmd_agent)

    v = sub.add_parser("version")
    v.set_defaults(func=lambda a: print(__version__) or 0)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
