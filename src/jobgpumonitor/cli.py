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

    tail: Deque[str] = collections.deque(maxlen=args.tail_lines)
    try:
        child = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, cwd=args.cwd or None)
    except OSError as e:
        print(f"jgm run: cannot start {cmd[0]!r}: {e}", file=sys.stderr)
        return 127

    run = Run(cfg, ctx, source="wrapper", hooks=False, monitor=True, probe_pid=child.pid)
    run.ctx["child_pid"] = child.pid
    run.start(emit_start=False)
    payload = run._start_payload()
    payload["command"] = cmd
    payload["child_pid"] = child.pid
    payload["name"] = args.name
    run.emit("run.start", payload)

    def tee() -> None:
        assert child.stderr is not None
        out = getattr(sys.stderr, "buffer", None)
        for raw in iter(child.stderr.readline, b""):
            try:
                if out is not None:
                    out.write(raw)
                    out.flush()
                else:
                    sys.stderr.write(raw.decode("utf-8", "replace"))
            except Exception:
                pass
            tail.append(raw.decode("utf-8", "replace").rstrip("\n")[:2000])

    t = threading.Thread(target=tee, name="jgm-stderr-tee", daemon=True)
    t.start()

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
    t.join(2.0)
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
    return exit_code


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
    probe = SchedulerProbe(adapter, base, user=user, cluster=cfg.cluster, interval_s=args.interval, refresh_s=args.refresh)
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
    url = args.url or os.environ.get("JGM_FORWARD_URL")
    token = args.token or os.environ.get("JGM_FORWARD_TOKEN")
    if not url or not token:
        print("jgm forward: need --url and --token (or JGM_FORWARD_URL / JGM_FORWARD_TOKEN)", file=sys.stderr)
        return 2
    base, origin = resolve_base_dir(cfg.dir)
    if not base:
        print("jgm forward: no event directory; set JGM_DIR", file=sys.stderr)
        return 1
    fw = Forwarder(base, url.rstrip("/"), token)
    print(f"jgm forward: {base} [{origin}] -> {url}  every {args.interval:.0f}s"
          + (f"  proxy={os.environ.get('https_proxy')}" if os.environ.get("https_proxy") else ""), file=sys.stderr)
    if args.once:
        r = fw.cycle()
        print(f"shipped={r['shipped']} batches={r['batches']} ok={bool(r['ok'])}", file=sys.stderr)
        return 0 if r["ok"] else 1
    fw.run_forever(args.interval)
    return 0


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
    sc.add_argument("-v", "--verbose", action="store_true")
    sc.set_defaults(func=cmd_scheduler)

    fw = sub.add_parser("forward", help="login-node relay: ship the JSONL event files to a jobgpumonitor-server")
    fw.add_argument("--url", help="ingest endpoint, e.g. https://host/jgm/ingest (or JGM_FORWARD_URL)")
    fw.add_argument("--token", help="ingest token (or JGM_FORWARD_TOKEN)")
    fw.add_argument("--interval", type=float, default=10.0, help="seconds between scans (default 10)")
    fw.add_argument("--once", action="store_true", help="ship what is new and exit")
    fw.add_argument("-v", "--verbose", action="store_true")
    fw.set_defaults(func=cmd_forward)

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
