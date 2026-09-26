"""``jgm agent``: the optional login-node companion, one process, self-managing.

It runs the scheduler probe (queue state, Slurm verdicts, ``.out``/``.err`` tailing) and
ships everything found in the event directory. Jobs do not depend on it: the wrapper inside
each job already pushes its own events. The agent adds what only the scheduler knows.

``jgm agent --install`` starts it detached from the terminal (no tmux), records its pid,
and registers a crontab line that restarts it if it ever dies or the node reboots.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

from ._log import warn

CRON_TAG = "# jobgpumonitor agent keepalive"


def pid_path(base_dir: str) -> str:
    return os.path.join(base_dir, "agent", "agent.pid")


def log_path(base_dir: str) -> str:
    return os.path.join(base_dir, "agent", "agent.log")


def running_pid(base_dir: str) -> Optional[int]:
    try:
        with open(pid_path(base_dir), encoding="utf-8") as f:
            pid = int(f.read().strip() or 0)
    except (OSError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    # same pid reused by another program after a reboot? (Linux: the kernel tells us)
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read()
        if b"jobgpumonitor" not in cmdline and b"jgm" not in cmdline:
            return None
    except OSError:
        pass
    return pid


def serve(base_dir: str, url: str, token: str, interval_s: float, cluster: Optional[str], user: Optional[str],
          scheduler: Optional[str], tail_logs: bool = True) -> int:
    """Foreground: probe + forwarder until SIGTERM/SIGINT."""
    from .forward import Forwarder
    from .scheduler import SchedulerProbe, detect_adapter

    os.makedirs(os.path.dirname(pid_path(base_dir)), exist_ok=True)
    with open(pid_path(base_dir), "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    stop = threading.Event()
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(s, lambda *_: stop.set())
        except (ValueError, OSError):
            pass
    fw = Forwarder(base_dir, url, token)
    threads = [threading.Thread(target=fw.run_forever, args=(max(5.0, interval_s / 3), stop), name="jgm-forward", daemon=True)]
    adapter = detect_adapter(scheduler)
    probe = None
    if adapter is None:
        warn("agent: no scheduler command found (squeue/oarstat); forwarding only")
    else:
        probe = SchedulerProbe(adapter, base_dir, user=user, cluster=cluster, interval_s=interval_s, tail_logs=tail_logs)

        def probe_loop() -> None:
            while not stop.is_set():
                t = time.monotonic()
                try:
                    probe.poll()
                except Exception as e:  # keep going whatever happens
                    warn(f"agent: poll failed: {e}")
                stop.wait(max(1.0, probe.interval_s - (time.monotonic() - t)))
            probe.close()

        threads.append(threading.Thread(target=probe_loop, name="jgm-probe", daemon=True))
    for t in threads:
        t.start()
    print(f"jgm agent: pid {os.getpid()} dir={base_dir} scheduler={adapter.name if adapter else 'none'} -> {url}", flush=True)
    try:
        while not stop.is_set():
            stop.wait(1.0)
    finally:
        stop.set()
        for t in threads:
            t.join(10.0)
        try:
            os.remove(pid_path(base_dir))
        except OSError:
            pass
    return 0


def start_detached(base_dir: str, extra_args: list, settle_s: float = 2.0) -> int:
    """Launch ``jgm agent serve`` as a daemon: new session, no controlling terminal, log file.

    The pid file is written here, before returning, so ``jgm agent status`` right after
    ``start`` sees it; the child rewrites the same value once up. Raises ``RuntimeError``
    (with the log tail) when the agent dies within ``settle_s``.
    """
    os.makedirs(os.path.dirname(log_path(base_dir)), exist_ok=True)
    cmd = [sys.executable, *_self_argv(), "agent", "serve", *extra_args]
    with open(log_path(base_dir), "ab") as log:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, close_fds=True, env=dict(os.environ, JGM_DIR=base_dir))
    with open(pid_path(base_dir), "w", encoding="utf-8") as f:
        f.write(str(p.pid))
    try:
        rc = p.wait(timeout=settle_s)
    except subprocess.TimeoutExpired:
        return p.pid
    try:
        os.remove(pid_path(base_dir))
    except OSError:
        pass
    raise RuntimeError(f"agent exited immediately (code {rc}); last log lines:\n{_log_tail(base_dir)}")


def _log_tail(base_dir: str, n: int = 15) -> str:
    try:
        with open(log_path(base_dir), "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            lines = f.read().decode("utf-8", "replace").splitlines()
        return "\n".join("  " + ln for ln in lines[-n:]) or "  (empty)"
    except OSError:
        return "  (no log)"


def _self_argv() -> list:
    """How to re-invoke this program: the zipapp path, or ``-m jobgpumonitor`` from a checkout."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isfile(here):  # inside jgm.pyz
        return [here]
    return ["-m", "jobgpumonitor"]


def install_cron(base_dir: str, extra_args: list) -> bool:
    """Add a keepalive line to the user's crontab (idempotent). Returns False when cron is unavailable."""
    if not shutil.which("crontab"):
        return False
    line = f"*/5 * * * * {sys.executable} {' '.join(_self_argv())} agent keepalive {' '.join(extra_args)} >/dev/null 2>&1 {CRON_TAG}"
    try:
        cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        lines = [ln for ln in (cur.stdout.splitlines() if cur.returncode == 0 else []) if CRON_TAG not in ln]
        lines.append(line)
        r = subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", capture_output=True, text=True)
        return r.returncode == 0
    except OSError:
        return False


def remove_cron() -> None:
    if not shutil.which("crontab"):
        return
    cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if cur.returncode != 0:
        return
    lines = [ln for ln in cur.stdout.splitlines() if CRON_TAG not in ln]
    subprocess.run(["crontab", "-"], input="\n".join(lines) + ("\n" if lines else ""), capture_output=True, text=True)


def stop(base_dir: str) -> bool:
    pid = running_pid(base_dir)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    for _ in range(50):
        if running_pid(base_dir) is None:
            break
        time.sleep(0.1)
    return True
