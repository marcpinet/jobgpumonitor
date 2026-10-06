from __future__ import annotations

import json
import subprocess
import sys

from conftest import first, last, posix_only, read_events

JGM = [sys.executable, "-m", "jobgpumonitor.cli"]


def test_run_wrapper_reports_exit_code_and_stderr_tail(runner):
    p = subprocess.run(
        JGM + ["run", "--name", "demo", "--", sys.executable, "-c", "import sys; sys.stderr.write('warn 1\\nboom\\n'); sys.exit(2)"],
        env=runner.env, capture_output=True, text=True, timeout=60,
    )
    assert p.returncode == 2
    assert "boom" in p.stderr  # stderr is passed through
    ev = [e for e in runner.events() if e["source"] == "wrapper"]
    start = first(ev, "run.start")["data"]
    assert start["command"][0] == sys.executable and start["child_pid"] > 0 and start["name"] == "demo"
    end = last(ev, "run.end")["data"]
    assert end["status"] == "error" and end["exit_code"] == 2
    assert end["stderr_tail"].splitlines() == ["warn 1", "boom"]


def test_run_wrapper_and_inner_process_share_run_id(runner):
    p = subprocess.run(
        JGM + ["run", "--", sys.executable, "-c", "import jobgpumonitor; jobgpumonitor.log(x=1)"],
        env=runner.env, capture_output=True, text=True, timeout=60,
    )
    assert p.returncode == 0, p.stderr
    ev = runner.events()
    sources = {e["source"] for e in ev}
    assert sources == {"wrapper", "process"}
    assert len({e["run_id"] for e in ev}) == 1
    inner_start = [e for e in ev if e["source"] == "process" and e["type"] == "run.start"][0]
    assert inner_start["data"]["wrapped"] is True


@posix_only
def test_run_wrapper_signal_kill(runner):
    import signal
    import time

    proc = subprocess.Popen(
        JGM + ["run", "--forward-signals", "always", "--", sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"],
        env=runner.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert proc.stdout.readline().strip() == "ready"
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=15)
    assert proc.returncode == 143
    ev = [e for e in runner.events() if e["source"] == "wrapper"]
    assert first(ev, "signal.received")["data"]["forwarded"] is True
    end = last(ev, "run.end")["data"]
    assert end["status"] == "killed" and end["signal"] == "SIGTERM" and end["forwarded_signals"] == ["SIGTERM"]


def test_emit_from_shell(runner):
    p = subprocess.run(
        JGM + ["emit", "stage", "name=eval", "n=3", "ok=true", "-m", "hello"],
        env=runner.env, capture_output=True, text=True, timeout=60,
    )
    assert p.returncode == 0, p.stderr
    ev = runner.events()
    assert len(ev) == 1
    e = ev[0]
    assert e["type"] == "custom.stage" and e["source"] == "shell" and e["emitter"].startswith("shell-")
    assert e["data"] == {"name": "eval", "n": 3, "ok": True, "message": "hello"}


def test_doctor_json(runner):
    p = subprocess.run(JGM + ["doctor", "--json"], env=runner.env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    rep = json.loads(p.stdout)
    assert rep["scheduler"] == "slurm" and rep["run_id"] == "test/42/0"
    assert rep["event_dir"] == str(runner.dir)


def test_ls_lists_runs(runner):
    subprocess.run([sys.executable, "-c", "import jobgpumonitor; jobgpumonitor.log(a=1)"], env=runner.env, check=True, timeout=60)
    p = subprocess.run(JGM + ["ls", "--json"], env=runner.env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    rows = json.loads(p.stdout)
    assert rows[0]["run_id"] == "test/42/0" and rows[0]["last"]["type"] == "run.end"
    assert read_events(runner.dir)


def test_events_validate_against_schema(runner):
    jsonschema = __import__("pytest").importorskip("jsonschema")
    from pathlib import Path

    schema = json.loads((Path(__file__).resolve().parents[1] / "schema" / "event.schema.json").read_text())
    p = subprocess.run(
        # 2.5 s: the first heartbeat (1 s) comes after the first sample, and nvidia-smi alone takes ~0.2 s
        [sys.executable, "-c", "import logging, jobgpumonitor.auto, time; jobgpumonitor.auto.run.log(l=1); logging.warning('w'); time.sleep(2.5); raise RuntimeError('x')"],
        env=runner.env, capture_output=True, text=True, timeout=60,
    )
    assert p.returncode == 1
    validator = jsonschema.Draft202012Validator(schema)
    ev = runner.events()
    assert {"run.start", "run.heartbeat", "resource.sample", "metric.log", "log.line", "run.exception", "run.end"} <= {e["type"] for e in ev}
    for e in ev:
        errors = list(validator.iter_errors(e))
        assert not errors, (e["type"], [err.message for err in errors])


@posix_only
def test_wrapper_does_not_relay_slurm_signals_twice(runner):
    """Slurm signals every process of the step: the command must see one SIGTERM, not two."""
    import os
    import signal
    import time

    child = (
        "import os, signal, sys, time\n"
        "n = [0]\n"
        "signal.signal(signal.SIGTERM, lambda s, f: n.__setitem__(0, n[0] + 1))\n"
        "print(os.getpid(), flush=True)\n"
        "t = time.time()\n"
        "while not n[0] and time.time() - t < 20:\n"
        "    time.sleep(0.05)\n"
        "time.sleep(1.5)\n"
        "sys.exit(10 + n[0])\n"
    )
    for mode, expected in (("auto", 11), ("always", 12)):
        proc = subprocess.Popen(JGM + ["run", "--forward-signals", mode, "--", sys.executable, "-c", child],
                                env=runner.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        pid = int(proc.stdout.readline())
        time.sleep(0.3)
        os.kill(pid, signal.SIGTERM)  # what Slurm does: every process of the step...
        time.sleep(0.3)
        proc.send_signal(signal.SIGTERM)  # ...the wrapper included
        proc.wait(timeout=30)
        assert proc.returncode == expected, (mode, proc.stderr.read())
        end = last([e for e in runner.events() if e["source"] == "wrapper"], "run.end")["data"]
        assert end["received_signals"] == ["SIGTERM"]
        assert end["forwarded_signals"] == ([] if mode == "auto" else ["SIGTERM"])


def test_chunk_emitter_folds_redraws_and_keeps_the_tail():
    from jobgpumonitor.cli import _ChunkEmitter

    class FakeRun:
        def __init__(self):
            self.chunks = []

        def emit(self, etype, data):
            self.chunks.append(data)
            return True

    r = FakeRun()
    ce = _ChunkEmitter(r, max_bytes=100, tail_bytes=40)
    ce.feed("stderr", b"".join(b"\r%3d%%" % i for i in range(100)) + b"\n")  # 501 bytes of redraws
    ce.flush()
    assert r.chunks[-1]["text"] == " 99%\n"  # what the terminal shows, 5 bytes of budget
    ce.feed("stderr", b"".join(b"line %02d\n" % i for i in range(40)))  # crosses the budget
    ce.flush()
    n = len(r.chunks)
    ce.feed("stderr", b"".join(b"more %02d\n" % i for i in range(40)) + b"Traceback\nboom\n")
    ce.flush()
    assert len(r.chunks) == n  # over budget: held back until the end
    ce.close("stdout")
    ce.close("stderr")
    ce.flush(final=True)
    tail = r.chunks[-1]
    assert tail["stream"] == "stderr" and tail["eof"] and tail["truncated"]
    assert tail["text"] == "more 37\nmore 38\nmore 39\nTraceback\nboom\n"
    assert tail["skipped"] == 296 and tail["offset"] == 501 + 320 + 296 and tail["size"] == 501 + 320 + 335
