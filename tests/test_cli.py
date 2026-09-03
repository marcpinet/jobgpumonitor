from __future__ import annotations

import json
import subprocess
import sys

from conftest import first, last, read_events

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


def test_run_wrapper_signal_kill(runner):
    import signal
    import time

    proc = subprocess.Popen(
        JGM + ["run", "--", sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"],
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
        [sys.executable, "-c", "import logging, jobgpumonitor.auto, time; jobgpumonitor.auto.run.log(l=1); logging.warning('w'); time.sleep(1.2); raise RuntimeError('x')"],
        env=runner.env, capture_output=True, text=True, timeout=60,
    )
    assert p.returncode == 1
    validator = jsonschema.Draft202012Validator(schema)
    ev = runner.events()
    assert {"run.start", "run.heartbeat", "resource.sample", "metric.log", "log.line", "run.exception", "run.end"} <= {e["type"] for e in ev}
    for e in ev:
        errors = list(validator.iter_errors(e))
        assert not errors, (e["type"], [err.message for err in errors])
