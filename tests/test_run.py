from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from conftest import first, last, types


def test_normal_run_emits_start_metric_end(runner):
    p = runner.script(
        """
        import jobgpumonitor
        jobgpumonitor.log(loss=0.5, step=1)
        jobgpumonitor.log({"acc": 0.9}, epoch=2)
        jobgpumonitor.emit("stage", name="eval")
        """
    )
    assert p.returncode == 0, p.stderr
    ev = runner.events()
    assert types(ev)[0] == "run.start"
    start = first(ev, "run.start")["data"]
    assert start["scheduler"]["name"] == "slurm" and start["scheduler"]["job_id"] == "42"
    assert "python" in start and start["resources"]["cpu_count"]
    assert "excepthook" in start["emitter_config"]["hooks"]
    m = first(ev, "metric.log")
    assert m["data"] == {"metrics": {"loss": 0.5}, "step": 1, "epoch": None}
    assert first(ev, "custom.stage")["data"] == {"name": "eval"}
    end = last(ev, "run.end")["data"]
    assert end["status"] == "ok" and end["exit_code"] == 0
    assert end["metrics"] == {"loss": 0.5, "acc": 0.9}
    seqs = [e["seq"] for e in ev]
    assert seqs == list(range(len(ev)))
    assert all(e["run_id"] == "test/42/0" and e["source"] == "process" for e in ev)


def test_uncaught_exception_is_reported(runner):
    p = runner.script(
        """
        import jobgpumonitor.auto
        def f(secret_key="hunter2"):
            raise ValueError("boom " + "x" * 10)
        f()
        """,
        env={"JGM_CAPTURE_LOCALS": "1"},
    )
    assert p.returncode == 1
    assert "ValueError: boom" in p.stderr  # previous excepthook still ran
    ev = runner.events()
    exc = first(ev, "run.exception")["data"]
    assert exc["type"] == "ValueError" and exc["message"].startswith("boom")
    assert exc["fatal"] is True and exc["kind"] == "error"
    assert exc["frames"][-1]["func"] == "f"
    assert exc["frames"][-1]["locals"]["secret_key"] == "***"
    end = last(ev, "run.end")["data"]
    assert end["status"] == "error" and end["exit_code"] == 1
    assert end["exception"]["type"] == "ValueError"


def test_sys_exit_code_is_tracked(runner):
    p = runner.script("import jobgpumonitor.auto, sys; sys.exit(3)")
    assert p.returncode == 3
    end = last(runner.events(), "run.end")["data"]
    assert end["status"] == "error" and end["exit_code"] == 3


def test_keyboard_interrupt_status(runner):
    p = runner.script("import jobgpumonitor.auto; raise KeyboardInterrupt")
    assert p.returncode != 0
    end = last(runner.events(), "run.end")["data"]
    assert end["status"] == "interrupted"


def test_sigterm_default_disposition_emits_killed_and_dies_by_signal(runner):
    proc = runner.popen(
        """
        import jobgpumonitor.auto, time, sys
        print("ready", flush=True)
        time.sleep(30)
        """
    )
    assert proc.stdout.readline().strip() == "ready"
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)
    assert proc.returncode == -signal.SIGTERM
    ev = runner.events()
    sig = first(ev, "signal.received")["data"]
    assert sig["signal"] == "SIGTERM" and sig["will_terminate"] is True and sig["previous_handler"] == "default"
    end = last(ev, "run.end")["data"]
    assert end["status"] == "killed" and end["signal"] == "SIGTERM" and end["exit_code"] == 143


def test_sigterm_user_handler_is_chained(runner):
    proc = runner.popen(
        """
        import signal, sys, time
        def h(s, f):
            print("user handler", flush=True); sys.exit(0)
        signal.signal(signal.SIGTERM, h)
        import jobgpumonitor.auto
        print("ready", flush=True)
        time.sleep(30)
        """
    )
    assert proc.stdout.readline().strip() == "ready"
    time.sleep(0.3)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=10)
    assert "user handler" in out
    assert proc.returncode == 0
    ev = runner.events()
    assert first(ev, "signal.received")["data"]["previous_handler"] == "h"
    assert last(ev, "run.end")["data"]["status"] == "ok"


def test_heartbeat_and_samples_flow(runner):
    p = runner.script("import jobgpumonitor.auto, time; time.sleep(2.6)")
    assert p.returncode == 0, p.stderr
    ev = runner.events()
    hbs = [e for e in ev if e["type"] == "run.heartbeat"]
    assert len(hbs) >= 2
    assert hbs[-1]["data"]["uptime_s"] >= 2
    samples = [e for e in ev if e["type"] == "resource.sample"]
    assert len(samples) >= 2
    assert samples[0]["data"]["proc"]["pid"] > 0
    assert types(ev)[-1] == "run.end"  # always the last line, even with the sampler racing
    end = last(ev, "run.end")["data"]
    assert end["summary"]["samples"] >= 2
    assert end["duration_s"] >= 2.5


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork only")
def test_forked_child_is_silent(runner):
    p = runner.script(
        """
        import jobgpumonitor, os, sys
        jobgpumonitor.watch()
        pid = os.fork()
        if pid == 0:
            jobgpumonitor.log(child=1)
            jobgpumonitor.emit("child")
            os._exit(0)
        os.waitpid(pid, 0)
        jobgpumonitor.log(parent=1)
        """
    )
    assert p.returncode == 0, p.stderr
    ev = runner.events()
    assert not any(e["type"] == "custom.child" for e in ev)
    assert first(ev, "metric.log")["data"]["metrics"] == {"parent": 1}
    assert len({e["pid"] for e in ev}) == 1


def test_multiprocessing_spawn_child_is_silent(runner, tmp_path):
    script = tmp_path / "prog.py"
    script.write_text(
        "import jobgpumonitor.auto\n"
        "import multiprocessing as mp\n"
        "def work(x):\n"
        "    import jobgpumonitor\n"
        "    jobgpumonitor.emit('worker')\n"
        "    return x * 2\n"
        "if __name__ == '__main__':\n"
        "    ctx = mp.get_context('spawn')\n"
        "    with ctx.Pool(2) as pool:\n"
        "        assert pool.map(work, [1, 2]) == [2, 4]\n"
        "    import jobgpumonitor\n"
        "    jobgpumonitor.emit('main')\n"
    )
    import subprocess

    p = subprocess.run([sys.executable, str(script)], env=runner.env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    ev = runner.events()
    assert any(e["type"] == "custom.main" for e in ev)
    assert not any(e["type"] == "custom.worker" for e in ev)


def test_disabled_writes_nothing(runner):
    p = runner.script("import jobgpumonitor; jobgpumonitor.log(a=1)", env={"JGM_DISABLED": "1"})
    assert p.returncode == 0, p.stderr
    assert runner.events() == []


def test_rank_nonzero_is_light(runner):
    p = runner.script(
        "import jobgpumonitor, time; jobgpumonitor.log(a=1); time.sleep(1.3)",
        env={"RANK": "2", "WORLD_SIZE": "4", "LOCAL_RANK": "2"},
    )
    assert p.returncode == 0, p.stderr
    ev = runner.events()
    assert all(e["rank"] == 2 for e in ev)
    assert "metric.log" not in types(ev)
    assert "resource.sample" not in types(ev)
    assert "run.heartbeat" in types(ev)
    assert first(ev, "run.start")["data"]["emitter_config"]["light"] is True
    assert ev[0]["emitter"].startswith("process-r2-")


def test_logging_records_are_forwarded(runner):
    p = runner.script(
        """
        import logging, jobgpumonitor.auto
        logging.getLogger("train").warning("lr too %s", "high")
        logging.getLogger("train").info("ignored")
        try:
            1/0
        except ZeroDivisionError:
            logging.getLogger("train").exception("caught")
        """
    )
    assert p.returncode == 0, p.stderr
    lines = [e["data"] for e in runner.events() if e["type"] == "log.line"]
    assert [ln["message"] for ln in lines] == ["lr too high", "caught"]
    assert lines[0]["level"] == "WARNING" and lines[0]["logger"] == "train"
    assert "ZeroDivisionError" in lines[1]["exception"]


def test_thread_exception_is_non_fatal(runner):
    p = runner.script(
        """
        import threading, jobgpumonitor.auto
        def boom(): raise RuntimeError("in thread")
        t = threading.Thread(target=boom); t.start(); t.join()
        """
    )
    assert p.returncode == 0
    ev = runner.events()
    exc = first(ev, "run.exception")["data"]
    assert exc["fatal"] is False and exc["kind"] == "thread" and exc["type"] == "RuntimeError"
    assert last(ev, "run.end")["data"]["status"] == "ok"


def test_tqdm_progress_events(runner):
    pytest.importorskip("tqdm")
    p = runner.script(
        """
        import time, jobgpumonitor.auto
        from tqdm import tqdm
        for _ in tqdm(range(30), desc="train", mininterval=0):
            time.sleep(0.02)
        for _ in tqdm(range(3), disable=True):
            pass
        """,
        env={"SLURM_JOB_START_TIME": "0", "SLURM_JOB_END_TIME": str(int(time.time()) + 3600)},
    )
    assert p.returncode == 0, p.stderr
    ev = runner.events()
    prog = [e["data"] for e in ev if e["type"] == "progress.update"]
    assert prog, types(ev)
    assert prog[0]["desc"] == "train" and prog[0]["total"] == 30
    assert prog[-1]["done"] is True and prog[-1]["n"] == 30
    assert len({p["bar_id"] for p in prog}) == 1  # the disabled bar emitted nothing
    mid = [p for p in prog if not p["done"] and p.get("eta_s") is not None]
    assert mid and "eta_vs_deadline_s" in mid[0]


def test_finish_is_idempotent_and_stops_emission(runner):
    p = runner.script(
        """
        import jobgpumonitor
        jobgpumonitor.watch()
        jobgpumonitor.finish(status="interactive")
        jobgpumonitor.finish()
        jobgpumonitor.log(after=1)
        """
    )
    assert p.returncode == 0, p.stderr
    ev = runner.events()
    assert types(ev).count("run.end") == 1
    assert last(ev, "run.end")["data"]["status"] == "interactive"
    assert "metric.log" not in types(ev)


def test_stderr_sink(runner):
    p = runner.script("import jobgpumonitor; jobgpumonitor.log(a=1)", env={"JGM_SINKS": "stderr"})
    assert p.returncode == 0
    assert '"type":"metric.log"' in p.stderr
    assert runner.events() == []


def test_unwritable_dir_falls_back_to_cwd(runner, tmp_path):
    cwd = tmp_path / "work"
    cwd.mkdir()
    p = runner.script(
        "import jobgpumonitor; jobgpumonitor.log(a=1)",
        env={"JGM_DIR": "/nonexistent/forbidden", "HOME": "/nonexistent/home2"},
        cwd=str(cwd),
    )
    assert p.returncode == 0, p.stderr
    from conftest import read_events

    assert read_events(cwd / ".jgm")
