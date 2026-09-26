"""JGM_AUTO: programs that never import jobgpumonitor are monitored anyway."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from conftest import ROOT, SRC, first, last, types

PROG = textwrap.dedent(
    """
    import logging, sys, time
    from tqdm import tqdm
    logging.warning("no import of jobgpumonitor anywhere in this file")
    for _ in tqdm(range(5), desc="train", mininterval=0):
        time.sleep(0.05)
    sys.stderr.write("10%\\r50%\\r100%\\n")
    raise RuntimeError("crash without any instrumentation")
    """
)


def _run_wrapped(jgm_cmd, runner, tmp_path):
    prog = tmp_path / "prog.py"
    prog.write_text(PROG)
    p = subprocess.run(jgm_cmd + ["run", "--", sys.executable, str(prog)], env=runner.env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 1, p.stderr
    assert "crash without any instrumentation" in p.stderr
    ev = runner.events()
    procs = [e for e in ev if e["source"] == "process"]
    assert procs, types(ev)
    assert first(procs, "run.start")["data"]["wrapped"] is True
    assert "progress.update" in types(procs) and "log.line" in types(procs)
    exc = first(procs, "run.exception")["data"]
    assert exc["type"] == "RuntimeError"
    assert last(procs, "run.end")["data"]["status"] == "error"
    wrapper_end = last([e for e in ev if e["source"] == "wrapper"], "run.end")["data"]
    assert wrapper_end["exit_code"] == 1
    # the stderr tail keeps whole lines, with tqdm-style carriage returns folded
    assert "100%" in wrapper_end["stderr_tail"] and "10%\r" not in wrapper_end["stderr_tail"]
    return ev


def test_jgm_run_from_checkout_auto_instruments_child(runner, tmp_path):
    pytest.importorskip("tqdm")
    _run_wrapped([sys.executable, "-m", "jobgpumonitor"], runner, tmp_path)


def test_jgm_run_from_zipapp_auto_instruments_child(runner, tmp_path):
    pytest.importorskip("tqdm")
    pyz = tmp_path / "jgm.pyz"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_zipapp.py"), str(pyz)], check=True, capture_output=True)
    env = {k: v for k, v in runner.env.items() if k != "PYTHONPATH"}  # the zipapp must be self-sufficient
    runner.env = env
    ev = _run_wrapped([sys.executable, str(pyz)], runner, tmp_path)
    # the child imported the library from inside the zipapp
    start = first([e for e in ev if e["source"] == "process"], "run.start")["data"]
    assert start["emitter_config"]["version"] != "?"


def test_pth_hook_with_plain_python_and_jgm_auto(runner, tmp_path):
    """pip-installed layout: the .pth in site-packages fires on JGM_AUTO=1, no wrapper at all."""
    pytest.importorskip("tqdm")
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv)], check=True)
    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    site = json.loads(subprocess.check_output([str(py), "-c", "import json,site;print(json.dumps(site.getsitepackages()))"]))[0]
    shutil.copytree(SRC / "jobgpumonitor", Path(site) / "jobgpumonitor", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(SRC / "jobgpumonitor_auto.pth", Path(site) / "jobgpumonitor_auto.pth")
    # tqdm/psutil live in the site-packages of the interpreter running the tests, which is
    # not the venv's "system" interpreter when pytest itself runs inside a virtualenv
    (Path(site) / "test_deps.pth").write_text("\n".join(p for p in sys.path if p.endswith("site-packages")) + "\n")
    prog = tmp_path / "prog.py"
    prog.write_text(PROG)
    env = {k: v for k, v in runner.env.items() if k != "PYTHONPATH"}
    # without JGM_AUTO: nothing
    p = subprocess.run([str(py), str(prog)], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 1 and runner.events() == []
    # with it: fully monitored
    env["JGM_AUTO"] = "1"
    p = subprocess.run([str(py), str(prog)], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 1
    ev = runner.events()
    assert {"run.start", "progress.update", "log.line", "run.exception", "run.end"} <= set(types(ev))
    # tooling is left alone
    n = len(ev)
    subprocess.run([str(py), "-m", "pip", "--version"], env=env, capture_output=True, timeout=60)
    assert len(runner.events()) == n


def test_autohook_skips_tooling(monkeypatch):
    from jobgpumonitor import _autohook

    monkeypatch.setenv("JGM_AUTO", "1")
    monkeypatch.delenv("JGM_IN_TREE", raising=False)
    cases = {
        ("python", "-m", "pip", "install", "x"): False,
        ("python", "/x/bin/pip", "install"): False,
        ("python", "/x/jgm.pyz", "run"): False,
        ("python", "-m", "jobgpumonitor", "forward"): False,
        ("python", "-m", "torch.distributed.run", "train.py"): True,
        ("python", "-u", "-W", "ignore", "train.py", "--lr", "1e-3"): True,
        ("python", "-c", "print(1)"): True,
        ("python",): False,
    }
    for argv, expected in cases.items():
        monkeypatch.setattr(sys, "orig_argv", list(argv), raising=False)
        monkeypatch.setattr(sys, "argv", [argv[-1]])
        assert _autohook._wanted() is expected, argv
    # Python < 3.10 without /proc: sys.argv is still ['-m'] at start-up -> stay silent
    monkeypatch.delattr(sys, "orig_argv", raising=False)
    monkeypatch.setattr(_autohook, "open", lambda *a, **k: (_ for _ in ()).throw(OSError()), raising=False)
    monkeypatch.setattr(sys, "argv", ["-m"])
    assert _autohook._wanted() is False
    monkeypatch.setattr(sys, "argv", ["-m", "--version"])  # what 3.9 shows for `-m pip --version`
    assert _autohook._wanted() is False
    monkeypatch.setattr(sys, "argv", ["/x/train.py"])
    assert _autohook._wanted() is True
    monkeypatch.delattr(_autohook, "open")
    monkeypatch.setattr(sys, "orig_argv", ["python", "train.py"], raising=False)
    monkeypatch.setenv("JGM_IN_TREE", "123")
    assert _autohook._wanted() is False  # helper interpreter of a monitored program
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert _autohook._wanted() is True  # a distributed rank reports even inside the tree
    monkeypatch.setenv("JGM_AUTO", "0")
    assert _autohook._wanted() is False
