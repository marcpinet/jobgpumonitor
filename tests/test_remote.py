"""Self-sufficient shipping: setup, the wrapper pushing its run to a server, the agent."""

from __future__ import annotations

import gzip
import http.server
import json
import os
import subprocess
import sys
import threading

from conftest import first, last
from jobgpumonitor import remote


class _Ingest(http.server.BaseHTTPRequestHandler):
    received: list = []
    token = "w"

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.headers.get("Authorization") != f"Bearer {self.token}":
            self.send_response(401)
            self.end_headers()
            return
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        _Ingest.received.extend(json.loads(body) if body else [])
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"accepted":0,"rejected":0}')

    def log_message(self, *a):
        pass


def _server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Ingest)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/ingest"


def test_remote_config_env_then_file(tmp_path, monkeypatch):
    monkeypatch.delenv("JGM_FORWARD_URL", raising=False)
    monkeypatch.delenv("JGM_FORWARD_TOKEN", raising=False)
    assert remote.load(str(tmp_path)) is None
    p = remote.save(str(tmp_path), "https://x/jgm/ingest/", "tok")
    assert oct(os.stat(p).st_mode & 0o777) == "0o600"
    assert remote.load(str(tmp_path)) == ("https://x/jgm/ingest", "tok")
    monkeypatch.setenv("JGM_FORWARD_URL", "https://env/ingest")
    monkeypatch.setenv("JGM_FORWARD_TOKEN", "envtok")
    assert remote.load(str(tmp_path)) == ("https://env/ingest", "envtok")


def test_remote_check_and_setup_command(tmp_path, runner):
    srv, url = _server()
    _Ingest.received.clear()
    assert remote.check(url, "w") == (True, "ok")
    assert remote.check(url, "bad")[0] is False
    p = subprocess.run([sys.executable, "-m", "jobgpumonitor", "setup", "--url", url, "--token", "w"], env=runner.env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert remote.load(str(runner.dir)) == (url, "w")
    p = subprocess.run([sys.executable, "-m", "jobgpumonitor", "setup", "--url", url, "--token", "nope"], env=runner.env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 1 and "401" in p.stderr
    srv.shutdown()


def test_wrapper_pushes_run_and_tees_both_streams(tmp_path, runner):
    """No login node, no agent: the job ships itself, stdout/stderr included."""
    srv, url = _server()
    _Ingest.received.clear()
    remote.save(str(runner.dir), url, "w")
    prog = tmp_path / "prog.py"
    prog.write_text("import sys\nprint('out line 1')\nprint('out line 2')\nsys.stderr.write('err line\\n')\nsys.exit(4)\n")
    p = subprocess.run([sys.executable, "-m", "jobgpumonitor", "run", "--", sys.executable, str(prog)], env=runner.env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 4
    assert "out line 1" in p.stdout and "err line" in p.stderr  # passed through unchanged
    srv.shutdown()
    ev = _Ingest.received
    assert ev, "nothing reached the server"
    sources = {e["source"] for e in ev}
    assert sources == {"wrapper", "process"}  # the child's own events were shipped too
    chunks = [e for e in ev if e["type"] == "log.chunk"]
    out = "".join(c["data"]["text"] for c in chunks if c["data"]["stream"] == "stdout")
    err = "".join(c["data"]["text"] for c in chunks if c["data"]["stream"] == "stderr")
    assert out == "out line 1\nout line 2\n" and "err line\n" in err
    assert any(c["data"]["eof"] for c in chunks if c["data"]["stream"] == "stdout")
    assert last([e for e in ev if e["source"] == "wrapper"], "run.end")["data"]["exit_code"] == 4
    assert first([e for e in ev if e["source"] == "process"], "run.start")["data"]["wrapped"] is True
    # everything written locally reached the server, nothing more (the server de-duplicates
    # if an agent on the login node ships the same files again)
    local = runner.events()
    assert len(ev) == len(local) and {(e["emitter"], e["seq"]) for e in ev} == {(e["emitter"], e["seq"]) for e in local}


def test_wrapper_without_server_still_writes_files(tmp_path, runner):
    prog = tmp_path / "prog.py"
    prog.write_text("print('x')\n")
    p = subprocess.run([sys.executable, "-m", "jobgpumonitor", "run", "--", sys.executable, str(prog)], env=runner.env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0
    ev = runner.events()
    assert any(e["type"] == "log.chunk" and e["data"]["stream"] == "stdout" and e["data"]["text"] == "x\n" for e in ev)


def test_agent_pid_and_keepalive(tmp_path, monkeypatch):
    from jobgpumonitor import agent

    base = str(tmp_path)
    assert agent.running_pid(base) is None
    os.makedirs(os.path.dirname(agent.pid_path(base)))
    with open(agent.pid_path(base), "w") as f:
        f.write("999999999")
    assert agent.running_pid(base) is None  # dead pid
    with open(agent.pid_path(base), "w") as f:
        f.write(str(os.getpid()))
    assert agent.running_pid(base) == os.getpid()
    assert agent._self_argv() in (["-m", "jobgpumonitor"],) or agent._self_argv()[0].endswith(".pyz")


def test_agent_start_status_stop(runner, tmp_path):
    srv, url = _server()
    env = dict(runner.env, JGM_FORWARD_URL=url, JGM_FORWARD_TOKEN="w", PATH=str(tmp_path / "nobin"))  # no crontab, no squeue
    jgm = [sys.executable, "-m", "jobgpumonitor", "agent"]
    assert subprocess.run(jgm + ["status"], env=env, capture_output=True, timeout=60).returncode == 3
    p = subprocess.run(jgm + ["start"], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    import time

    for _ in range(50):
        if subprocess.run(jgm + ["status"], env=env, capture_output=True, timeout=60).returncode == 0:
            break
        time.sleep(0.2)
    st = subprocess.run(jgm + ["status"], env=env, capture_output=True, text=True, timeout=60)
    assert st.returncode == 0 and "running" in st.stdout
    assert subprocess.run(jgm + ["start"], env=env, capture_output=True, text=True, timeout=60).stderr.strip().startswith("jgm agent: already running")
    assert subprocess.run(jgm + ["stop"], env=env, capture_output=True, timeout=60).returncode == 0
    assert subprocess.run(jgm + ["status"], env=env, capture_output=True, timeout=60).returncode == 3
    srv.shutdown()
    assert "forwarding only" in (runner.dir / "agent" / "agent.log").read_text()
