from __future__ import annotations

import gzip
import json

from jobgpumonitor.forward import Forwarder


def ev(seq, run_id="c/1/0", emitter="process-r0-h-1"):
    return {"v": 1, "seq": seq, "run_id": run_id, "emitter": emitter, "pid": 1, "source": "process", "type": "x.y", "data": {"seq": seq}, "ts": "t"}


def write(base, run_id, name, events, mode="a"):
    d = base / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    with open(d / name, mode) as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


class FakeServer:
    def __init__(self, fail_first=0):
        self.batches = []
        self.fail_first = fail_first
        self.headers = None

    def __call__(self, body, headers):
        self.headers = headers
        if self.fail_first > 0:
            self.fail_first -= 1
            return 503, "down"
        self.batches.append(json.loads(gzip.decompress(body)))
        return 200, "{}"


def test_forward_ships_new_lines_and_resumes(tmp_path):
    base = tmp_path / "ev"
    write(base, "c/1/0", "process-r0-h-1.jsonl", [ev(0), ev(1)])
    srv = FakeServer()
    fw = Forwarder(str(base), "https://x/ingest", "tok", post=srv, host="login")
    r = fw.cycle()
    assert r == {"shipped": 2, "batches": 1, "ok": 1}
    assert srv.headers["Authorization"] == "Bearer tok" and srv.headers["Content-Encoding"] == "gzip" and srv.headers["X-JGM-Host"] == "login"
    assert [e["seq"] for e in srv.batches[0]] == [0, 1]
    assert fw.cycle()["shipped"] == 0  # nothing new
    # partial last line is not shipped until complete
    f = base / "runs" / "c" / "1" / "0" / "process-r0-h-1.jsonl"
    with open(f, "a") as fh:
        fh.write(json.dumps(ev(2))[:-3])
    assert fw.cycle()["shipped"] == 0
    with open(f, "a") as fh:
        fh.write(json.dumps(ev(2))[-3:] + "\n")
    assert fw.cycle()["shipped"] == 1 and srv.batches[-1][0]["seq"] == 2
    # a new process restarts from the saved state, not from zero
    fw2 = Forwarder(str(base), "https://x/ingest", "tok", post=srv, host="login")
    assert fw2.cycle()["shipped"] == 0


def test_forward_keeps_offsets_when_server_down(tmp_path):
    base = tmp_path / "ev"
    write(base, "c/1/0", "process-r0-h-1.jsonl", [ev(0)])
    write(base, "c/2/0", "scheduler-login.jsonl", [ev(0, "c/2/0", "scheduler-login")])
    srv = FakeServer(fail_first=1)
    fw = Forwarder(str(base), "https://x/ingest", "tok", post=srv, host="l")
    r = fw.cycle()
    assert r["ok"] == 0 and r["shipped"] == 0 and fw.failures == 1
    r = fw.cycle()  # server back: everything is shipped, nothing was lost
    assert r["ok"] == 1 and r["shipped"] == 2 and fw.failures == 0
    assert sorted(e["run_id"] for b in srv.batches for e in b) == ["c/1/0", "c/2/0"]


def test_forward_batches_and_truncation(tmp_path):
    base = tmp_path / "ev"
    write(base, "c/1/0", "p.jsonl", [ev(i) for i in range(1200)])
    srv = FakeServer()
    fw = Forwarder(str(base), "https://x/ingest", "tok", batch_events=500, post=srv, host="l")
    r = fw.cycle()
    assert r["shipped"] == 1200 and r["batches"] == 3 and [len(b) for b in srv.batches] == [500, 500, 200]
    # file truncated and rewritten (e.g. copied back): re-read from the start
    write(base, "c/1/0", "p.jsonl", [ev(0)], mode="w")
    assert fw.cycle()["shipped"] == 1
