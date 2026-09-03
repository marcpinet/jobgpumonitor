from __future__ import annotations

import json
import math

from jobgpumonitor import events as E


def test_ulid_shape_and_ordering():
    a = E.ulid(1000)
    b = E.ulid(2000)
    assert len(a) == 26 and len(b) == 26
    assert a < b
    assert a[:10] == E.ulid(1000)[:10]  # same timestamp prefix


def test_utc_iso():
    assert E.utc_iso(0) == "1970-01-01T00:00:00.000Z"


def test_normalize_type():
    assert E.normalize_type("stage") == "custom.stage"
    assert E.normalize_type("My Stage!") == "custom.my_stage_"
    assert E.normalize_type("run.start") == "run.start"
    assert E.normalize_type("a.b.c") == "a.b.c"


def test_sanitize_handles_nan_and_objects():
    class Scalar:
        shape = ()

        def item(self):
            return 3.5

    class Arr:
        shape = (2, 2)
        dtype = "float32"

        def tolist(self):
            return [[1, 2], [3, 4]]

    class Big:
        shape = (1000, 1000)
        dtype = "float32"

        def tolist(self):  # pragma: no cover - must not be called
            raise AssertionError

    out = E.sanitize({"nan": math.nan, "inf": math.inf, "s": Scalar(), "a": Arr(), "big": Big(), "obj": object(), "t": (1, 2)})
    assert out["nan"] is None and out["inf"] is None
    assert out["s"] == 3.5
    assert out["a"] == [[1, 2], [3, 4]]
    assert out["big"]["__array__"] is True
    assert out["obj"].startswith("<object")
    assert out["t"] == [1, 2]
    json.loads(E.to_json(out))


def test_to_json_falls_back_on_nan():
    s = E.to_json({"x": math.nan})
    assert json.loads(s) == {"x": None}


def test_envelope_fields():
    env = E.make_envelope(run_id="c/1/0", emitter="process-r0-h-1", pid=1, source="process", rank=None, seq=3, etype="run.heartbeat", data={"a": 1}, mono=1.23456)
    assert env["v"] == 1 and env["seq"] == 3 and env["mono"] == 1.235
    assert set(env) == {"v", "id", "seq", "ts", "mono", "run_id", "emitter", "pid", "source", "rank", "type", "data"}
