from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

BASE_ENV_KEYS_TO_STRIP = ("SLURM_", "OAR_", "PBS_", "LSB_", "JGM_", "RANK", "WORLD_SIZE", "LOCAL_RANK")


def clean_env(**extra: str) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(BASE_ENV_KEYS_TO_STRIP)}
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra)
    return env


def read_events(base_dir: Path, run_id: str = "test/42/0") -> List[Dict[str, Any]]:
    files = sorted(glob.glob(str(base_dir / "runs" / run_id / "*.jsonl")))
    out: List[Dict[str, Any]] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    out.sort(key=lambda e: (e["emitter"], e["pid"], e["seq"]))
    return out


class Runner:
    def __init__(self, tmp_path: Path) -> None:
        self.dir = tmp_path / "events"
        self.env = clean_env(
            JGM_DIR=str(self.dir), JGM_CLUSTER="test", SLURM_JOB_ID="42", SLURM_JOB_NAME="unit",
            JGM_HEARTBEAT_S="1", JGM_SAMPLE_S="1", JGM_PROGRESS_S="0.1",
        )

    def script(self, code: str, timeout: float = 60, env: Optional[Dict[str, str]] = None, **popen: Any) -> subprocess.CompletedProcess:
        e = dict(self.env)
        if env:
            e.update(env)
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)], env=e, capture_output=True, text=True, timeout=timeout, **popen
        )

    def popen(self, code: str, env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
        e = dict(self.env)
        if env:
            e.update(env)
        return subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(code)], env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    def events(self, run_id: str = "test/42/0") -> List[Dict[str, Any]]:
        return read_events(self.dir, run_id)


@pytest.fixture
def runner(tmp_path: Path) -> Runner:
    return Runner(tmp_path)


def types(events: List[Dict[str, Any]]) -> List[str]:
    return [e["type"] for e in events]


def first(events: List[Dict[str, Any]], etype: str) -> Dict[str, Any]:
    for e in events:
        if e["type"] == etype:
            return e
    raise AssertionError(f"no {etype} in {types(events)}")


def last(events: List[Dict[str, Any]], etype: str) -> Dict[str, Any]:
    for e in reversed(events):
        if e["type"] == etype:
            return e
    raise AssertionError(f"no {etype} in {types(events)}")
