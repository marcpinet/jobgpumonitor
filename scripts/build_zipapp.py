#!/usr/bin/env python3
"""Build ``dist/jgm.pyz``: the whole zero-dependency CLI in one file, runnable with any
Python >= 3.9 (``python3 jgm.pyz scheduler``). Stdlib only, no pip/venv needed to run it."""

import pathlib
import shutil
import sys
import tempfile
import zipapp

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "jobgpumonitor"
OUT = ROOT / "dist" / "jgm.pyz"


# Imported by every interpreter that has the zipapp on PYTHONPATH (``jgm run`` sets it):
# runs the JGM_AUTO hook, then hands over to the environment's own sitecustomize if any.
SITECUSTOMIZE = """\
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
try:
    import jobgpumonitor._autohook  # noqa: F401
except Exception:
    pass
try:
    import importlib.machinery, importlib.util
    _others = [p for p in sys.path if os.path.abspath(p) != _here]
    _spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _others)
    if _spec is not None and _spec.origin != __file__ and _spec.loader is not None:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
except Exception:
    pass
"""


def main() -> int:
    global OUT
    if len(sys.argv) > 1:
        OUT = pathlib.Path(sys.argv[1]).resolve()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pkg = pathlib.Path(tmp) / "jobgpumonitor"
        shutil.copytree(SRC, pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pth"))
        (pathlib.Path(tmp) / "__main__.py").write_text("import sys\nfrom jobgpumonitor.cli import main\nsys.exit(main())\n")
        (pathlib.Path(tmp) / "sitecustomize.py").write_text(SITECUSTOMIZE)
        zipapp.create_archive(tmp, OUT, interpreter="/usr/bin/env python3", compressed=True)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
