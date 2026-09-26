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


def main() -> int:
    global OUT
    if len(sys.argv) > 1:
        OUT = pathlib.Path(sys.argv[1]).resolve()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pkg = pathlib.Path(tmp) / "jobgpumonitor"
        shutil.copytree(SRC, pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pth"))
        (pathlib.Path(tmp) / "__main__.py").write_text("import sys\nfrom jobgpumonitor.cli import main\nsys.exit(main())\n")
        zipapp.create_archive(tmp, OUT, interpreter="/usr/bin/env python3", compressed=True)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
