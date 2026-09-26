"""Interpreter start-up hook: ``JGM_AUTO=1`` makes every Python process monitor itself.

Reached in two ways, both before the user's first line runs:

* ``jobgpumonitor_auto.pth`` in site-packages (installed with the wheel), the same
  mechanism coverage.py uses to trace subprocesses;
* ``sitecustomize.py`` at the root of the ``jgm.pyz`` zipapp, when the zipapp is on
  ``PYTHONPATH`` (``jgm run`` does that for the command it launches).

It does nothing unless ``JGM_AUTO`` is set, and never raises: a monitoring hook that
broke an interpreter would be worse than no monitoring at all.
"""

import os
import sys

_SKIP_PROGRAMS = {"jgm", "jgm.pyz", "pip", "pip3", "uv", "conda", "pytest", "py.test", "sphinx-build"}
_SKIP_MODULES = {"pip", "ensurepip", "venv", "build", "wheel", "setuptools", "jobgpumonitor", "pytest", "site", "sysconfig"}


def _program() -> "tuple[str, str]":
    """(kind, name) of what this interpreter runs: ("module", "pip"), ("script", "train.py"),
    ("command", "-c")... At start-up ``sys.argv`` is not final yet for ``-m``/``-c`` runs,
    ``sys.orig_argv`` (3.10+) has the real command line."""
    orig = getattr(sys, "orig_argv", None)
    args = list(orig)[1:] if orig else list(sys.argv)
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-m", "-c"):
            return ("module" if a == "-m" else "command", args[i + 1] if i + 1 < len(args) else "")
        if a == "-":
            return ("stdin", "-")
        if a.startswith("-"):
            i += 2 if a in ("-W", "-X", "--check-hash-based-pycs") else 1  # options with a value
            continue
        return ("script", a)
    return ("interactive", "")


def _wanted() -> bool:
    if os.environ.get("JGM_AUTO", "").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    kind, name = _program()
    if kind == "module" and (name.split(".")[0] in _SKIP_MODULES):
        return False
    if kind == "script":
        base = os.path.basename(name)
        norm = name.replace("\\", "/")
        if base in _SKIP_PROGRAMS or base.startswith("pip") or any(f"/{m}/" in norm for m in _SKIP_MODULES):
            return False
    if kind == "interactive":
        return False
    # Inside a process tree that is already monitored (a Python job spawning helper
    # interpreters, ``python -c`` probes, workers...): only distributed ranks report,
    # the parent already covers the rest.
    if os.environ.get("JGM_IN_TREE"):
        try:
            from jobgpumonitor.context import detect_rank

            if detect_rank(os.environ) is None:
                return False
        except Exception:
            return False
    return True


def install() -> None:
    try:
        if not _wanted():
            return
        import jobgpumonitor.auto  # noqa: F401  (side effect: watch())
    except Exception:  # pragma: no cover - never break the host interpreter
        if os.environ.get("JGM_DEBUG"):
            import traceback

            traceback.print_exc()


install()
