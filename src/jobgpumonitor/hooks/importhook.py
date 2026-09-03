"""Run a callback right after a module is imported (or immediately if it already is).

Used to patch ``tqdm`` lazily: we must not import it ourselves (it may not be installed,
and importing it changes the user's import order/side effects).
"""

from __future__ import annotations

import importlib.abc
import sys
import threading
from typing import Any, Callable, Dict, List

from .._log import dbg

Callback = Callable[[Any], None]


class _LoaderProxy:
    def __init__(self, loader: Any, callbacks: List[Callback]) -> None:
        self._loader = loader
        self._callbacks = callbacks

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def exec_module(self, module: Any) -> None:
        self._loader.exec_module(module)
        for cb in self._callbacks:
            try:
                cb(module)
            except Exception as e:  # never break the user's import
                dbg(f"post-import hook for {module.__name__} failed: {e}")


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._hooks: Dict[str, List[Callback]] = {}
        self._local = threading.local()
        self._lock = threading.Lock()

    def register(self, name: str, cb: Callback) -> None:
        with self._lock:
            self._hooks.setdefault(name, []).append(cb)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname not in self._hooks or getattr(self._local, "busy", False):
            return None
        self._local.busy = True
        try:
            spec = None
            for finder in list(sys.meta_path):
                if finder is self:
                    continue
                fs = getattr(finder, "find_spec", None)
                if fs is None:
                    continue
                try:
                    spec = fs(fullname, path, target)
                except Exception:
                    spec = None
                if spec is not None:
                    break
        finally:
            self._local.busy = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        with self._lock:
            callbacks = self._hooks.pop(fullname, [])
        spec.loader = _LoaderProxy(spec.loader, callbacks)
        return spec


_finder: _PostImportFinder = _PostImportFinder()


def on_import(module_name: str, callback: Callback) -> None:
    mod = sys.modules.get(module_name)
    if mod is not None:
        try:
            callback(mod)
        except Exception as e:
            dbg(f"import hook for {module_name} failed: {e}")
        return
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)
    _finder.register(module_name, callback)
