"""``import jobgpumonitor.auto`` starts monitoring as a side effect of the import.

Safe to import from a module that ``multiprocessing`` re-imports in spawned children:
the child detection in :func:`jobgpumonitor.watch` makes it a no-op there.
"""

from . import watch as _watch

run = _watch()
