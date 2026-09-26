"""``python -m jobgpumonitor`` and the ``jgm.pyz`` zipapp entry point."""

import sys

from .cli import main

sys.exit(main())
