"""PyInstaller entry point.

PyInstaller analyzes a script, not a ``python -m package`` invocation, so this is a thin
stand-in for ``python -m chaselate``. ``import chaselate`` must be the first import: its
``__init__.py`` pins the system Visual C++ runtime before PyQt5 can load its own older copy --
see ``chaselate/_runtime.py`` for why skipping this crashes the process outright.
"""

import sys

import chaselate  # noqa: F401
from chaselate.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
