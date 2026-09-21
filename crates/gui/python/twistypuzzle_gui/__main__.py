"""Run the interface with ``python -m twistypuzzle_gui [OPTIONS] [PUZZLE]``.

Every argument is forwarded to the ``twistypuzzle-gui`` command, so
``python -m twistypuzzle_gui --help`` prints the interface's own usage.
"""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
