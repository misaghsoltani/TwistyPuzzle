"""Launcher for the twisty puzzle desktop interface.

The interface itself is a native executable that ships in this wheel. It owns a
windowing event loop, and on macOS that loop has to be the process's main
thread, so it runs as its own process rather than inside the interpreter that
asked for it.

    >>> import twistypuzzle_gui  # doctest: +SKIP
    >>> twistypuzzle_gui.launch()  # doctest: +SKIP

From a shell, the wheel installs a `twistypuzzle-gui` command.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import sysconfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subprocess import Popen


__all__: list[str] = ["executable", "launch", "main"]

#: The name the executable is installed under.
_PROGRAM: str = "twistypuzzle-gui"


def executable() -> Path:
    """Where the interface's executable was installed.

    Returns:
        The path to it.

    Raises:
        FileNotFoundError: If the wheel is installed but its executable is not
            where this interpreter's scripts live.
    """
    suffix = ".exe" if sys.platform == "win32" else ""
    candidates = [
        Path(sysconfig.get_path("scripts")) / f"{_PROGRAM}{suffix}",
        Path(sys.prefix) / "bin" / f"{_PROGRAM}{suffix}",
        Path(sys.prefix) / "Scripts" / f"{_PROGRAM}{suffix}",
    ]
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(p) for p in candidates)
    msg = f"could not find the {_PROGRAM} executable; looked in {searched}"
    raise FileNotFoundError(msg)


def launch(*, wait: bool = True) -> int:
    """Open the interface.

    Args:
        wait: Block until the window is closed. With ``False`` the process is
            started and left running, which is what a notebook or a script that
            has other work to do wants.

    Returns:
        The exit status, or 0 when not waiting.
    """
    process: Popen[bytes] = subprocess.Popen([str(executable())])
    if not wait:
        return 0
    return process.wait()


def main() -> int:
    """Entry point for the console script.

    Returns:
        The exit status.
    """
    return launch()
