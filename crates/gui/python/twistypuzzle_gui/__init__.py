"""Launcher for the twisty puzzle desktop interface.

The interface itself is a native executable that ships in this wheel. It owns a
windowing event loop, and on macOS that loop has to be the process's main thread,
so it runs as its own process instead of inside the interpreter that asked for it.

The wheel installs that executable as the ``twistypuzzle-gui`` command, which
is the interface's command line:

.. code-block:: console

    $ twistypuzzle-gui --help
    $ twistypuzzle-gui "Rubik's Cube (3x3x3)" --scramble 20 --seed 7
    $ twistypuzzle-gui '?shell=C$1&cut=C$1/3' --theme dark

Everything that command line takes, this module passes through, so a notebook
or a script can open the same window without a shell::

    >>> import twistypuzzle_gui  # doctest: +SKIP
    >>> twistypuzzle_gui.launch("Megaminx", "--scramble", "12")  # doctest: +SKIP

``python -m twistypuzzle_gui ARGS...`` forwards command-line arguments, and
:func:`capture` returns process output for flags such as ``--list`` and ``--help``.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import sysconfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from subprocess import CompletedProcess, Popen


__all__: list[str] = ["capture", "executable", "launch", "main", "run"]

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
    raise FileNotFoundError(f"could not find the {_PROGRAM} executable. Looked in {searched}")


def run(*args: str, wait: bool = True) -> int:
    """Run the interface's command line with ``args``.

    The arguments are the ones ``twistypuzzle-gui --help`` lists, passed through untouched.
    One that only prints something (``--help``, ``--list``, ``--version``, ``--polyhedra``)
    prints it and opens no window, invoking the executable directly.

    Args:
        *args: Command-line arguments, one per element, not a single shell string:
            ``run("--moves", "A B' C2")``, with the sequence as its own argument.
        wait: Block until the process ends. With ``False`` it is started and left
            running, which is what a notebook or a script with other work to do wants.

    Returns:
        The exit status, or 0 when not waiting. 2 means the command line was
        not understood, and the reason is on standard error.
    """
    process: Popen[bytes] = subprocess.Popen([str(executable()), *args])
    if not wait:
        return 0
    return process.wait()


def capture(*args: str) -> str:
    """Run the command line and return standard output as a string.

    For the arguments that only print (``--list``, ``--help``, ``--version``, ``--polyhedra``),
    so a caller can read the catalog the interface offers without opening a window or parsing
    the interface's own tables.

    Args:
        *args: Command-line arguments, one per element.

    Returns:
        Everything the command wrote to standard output. A command line the
        interface does not understand raises `subprocess.CalledProcessError`.
    """
    done: CompletedProcess[str] = subprocess.run([str(executable()), *args], capture_output=True, text=True, check=True)
    return done.stdout


def launch(*args: str, wait: bool = True) -> int:
    """Open the interface.

    Args:
        *args: Command-line arguments, as :func:`run` takes them.
        wait: Block until the window is closed.

    Returns:
        The exit status, or 0 when not waiting.
    """
    return run(*args, wait=wait)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m twistypuzzle_gui``.

    Args:
        argv: Arguments to forward, or this process's own when omitted.

    Returns:
        The exit status.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        return run(*args)
    except FileNotFoundError as exc:
        print(f"{_PROGRAM}: {exc}", file=sys.stderr)
        return 1
