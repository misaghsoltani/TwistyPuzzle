"""The optional extras must stay optional.

`twistypuzzle` declares no dependencies. Gymnasium and NumPy arrive only with
`twistypuzzle[gym]`, and the desktop application only with `twistypuzzle[gui]`.

Nothing enforces that on its own. A single top-level `import numpy` added to
the package would make a bare install fail at import time, and every other test
in this suite would keep passing, because the environment they run in has the
extras installed. So the checks here run in a *subprocess* and look at what
`import twistypuzzle` actually loaded, which is a question that has the same
answer whether or not the extras happen to be present.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import twistypuzzle as tp


def _in_subprocess(body: str) -> str:
    """Run `body` in a fresh interpreter and return its stdout.

    A fresh one because `sys.modules` in this process is already full of
    whatever the rest of the suite imported, which is exactly what must not be
    consulted.
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_importing_the_package_does_not_import_the_optional_extras() -> None:
    out = _in_subprocess("""
        import sys

        import twistypuzzle

        leaked = sorted(m for m in ("gymnasium", "numpy") if m in sys.modules)
        print(",".join(leaked))
    """)
    assert out.strip() == "", (
        f"importing twistypuzzle pulled in {out.strip()}, which the base package does not depend on"
    )


def test_the_package_is_usable_with_the_extras_unimportable() -> None:
    """Verify package behavior when optional dependencies cannot be imported.

    Blocking the modules outright is the difference between "we happen not to
    import it at module scope" and "a user without the extra can build and turn
    a puzzle", which is the promise the empty dependency list makes.
    """
    out = _in_subprocess("""
        import sys

        for blocked in ("gymnasium", "numpy"):
            sys.modules[blocked] = None  # any import of these now raises

        import twistypuzzle as tp

        puzzle = tp.Puzzle.named("Rubik's Cube (3x3x3)")
        # Turning is the operation the whole exact-arithmetic core exists for,
        # and it is the one most likely to reach for an array library.
        puzzle.turn(0, 1)
        puzzle.scramble(3)
        print(puzzle.piece_count)
    """)
    assert int(out.strip()) > 0


def test_the_gym_subpackage_is_not_imported_by_the_package() -> None:
    out = _in_subprocess("""
        import sys

        import twistypuzzle

        print("twistypuzzle.gym" in sys.modules)
    """)
    assert out.strip() == "False"


def test_the_gym_extra_names_its_own_dependencies() -> None:
    """Whatever `[gym]` resolves to must include gymnasium.

    Read from the installed metadata instead of from pyproject.toml, so this
    tests the wheel a user receives and not a file that only exists in a
    checkout.
    """
    from importlib.metadata import metadata

    requires = metadata("twistypuzzle").get_all("Requires-Dist") or []
    gym_requirements = [r for r in requires if "extra ==" in r and "gym" in r]
    assert gym_requirements, f"no extra declares any dependency, got {requires}"
    assert any("gymnasium" in r for r in gym_requirements), (
        f"the gym extra does not bring gymnasium: {gym_requirements}"
    )


def test_the_gui_is_a_separate_distribution() -> None:
    """The desktop application must not ride along with the library.

    `[gui]` pulls a second distribution, `twistypuzzle-gui`, so that the
    library does not carry a windowing toolkit for the many users who only
    want the simulator.
    """
    from importlib.metadata import metadata

    requires = metadata("twistypuzzle").get_all("Requires-Dist") or []
    gui_requirements = [r for r in requires if "extra ==" in r and "gui" in r]
    assert gui_requirements, "no [gui] extra is declared"
    assert any("twistypuzzle-gui" in r.replace("_", "-") for r in gui_requirements), (
        f"the gui extra does not name the desktop distribution: {gui_requirements}"
    )

    # And it must not be a dependency of the library proper.
    unconditional = [r for r in requires if "extra ==" not in r]
    assert unconditional == [], f"twistypuzzle must have no unconditional dependencies, found {unconditional}"


def test_the_version_is_importable_without_anything_else() -> None:
    assert isinstance(tp.__version__, str)
