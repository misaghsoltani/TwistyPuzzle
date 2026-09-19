"""Collection rules for the parts of the suite that need an optional extra.

The package installs nothing but itself. Gymnasium and NumPy arrive only with
``twistypuzzle[gym]``, so an environment that has the wheel and nothing else is
a supported environment and has to produce a passing run rather than a
collection error.

``test_gym.py`` imports ``gymnasium`` at module scope, which is correct, as it
is a test *of* the extra and there is nothing meaningful to test without it.
Module-scope imports fail at collection, though, and pytest reports that as an
error for the whole run, not as a skip. So the decision is made here instead.

This is not a way to let the Gymnasium tests quietly stop running. Every job
that is supposed to test the extra installs ``twistypuzzle[gym]`` and
asserts the environments work, so a genuinely broken extra fails there.
"""

from __future__ import annotations

import importlib.util

#: Modules that ``test_gym.py`` needs before it can even be imported.
_GYM_EXTRA: tuple[str, ...] = ("gymnasium", "numpy")

collect_ignore: list[str] = []

if any(importlib.util.find_spec(name) is None for name in _GYM_EXTRA):
    collect_ignore.append("test_gym.py")
