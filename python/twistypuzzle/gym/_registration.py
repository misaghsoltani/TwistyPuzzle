"""Gymnasium ids for the puzzles that have a state vector.

One generic id, ``twistypuzzle/Puzzle-v0``, takes any puzzle through its
``recipe`` argument. Alongside it sits one id per cataloged puzzle that does
not jumble, so ``gymnasium.make("twistypuzzle/Megaminx-v0")`` works without
knowing the recipe.

Every id carries both an ``entry_point`` and a ``vector_entry_point``, so
``make_vec`` uses the batched implementation by default while ``"sync"`` and
``"async"`` remain available.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from gymnasium.envs.registration import register, registry

from twistypuzzle import catalog_entries
from twistypuzzle._native import non_jumbling

from ._env import DEFAULT_MAX_EPISODE_STEPS, DEFAULT_RECIPE

if TYPE_CHECKING:
    from typing import Final


__all__: list[str] = ["NAMESPACE", "env_ids", "register_envs"]

#: The namespace every id lives in.
NAMESPACE: Final[str] = "twistypuzzle"

#: The id that takes any puzzle through its ``recipe`` argument.
GENERIC_ID: Final[str] = f"{NAMESPACE}/Puzzle-v0"

_NOT_ID_SAFE: Final[re.Pattern[str]] = re.compile(r"[^0-9A-Za-z]+")

#: Given as import strings instead of as the classes themselves: that is what
#: the registry's own signature asks for, and a string survives the trip into
#: the subprocesses `AsyncVectorEnv` starts.
ENTRY_POINT: Final[str] = "twistypuzzle.gym._env:TwistyPuzzleEnv"
VECTOR_ENTRY_POINT: Final[str] = "twistypuzzle.gym._vector:TwistyPuzzleVectorEnv"

_REGISTERED: dict[str, str] = {}


def _slug(name: str) -> str:
    """Turn a puzzle's name into something a Gymnasium id will accept.

    Args:
        name: The catalog name, which may hold spaces, quotes and brackets.

    Returns:
        The name with everything but letters and digits removed.
    """
    return _NOT_ID_SAFE.sub("", name) or "Puzzle"


def env_ids() -> dict[str, str]:
    """Every registered id, mapped to the recipe it builds.

    Returns:
        Ids in registration order, and empty until :func:`register_envs` has run.
    """
    return dict(_REGISTERED)


def register_envs() -> None:
    """Register the generic id and one per non-jumbling catalog entry.

    Called when :mod:`twistypuzzle.gym` is imported, and safe to call again:
    ids already in Gymnasium's registry are left alone, so importing the module
    twice does not warn about overriding them.
    """
    if GENERIC_ID not in registry:
        register(
            id=GENERIC_ID,
            entry_point=ENTRY_POINT,
            vector_entry_point=VECTOR_ENTRY_POINT,
            max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
            kwargs={"recipe": DEFAULT_RECIPE},
        )
    _REGISTERED[GENERIC_ID] = DEFAULT_RECIPE

    playable = set(non_jumbling())
    taken: set[str] = set()
    for entry in catalog_entries():
        if entry.recipe not in playable:
            continue
        # Ten catalog entries share the placeholder name "Unknown", and a
        # slug can collide anyway, so the first keeps the bare name and the
        # rest are numbered.
        base = _slug(entry.name)
        slug = base
        n = 2
        while slug in taken:
            slug = f"{base}{n}"
            n += 1
        taken.add(slug)

        env_id = f"{NAMESPACE}/{slug}-v0"
        if env_id not in registry:
            register(
                id=env_id,
                entry_point=ENTRY_POINT,
                vector_entry_point=VECTOR_ENTRY_POINT,
                max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
                kwargs={"recipe": entry.recipe},
            )
        _REGISTERED[env_id] = entry.recipe
