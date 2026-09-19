"""Pieces the single and vectorized environments share."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Literal

from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np

from twistypuzzle._native import jumbles, non_jumbling

if TYPE_CHECKING:
    from typing import Final, TypeAlias

    from gymnasium.spaces import Space
    from numpy.typing import NDArray

    from twistypuzzle._native import Puzzle


__all__ = [
    "ObsArray",
    "ObsSpaceArray",
    "Observation",
    "Permutations",
    "RewardScheme",
    "StepArray",
    "action_space",
    "clear_tables",
    "encode",
    "observation_space",
    "permutation_table",
    "refuse_jumbling",
    "text_board",
]

#: How a state is handed to the agent.
#:
#: ``"colors"`` is one categorical value per sticker.``"onehot"`` is that array expanded to
#: ``sticker_count * color_count`` indicator bytes.
Observation: TypeAlias = Literal["colors", "onehot"]

#: What a step is worth.
#:
#: ``"cost"`` charges 1 for every move, so the return of an episode is minus the
#: length of the path, which is the cost-to-go.``"sparse"`` pays 1 only for solving it.
RewardScheme: TypeAlias = Literal["cost", "sparse"]

#: A batch of observations, or a single one. Always bytes: a sticker color and
#: a one-hot indicator both fit in one.
ObsArray: TypeAlias = "NDArray[np.uint8]"

#: What an observation is as far as Gymnasium's spaces can say. `MultiDiscrete`
#: is declared over `NDArray[np.integer]` and `Box` over `NDArray[Any]`, so no
#: space can promise `uint8` specifically, though what comes out of one always is.
ObsSpaceArray: TypeAlias = "NDArray[np.integer]"

#: Rewards and flags share one parameter in `VectorEnv`, which needs both.
StepArray: TypeAlias = "NDArray[np.float64] | NDArray[np.bool_]"

#: One row per move: `new[i] = old[perm[a][i]]`.
Permutations: TypeAlias = "NDArray[np.intp]"

#: Derived tables, by recipe, so a batch of environments pays for one. The
#: value is a one-tuple so that a missing key ("not derived yet") is distinct
#: from a `None` table ("derived, and this puzzle has none").
_TABLES: Final[dict[str, tuple[Permutations | None]]] = {}

#: One lock per recipe, so that threads racing for the same table derive it
#: once between them rather than once each, while threads after *different*
#: tables still derive in parallel. Guarded by `_REGISTRY`, which is held only
#: for the lookup and never across a derivation.
_LOCKS: Final[dict[str, threading.Lock]] = {}
_REGISTRY: Final[threading.Lock] = threading.Lock()


def clear_tables() -> None:
    """Forget the derived permutation tables.

    Only useful in a test that wants to measure the cost of deriving one.
    """
    with _REGISTRY:
        _TABLES.clear()
        _LOCKS.clear()


def permutation_table(puzzle: Puzzle) -> Permutations | None:
    """Every move of `puzzle` as a permutation of its sticker slots.

    Turning the geometry costs about two milliseconds a move because it
    re-derives the whole cut structure, whereas a gather costs nanoseconds. The table
    is derived once per recipe and shared, so a batch of environments pays for
    one.

    Safe to call from several threads at once, and worth it: without the GIL,
    eight threads building the same environment would otherwise each pay the
    full derivation. The first one through derives it and the rest wait.

    Args:
        puzzle: A puzzle to derive it from. Left as it was found.

    Returns:
        A ``(actions, stickers)`` array, or ``None`` for a puzzle whose moves
        are not fixed permutations, which must be turned move by move.
    """
    key = puzzle.query
    held = _TABLES.get(key)
    if held is not None:
        return held[0]

    with _REGISTRY:
        lock = _LOCKS.setdefault(key, threading.Lock())
    with lock:
        # Another thread may have finished while this one waited.
        held = _TABLES.get(key)
        if held is None:
            rows = puzzle.action_permutations()
            held = (None if rows is None else np.asarray(rows, dtype=np.intp),)
            _TABLES[key] = held
    return held[0]


#: Recipes known not to jumble, so the usual ones cost nothing to check.
_KNOWN_GOOD: Final[frozenset[str]] = frozenset(non_jumbling())

#: Answers for recipes that are not cataloged, so a batch of environments pays
#: for the probe once rather than once each.
_PROBED: Final[dict[str, bool]] = {}


def refuse_jumbling(query: str, label: str) -> None:
    """Raise unless this puzzle's stickers stay on the solved lattice.

    A puzzle that *jumbles* (such as a Radiolarian, a jumble prism, or the Big Chop) has
    legal turns that leave pieces where no piece sits when it is solved, so
    there is no fixed set of sticker slots to number and no state vector to
    observe. Thirty-six of the eighty-five cataloged puzzles do not jumble.

    Args:
        query: The puzzle's recipe, as a canonical query string.
        label: How the caller named it, for the message.

    Raises:
        ValueError: If the puzzle jumbles.
    """
    if query in _KNOWN_GOOD:
        return
    answer = _PROBED.get(query)
    if answer is None:
        answer = jumbles(query)
        _PROBED[query] = answer
    if answer:
        msg = (
            f"{label!r} jumbles: some of its turns leave stickers where no sticker sits "
            f"when it is solved, so it has no fixed set of slots to observe. "
            f"twistypuzzle.non_jumbling_entries() lists the puzzles that do have one."
        )
        raise ValueError(msg)


def observation_space(sticker_count: int, color_count: int, mode: Observation) -> Space[ObsSpaceArray]:
    """The space one environment's observations live in.

    Args:
        sticker_count: How many slots the puzzle has.
        color_count: How many distinct sticker colors it has.
        mode: Which encoding to describe.

    Returns:
        A ``MultiDiscrete`` of sticker colors, or a ``Box`` of indicator bytes.
    """
    if mode == "colors":
        return MultiDiscrete(np.full(sticker_count, color_count, dtype=np.int64), dtype=np.uint8)

    return Box(low=0, high=1, shape=(sticker_count * color_count,), dtype=np.uint8)


def action_space(action_count: int) -> Discrete[np.int_]:
    """The space one environment's actions live in.

    Args:
        action_count: How many moves the puzzle has: two per grip.

    Returns:
        A ``Discrete`` over the move indices.
    """
    return Discrete(action_count, start=0)


def encode(colors: ObsArray, color_count: int, mode: Observation) -> ObsArray:
    """Turn a batch of sticker arrays into observations.

    Args:
        colors: Sticker colors, shaped ``(sticker_count,)`` or ``(n, sticker_count)``.
        color_count: How many distinct colors the puzzle has.
        mode: Which encoding to produce.

    Returns:
        The array itself for ``"colors"``, or its flattened one-hot expansion.
    """
    if mode == "colors":
        return colors
    # `arange == value` builds the indicator directly, which is both faster than
    # an `eye` lookup and free of the intermediate float array one would make.
    wide = (colors[..., None] == np.arange(color_count, dtype=colors.dtype)).astype(np.uint8)
    return wide.reshape(*colors.shape[:-1], colors.shape[-1] * color_count)


#: One character per color, so a board reads at a glance.
_GLYPHS: Final[str] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def text_board(colors: ObsArray, solved: ObsArray) -> str:
    """The state as text, one line per face.

    Slots are numbered in color order, so grouping them by the color they hold
    when solved puts each face on its own line (for a 3x3x3, six rows of nine).

    Args:
        colors: The color now in each slot.
        solved: The color in each slot when the puzzle is solved.

    Returns:
        One line per face, a character per sticker.
    """
    lines: list[str] = []
    start = 0
    for face in range(int(np.max(solved)) + 1 if solved.size else 0):
        end = start + int(np.count_nonzero(solved == face))
        run = colors[start:end]
        lines.append("".join(_GLYPHS[int(c) % len(_GLYPHS)] for c in run))
        start = end
    return "\n".join(lines)
