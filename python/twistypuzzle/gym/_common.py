"""Pieces the single and vectorized environments share."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Literal

from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np

from twistypuzzle._native import Layout, jumbles, non_jumbling

if TYPE_CHECKING:
    from typing import Final, TypeAlias

    from gymnasium.spaces import Space
    from numpy.typing import NDArray

    from twistypuzzle._native import Puzzle, PuzzleBatch


__all__: list[str] = [
    "DEFAULT_VIEWS",
    "OBSERVATIONS",
    "Depth",
    "ObsArray",
    "ObsSpaceArray",
    "Observation",
    "Permutations",
    "RewardScheme",
    "StepArray",
    "View",
    "action_space",
    "clear_tables",
    "depth_range",
    "draw",
    "image_shape",
    "layout_for",
    "observation_space",
    "permutation_table",
    "refuse_jumbling",
    "refuse_unknown",
    "smallest",
    "solved_stickers",
    "text_board",
]

#: How a state is handed to the agent.
#:
#: ``"colors"`` is one categorical value per sticker slot. ``"onehot"`` is that
#: array expanded to ``sticker_count * color_count`` indicator bytes.
#: ``"ids"`` names the sticker in each slot instead of its color, which tells
#: apart states the colors cannot. ``"facelets"`` is the colors in the puzzle's
#: own face-by-face numbering, which for a cube is the one cubes are usually
#: written in. ``"image"`` is a rendered picture, one or more views deep.
Observation: TypeAlias = Literal["colors", "onehot", "ids", "facelets", "image"]

#: The five of them, to check a string against. A `Literal` says what a caller
#: should pass and not what one did, and an environment is built from
#: arguments that often came from a registry or a configuration file.
OBSERVATIONS: Final[tuple[str, ...]] = ("colors", "ids", "onehot", "facelets", "image")

#: What a step is worth.
#:
#: ``"cost"`` charges 1 for every move, so the return of an episode is minus the
#: length of the path, which is the cost-to-go. ``"sparse"`` pays 1 only for solving it.
RewardScheme: TypeAlias = Literal["cost", "sparse"]

#: How far from solved a reset leaves a puzzle: a fixed number of moves, or
#: ``(low, high)`` to draw one per episode.
Depth: TypeAlias = "int | tuple[int, int]"

#: Where a camera sits and which way is up for it, both in puzzle coordinates.
View: TypeAlias = "tuple[tuple[float, float, float], tuple[float, float, float]]"

#: A batch of observations, or a single one.
#:
#: Which whole-number width it is depends on the puzzle (`SEMANTICS.md` §13),
#: so all three are named here and the space an environment provides specifies
#: which one it produces.
ObsArray: TypeAlias = "NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.uint32] | NDArray[np.float32]"

#: What an observation is as far as Gymnasium's own generics can say.
#:
#: The five encodings have numeric element types (unsigned integers for
#: colors, identities, and indicators, and floats for images).
#: An environment is typed over this, and the space it provides
#: specifies what the numbers actually are.
ObsSpaceArray: TypeAlias = "NDArray[np.number]"

#: Rewards and flags share one parameter in `VectorEnv`, which needs both.
StepArray: TypeAlias = "NDArray[np.float64] | NDArray[np.bool_]"

#: One row per move: `new[i] = old[perm[a][i]]`.
#:
#: As narrow as the puzzle's slot count allows, matching the representation
#: returned by the underlying library. NumPy indexes with any integer width, so
#: ``state[table[a]]`` reads the same whichever this turns out to be.
Permutations: TypeAlias = "NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.uint32]"

#: Two corner views, from opposite corners, so that between them every face of
#: a cube is in one of the pictures.
#:
#: Closer in than the camera the library draws a puzzle from by default: an
#: image observation is small, usually 32 pixels square, and a picture in
#: which the puzzle is a quarter of the frame spends three quarters of a
#: network's input on the background. At this distance it fills a little under
#: half of it and still clears the edge.
_CORNER: Final[float] = 8.0 / 3.0**0.5
DEFAULT_VIEWS: Final[tuple[View, ...]] = (
    ((_CORNER, _CORNER, _CORNER), (0.0, 1.0, 0.0)),
    ((-_CORNER, -_CORNER, -_CORNER), (0.0, 1.0, 0.0)),
)


def depth_range(depth: Depth) -> tuple[int, int]:
    """Read a scramble depth, however it was given.

    Args:
        depth: A number of moves, or ``(low, high)`` to draw one per episode.

    Returns:
        The inclusive range to draw from.

    Raises:
        ValueError: If it is negative, or the range runs backward.
    """
    if isinstance(depth, tuple):
        low, high = int(depth[0]), int(depth[1])
    else:
        low = high = int(depth)
    if low < 0:
        msg = f"scramble must not be negative, got {low}"
        raise ValueError(msg)
    if high < low:
        msg = f"scramble range runs backward: {low} to {high}"
        raise ValueError(msg)
    return low, high


#: Recipes known not to jumble, so the usual ones cost nothing to check.
_KNOWN_GOOD: Final[frozenset[str]] = frozenset(non_jumbling())

#: Answers for recipes that are not cataloged, so a batch of environments pays
#: for the probe once instead of once each.
_PROBED: Final[dict[str, bool]] = {}

#: Face-by-face layouts, by recipe, so a batch of environments derives one between them.
_LAYOUTS: Final[dict[str, Layout]] = {}

#: Derived permutation tables, by recipe. The value is a one-tuple so that a
#: missing key ("not derived yet") is distinct from a `None` table ("derived,
#: and this puzzle has none").
_TABLES: Final[dict[str, tuple[Permutations | None]]] = {}

#: One lock per recipe, so that threads racing for the same table derive it
#: once between them instead of once each, while threads after *different*
#: tables still derive in parallel. Guarded by `_REGISTRY`, which is held only
#: for the lookup and never across a derivation.
_LOCKS: Final[dict[str, threading.Lock]] = {}
_REGISTRY: Final[threading.Lock] = threading.Lock()


def clear_tables() -> None:
    """Forget the derived tables and layouts.

    Only useful in a test that wants to measure the cost of deriving one.
    """
    with _REGISTRY:
        _TABLES.clear()
        _LOCKS.clear()
        _LAYOUTS.clear()


def permutation_table(puzzle: Puzzle) -> Permutations | None:
    """Every move of ``puzzle`` as a permutation of its sticker slots.

    Derived once per recipe and shared. The environments do not need this to
    step (the batch applies moves in Rust), but an agent that wants to look
    ahead without stepping does.

    Safe to call from several threads at once, and worth it: without the GIL,
    eight threads asking for the same table would otherwise each pay the full
    derivation. The first one through derives it and the rest wait.

    Args:
        puzzle: A puzzle to derive it from. Left as it was found.

    Returns:
        An ``(actions, stickers)`` array, or ``None`` for a puzzle whose moves
        are not fixed permutations.
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
            held = (None if rows is None else np.asarray(rows),)
            _TABLES[key] = held
    return held[0]


def layout_for(batch: PuzzleBatch) -> Layout:
    """The face-by-face layout of a batch's puzzle, derived once per recipe.

    Args:
        batch: The batch whose puzzle to describe.

    Returns:
        Its layout: for a cube, the numbering cubes are usually written in.
        For anything else, the puzzle's own.

    Raises:
        ValueError: If the puzzle jumbles, so that a move is not a
            permutation of its stickers and there is nothing to number.
    """
    key = batch.query
    held = _LAYOUTS.get(key)
    if held is not None:
        return held
    with _REGISTRY:
        lock = _LOCKS.setdefault(f"layout:{key}", threading.Lock())
    with lock:
        held = _LAYOUTS.get(key)
        if held is not None:
            return held
        try:
            held = Layout(key)
        except Exception as exc:
            msg = (
                f"{key!r} has no face-by-face layout: {exc}. "
                f"Use observation='colors' or 'ids' for a puzzle whose moves are not "
                f"permutations of its stickers."
            )
            raise ValueError(msg) from exc
        _LAYOUTS[key] = held
    return held


def refuse_jumbling(query: str, label: str) -> None:
    """Raise unless this puzzle's stickers stay on the solved lattice.

    A puzzle that *jumbles* (such as a Radiolarian, a jumble prism, or the Big Chop) has
    legal turns that leave pieces where no piece sits when it is solved, so
    there is no fixed set of slots to number and no state vector to observe.
    Thirty-six of the eighty-five cataloged puzzles do not jumble.

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


def refuse_unknown(mode: str) -> None:
    """Refuse an encoding that is not one of the five.

    Takes a plain string instead of the `Observation` alias on purpose: the
    alias says what a caller *should* pass, and this is here for the calls
    that did not, which arrive from registries and configuration files where
    nothing was checked.

    Args:
        mode: Whatever was asked for.

    Raises:
        ValueError: If it is not an encoding this package has.
    """
    if mode not in OBSERVATIONS:
        listed = ", ".join(OBSERVATIONS)
        msg = f"observation must be one of {listed}, not {mode!r}"
        raise ValueError(msg)


def image_shape(views: int, width: int, height: int) -> tuple[int, int, int]:
    """The shape one image observation has.

    Args:
        views: How many viewpoints are stacked.
        width: Frame width in pixels.
        height: Frame height.

    Returns:
        ``(3 * views, height, width)``: channels first, three per view.
    """
    return (3 * views, height, width)


def observation_space(
    batch: PuzzleBatch, mode: Observation, *, views: int = 1, width: int = 32, height: int = 32
) -> Space[ObsSpaceArray]:
    """The space one environment's observations live in.

    Args:
        batch: The batch the observations come from.
        mode: Which encoding to describe.
        views: How many viewpoints an image observation stacks.
        width: Frame width for an image observation.
        height: Frame height.

    Returns:
        The matching space. An encoding that is not one of the five is
        refused.
    """
    refuse_unknown(mode)
    k, colors = batch.sticker_count, batch.color_count
    if mode == "colors":
        return MultiDiscrete(np.full(k, colors, dtype=smallest(colors + 1)), dtype=smallest(colors))
    if mode == "ids":
        return MultiDiscrete(np.full(k, k, dtype=smallest(k + 1)), dtype=smallest(k))
    if mode == "onehot":
        return Box(low=0, high=1, shape=(k * colors,), dtype=np.uint8)
    if mode == "facelets":
        faces = layout_for(batch).face_count
        return MultiDiscrete(np.full(k, faces, dtype=smallest(faces + 1)), dtype=smallest(faces))
    return Box(low=0.0, high=1.0, shape=image_shape(views, width, height), dtype=np.float32)


def smallest(bound: int) -> type[np.uint8 | np.uint16 | np.uint32]:
    """The narrowest unsigned dtype that holds every value below ``bound``.

    The arrays returned by the library are as narrow as the puzzle allows, and a
    space whose dtype does not match the samples it is meant to describe is a
    space ``contains()`` rejects. The selection rule is centralized here to maintain consistency.

    Args:
        bound: One past the largest value: the number of colors, of slots or of faces.

    Returns:
        ``numpy.uint8``, ``numpy.uint16`` or ``numpy.uint32``.
    """
    if bound <= 2**8:
        return np.uint8
    if bound <= 2**16:
        return np.uint16
    return np.uint32


def action_space(action_count: int) -> Discrete[np.int_]:
    """The space one environment's actions live in.

    Args:
        action_count: How many moves the puzzle has: two per grip.

    Returns:
        A ``Discrete`` over the move indices.
    """
    return Discrete(action_count, start=0)


def draw(batch: PuzzleBatch, views: tuple[View, ...], width: int, height: int) -> NDArray[np.float32]:
    """Draw every puzzle in a batch, from each viewpoint in turn.

    The batch caches a drawing table per viewpoint, avoiding recomputation
    after the initial frame for each viewpoint.

    Args:
        batch: The puzzles to draw.
        views: Where to draw them from.
        width: Frame width in pixels.
        height: Frame height.

    Returns:
        ``(n, 3 * views, height, width)`` in ``[0, 1]``, the views stacked on the channel axis.

    Raises:
        ValueError: If there are no viewpoints to draw from.
    """
    if not views:
        msg = "an image observation needs at least one viewpoint"
        raise ValueError(msg)
    # Where the camera was, so that drawing an observation does not move it:
    # the same batch also draws the frames `render_mode` asks for, and those
    # are meant to come from where the environment was pointed.
    was = batch.camera
    frames: list[NDArray[np.float32]] = []
    for position, up in views:
        batch.configure(camera_position=position, camera_up=up)
        frames.append(np.asarray(batch.render(width, height, channels=3, channels_first=True, dtype="f4")))
    if was is not None:
        batch.configure(camera_position=was[0], camera_target=was[1], camera_up=was[2])
    return frames[0] if len(frames) == 1 else np.concatenate(frames, axis=1)


#: One character per color, so a board reads at a glance.
_GLYPHS: Final[str] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def solved_stickers(batch: PuzzleBatch) -> ObsArray:
    """The solved sticker array with the narrowest unsigned integer dtype.

    Args:
        batch: The puzzle batch to read solved stickers from.

    Returns:
        An array of solved sticker values with the appropriate unsigned integer dtype.
    """
    bound = batch.color_count
    if bound <= 2**8:
        return np.asarray(batch.solved_stickers, dtype=np.uint8)
    if bound <= 2**16:
        return np.asarray(batch.solved_stickers, dtype=np.uint16)
    return np.asarray(batch.solved_stickers, dtype=np.uint32)


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


def encode(
    batch: PuzzleBatch,
    mode: Observation,
    *,
    layout: Layout | None = None,
    views: tuple[View, ...] = DEFAULT_VIEWS,
    width: int = 32,
    height: int = 32,
) -> ObsArray:
    """Read a whole batch's state, in whichever encoding was asked for.

    Args:
        batch: The puzzles to read.
        mode: Which encoding to produce.
        layout: The face-by-face layout, for ``"facelets"``.
        views: Viewpoints, for ``"image"``.
        width: Frame width, for ``"image"``.
        height: Frame height, for ``"image"``.

    Returns:
        An ``(n, ...)`` array in that encoding. An encoding that is not one of the five is refused.
    """
    refuse_unknown(mode)
    if mode == "colors":
        return np.asarray(batch.observations())
    if mode == "ids":
        return np.asarray(batch.sticker_ids())
    if mode == "onehot":
        return np.asarray(batch.one_hot())
    if mode == "facelets":
        if layout is None:
            layout = layout_for(batch)
        return np.asarray(batch.facelets(layout))
    return draw(batch, views, width, height)
