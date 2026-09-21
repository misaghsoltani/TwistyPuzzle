"""Exact-arithmetic twisty puzzle simulator with an integrated renderer.

Every cut and every turn is computed in an algebraic number field, so the
geometry never drifts: a face that should meet another meets it exactly, and a
rotation that should be a fifth of a turn is a fifth of a turn.

    >>> import twistypuzzle as tp
    >>> p = tp.Puzzle("Rubik's Cube (3x3x3)")
    >>> p.piece_count
    26
    >>> p.apply("A B' A")
    3
    >>> p.is_solved
    False

A puzzle can be named, written as a recipe query string, or spelled out shape by
shape, and every view setting is both a keyword and a property::

    >>> q = tp.Puzzle(shell=tp.Shape.polyhedron("C"),
    ...               cuts=tp.Shape.polyhedron("C", "1/3"),
    ...               background=(255, 255, 255, 255), supersample=2)
    >>> q.query
    '?shell=C$1&cut=C$1/3'

Rendering is built in and needs no GPU, display server or browser. Frames come
back as :class:`Image`, which NumPy can view without copying::

    >>> import numpy as np                      # doctest: +SKIP
    >>> a = np.asarray(q.render(256, 256))      # doctest: +SKIP
    >>> a.shape                                 # doctest: +SKIP
    (256, 256, 4)

A state can be read back as the integer array.

    >>> p = tp.Puzzle("Rubik's Cube (3x3x3)")
    >>> p.solved_stickers == [i // 9 for i in range(54)]
    True

Many states of one puzzle are turned together by :class:`PuzzleBatch`, in one
call into Rust across every core. For a puzzle that does not jumble, a move is
a fixed permutation of its slots, so a turn is a gather instead of a re-derivation
of its geometry, and a frame is a lookup per pixel instead of a rasterization::

    >>> b = tp.PuzzleBatch("Rubik's Cube (3x3x3)", 256, seed=0)
    >>> b.reset(scramble=20)
    >>> b.observations().shape
    (256, 54)
    >>> solved, applied = b.step([0] * 256)

Batch queries return instances of :class:`Array`: contiguous memory buffers
that ``numpy.asarray`` inspects without copying, and that ``tolist()`` reads
directly without requiring NumPy, since the package has no external dependencies.

:class:`Layout` numbers a puzzle's stickers face by face: for a cube, the way
cubes are usually numbered, six faces in a fixed order each read as a grid.
For anything else, the puzzle's own order, which is already canonical. Its moves
are permutations of a small array, so a layout is a complete engine for
its puzzle with no geometry in it at all::

    >>> c = tp.Layout()
    >>> c.goal_colors.tolist() == [i // 9 for i in range(54)]
    True
    >>> after = c.next_states(c.goal_colors, c.move_index("R"))

Every array is as narrow as its puzzle allows: a 3x3x3 has 54 slots and 6
colors, so its states, slot numbers and moves are all bytes, while a 9x9x9
needs two bytes for a slot number and still one for a color. ``Array.typestr``
says which, and the Gymnasium spaces are built from the same rule.

An optional Gymnasium environment lives in :mod:`twistypuzzle.gym`, and an
optional desktop interface installs with the ``gui`` extra, which brings a
``twistypuzzle-gui`` command with it.
"""

from __future__ import annotations

from itertools import starmap
from typing import NamedTuple

from ._native import (
    Array,
    Fraction,
    Grip,
    Image,
    Layout,
    Puzzle,
    PuzzleBatch,
    PuzzleError,
    Real,
    Recipe,
    Shape,
    __version__,
    build_many,
    catalog,
    evaluate,
    factor_polynomial,
    free_threaded,
    jumbles,
    non_jumbling,
    polyhedra,
    render_many,
    thread_count,
)
from ._types import (
    ArrayInterface,
    ArrayLike,
    Atom,
    BlendMode,
    ExprLike,
    Model,
    PieceData,
    PieceFace,
    RecipeLike,
    Rgba,
    ShapeLike,
)

__all__: list[str] = [
    "Array",
    "ArrayInterface",
    "ArrayLike",
    "Atom",
    "BlendMode",
    "CatalogEntry",
    "ExprLike",
    "Fraction",
    "Grip",
    "Image",
    "Layout",
    "Model",
    "PieceData",
    "PieceFace",
    "Puzzle",
    "PuzzleBatch",
    "PuzzleError",
    "Real",
    "Recipe",
    "RecipeLike",
    "Rgba",
    "Shape",
    "ShapeLike",
    "__version__",
    "build_many",
    "catalog",
    "catalog_entries",
    "catalog_names",
    "evaluate",
    "factor_polynomial",
    "free_threaded",
    "ground_model",
    "jumbles",
    "non_jumbling",
    "non_jumbling_entries",
    "polyhedra",
    "render_many",
    "render_puzzle",
    "thread_count",
]


class CatalogEntry(NamedTuple):
    """One puzzle in the shipped catalog.

    Names are not unique (ten entries carry the placeholder ``"Unknown"``),
    but recipes are, so use :attr:`recipe` to identify a puzzle exactly.
    """

    name: str
    """Common name, e.g. ``"Rubik's Cube (3x3x3)"``."""
    family: str
    """Shell family, e.g. ``"Cubes"``."""
    kind: str
    """Turning style, e.g. ``"Face-turning"``."""
    recipe: str
    """Recipe query string, e.g. ``"?shell=C$1&cut=C$1/3"``."""


def catalog_entries() -> list[CatalogEntry]:
    """Every cataloged puzzle, as named tuples.

    Returns:
        The catalog, in order.
    """
    return list(starmap(CatalogEntry, catalog()))


def catalog_names() -> list[str]:
    """Names of every cataloged puzzle in the library.

    Returns:
        List of puzzle names available in the catalog.
    """
    return [name for name, _family, _kind, _recipe in catalog()]


def non_jumbling_entries() -> list[CatalogEntry]:
    """The cataloged puzzles that have a sticker array.

    A puzzle that *jumbles* (such as a Radiolarian, a jumble prism, or the Big Chop) has
    legal turns that leave pieces where no piece sits when it is solved, so
    there is no fixed set of sticker slots to number. Thirty-six of the
    eighty-five cataloged puzzles do not jumble, and only those can be read as
    a state vector or wrapped in a learning environment.

    Returns:
        The catalog entries whose stickers stay on the solved lattice.
    """
    encodable = set(non_jumbling())
    return [e for e in catalog_entries() if e.recipe in encodable]


def ground_model(puzzle: Puzzle) -> Model:
    """A puzzle's state as a set of ground atoms.

    A state is a set of ``("color", "s<slot>", "c<color>")`` atoms,
    so it can be compared against a partial goal without going through the array.

    Args:
        puzzle: The puzzle to describe. Reading its state takes it exclusively.

    Returns:
        One atom per sticker slot.
    """
    return frozenset(puzzle.ground_atoms())


def render_puzzle(
    recipe: RecipeLike,
    width: int = 512,
    height: int = 512,
    *,
    background: Rgba = (255, 255, 255, 255),
    supersample: int = 2,
    show_arrows: bool = False,
    show_edges: bool = True,
    scramble: int = 0,
    seed: int | None = None,
    yaw: float = 0.0,
    pitch: float = 0.0,
    distance: float | None = None,
) -> Image:
    """Build a puzzle and render one frame of it.

    A convenience wrapper for the common case of obtaining a picture of a
    puzzle. For more than one, :func:`render_many` does the whole batch across
    every core.

    Args:
        recipe: Catalog name, recipe query string, or :class:`Recipe`.
        width: Image width in pixels.
        height: Image height in pixels.
        background: RGBA background color.
        supersample: Supersampling factor, where 1 disables antialiasing.
        show_arrows: Whether to draw the arrows that drive the turns.
        show_edges: Whether to draw piece outlines.
        scramble: Number of random moves to apply before rendering.
        seed: Seed for the scrambler, for a reproducible scramble.
        yaw: Degrees to swing the camera sideways from head-on.
        pitch: Degrees to raise it.
        distance: How far out the camera sits, or the default if omitted.

    Returns:
        The rendered frame.
    """
    puzzle = Puzzle(
        recipe,
        background=background,
        supersample=supersample,
        show_arrows=show_arrows,
        show_edges=show_edges,
        yaw=yaw,
        pitch=pitch,
        distance=distance,
        seed=seed,
        scramble=scramble,
    )
    return puzzle.render(width, height)
