"""Structural types for the values the native module accepts and returns.

Most of these are ``TypedDict`` declarations and type aliases, not classes: the
native module returns standard dictionaries and accepts strings, and these type
definitions formalize their schemas for static type checkers and documentation.
They live here instead of in the extension because a compiled module cannot carry
typing constructs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TypeAlias

    from ._native import Fraction, Recipe, Shape


__all__: list[str] = [
    "ArrayInterface",
    "ArrayLike",
    "Atom",
    "BlendMode",
    "ExprLike",
    "Model",
    "PieceData",
    "PieceFace",
    "RecipeLike",
    "Rgba",
    "ShapeLike",
]

_ArrayInterfaceValue: TypeAlias = (
    "tuple[int, ...] | str | int | tuple[int, int | bool] | bool | list[tuple[str, str]] | None"
)

#: How :meth:`Image.blit` and :meth:`Image.fill_rect` combine source with
#: destination: replace, alpha-composite, or add.
BlendMode: TypeAlias = Literal["copy", "over", "add"]

#: A color as red, green, blue and alpha bytes.
Rgba: TypeAlias = "tuple[int, int, int, int]"

#: An exact number, written as an expression (``"sqrt(5)/5"``), as a whole
#: number, or as a :class:`~twistypuzzle.Fraction`.
#:
#: Deliberately not ``float``: a float is not the number it is spelled. ``1/3``
#: has no ``double``, and a cube cut at 0.3333333333333333 is a different puzzle
#: from one cut at a third.
ExprLike: TypeAlias = "str | int | Fraction"

#: A shell or a cut, as a :class:`~twistypuzzle.Shape` or as its text
#: (``"C$1/3"``).
ShapeLike: TypeAlias = "Shape | str"

#: Whatever names a puzzle: a recipe query string, a catalog name, or a
#: :class:`~twistypuzzle.Recipe`.
RecipeLike: TypeAlias = "str | Recipe"

#: One ground atom: a predicate followed by its arguments, all text.
#:
#: A state can be compared against a partial goal without going through the sticker array.
Atom: TypeAlias = "tuple[str, ...]"

#: A set of ground atoms describing one state.
Model: TypeAlias = "frozenset[Atom]"


class ArrayInterface(TypedDict):
    """The NumPy array interface an :class:`Image` exposes, for zero-copy views."""

    shape: tuple[int, int, int]
    typestr: str
    version: int
    data: tuple[int, int]


@runtime_checkable
class ArrayLike(Protocol):
    """Anything that describes its own block of memory the way NumPy does.

    A NumPy array, this package's own :class:`~twistypuzzle.Array`, and
    anything else carrying ``__array_interface__``. Input arrays are read as
    contiguous memory buffers instead of per-element iterations, processing
    state arrays across large batches in a single linear memory pass without
    per-element Python interpreter transitions.

    Declared structurally because the package depends on nothing: NumPy is an
    optional extra, and its arrays satisfy this without either side importing
    the other.
    """

    @property
    def __array_interface__(self) -> Mapping[str, _ArrayInterfaceValue]: ...  # ruff: ignore[bad-dunder-method-name]


class PieceFace(TypedDict):
    """One face of a piece: indices into the piece's vertices, plus appearance.

    ``color`` is packed 0xRRGGBB in sRGB, the usual web encoding.
    ``interior`` marks a face produced by a cut instead of the puzzle's
    outer shell, so it is hidden unless the puzzle
    is turned.
    """

    vertices: list[int]
    color: int
    interior: bool


class PieceData(TypedDict):
    """A single piece: its vertices, its faces, and its exact rotation.

    ``rotation`` is the quaternion in exact form as a string, because its
    components are algebraic numbers that no float can hold.
    """

    vertices: list[tuple[float, float, float]]
    faces: list[PieceFace]
    rotation: str
