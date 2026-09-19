"""The fine-grained API: recipes, grips, turns, and the typing stub.

These cover the surface the extension offers rather than the geometry underneath it, which the Rust suite pins.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import re

import pytest

import twistypuzzle as tp
from twistypuzzle import _native

CUBE: str = "?shell=C$1&cut=C$1/3"


# --------------------------------------------------------------------------- #
#  The stub and the extension must agree                                       #
# --------------------------------------------------------------------------- #


def _stub_members() -> dict[str, set[str]]:
    """Every class in `_native.pyi`, with the names declared on it.

    Parsed rather than imported: a stub is not runnable, and the point is to
    compare what it *claims* against what the compiled module has.
    """
    text = (Path(_native.__file__).parent / "_native.pyi").read_text(encoding="utf-8")
    classes: dict[str, set[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        klass = re.match(r"class (\w+)", line)
        if klass:
            current = klass.group(1)
            classes[current] = set()
            continue
        if not line.startswith("    ") or current is None:
            if line and not line.startswith(" "):
                current = None
            continue
        member = re.match(r"    (?:def )?(\w+)", line)
        if member and not member.group(1).startswith("__"):
            classes[current].add(member.group(1))
    return classes


def test_the_stub_describes_the_module_that_is_actually_there() -> None:
    """A stub that has drifted typechecks clean and fails at runtime.

    pyrefly checks Python against `_native.pyi`, never against the `.so`, so
    nothing but a test like this notices when the two disagree.
    """
    missing: list[str] = []
    for name, members in _stub_members().items():
        obj = getattr(_native, name, None)
        assert obj is not None, f"the stub declares {name}, which the module does not have"
        missing.extend(f"{name}.{m}" for m in sorted(members) if not hasattr(obj, m))
    assert not missing, f"declared in the stub but absent from the extension: {missing}"


def test_every_module_level_name_is_in_the_stub() -> None:
    text = (Path(_native.__file__).parent / "_native.pyi").read_text(encoding="utf-8")
    public = {n for n in dir(_native) if not n.startswith("_") or n == "__version__"}
    undeclared = sorted(n for n in public if not re.search(rf"\b{re.escape(n)}\b", text))
    assert not undeclared, f"the extension exports names the stub does not mention: {undeclared}"


# --------------------------------------------------------------------------- #
#  Recipes                                                                     #
# --------------------------------------------------------------------------- #


def test_a_recipe_round_trips_through_its_query() -> None:
    r = tp.Recipe.parse(CUBE)
    assert r.query == CUBE
    assert str(r) == CUBE
    assert r == tp.Recipe.parse(CUBE)
    assert hash(r) == hash(tp.Recipe.parse(CUBE))
    assert len(r.shell) == 1
    assert len(r.cuts) == 1


def test_a_recipe_can_be_built_shape_by_shape() -> None:
    r = tp.Recipe([tp.Shape.polyhedron("C")], [tp.Shape.polyhedron("C", "1/3")])
    assert r == tp.Recipe.parse(CUBE)
    assert r.build().piece_count == 26


def test_shapes_describe_themselves() -> None:
    poly = tp.Shape.polyhedron("C", "1/3")
    assert poly.kind == "polyhedron"
    assert poly.name == "C"
    assert poly.size == "1/3"
    assert poly.coefficients is None
    assert str(poly) == "C$1/3"

    plane = tp.Shape.plane(1, 0, 0, "1/2")
    assert plane.kind == "plane"
    assert plane.name is None
    assert plane.coefficients == ("1", "0", "0")
    assert str(plane) == "1,0,0$1/2"


def test_an_offset_is_exact_not_a_float() -> None:
    """A cut at a third is at a third, which no double is."""
    third = tp.Shape.polyhedron("C", "1/3").offset()
    assert float(third * tp.Real.from_int(3)) == 1.0
    root_five = tp.Shape.polyhedron("D", "sqrt(5)/5").offset()
    assert float(root_five * root_five * tp.Real.from_int(5)) == pytest.approx(1.0)


def test_offsets_may_be_written_three_ways() -> None:
    by_text = tp.Shape.polyhedron("C", "1/3")
    by_fraction = tp.Shape.polyhedron("C", tp.Fraction(1, 3))
    assert by_text == by_fraction
    assert tp.Shape.polyhedron("C", 2).size == "2"


def test_a_float_offset_is_refused() -> None:
    """Because 1/3 has no double, and a cube cut at 0.333... is another puzzle."""
    with pytest.raises(TypeError, match="expression"):
        tp.Shape.polyhedron("C", 0.3333333333333333)


def test_recipe_edits_leave_the_original_alone() -> None:
    r = tp.Recipe.parse(CUBE)
    deeper = r.with_cuts([tp.Shape.polyhedron("C", "1/2")])
    assert r.query == CUBE
    assert deeper.query == "?shell=C$1&cut=C$1/2"
    assert r.adding_cuts([tp.Shape.polyhedron("C", "1/2")]).query == ("?shell=C$1&cut=C$1/3&cut=C$1/2")


# --------------------------------------------------------------------------- #
#  Building a puzzle                                                           #
# --------------------------------------------------------------------------- #


def test_a_puzzle_can_be_named_queried_or_spelled_out() -> None:
    by_query = tp.Puzzle(CUBE)
    by_name = tp.Puzzle("Rubik's Cube (3x3x3)")
    by_recipe = tp.Puzzle(tp.Recipe.parse(CUBE))
    by_parts = tp.Puzzle(shell=tp.Shape.polyhedron("C"), cuts=tp.Shape.polyhedron("C", "1/3"))
    by_text = tp.Puzzle(shell="C$1", cuts=["C$1/3"])
    for p in (by_query, by_name, by_recipe, by_parts, by_text):
        assert p.piece_count == 26
        assert p.query == CUBE


def test_a_recipe_and_loose_parts_together_are_refused() -> None:
    with pytest.raises(TypeError, match="not both"):
        tp.Puzzle(CUBE, shell="C$1")


def test_a_puzzle_with_no_shell_is_refused() -> None:
    with pytest.raises(TypeError, match="shell"):
        tp.Puzzle()


def test_view_settings_are_keywords_as_well_as_properties() -> None:
    p = tp.Puzzle(
        CUBE,
        background=(1, 2, 3, 255),
        supersample=3,
        show_arrows=True,
        show_edges=False,
        show_pieces=True,
        line_width=2.0,
        fov=20.0,
        distance=9.0,
    )
    assert p.background == (1, 2, 3, 255)
    assert p.supersample == 3
    assert p.show_arrows is True
    assert p.show_edges is False
    assert p.show_pieces is True
    assert p.line_width == 2.0
    assert p.fov == 20.0
    assert p.distance == pytest.approx(9.0)


def test_every_setter_has_a_getter() -> None:
    """These were write-only, so reading one raised `AttributeError`."""
    p = tp.Puzzle(CUBE)
    for name, value in [
        ("background", (9, 8, 7, 6)),
        ("supersample", 4),
        ("show_arrows", False),
        ("show_edges", False),
        ("show_pieces", False),
        ("line_width", 3.0),
        ("fov", 30.0),
        ("camera_position", (1.0, 2.0, 3.0)),
        ("camera_up", (0.0, 0.0, 1.0)),
        ("camera_target", (0.5, 0.0, 0.0)),
    ]:
        setattr(p, name, value)
        assert getattr(p, name) == value, name


def test_a_seeded_scramble_repeats() -> None:
    a = tp.Puzzle(CUBE, seed=7, scramble=12)
    b = tp.Puzzle(CUBE, seed=7, scramble=12)
    c = tp.Puzzle(CUBE, seed=8, scramble=12)
    assert a.stickers() == b.stickers()
    assert a.stickers() != c.stickers()


# --------------------------------------------------------------------------- #
#  Turning                                                                     #
# --------------------------------------------------------------------------- #


def test_grips_describe_the_layers() -> None:
    p = tp.Puzzle(CUBE)
    grips = p.grips()
    assert len(grips) == p.grip_count == 6
    for i, g in enumerate(grips):
        assert g.index == i
        assert g.piece_count + g.other_count == p.piece_count
        assert sum(c * c for c in g.axis) == pytest.approx(1.0)
        assert g.plane


def test_a_turn_can_land_on_a_named_stop() -> None:
    p = tp.Puzzle(CUBE)
    stops = p.stops(0)
    assert stops == pytest.approx([0.0, 90.0, 180.0, 270.0])

    p.turn_to(0, 2)  # a half turn, which no single direction reaches
    p.settle()
    half = p.stickers()

    q = tp.Puzzle(CUBE)
    q.turn(0, 1, repeat=2)
    q.settle()
    assert q.stickers() == half


def test_a_turn_out_of_range_is_refused() -> None:
    p = tp.Puzzle(CUBE)
    with pytest.raises(ValueError, match="grip index"):
        p.turn(99, 1)
    with pytest.raises(ValueError, match="stop index"):
        p.turn_to(0, 99)


def test_undo_takes_back_a_turn_however_it_was_made() -> None:
    p = tp.Puzzle(CUBE)
    solved = p.stickers()
    assert p.undo() is False

    p.turn_to(0, 2)
    p.settle()
    assert p.stickers() != solved
    assert p.undo() is True
    assert p.stickers() == solved
    assert p.history_length == 0

    p.apply("A B' C2")
    assert p.history_length == 4
    while p.undo():
        pass
    assert p.stickers() == solved


def test_reset_keeps_the_camera() -> None:
    p = tp.Puzzle(CUBE, background=(1, 2, 3, 4), fov=42.0)
    p.scramble(5)
    p.settle()
    assert not p.is_solved
    p.reset()
    assert p.is_solved
    assert p.background == (1, 2, 3, 4)
    assert p.fov == 42.0


# --------------------------------------------------------------------------- #
#  Rendering                                                                   #
# --------------------------------------------------------------------------- #


def test_render_overrides_do_not_stick() -> None:
    p = tp.Puzzle(CUBE, background=(255, 255, 255, 255), supersample=2)
    before = p.render(32, 32)
    other = p.render(32, 32, background=(0, 0, 0, 255), supersample=1)
    assert p.background == (255, 255, 255, 255)
    assert p.supersample == 2
    assert other.to_bytes() != before.to_bytes()
    assert p.render(32, 32).to_bytes() == before.to_bytes()


def test_render_and_piece_take_the_puzzle_exclusively() -> None:
    """Reading a coordinate as a float refines the field it shares.

    The numbers that come out do not depend on the interleaving, but the
    mutation is real (the shared interval narrows and the memo caches fill),
    so two threads doing it to one puzzle would be a data race. The borrow
    check refuses it rather than letting it happen.
    """
    for name in ("render", "piece", "grips", "stickers"):
        sig = inspect.signature(getattr(tp.Puzzle, name))
        assert "self" in sig.parameters, name


def test_render_many_accepts_recipes_as_well_as_names() -> None:
    images = tp.render_many([tp.Recipe.parse(CUBE), "Megaminx"], 24, 24)
    assert len(images) == 2
    assert all(i.width == 24 for i in images)
