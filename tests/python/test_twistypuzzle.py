"""Tests for the installed `twistypuzzle` wheel.

These test the package as a user gets it, through the extension module
rather than the Rust crate, so they catch things the Rust test suite structurally
cannot: a stub that disagrees with the runtime, a name exported that does not
exist, a GIL interaction, a buffer whose shape is wrong.

Run against a built wheel::

    maturin build --release && pip install target/wheels/*.whl pytest
    pytest tests/python
"""

from __future__ import annotations

import math

import pytest

import twistypuzzle as tp

# Puzzles with hand-checked piece and grip counts, spanning the three number
# fields the catalog uses: Q, Q(sqrt 2) and Q(sqrt 5).
KNOWN = [
    # name, pieces, grips, recipe
    ("2x2x2 Cube", 8, 6, "?shell=C$1&cut=C$0"),
    ("Rubik's Cube (3x3x3)", 26, 6, "?shell=C$1&cut=C$1/3"),
    ("Pyraminx", 14, 8, "?shell=T$1&cut=T$-1/3&cut=T$-5/3"),
    ("Megaminx", 62, 12, "?shell=D$1&cut=D$2/(sqrt(5)+1)"),
]


# --------------------------------------------------------------------------
# packaging
# --------------------------------------------------------------------------


def test_every_exported_name_exists() -> None:
    """`__all__` must not promise anything the module does not have."""
    missing = [n for n in tp.__all__ if not hasattr(tp, n)]
    assert missing == []


def test_version_is_a_string() -> None:
    assert isinstance(tp.__version__, str)
    assert tp.__version__.count(".") >= 1


def test_puzzle_error_is_an_exception() -> None:
    assert issubclass(tp.PuzzleError, Exception)


# --------------------------------------------------------------------------
# catalog
# --------------------------------------------------------------------------


def test_catalog_is_complete_and_well_formed() -> None:
    entries = tp.catalog()
    assert len(entries) == 85
    for name, family, kind, recipe in entries:
        assert name
        assert family
        assert kind
        assert recipe.startswith("?shell=")
    # Recipes identify a puzzle, whereas names do not. Ten catalog entries carry
    # the placeholder name "Unknown" (see `src/catalog.rs`).
    assert len({recipe for _, _, _, recipe in entries}) == 85
    names = tp.catalog_names()
    assert len(names) == 85
    assert names.count("Unknown") == 10


def test_an_ambiguous_name_is_refused() -> None:
    """`named` must not silently pick one of ten puzzles called "Unknown"."""
    with pytest.raises(ValueError, match="build one by recipe"):
        tp.Puzzle.named("Unknown")
    # The recipes it points at do work.
    assert tp.Puzzle("?shell=O$1&cut=jC$2/9").piece_count > 0


def test_polyhedra_are_listed() -> None:
    polys = tp.polyhedra()
    assert ("T", "Tetrahedron") in polys
    assert all(len(code) >= 1 and label for code, label in polys)


@pytest.mark.parametrize(("name", "pieces", "grips", "recipe"), KNOWN)
def test_known_puzzles_have_the_expected_shape(name: str, pieces: int, grips: int, recipe: str) -> None:
    p = tp.Puzzle.named(name)
    assert p.piece_count == pieces
    assert p.grip_count == grips
    assert p.query == recipe
    assert p.recipe == tp.Recipe.parse(recipe)
    # Building from the recipe string must give the same puzzle.
    assert tp.Puzzle(recipe).piece_count == pieces


def test_unknown_puzzle_name_is_rejected() -> None:
    with pytest.raises(Exception):
        tp.Puzzle.named("Not A Puzzle")


def test_malformed_recipe_is_rejected() -> None:
    with pytest.raises(Exception):
        tp.Puzzle("?shell=NotAPolyhedron$1")


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_piece_geometry_is_consistent() -> None:
    p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    for i in range(p.piece_count):
        d = p.piece(i)
        verts = d["vertices"]
        assert verts, "a piece always has vertices"
        for v in verts:
            assert len(v) == 3
            assert all(isinstance(c, float) and math.isfinite(c) for c in v)
        for f in d["faces"]:
            assert len(f["vertices"]) >= 3, "a face is at least a triangle"
            assert all(0 <= k < len(verts) for k in f["vertices"])
            assert 0 <= f["color"] <= 0xFFFFFF
            assert isinstance(f["interior"], bool)
        assert d["rotation"]


def test_a_solved_cube_shows_six_colors() -> None:
    p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    exterior = {f["color"] for i in range(p.piece_count) for f in p.piece(i)["faces"] if not f["interior"]}
    # The six Rubik's Cube colors.
    assert exterior == {0xFFFFFF, 0xC41E3A, 0x009E60, 0x0051BA, 0xFF5800, 0xFFD500}


def test_piece_index_is_bounds_checked() -> None:
    p = tp.Puzzle.named("2x2x2 Cube")
    with pytest.raises(IndexError):
        p.piece(p.piece_count)


def test_cube_stops_are_quarter_turns() -> None:
    p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    assert p.stops(0) == pytest.approx([0.0, 90.0, 180.0, 270.0], abs=1e-9)


def test_megaminx_stops_are_fifths_of_a_turn() -> None:
    """Fifths of a turn.

    The rotation is exact. The *angle in degrees* is a float, so the tolerance
    here is floating point's: an inverse cosine away from the exact value,
    not an artifact of how the conversion was done.
    """
    p = tp.Puzzle.named("Megaminx")
    assert p.stops(0) == pytest.approx([0.0, 72.0, 144.0, 216.0, 288.0], abs=1e-9)


# --------------------------------------------------------------------------
# exact arithmetic
# --------------------------------------------------------------------------


def test_square_roots_are_exact_not_approximate() -> None:
    """The point of the library: sqrt(2)**2 is 2 on the nose, not 2 + 4e-16."""
    two = tp.Real.from_int(2)
    r = tp.evaluate("sqrt(2)")
    assert (r * r - two).sign() == 0
    assert r * r == two
    assert r > tp.Real.from_int(1)
    assert r < two


def test_golden_ratio_identity_holds_exactly() -> None:
    phi = tp.evaluate("(1+sqrt(5))/2")
    one = tp.Real.from_int(1)
    assert (phi * phi - phi - one).sign() == 0


def test_floats_are_correctly_rounded() -> None:
    """Every conversion to float is correctly rounded. There is only one."""
    assert float(tp.evaluate("sqrt(2)")) == math.sqrt(2)
    assert float(tp.evaluate("sqrt(3)")) == math.sqrt(3)
    assert float(tp.evaluate("(1+sqrt(5))/2")) == (1 + math.sqrt(5)) / 2
    assert float(tp.evaluate("-sqrt(2)")) == -math.sqrt(2)
    assert float(tp.Fraction(1, 3)) == 1 / 3
    assert float(tp.Fraction(-22, 7)) == -22 / 7

    # One conversion, and the renderer reads coordinates through it. A second
    # entry point offering a cheaper, less accurate answer would be a trap.
    assert not hasattr(tp.evaluate("sqrt(2)"), "to_number")
    assert not hasattr(tp.Fraction(1, 3), "to_number")


def test_precision_refinement_does_not_disturb_the_puzzle() -> None:
    """Reading floats must not change what the renderer draws.

    Refining a field narrows an interval shared by every number in it, so
    reading one float could in principle move another. It cannot: a conversion
    refines until the answer is the correctly rounded double, and that answer
    does not depend on how much refining came before it.
    """
    p = tp.Puzzle.named("Megaminx")
    before = p.render(64, 64).to_bytes()
    for _ in range(20):
        float(tp.evaluate("(1+sqrt(5))/2"))
    assert p.render(64, 64).to_bytes() == before


def test_fraction_arithmetic() -> None:
    a, b = tp.Fraction(1, 2), tp.Fraction(1, 3)
    assert float(a + b) == pytest.approx(5 / 6)
    assert float(a - b) == pytest.approx(1 / 6)
    assert float(a * b) == pytest.approx(1 / 6)
    assert float(a / b) == pytest.approx(3 / 2)
    assert float(-a) == -0.5
    assert float(abs(tp.Fraction(-1, 2))) == 0.5
    assert a > b
    assert b < a


def test_division_by_zero_is_reported() -> None:
    with pytest.raises(ZeroDivisionError):
        tp.Fraction(1, 0)


def test_polynomial_factoring() -> None:
    # x^2 - 1 = (x - 1)(x + 1), with coefficients lowest-degree first.
    factors = tp.factor_polynomial([-1, 0, 1])
    assert sorted(factors) == [["-1", "1"], ["1", "1"]]
    # x^2 + 1 is irreducible over Q.
    assert tp.factor_polynomial([1, 0, 1]) == [["1", "0", "1"]]


def test_real_interval_brackets_the_value() -> None:
    r = tp.evaluate("sqrt(2)")
    lo, hi = r.interval()
    assert float(lo) <= math.sqrt(2) <= float(hi)
    assert r.field


# --------------------------------------------------------------------------
# images and blitting
# --------------------------------------------------------------------------


def test_image_basics() -> None:
    im = tp.Image(8, 4, (10, 20, 30, 255))
    assert (im.width, im.height, im.stride) == (8, 4, 32)
    assert im.pixel(0, 0) == (10, 20, 30, 255)
    assert im.pixel(7, 3) == (10, 20, 30, 255)
    im.set_pixel(2, 1, (1, 2, 3, 4))
    assert im.pixel(2, 1) == (1, 2, 3, 4)
    assert len(im.to_bytes()) == 8 * 4 * 4
    im.fill((0, 0, 0, 0))
    assert im.pixel(2, 1) == (0, 0, 0, 0)


def test_image_copy_is_independent() -> None:
    a = tp.Image(4, 4, (1, 1, 1, 255))
    b = a.copy()
    b.fill((2, 2, 2, 255))
    assert a.pixel(0, 0) == (1, 1, 1, 255)
    assert b.pixel(0, 0) == (2, 2, 2, 255)


def test_blit_copies_the_source() -> None:
    dst = tp.Image(8, 8, (0, 0, 0, 255))
    src = tp.Image(2, 2, (255, 0, 0, 255))
    dst.blit(src, 3, 3)
    assert dst.pixel(3, 3) == (255, 0, 0, 255)
    assert dst.pixel(4, 4) == (255, 0, 0, 255)
    assert dst.pixel(5, 5) == (0, 0, 0, 255)
    assert dst.pixel(2, 2) == (0, 0, 0, 255)


def test_blit_clips_on_both_surfaces() -> None:
    """A blit that runs off either edge must clip, not wrap or crash."""
    dst = tp.Image(4, 4, (0, 0, 0, 255))
    src = tp.Image(4, 4, (9, 9, 9, 255))
    dst.blit(src, 2, 2)  # bottom-right corner only
    assert dst.pixel(3, 3) == (9, 9, 9, 255)
    assert dst.pixel(1, 1) == (0, 0, 0, 255)

    dst.fill((0, 0, 0, 255))
    dst.blit(src, -2, -2)  # off the top-left
    assert dst.pixel(0, 0) == (9, 9, 9, 255)
    assert dst.pixel(1, 1) == (9, 9, 9, 255)
    assert dst.pixel(2, 2) == (0, 0, 0, 255)

    dst.fill((0, 0, 0, 255))
    dst.blit(src, 100, 100)  # entirely outside
    assert dst.pixel(0, 0) == (0, 0, 0, 255)
    assert dst.pixel(3, 3) == (0, 0, 0, 255)


def test_blit_over_alpha_composites() -> None:
    dst = tp.Image(2, 2, (0, 0, 0, 255))
    src = tp.Image(2, 2, (255, 255, 255, 128))
    dst.blit(src, mode="over")
    r, g, b, a = dst.pixel(0, 0)
    assert a == 255
    assert 120 <= r <= 136
    assert r == g == b


def test_fill_rect_is_clipped_and_blended() -> None:
    im = tp.Image(4, 4, (0, 0, 0, 255))
    im.fill_rect(2, 2, 10, 10, (5, 5, 5, 255))
    assert im.pixel(3, 3) == (5, 5, 5, 255)
    assert im.pixel(1, 1) == (0, 0, 0, 255)


def test_downsample_averages() -> None:
    im = tp.Image(4, 4, (0, 0, 0, 255))
    im.fill_rect(0, 0, 2, 2, (255, 255, 255, 255))
    small = im.downsample(2)
    assert (small.width, small.height) == (2, 2)
    assert small.pixel(0, 0) == (255, 255, 255, 255)
    assert small.pixel(1, 1) == (0, 0, 0, 255)


def test_resized_changes_dimensions() -> None:
    im = tp.Image(8, 8, (7, 7, 7, 255))
    out = im.resized(4, 16)
    assert (out.width, out.height) == (4, 16)


def test_from_bytes_round_trips() -> None:
    im = tp.Image(3, 2, (4, 5, 6, 255))
    again = tp.Image.from_bytes(3, 2, im.to_bytes())
    assert again.to_bytes() == im.to_bytes()


def test_png_encodes() -> None:
    png = tp.Image(16, 16, (1, 2, 3, 255)).to_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IEND" in png


def test_save_writes_a_png(tmp_path) -> None:
    out = tmp_path / "img.png"
    tp.Image(8, 8, (1, 2, 3, 255)).save(str(out))
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_render_is_deterministic() -> None:
    """Two renders of one unchanged puzzle must agree byte for byte."""
    p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    assert p.render(128, 128).to_bytes() == p.render(128, 128).to_bytes()


def test_two_identical_puzzles_render_identically() -> None:
    a = tp.Puzzle.named("Megaminx")
    b = tp.Puzzle.named("Megaminx")
    assert a.render(96, 96).to_bytes() == b.render(96, 96).to_bytes()


def test_render_draws_the_puzzle() -> None:
    p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    p.background = (255, 255, 255, 255)
    im = p.render(128, 128)
    assert (im.width, im.height) == (128, 128)
    # The center lands on a face, and the corners on the background.
    assert im.pixel(64, 64) != (255, 255, 255, 255)
    assert im.pixel(0, 0) == (255, 255, 255, 255)


def test_background_is_honored() -> None:
    p = tp.Puzzle.named("2x2x2 Cube")
    p.background = (1, 2, 3, 255)
    assert p.render(64, 64).pixel(0, 0) == (1, 2, 3, 255)


def test_supersampling_smooths_without_moving_anything() -> None:
    p = tp.Puzzle.named("2x2x2 Cube")
    p.supersample = 1
    flat = p.render(96, 96)
    p.supersample = 3
    smooth = p.render(96, 96)
    assert flat.to_bytes() != smooth.to_bytes()
    # Well inside a face, away from the edge lines, the color is unchanged:
    # supersampling resolves edges, it does not shift the geometry.
    assert flat.pixel(30, 30) == smooth.pixel(30, 30) == (235, 38, 71, 255)


def test_toggling_arrows_and_edges() -> None:
    p = tp.Puzzle.named("2x2x2 Cube")
    p.show_edges = False
    p.show_arrows = False
    plain = p.render(96, 96).to_bytes()
    p.show_edges = True
    assert p.render(96, 96).to_bytes() != plain


def test_arrow_hover_and_clear_hover() -> None:
    p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    p.look_from(-28.0, 20.0, 12.0)
    assert p.hovered_arrow is None
    p.pointer_move(0.050, 0.775)
    assert p.hovered_arrow == 6
    p.clear_hover()
    assert p.hovered_arrow is None
    # When arrows are hidden, picking and hovering arrows are disabled
    p.show_arrows = False
    assert p.arrow_at(0.050, 0.775) is None
    p.pointer_move(0.050, 0.775)
    assert p.hovered_arrow is None


def test_render_puzzle_convenience() -> None:
    im = tp.render_puzzle("Rubik's Cube (3x3x3)", 64, 64, background=(9, 9, 9, 255))
    assert (im.width, im.height) == (64, 64)
    assert im.pixel(0, 0) == (9, 9, 9, 255)
    # Accepts a recipe string as well as a catalog name.
    assert tp.render_puzzle("?shell=C$1&cut=C$1/3", 32, 32).width == 32


# --------------------------------------------------------------------------
# moves and animation
# --------------------------------------------------------------------------


def test_turning_changes_the_picture() -> None:
    p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    before = p.render(96, 96).to_bytes()
    p.turn(0, 1)
    p.settle()
    assert p.move_count == 1
    assert p.render(96, 96).to_bytes() != before


def test_four_quarter_turns_restore_the_cube() -> None:
    """Exact arithmetic means this is byte-identical, not merely close."""
    p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    before = p.render(96, 96).to_bytes()
    for _ in range(4):
        p.turn(0, 1)
        p.settle()
    assert p.move_count == 4
    assert p.render(96, 96).to_bytes() == before


def test_a_turn_and_its_inverse_cancel() -> None:
    p = tp.Puzzle.named("Megaminx")
    before = p.render(96, 96).to_bytes()
    p.turn(0, 1)
    p.settle()
    p.turn(0, -1)
    p.settle()
    assert p.render(96, 96).to_bytes() == before


def test_scrambling_is_reproducible_from_a_seed() -> None:
    def scrambled() -> bytes:
        p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
        p.seed(12345)
        p.scramble(12)
        p.settle()
        assert p.move_count == 12
        return p.render(96, 96).to_bytes()

    assert scrambled() == scrambled()


def test_settle_matches_the_animated_path() -> None:
    """Settling must land exactly where advancing frame by frame lands."""
    a = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    a.seed(7)
    a.scramble(5)
    a.settle()

    b = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    b.seed(7)
    b.scramble(5)
    guard = 0
    while (b.animating or b.move_count < 5) and guard < 5000:
        b.advance(16.0)
        guard += 1

    assert not a.animating
    assert not b.animating
    assert a.move_count == b.move_count == 5
    assert a.render(96, 96).to_bytes() == b.render(96, 96).to_bytes()


def test_animation_is_visible_midway() -> None:
    p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    rest = p.render(96, 96).to_bytes()
    p.turn(0, 1)
    assert p.animating
    p.advance(8.0)
    assert p.render(96, 96).to_bytes() != rest
    p.settle()
    assert not p.animating


def test_frame_advances_and_renders() -> None:
    p = tp.Puzzle.named("2x2x2 Cube")
    im = p.frame(16.0, 64, 64)
    assert (im.width, im.height) == (64, 64)


def test_grips_are_addressable() -> None:
    p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    assert p.grip_count == 6
    assert all(p.grip(i) for i in range(p.grip_count))
    with pytest.raises(IndexError):
        p.grip(p.grip_count)


def test_orbit_moves_the_camera() -> None:
    p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    before = p.render(96, 96).to_bytes()
    p.orbit(0.3, 0.2)
    assert p.render(96, 96).to_bytes() != before


# --------------------------------------------------------------------------
# NumPy interoperability
# --------------------------------------------------------------------------


def test_numpy_sees_the_pixels_without_copying() -> None:
    np = pytest.importorskip("numpy")
    im = tp.Image(6, 4, (1, 2, 3, 4))
    a = np.asarray(im)
    assert a.shape == (4, 6, 4)  # height, width, RGBA
    assert a.dtype == np.uint8
    assert tuple(a[0, 0]) == (1, 2, 3, 4)
    # A view, not a copy: writing through it changes the image.
    a[0, 0] = (9, 8, 7, 6)
    assert im.pixel(0, 0) == (9, 8, 7, 6)


def test_numpy_view_of_a_rendered_frame() -> None:
    np = pytest.importorskip("numpy")
    p = tp.Puzzle.named("Rubik's Cube (3x3x3)")
    p.background = (255, 255, 255, 255)
    a = np.asarray(p.render(64, 64))
    assert a.shape == (64, 64, 4)
    assert a[:, :, 3].min() == 255  # opaque background, opaque puzzle
    assert a[:, :, :3].std() > 0  # something was actually drawn


# --------------------------------------------------------------------------
# Batch and threading
# --------------------------------------------------------------------------


def test_build_many_matches_one_at_a_time() -> None:
    names = ["Rubik's Cube (3x3x3)", "Megaminx", "Skewb", "Pyraminx"]
    batch = tp.build_many(names)
    assert [p.piece_count for p in batch] == [tp.Puzzle.named(n).piece_count for n in names]
    assert [p.field for p in batch] == [tp.Puzzle.named(n).field for n in names]


def test_build_many_keeps_the_input_order() -> None:
    names = ["Megaminx", "2x2x2 Cube", "Skewb Diamond"]
    assert [p.piece_count for p in tp.build_many(names)] == [tp.Puzzle.named(n).piece_count for n in names]
    assert [p.piece_count for p in tp.build_many(list(reversed(names)))] == [
        tp.Puzzle.named(n).piece_count for n in reversed(names)
    ]


def test_build_many_names_the_entry_that_failed() -> None:
    with pytest.raises(ValueError, match="Definitely Not A Puzzle"):
        tp.build_many(["Skewb", "Definitely Not A Puzzle"])


def test_build_many_accepts_recipes_and_names_alike() -> None:
    by_name = tp.build_many(["Rubik's Cube (3x3x3)"])[0]
    by_recipe = tp.build_many(["?shell=C$1&cut=C$1/3"])[0]
    assert by_name.piece_count == by_recipe.piece_count == 26


def test_render_many_matches_rendering_one_at_a_time() -> None:
    names = ["Rubik's Cube (3x3x3)", "Skewb"]
    batch = tp.render_many(names, 48, 48, background=(255, 255, 255, 255), supersample=1)
    for name, got in zip(names, batch, strict=True):
        p = tp.Puzzle.named(name)
        p.background = (255, 255, 255, 255)
        p.supersample = 1
        p.show_arrows = False
        assert got.to_bytes() == p.render(48, 48).to_bytes(), name


def test_render_many_is_reproducible_when_scrambled() -> None:
    names = ["Rubik's Cube (3x3x3)", "Skewb", "Megaminx"]
    a = tp.render_many(names, 40, 40, supersample=1, scramble=6, seed=7)
    b = tp.render_many(names, 40, 40, supersample=1, scramble=6, seed=7)
    assert [i.to_bytes() for i in a] == [i.to_bytes() for i in b]
    c = tp.render_many(names, 40, 40, supersample=1, scramble=6, seed=8)
    assert [i.to_bytes() for i in a] != [i.to_bytes() for i in c]


def test_the_whole_catalog_builds_in_one_batch() -> None:
    entries = tp.catalog_entries()
    puzzles = tp.build_many([e.recipe for e in entries])
    assert len(puzzles) == len(entries)
    assert all(p.piece_count > 0 for p in puzzles)


def test_catalog_entries_are_named_tuples() -> None:
    entries = tp.catalog_entries()
    assert len(entries) == len(tp.catalog())
    first = entries[0]
    assert first.name
    assert first.family
    assert first.kind
    assert first.recipe.startswith("?")
    assert tuple(first) == tp.catalog()[0]


def test_thread_count_is_positive() -> None:
    assert tp.thread_count() >= 1


def test_free_threaded_agrees_with_the_interpreter() -> None:
    import sys
    import sysconfig

    is_free_threaded_build = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    assert tp.free_threaded == is_free_threaded_build
    if is_free_threaded_build:
        is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
        assert callable(is_gil_enabled)
        assert not is_gil_enabled()


def test_threads_get_the_same_answers_as_one_thread() -> None:
    """Concurrency must not change a coordinate.

    Each puzzle owns its number field, so several threads driving several
    puzzles is safe, but refinement mutates state shared *within* a puzzle,
    so this checks rather than assumes.
    """
    from concurrent.futures import ThreadPoolExecutor

    names = ["Rubik's Cube (3x3x3)", "Megaminx", "Skewb Diamond", "Pyraminx", "Starminx"]

    def shoot(name: str) -> bytes:
        p = tp.Puzzle.named(name)
        p.background = (255, 255, 255, 255)
        p.supersample = 1
        p.show_arrows = False
        return p.render(56, 56).to_bytes()

    serial = [shoot(n) for n in names]
    with ThreadPoolExecutor(max_workers=8) as pool:
        # Several rounds, so the work actually overlaps.
        threaded = list(pool.map(shoot, names * 4))
    assert threaded == serial * 4


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def test_text_is_drawn_where_it_is_measured() -> None:
    w, h = tp.Image.text_size("Megaminx", 2)
    im = tp.Image(w + 8, h + 8, (0, 0, 0, 255))
    assert im.draw_text(4, 4, "Megaminx", (255, 255, 255, 255), 2) == w
    inked = [(x, y) for y in range(im.height) for x in range(im.width) if im.pixel(x, y) == (255, 255, 255, 255)]
    assert inked, "nothing was drawn"
    assert min(x for x, _ in inked) == 4
    assert max(x for x, _ in inked) < 4 + w
    assert min(y for _, y in inked) >= 4
    assert max(y for _, y in inked) < 4 + h


def test_text_size_grows_with_scale_and_length() -> None:
    assert tp.Image.text_size("ab", 1)[0] < tp.Image.text_size("abcd", 1)[0]
    assert tp.Image.text_size("abc", 1)[0] * 3 == tp.Image.text_size("abc", 3)[0]
    assert tp.Image.text_size("a\nb", 1)[1] > tp.Image.text_size("a", 1)[1]


def test_fit_text_respects_the_budget() -> None:
    long = "Circo-Radiolarian (Radiolarian 10)"
    for budget in (0, 10, 40, 120, 10_000):
        cut = tp.Image.fit_text(long, budget, 2)
        assert tp.Image.text_size(cut, 2)[0] <= budget
    assert tp.Image.fit_text(long, 10_000, 2) == long
    assert tp.Image.fit_text(long, 60, 2).endswith("...")


def test_centered_text_is_centered() -> None:
    im = tp.Image(101, 20, (0, 0, 0, 255))
    im.draw_text(50, 2, "II", (255, 255, 255, 255), 2, center=True)
    xs = [x for y in range(im.height) for x in range(im.width) if im.pixel(x, y) == (255, 255, 255, 255)]
    assert xs
    assert abs((min(xs) + max(xs)) / 2 - 50) <= 2


def test_unmapped_characters_fall_back_rather_than_fail() -> None:
    im = tp.Image(60, 20, (0, 0, 0, 255))
    # Non-ASCII has no glyph, so it must draw something, not raise.
    assert im.draw_text(2, 2, "é中", (255, 255, 255, 255), 1) > 0


def test_one_puzzle_cannot_be_rendered_from_two_threads_at_once() -> None:
    """The overlap is refused rather than silently producing different pixels.

    Rendering narrows the isolating interval the puzzle's numbers share, so
    two threads inside `render` on the same object would make the coordinates
    depend on the interleaving. Building one puzzle per thread is the rule
    (`SEMANTICS.md` §9). This is what enforces it.
    """
    import threading

    p = tp.Puzzle.named("Megaminx")
    p.supersample = 1
    entered = threading.Event()
    errors: list[BaseException] = []

    # Hold a mutable borrow open by blocking inside a property read is not
    # possible, so take the borrow from another thread and race it instead.
    def hold() -> None:
        try:
            entered.set()
            for _ in range(40):
                p.render(64, 64)
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=hold)
    t.start()
    entered.wait()
    for _ in range(40):
        try:
            p.render(64, 64)
        except RuntimeError as exc:
            errors.append(exc)
    t.join()

    # Either the two never actually overlapped (fast machine, short renders)
    # or the overlap was refused. What must not happen is a quiet success on
    # both sides producing different pixels, which is what `&mut self` rules
    # out structurally.
    assert all(isinstance(e, RuntimeError) for e in errors), errors


def test_render_many_is_the_supported_way_to_use_many_threads() -> None:
    names = ["Megaminx", "Skewb", "Pyraminx"]
    a = tp.render_many(names, 48, 48, supersample=1)
    b = tp.render_many(names, 48, 48, supersample=1)
    assert [i.to_bytes() for i in a] == [i.to_bytes() for i in b]
