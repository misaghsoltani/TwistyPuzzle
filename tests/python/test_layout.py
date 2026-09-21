"""A puzzle numbered face by face, as a Python user meets it.

The cube conventions themselves are checked against recorded permutations in
the Rust suite (`tests/layout.rs`, against `tests/data/facelet_cube.txt`).
Tests verify that round-trip state representations preserve invariants, move
applications produce identical states whether executed via layout permutations
or geometric simulation, and state observation encodings accurately represent
the underlying state across all puzzle types.

No NumPy at module scope: a layout is part of the package, not of the ``gym``
extra.
"""

from __future__ import annotations

import pytest

import twistypuzzle as tp

CUBE: str = "Rubik's Cube (3x3x3)"
MEGAMINX: str = "Megaminx"
PYRAMINX: str = "Pyraminx"


def test_a_layout_describes_the_cube_it_came_from() -> None:
    layout = tp.Layout()
    assert layout.is_cube
    assert layout.size == 3
    assert layout.facelet_count == 54
    assert layout.face_count == 6
    assert layout.move_count == 12
    assert layout.faces == ["U", "D", "L", "R", "B", "F"]
    assert layout.face_sizes.tolist() == [9] * 6
    assert layout.move_names == [f"{f}{d}" for f in layout.faces for d in (-1, 1)]
    assert sorted(layout.facelet_of_slot) == list(range(54))
    assert sorted(layout.slot_of_facelet) == list(range(54))
    for slot, facelet in enumerate(layout.facelet_of_slot):
        assert layout.slot_of_facelet[facelet] == slot


def test_the_goal_is_the_array_a_cube_is_usually_written_with() -> None:
    layout = tp.Layout()
    assert layout.goal_colors.tolist() == [i // 9 for i in range(54)]
    assert layout.goal_ids.tolist() == list(range(54))


def test_a_move_can_be_named_three_ways() -> None:
    layout = tp.Layout()
    for face, name in enumerate(layout.faces):
        assert layout.move_index(f"{name}-1") == face * 2
        assert layout.move_index(f"{name}'") == face * 2
        assert layout.move_index(f"{name}1") == face * 2 + 1
        assert layout.move_index(name) == face * 2 + 1
    assert layout.move_index("U2") is None
    assert layout.move_index("nonsense") is None


@pytest.mark.parametrize("recipe", [CUBE, MEGAMINX, PYRAMINX])
def test_a_move_is_the_turn_the_puzzle_makes(recipe: str) -> None:
    layout = tp.Layout(recipe)
    for move, action in enumerate(layout.actions):
        assert action is not None, f"{layout.move_names[move]} has no action behind it"
        puzzle = tp.Puzzle(recipe)
        assert puzzle.apply_action(action)
        through_puzzle = layout.to_facelets(puzzle.stickers()).tolist()
        through_layout = layout.next_states(layout.goal_colors, move).tolist()
        assert through_puzzle == through_layout, layout.move_names[move]
    assert layout.moves_to_actions(range(layout.move_count)) == [a for a in layout.actions if a is not None]


@pytest.mark.parametrize("recipe", [CUBE, MEGAMINX, PYRAMINX])
def test_a_state_survives_the_trip_out_and_back(recipe: str) -> None:
    puzzle = tp.Puzzle(recipe)
    puzzle.seed(3)
    puzzle.scramble(20)
    puzzle.settle()
    layout = tp.Layout(recipe)
    facelets = layout.to_facelets(puzzle.stickers())
    assert layout.from_facelets(facelets).tolist() == puzzle.stickers()
    # Each face still has all of its own stickers somewhere.
    values = facelets.tolist()
    for face, size in enumerate(layout.face_sizes.tolist()):
        assert values.count(face) == size


def test_identities_say_more_than_colors() -> None:
    layout = tp.Layout()
    puzzle = tp.Puzzle(CUBE)
    puzzle.apply("A")
    colors = layout.to_facelets(puzzle.stickers()).tolist()
    ids = layout.to_facelet_ids(puzzle.sticker_ids()).tolist()
    assert sorted(ids) == list(range(54))
    # The identity of each sticker names a facelet of the same color.
    for facelet, sticker in enumerate(ids):
        assert colors[facelet] == sticker // 9


@pytest.mark.parametrize("recipe", [CUBE, MEGAMINX])
def test_a_move_and_its_inverse_cancel(recipe: str) -> None:
    layout = tp.Layout(recipe)
    goal = layout.goal_ids
    for move in range(layout.move_count):
        once = layout.next_states(goal, move)
        assert once.tolist() != goal.tolist()
        back = layout.next_states(once, layout.inverse_moves([move])[0])
        assert back.tolist() == goal.tolist()


def test_a_tip_turns_its_own_three_stickers_and_nothing_else() -> None:
    # A grip holding a single piece (a tip) still turns, spinning that
    # piece in its socket. Eight of the cataloged Pyraminx's sixteen actions
    # are its four tips, each moving exactly the three stickers of the piece
    # it holds, and none of its actions turns nothing at all.
    layout = tp.Layout(PYRAMINX)
    goal = layout.goal_ids.tolist()
    tips = []
    for move in range(layout.move_count):
        after = layout.next_states(goal, move).tolist()
        moved = sum(1 for a, b in zip(goal, after, strict=True) if a != b)
        assert moved > 0, f"move {move} of the Pyraminx turns nothing"
        if moved == 3:
            tips.append(move)
    assert len(tips) == 8
    # A third of a turn: the two directions undo each other, and neither is its own inverse.
    inverses = layout.inverse_moves(tips).tolist()
    assert sorted(inverses) == tips
    assert all(inv != move for move, inv in zip(tips, inverses, strict=True))


def test_a_whole_batch_of_states_steps_at_once() -> None:
    layout = tp.Layout()
    states = layout.scrambled(8, 5, seed=1)
    assert states.shape == (8, 54)
    moves = [i % layout.move_count for i in range(8)]
    stepped = layout.next_states(states, moves)
    assert stepped.shape == (8, 54)
    for row in range(8):
        alone = layout.next_states(states[row], moves[row])
        assert stepped[row] == alone.tolist()


@pytest.mark.parametrize("recipe", [CUBE, MEGAMINX])
def test_a_walk_of_k_moves_is_k_moves_from_solved(recipe: str) -> None:
    layout = tp.Layout(recipe)
    k = layout.facelet_count
    states, moves = layout.trajectories(6, 4, seed=2)
    assert states.shape == (4, 7, k)
    assert moves.shape == (4, 6)
    for row in range(4):
        path = states[row]
        assert path[:k] == layout.goal_colors.tolist()
        # Walking the moves back from the end comes home.
        state = path[-k:]
        for move in reversed(moves[row]):
            state = layout.next_states(state, layout.inverse_moves([move])[0]).tolist()
        assert state == layout.goal_colors.tolist()


def test_a_scramble_range_gives_a_spread_of_depths() -> None:
    layout = tp.Layout()
    _, moves = layout.trajectories((1, 12), 64, seed=3)
    # Rows that stopped early repeat their last state, and the moves they did
    # not make are left as the mark one past the last move, so the lengths
    # differ row to row.
    lengths = {len([m for m in row if m < layout.move_count]) for row in moves}
    assert len(lengths) > 1, "a range of depths produced only one length"


def test_macro_moves_are_the_moves_they_are_made_of() -> None:
    layout = tp.Layout()
    triples = layout.with_moves([[a, b, c] for a in range(12) for b in range(12) for c in range(12)])
    assert triples.move_count == 12**3
    assert triples.facelet_count == 54
    assert triples.move_names[0] == "U-1 U-1 U-1"
    assert triples.actions == [None] * 12**3, "a macro is not one turn of the puzzle"

    goal = layout.goal_colors
    for macro, parts in [(0, [0, 0, 0]), (5, [0, 0, 5]), (12**3 - 1, [11, 11, 11])]:
        one_go = triples.next_states(goal, macro).tolist()
        step_by_step = goal
        for move in parts:
            step_by_step = layout.next_states(step_by_step, move)
        assert one_go == step_by_step.tolist()


def test_a_macro_is_undone_by_the_macro_that_undoes_it() -> None:
    # Composed moves have no numbering to lean on, so which one undoes which
    # is worked out from the permutations. Every triple's reverse is in the
    # set of all triples, so every one of them has an inverse.
    layout = tp.Layout()
    parts = [[a, b] for a in range(12) for b in range(12)]
    pairs = layout.with_moves(parts)
    goal = pairs.goal_colors
    backs = pairs.inverse_moves(range(pairs.move_count)).tolist()
    for m, back in enumerate(backs):
        once = pairs.next_states(goal, m)
        assert pairs.next_states(once, back).tolist() == goal.tolist()
    # Which macro is named is not always the reverse sequence, because a
    # macro can have several inverses (for example, `U-1 D-1` is undone by `D1 U1` and
    # equally by `U1 D1`, since opposite faces commute), and a macro that
    # undoes itself answers with itself.
    assert parts[backs[0]] == parts[0], "U-1 U-1 is a half turn, and undoes itself"


@pytest.mark.parametrize("recipe", [CUBE, MEGAMINX])
def test_the_encodings_describe_the_state(recipe: str) -> None:
    layout = tp.Layout(recipe)
    k, faces = layout.facelet_count, layout.face_count
    states = layout.scrambled(3, 7, seed=4)
    hot = layout.one_hot(states)
    assert hot.shape == (3, k * faces)
    for row in range(3):
        indicator = hot[row]
        for facelet, face in enumerate(states[row]):
            block = indicator[facelet * faces : (facelet + 1) * faces]
            assert sum(block) == 1
            assert block[face] == 1
    assert layout.solved(states).tolist() == [False, False, False]
    assert layout.solved(layout.goal_colors).tolist() == [True]


def test_a_board_reads_as_a_net() -> None:
    assert tp.Layout().text(tp.Layout().goal_colors).splitlines()[0] == "U UUU/UUU/UUU"
    # A puzzle with no grid to read gets one line per face all the same.
    board = tp.Layout(MEGAMINX).text(tp.Layout(MEGAMINX).goal_colors).splitlines()
    assert len(board) == 12
    assert board[0] == "0 " + "0" * 11


@pytest.mark.parametrize("recipe", [CUBE, MEGAMINX])
def test_a_batch_reads_out_in_face_order(recipe: str) -> None:
    layout = tp.Layout(recipe)
    batch = tp.PuzzleBatch(recipe, 3, seed=5)
    batch.reset(scramble=6)
    facelets = batch.facelets(layout)
    assert facelets.shape == (3, layout.facelet_count)
    for row in range(3):
        assert facelets[row] == layout.to_facelets(batch.observations()[row]).tolist()
    ids = batch.facelet_ids(layout)
    for row in range(3):
        assert ids[row] == layout.to_facelet_ids(batch.sticker_ids()[row]).tolist()


def test_a_bigger_cube_keeps_the_conventions_and_adds_its_slices() -> None:
    layout = tp.Layout("?shell=C$1&cut=C$1/2&cut=C$0")
    assert layout.size == 4
    assert layout.facelet_count == 96
    assert layout.goal_colors.tolist().count(0) == 16
    # Twelve face turns under the usual names, then the slices behind them
    # under the puzzle's own, marked so that `F1` cannot mean two things.
    assert layout.move_count == 24
    assert layout.move_names[:2] == ["U-1", "U1"]
    assert all(name.startswith(":") for name in layout.move_names[12:])
    assert layout.move_index("F1") == 11
    assert layout.move_index(":F1") == 22


def test_a_puzzle_that_is_not_a_cube_is_numbered_in_its_own_order() -> None:
    layout = tp.Layout(MEGAMINX)
    assert not layout.is_cube
    assert layout.size is None
    assert layout.face_count == 12
    assert layout.faces == [str(i) for i in range(12)]
    k = layout.facelet_count
    assert layout.facelet_of_slot.tolist() == list(range(k))
    assert layout.slot_of_facelet.tolist() == list(range(k))
    # Slots are numbered in color order, so the goal is still a run per face.
    goal = layout.goal_colors.tolist()
    assert goal == sorted(goal)
    assert goal.count(0) == k // 12


def test_a_puzzle_that_jumbles_has_no_layout() -> None:
    with pytest.raises(tp.PuzzleError, match="jumbles|permutations"):
        tp.Layout("Pyramorphix")


def test_an_array_is_only_as_wide_as_the_puzzle_needs() -> None:
    small = tp.Layout()
    assert small.goal_colors.typestr == "|u1"
    assert small.goal_ids.typestr == "|u1"
    assert small.moves.typestr == "|u1"
    assert small.moves.shape == (12, 54)
    # A 9x9x9 has 486 facelets, which a byte cannot name.
    big = tp.Layout("?shell=C$1&cut=C$1/9&cut=C$3/9&cut=C$5/9&cut=C$7/9")
    assert big.facelet_count == 486
    assert big.goal_colors.typestr == "|u1", "six faces still fit a byte"
    assert big.goal_ids.typestr in {"<u2", ">u2"}
    assert big.goal_ids.tolist() == list(range(486))


def test_a_state_keeps_the_width_it_arrived_in() -> None:
    layout = tp.Layout()
    colors = layout.goal_colors
    assert layout.next_states(colors, 0).typestr == "|u1"
    # A list of numbers is read at the narrowest width that holds it.
    assert layout.next_states(colors.tolist(), 0).typestr == "|u1"


def test_numpy_can_step_a_puzzle_with_no_puzzle_in_sight() -> None:
    np = pytest.importorskip("numpy")
    layout = tp.Layout()
    states = np.asarray(layout.scrambled(16, 10, seed=6))
    assert states.shape == (16, 54)
    assert states.dtype == np.uint8
    moves = np.arange(16) % 12
    stepped = np.asarray(layout.next_states(states, moves.tolist()))
    back = np.asarray(layout.next_states(stepped, (moves ^ 1).tolist()))
    assert np.array_equal(back, states)


def test_a_numpy_array_is_read_as_a_block_whatever_its_dtype() -> None:
    """Every integer dtype NumPy produces is read, and read the same way.

    Platform default integers (e.g. ``int64``) and narrow package representations
    (``uint8``) must both parse consistently into contiguous buffers without
    per-element Python iterations.
    """
    np = pytest.importorskip("numpy")
    layout = tp.Layout()
    want = layout.next_states(layout.goal_colors, 3).tolist()
    for dtype in (np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64):
        states = np.asarray(layout.goal_colors, dtype=dtype)
        assert layout.next_states(states, 3).tolist() == want, dtype
    # A negative number is not a face, and says so instead of wrapping.
    with pytest.raises(ValueError, match="negative"):
        layout.next_states(np.full(54, -1, dtype=np.int64), 0)


def test_a_layout_steps_a_hundred_thousand_states_at_once() -> None:
    np = pytest.importorskip("numpy")
    layout = tp.Layout()
    states = np.asarray(layout.scrambled(100_000, (1, 30), seed=7))
    assert states.shape == (100_000, 54)
    moves = np.arange(100_000) % 12
    stepped = np.asarray(layout.next_states(states, moves.tolist()))
    assert stepped.shape == (100_000, 54)
    # Every row is still a whole puzzle: nine of each face.
    counts = np.apply_along_axis(lambda r: np.bincount(r, minlength=6), 1, stepped[:100])
    assert np.array_equal(counts, np.full((100, 6), 9))


def test_every_puzzle_that_can_have_a_layout_has_one() -> None:
    """All thirty-six of them, and each one describes its own puzzle.

    "Face by face" is meant to be something every puzzle has, not something
    cubes have. What stands in the way is not the shape: a puzzle whose moves
    are not fixed permutations of its stickers (one that jumbles, or one
    with a layer that can lock) has no permutation to number, and those are
    refused with a reason. Everything else is covered, and this checks it
    instead of asserting it.
    """
    good = set(tp.non_jumbling())
    entries = [(entry[0], entry[-1]) for entry in tp.catalog()]
    covered = [(name, query) for name, query in entries if query in good]
    assert len(covered) == 36, "the catalog changed: this is a count, not a law"

    for name, query in covered:
        layout = tp.Layout(query)
        puzzle = tp.Puzzle(query)
        assert layout.facelet_count == puzzle.sticker_count, name
        assert layout.face_count == puzzle.color_count, name
        assert sum(layout.face_sizes.tolist()) == layout.facelet_count, name
        # The goal is a run of each face in turn, whatever the shape.
        goal = layout.goal_colors.tolist()
        assert goal == sorted(goal), name
        # Every move is one of the puzzle's actions, and has an inverse here.
        assert layout.move_count >= puzzle.action_count, name
        assert len(layout.moves_to_actions(range(puzzle.action_count))) == puzzle.action_count, name
        backs = layout.inverse_moves(range(layout.move_count)).tolist()
        state = layout.goal_ids
        for move, back in zip(range(layout.move_count), backs, strict=True):
            once = layout.next_states(state, move)
            assert layout.next_states(once, back).tolist() == state.tolist(), f"{name}: move {move}"


def test_a_layout_for_another_puzzle_is_refused() -> None:
    """Two puzzles of a size have arrays of a length, and nothing else in common.

    Nothing about the shapes would catch the mix-up, so the answer would just mean
    something else, and for ``set_facelets`` it would be written into the batch.
    """
    pyraminx = tp.Layout(PYRAMINX)
    batch = tp.PuzzleBatch("Pyraminx (no tips)", 2)
    assert batch.sticker_count != pyraminx.facelet_count
    for call in (
        lambda: batch.facelets(pyraminx),
        lambda: batch.facelet_ids(pyraminx),
        lambda: batch.set_facelets(pyraminx, pyraminx.goal_colors),
    ):
        with pytest.raises(ValueError, match="describes"):
            call()
    # And its own layout is accepted.
    own = tp.Layout(batch.query)
    assert batch.facelets(own).shape == (2, batch.sticker_count)
