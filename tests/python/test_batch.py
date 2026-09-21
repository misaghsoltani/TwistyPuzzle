"""Parallel puzzle simulator batches and their exported numerical arrays.

Deliberately written without NumPy at module scope: a batch is part of the
package, not of the ``gym`` extra, and everything here has to work for someone
who installed the wheel and nothing else. The tests that check NumPy sees the
same numbers ask for it when they run.
"""

from __future__ import annotations

import pytest

import twistypuzzle as tp

CUBE: str = "Rubik's Cube (3x3x3)"
JUMBLER: str = "?shell=C$3*sqrt(2)/4&cut=jC$0"  # the Little Chop: it jumbles


def test_a_batch_of_one_follows_the_puzzle_it_copies() -> None:
    batch = tp.PuzzleBatch(CUBE, 1)
    puzzle = tp.Puzzle(CUBE)
    assert batch.sticker_count == puzzle.sticker_count
    assert batch.action_count == puzzle.action_count
    assert batch.color_count == puzzle.color_count
    assert batch.actions == puzzle.actions
    assert batch.solved_stickers == puzzle.solved_stickers
    assert batch.palette == puzzle.palette
    assert batch.query == puzzle.query

    for action in (3, 7, 1, 10):
        batch.step([action])
        puzzle.apply_action(action)
        assert batch.observations()[0] == puzzle.stickers()
        assert batch.sticker_ids()[0] == puzzle.sticker_ids()
        assert bool(batch.solved()[0]) is puzzle.is_solved


def test_every_copy_turns_on_its_own() -> None:
    batch = tp.PuzzleBatch(CUBE, 4)
    singles = [tp.Puzzle(CUBE) for _ in range(4)]
    for round_ in range(6):
        actions = [(i * 5 + round_) % batch.action_count for i in range(4)]
        batch.step(actions)
        for puzzle, action in zip(singles, actions, strict=True):
            puzzle.apply_action(action)
    seen = batch.observations()
    for i, puzzle in enumerate(singles):
        assert seen[i] == puzzle.stickers(), f"copy {i} differs"


def test_an_array_describes_itself() -> None:
    batch = tp.PuzzleBatch(CUBE, 3, seed=0)
    states = batch.observations()
    assert states.shape == (3, 54)
    assert states.typestr == "|u1"
    assert states.itemsize == 1
    assert states.size == 162
    assert len(states) == 3
    assert len(states[0]) == 54
    assert states.tolist() == [c for _ in range(3) for c in batch.solved_stickers]
    assert len(states.to_bytes()) == 162
    assert states[-1] == states[2]
    with pytest.raises(IndexError):
        _ = states[3]

    ids = batch.sticker_ids()
    # A cube has 54 slots, so a slot number is a byte. The width follows the
    # puzzle, not the kind of thing being counted.
    assert ids.typestr == "|u1"
    assert ids.itemsize == 1
    assert ids[0] == list(range(54))

    flags = batch.solved()
    assert flags.shape == (3,)
    assert flags.typestr == "|b1"
    assert flags.tolist() == [True, True, True]


def test_numpy_sees_the_same_numbers_without_copying() -> None:
    np = pytest.importorskip("numpy")
    batch = tp.PuzzleBatch(CUBE, 2, seed=1)
    batch.reset(scramble=5)
    states = batch.observations()
    view = np.asarray(states)
    assert view.shape == (2, 54)
    assert view.dtype == np.uint8
    assert view.tolist() == [states[0], states[1]]
    # The array owns the buffer, so the view is of that buffer and not a copy.
    assert view.base is states
    # Subsequent batch stepping does not modify previously exported buffers.
    before = view.copy()
    batch.step([0, 1])
    assert np.array_equal(view, before)


def test_a_seeded_batch_repeats_and_its_copies_differ() -> None:
    first = tp.PuzzleBatch(CUBE, 4, seed=7)
    second = tp.PuzzleBatch(CUBE, 4, seed=7)
    first.reset(scramble=12)
    second.reset(scramble=12)
    assert first.observations().tolist() == second.observations().tolist()
    rows = [first.observations()[i] for i in range(4)]
    assert len({tuple(r) for r in rows}) == 4, "the copies all walked together"


def test_a_scramble_of_k_moves_is_k_moves_long() -> None:
    batch = tp.PuzzleBatch(CUBE, 3, seed=2)
    batch.reset(scramble=9)
    assert batch.move_counts == [9, 9, 9]
    # And undoing them, newest first, solves it again: a walk of nine moves
    # is nine moves from solved and not fewer, because it never steps back
    # over itself.
    walks = [list(batch.history(i)) for i in range(3)]
    for step in range(9):
        batch.step([walk[-1 - step] ^ 1 for walk in walks])
        assert all(batch.solved().tolist()) is (step == 8)


def test_history_puts_a_puzzle_where_a_copy_stands() -> None:
    batch = tp.PuzzleBatch(CUBE, 2, seed=3)
    batch.reset(scramble=7)
    for i in range(2):
        puzzle = tp.Puzzle(CUBE)
        for action in batch.history(i):
            assert puzzle.apply_action(action)
        assert puzzle.stickers() == batch.observations()[i]


def test_a_reset_touches_only_the_rows_it_is_given() -> None:
    batch = tp.PuzzleBatch(CUBE, 4, seed=4)
    batch.reset(scramble=8)
    before = [batch.observations()[i] for i in range(4)]
    batch.reset_where([0, 2], scramble=8)
    after = [batch.observations()[i] for i in range(4)]
    assert after[1] == before[1]
    assert after[3] == before[3]
    assert after[0] != before[0]
    assert after[2] != before[2]

    # Flags say the same thing as indices.
    batch.reset_where([True, False, False, False], scramble=0)
    assert bool(batch.solved()[0]) is True
    assert bool(batch.solved()[2]) is False


def test_scramble_takes_a_depth_for_each_copy() -> None:
    batch = tp.PuzzleBatch(CUBE, 3, seed=5)
    batch.scramble([0, 4, 11])
    assert batch.move_counts == [0, 4, 11]
    assert batch.solved().tolist() == [True, False, False]


def test_one_hot_marks_the_color_in_each_slot() -> None:
    batch = tp.PuzzleBatch(CUBE, 2, seed=6)
    batch.reset(scramble=6)
    colors = batch.observations()
    hot = batch.one_hot()
    assert hot.shape == (2, 54 * 6)
    for row in range(2):
        indicator = hot[row]
        for slot in range(54):
            block = indicator[slot * 6 : (slot + 1) * 6]
            assert sum(block) == 1
            assert block[colors[row][slot]] == 1


def test_a_batch_draws_every_copy() -> None:
    batch = tp.PuzzleBatch(CUBE, 3, seed=8)
    batch.reset(scramble=5)
    frames = batch.render(24, 24)
    assert frames.shape == (3, 24, 24, 4)
    images = batch.frames(24, 24)
    assert len(images) == 3
    assert images[0].to_bytes() == bytes(frames[0])

    # Channels and layout are the caller's to choose.
    chw = batch.render(24, 24, channels=3, channels_first=True)
    assert chw.shape == (3, 3, 24, 24)
    real = batch.render(24, 24, channels=3, dtype="f4")
    assert real.typestr in {"<f4", ">f4"}
    assert 0.0 <= min(real.tolist()) <= max(real.tolist()) <= 1.0
    with pytest.raises(ValueError, match="dtype"):
        batch.render(8, 8, dtype="i8")


def test_a_drawn_frame_is_the_frame_a_puzzle_draws() -> None:
    batch = tp.PuzzleBatch(CUBE, 1)
    puzzle = tp.Puzzle(CUBE, show_arrows=False)
    assert bytes(batch.render(32, 32)[0]) == puzzle.render(32, 32).to_bytes()


def test_states_from_elsewhere_can_be_drawn() -> None:
    batch = tp.PuzzleBatch(CUBE, 2, seed=9)
    batch.reset(scramble=4)
    states = batch.observations()
    assert bytes(batch.render_states(states, 16, 16)[0]) == bytes(batch.render(16, 16)[0])
    # The batch's own states are untouched by drawing someone else's.
    assert batch.observations().tolist() == states.tolist()
    # And one state on its own is a batch of one picture.
    assert batch.render_states(states[0], 16, 16).shape == (1, 16, 16, 4)


def test_a_jumbling_puzzle_still_works_without_a_table() -> None:
    batch = tp.PuzzleBatch(JUMBLER, 2)
    assert batch.is_tabular is False
    assert tp.PuzzleBatch(CUBE, 2).is_tabular is True
    solved, applied = batch.step([0, 1])
    assert applied.tolist() == [True, True]
    # A turn may take a sticker off the solved lattice, and a puzzle in such a
    # state is certainly not solved.
    assert solved.tolist() == [False, False]
    assert batch.move_counts == [1, 1]
    # It has no drawing table, and says so instead of drawing the wrong thing.
    with pytest.raises(tp.PuzzleError, match="drawing table"):
        batch.render_states([0] * batch.sticker_count, 8, 8)
    # Drawing it the slow way still works.
    assert batch.render(8, 8).shape == (2, 8, 8, 4)


def test_a_batch_refuses_what_it_cannot_do() -> None:
    batch = tp.PuzzleBatch(CUBE, 2)
    with pytest.raises(ValueError, match="a batch needs at least one puzzle"):
        tp.PuzzleBatch(CUBE, 0)
    with pytest.raises(ValueError, match="2 puzzles"):
        batch.step([0])
    with pytest.raises(ValueError, match="out of range"):
        batch.step([0, 99])
    # A puzzle number this batch does not have is an index error, as it
    # would be for any sequence.
    with pytest.raises(IndexError, match="out of range"):
        batch.seed(5, 1)
    with pytest.raises(IndexError, match="out of range"):
        batch.history(5)
    with pytest.raises(IndexError, match="out of range"):
        batch.reset_one(5)


def test_the_view_can_be_changed_after_building() -> None:
    batch = tp.PuzzleBatch(CUBE, 1)
    first = bytes(batch.render(16, 16)[0])
    batch.configure(background=(0, 0, 0, 255))
    assert bytes(batch.render(16, 16)[0]) != first
    batch.configure(camera_position=(0.0, 0.0, 12.0), camera_up=(0.0, 1.0, 0.0))
    head_on = bytes(batch.render(16, 16)[0])
    batch.configure(camera_position=(12.0, 0.0, 0.0))
    assert bytes(batch.render(16, 16)[0]) != head_on
    # Going back to a viewpoint already drawn gives the same picture again,
    # which is what keeping more than one table is for.
    batch.configure(camera_position=(0.0, 0.0, 12.0))
    assert bytes(batch.render(16, 16)[0]) == head_on


def test_a_written_sequence_turns_every_copy() -> None:
    batch = tp.PuzzleBatch(CUBE, 2)
    puzzle = tp.Puzzle(CUBE)
    assert batch.apply("A B2' C") == 4  # a repeat counts once per turn
    assert puzzle.apply("A B2' C") == 4
    assert batch.move_counts == [4, 4]
    for row in range(2):
        assert batch.observations()[row] == puzzle.stickers()
    with pytest.raises(tp.PuzzleError, match="Q"):
        batch.apply("Q")


def test_undo_takes_back_one_move_at_a_time() -> None:
    batch = tp.PuzzleBatch(CUBE, 3, seed=12)
    batch.reset(scramble=5)
    assert batch.move_counts == [5, 5, 5]

    # One copy at a time, so the others are left where they were.
    assert batch.undo([0]).tolist() == [True, False, False]
    assert batch.move_counts == [4, 5, 5]

    while any(batch.undo().tolist()):
        pass
    assert batch.move_counts == [0, 0, 0]
    assert batch.solved().tolist() == [True, True, True]
    # Verification that history is fully depleted.
    assert batch.undo().tolist() == [False, False, False]


def test_no_batch_draws_the_arrows() -> None:
    """A batch draws states, not controls, whichever way it turns them.

    The arrows are a thing to click on, they are blended over a frame rather
    than written into it, and a drawing table cannot express them. A batch
    that keeps geometry could draw them and must not, or two puzzles would
    disagree about what a frame of a state looks like for no reason a caller
    could see.
    """
    plain = tp.Puzzle(CUBE, show_arrows=False)
    tabular = tp.PuzzleBatch(CUBE, 1)
    assert tabular.is_tabular is True
    assert bytes(tabular.render(32, 32)[0]) == plain.render(32, 32).to_bytes()

    geometric = tp.PuzzleBatch(JUMBLER, 1)
    assert geometric.is_tabular is False
    bare = tp.Puzzle(JUMBLER, show_arrows=False)
    assert bytes(geometric.render(32, 32)[0]) == bare.render(32, 32).to_bytes()
    # And emphatically not the frame with them.
    with_arrows = tp.Puzzle(JUMBLER, show_arrows=True)
    assert bytes(geometric.render(32, 32)[0]) != with_arrows.render(32, 32).to_bytes()


def test_the_camera_can_be_read_back_and_put_back() -> None:
    batch = tp.PuzzleBatch(CUBE, 1)
    was = batch.camera
    assert was is not None
    position, target, up, fov = was
    assert len(position) == len(target) == len(up) == 3
    assert fov > 0
    first = bytes(batch.render(24, 24)[0])

    batch.configure(camera_position=(0.0, 12.0, 0.0), camera_up=(0.0, 0.0, 1.0))
    assert bytes(batch.render(24, 24)[0]) != first
    batch.configure(camera_position=position, camera_target=target, camera_up=up)
    assert bytes(batch.render(24, 24)[0]) == first


def test_an_array_is_only_as_wide_as_its_puzzle_needs() -> None:
    """The element type follows the puzzle, and never silently truncates.

    A slot number of a 3x3x3 fits a byte and a slot number of a 9x9x9 does
    not, so exported arrays select dynamic integer widths: they are
    the narrowest that can hold every value the puzzle can produce. Which is
    also what makes them safe to widen: nothing is ever lost on the way out.
    """
    small = tp.PuzzleBatch(CUBE, 2)
    assert small.sticker_count == 54
    assert small.sticker_ids().typestr == "|u1"
    assert small.observations().typestr == "|u1"
    assert small.permutations.typestr == "|u1"

    # A 9x9x9: 486 slots, so a slot number needs two bytes, while its six
    # colors still need one.
    big = tp.PuzzleBatch("?shell=C$1&cut=C$1/9&cut=C$3/9&cut=C$5/9&cut=C$7/9", 2)
    assert big.sticker_count == 486
    assert big.sticker_ids().typestr in {"<u2", ">u2"}
    assert big.observations().typestr == "|u1"
    assert big.sticker_ids()[0] == list(range(486))


def test_a_puzzle_with_more_colors_than_a_byte_is_read_and_not_refused() -> None:
    """Three hundred twelve faces, and a state of it still reads.

    Planes tangent to one sphere each contribute a face, so a shell can have
    as many as it is given. The colors of this one do not fit a byte, and the
    answer is a wider array instead of a refusal.
    """
    n = 314
    normals = [
        (a, b, c) for a in range(-18, 19) for b in range(-18, 19) for c in range(-18, 19) if a * a + b * b + c * c == n
    ]
    assert len(normals) == 312
    recipe = "?" + "&".join(f"shell={a},{b},{c}$-{n}" for a, b, c in normals)
    batch = tp.PuzzleBatch(recipe, 2)
    assert batch.color_count == 312
    states = batch.observations()
    assert states.typestr in {"<u2", ">u2"}
    assert states.shape == (2, 312)
    assert states[0] == list(range(312))


def test_a_two_byte_array_reads_the_same_three_ways() -> None:
    """An element read at the wrong width is a wrong number, not an error.

    So every way of getting at a two-byte array (NumPy's zero-copy view,
    `tolist()`, indexing a row, indexing an element) has to agree, and has
    to agree with what the puzzle actually holds.
    """
    np = pytest.importorskip("numpy")
    batch = tp.PuzzleBatch("?shell=C$1&cut=C$1/9&cut=C$3/9&cut=C$5/9&cut=C$7/9", 3, seed=2)
    batch.reset(scramble=4)
    ids = batch.sticker_ids()
    assert ids.typestr in {"<u2", ">u2"}
    assert ids.itemsize == 2

    viewed = np.asarray(ids)
    assert viewed.dtype == np.uint16
    assert viewed.shape == (3, 486)
    assert viewed.tolist() == [ids[row] for row in range(3)]
    flat = ids.tolist()
    assert flat == viewed.reshape(-1).tolist()
    assert len(ids.to_bytes()) == 3 * 486 * 2
    # Every row is still a permutation of the slots, which it would not look
    # like if the bytes were being paired up wrongly.
    for row in range(3):
        assert sorted(ids[row]) == list(range(486))

    # And a one-dimensional two-byte array indexes element by element.
    goal = tp.Layout("?shell=C$1&cut=C$1/9&cut=C$3/9&cut=C$5/9&cut=C$7/9").goal_ids
    assert goal.typestr in {"<u2", ">u2"}
    assert [goal[i] for i in (0, 1, 255, 256, 300, 485)] == [0, 1, 255, 256, 300, 485]
    assert goal[-1] == 485


def test_a_state_from_somewhere_else_goes_in() -> None:
    """The way in, which is what makes a batch drivable instead of only walkable.

    A solver, a file, or a `Layout` with no puzzle in it produces states.
    A batch that could only reach the states it had walked to itself could not
    be handed any of them.
    """
    layout = tp.Layout(CUBE)
    batch = tp.PuzzleBatch(CUBE, 4, seed=1)
    batch.reset(scramble=5)

    # In by facelet, out by facelet.
    wanted = layout.scrambled(4, 8, seed=3)
    batch.set_facelets(layout, wanted)
    assert batch.facelets(layout).tolist() == wanted.tolist()
    assert batch.solved().tolist() == [False] * 4
    # And the moves it had made no longer lead anywhere, so they are gone.
    assert batch.move_counts == [0, 0, 0, 0]

    # In by identity, which says more than the colors do.
    other = tp.PuzzleBatch(CUBE, 4, seed=2)
    other.reset(scramble=7)
    ids = other.sticker_ids()
    batch.set_states(ids, ids=True)
    assert batch.sticker_ids().tolist() == ids.tolist()
    assert batch.observations().tolist() == other.observations().tolist()

    # In by color: the identities are chosen, since colors do not fix them,
    # but what a caller reading colors sees is exactly what it asked for.
    colors = other.observations()
    batch.set_states(colors)
    assert batch.observations().tolist() == colors.tolist()

    # One state goes into every puzzle.
    batch.set_states(layout.from_facelets(layout.goal_colors))
    assert batch.solved().tolist() == [True] * 4


def test_a_state_that_is_not_one_is_refused() -> None:
    layout = tp.Layout(CUBE)
    batch = tp.PuzzleBatch(CUBE, 2)
    with pytest.raises(ValueError, match="arrangement"):
        batch.set_states([0] * 54, ids=True)
    with pytest.raises(ValueError, match="colors of this puzzle"):
        batch.set_states([0] * 54)
    with pytest.raises(ValueError, match="not a state each"):
        batch.set_states([0] * 53)
    jumbler = tp.PuzzleBatch("Pyramorphix", 2)
    with pytest.raises(tp.PuzzleError, match="geometry"):
        jumbler.set_states([0] * (jumbler.sticker_count * 2), ids=True)
    # An arrangement no sequence of turns reaches is held anyway: deciding
    # which those are is the puzzle's whole difficulty, not a batch's job.
    # One transposition is an odd permutation, and no turn of a cube is.
    swapped = layout.goal_ids.tolist()
    swapped[0], swapped[9] = swapped[9], swapped[0]
    batch.set_states(swapped, ids=True)
    assert batch.sticker_ids()[0] == swapped
    assert batch.solved().tolist() == [False, False]

    # Swapping two stickers of one color, on the other hand, is a state the
    # colors cannot tell from solved, and the batch agrees with the colors.
    alike = layout.goal_ids.tolist()
    alike[0], alike[1] = alike[1], alike[0]
    batch.set_states(alike, ids=True)
    assert batch.solved().tolist() == [True, True]


def test_states_from_outside_can_be_drawn_and_stepped() -> None:
    np = pytest.importorskip("numpy")
    layout = tp.Layout(CUBE)
    batch = tp.PuzzleBatch(CUBE, 3, seed=4)
    walked = layout.scrambled(3, 6, seed=9)
    batch.set_facelets(layout, walked)
    # Drawn: the same frames as if the batch had walked there itself.
    put_in = np.asarray(batch.render(32, 32))
    assert put_in.shape == (3, 32, 32, 4)
    # Stepped: a move made in the batch and the same move made in the layout
    # land in the same place.
    batch.step(layout.moves_to_actions([2, 2, 2]))
    assert batch.facelets(layout).tolist() == layout.next_states(walked, [2, 2, 2]).tolist()
