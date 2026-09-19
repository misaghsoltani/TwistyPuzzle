"""The sticker array and the move names, as a Python user meets them."""

from __future__ import annotations

import pytest

import twistypuzzle as tp

CUBE: str = "Rubik's Cube (3x3x3)"
JUMBLER: str = "?shell=I$1&cut=I$1/3"  # a face-turning icosahedron: it jumbles


def test_a_three_by_three_reads_exactly_like_deepxubes_cube3() -> None:
    p = tp.Puzzle(CUBE)
    assert p.sticker_count == 54
    assert p.color_count == 6
    assert p.action_count == 12
    # DeepXube's `goal_colors = (arange(54) // 9).astype(uint8)`, value for value.
    assert p.solved_stickers == [i // 9 for i in range(54)]
    assert p.stickers() == p.solved_stickers
    assert p.is_solved


def test_a_solved_permutation_is_the_identity() -> None:
    p = tp.Puzzle(CUBE)
    assert p.sticker_ids() == list(range(54))
    p.apply("A")
    assert p.sticker_ids() != list(range(54))
    assert sorted(p.sticker_ids()) == list(range(54))


def test_the_permutation_says_more_than_the_colors() -> None:
    """Two identically colored stickers may be swapped, and only ids see it."""
    p = tp.Puzzle(CUBE)
    p.apply("A")
    colors = p.stickers()
    ids = p.sticker_ids()
    solved = p.solved_stickers
    assert [solved[i] for i in ids] == colors


def test_move_names_round_trip() -> None:
    p = tp.Puzzle(CUBE)
    assert p.actions == ["A", "A'", "B", "B'", "C", "C'", "D", "D'", "E", "E'", "F", "F'"]
    assert p.grip_names == ["A", "B", "C", "D", "E", "F"]
    for i, name in enumerate(p.actions):
        assert p.action_index(name) == i
    assert p.action_index("nope") is None


def test_a_sequence_and_its_inverse_cancel() -> None:
    p = tp.Puzzle(CUBE)
    solved = p.stickers()
    assert p.apply("A B' C2 D") == 5
    assert p.stickers() != solved
    assert p.apply("D' C2' B A'") == 5
    assert p.stickers() == solved


def test_an_unknown_move_names_the_grips_it_could_have_meant() -> None:
    p = tp.Puzzle(CUBE)
    with pytest.raises(tp.PuzzleError, match="A, B, C, D, E, F"):
        p.apply("Z")


def test_actions_and_names_agree() -> None:
    p = tp.Puzzle(CUBE)
    by_name = tp.Puzzle(CUBE)
    for i, name in enumerate(p.actions):
        p.reset()
        by_name.reset()
        assert p.apply_action(i)
        by_name.apply(name)
        assert p.stickers() == by_name.stickers(), name


def test_every_layer_of_a_solved_cube_turns() -> None:
    p = tp.Puzzle(CUBE)
    assert p.action_mask() == [True] * 12
    for a in range(p.action_count):
        assert p.action_grip(a) is not None


def test_ground_atoms_match_the_array() -> None:
    p = tp.Puzzle(CUBE)
    p.apply("A B'")
    colors = p.stickers()
    atoms = p.ground_atoms()
    assert len(atoms) == len(colors)
    assert atoms[0] == ("color", "s0", f"c{colors[0]}")
    model = tp.ground_model(p)
    assert isinstance(model, frozenset)
    assert len(model) == len(set(atoms))


def test_a_jumbling_puzzle_says_so_rather_than_guessing() -> None:
    assert tp.jumbles(JUMBLER) is True
    assert tp.jumbles(CUBE) is False

    p = tp.Puzzle(JUMBLER)
    assert p.stickers(), "a solved puzzle always reads"
    refused = False
    for a in range(p.action_count):
        q = tp.Puzzle(JUMBLER)
        if not q.apply_action(a):
            continue
        try:
            q.stickers()
        except tp.PuzzleError as e:
            assert "not at any solved sticker position" in str(e)
            refused = True
    assert refused, "the icosahedron was expected to jumble"


def test_the_non_jumbling_list_is_the_catalog_minus_the_jumblers() -> None:
    entries = tp.non_jumbling_entries()
    assert len(entries) == len(tp.non_jumbling()) == 36
    assert len(entries) < len(tp.catalog_entries())
    recipes = {e.recipe for e in entries}
    assert set(tp.non_jumbling()) == recipes
    assert CUBE in {e.name for e in entries}


@pytest.mark.parametrize("name", ["Rubik's Cube (3x3x3)", "Megaminx", "Pyraminx", "2x2x2 Cube"])
def test_a_scramble_and_its_undo_restore_the_puzzle(name: str) -> None:
    p = tp.Puzzle(name, seed=3)
    solved = p.stickers()
    p.scramble(15)
    p.settle()
    assert p.stickers() != solved
    assert not p.is_solved
    while p.undo():
        pass
    assert p.stickers() == solved
    assert p.is_solved
