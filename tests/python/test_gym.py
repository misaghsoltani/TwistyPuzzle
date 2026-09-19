"""The Gymnasium environments: single, vectorized, and all three autoreset modes."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
import numpy as np
import pytest

import twistypuzzle as tp
import twistypuzzle.gym as tpg

CUBE_ID: str = "twistypuzzle/Puzzle-v0"


# --------------------------------------------------------------------------- #
#  Registration                                                                #
# --------------------------------------------------------------------------- #


def test_importing_the_module_registers_the_ids() -> None:
    ids = tpg.env_ids()
    assert CUBE_ID in ids
    # One generic id plus one per non-jumbling catalog entry.
    assert len(ids) == 1 + len(tp.non_jumbling())
    assert "twistypuzzle/Megaminx-v0" in ids


def test_registering_twice_is_harmless() -> None:
    before = dict(tpg.env_ids())
    tpg.register_envs()
    assert tpg.env_ids() == before


def test_no_registered_id_is_a_jumbling_puzzle() -> None:
    playable = set(tp.non_jumbling())
    for env_id, recipe in tpg.env_ids().items():
        if env_id == CUBE_ID:
            continue
        assert recipe in playable, env_id


# --------------------------------------------------------------------------- #
#  One environment                                                             #
# --------------------------------------------------------------------------- #


def test_reset_and_step_follow_the_api() -> None:
    env = gym.make(CUBE_ID, scramble=4)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.dtype == np.uint8
    assert set(info) == {"is_solved", "moves", "applied", "action_mask"}

    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert reward == -1.0
    assert terminated is False
    assert truncated is False
    env.close()


def test_the_observation_is_deepxubes_array() -> None:
    env = gym.make(CUBE_ID, scramble=0)
    obs, _ = env.reset(seed=0)
    assert obs.tolist() == [i // 9 for i in range(54)]


def test_a_solved_puzzle_terminates() -> None:
    """One move from solved, undone, ends the episode."""
    env = gym.make(CUBE_ID, scramble=1)
    env.reset(seed=1)
    base = env.unwrapped
    assert isinstance(base, tpg.TwistyPuzzleEnv)
    # The scramble made exactly one move, and its inverse is the other direction.
    action = next(i for i in range(base.action_space.n) if _solves(base, i))
    _, reward, terminated, truncated, info = env.step(action)
    assert terminated is True
    assert truncated is False
    assert info["is_solved"] is True
    assert reward == -1.0
    env.close()


def _solves(env: tpg.TwistyPuzzleEnv, action: int) -> bool:
    """Would this move solve the puzzle? Asked without disturbing it."""
    puzzle = env.puzzle
    if not puzzle.apply_action(action):
        return False
    solved = puzzle.is_solved
    puzzle.undo()
    return solved


def test_a_seeded_reset_repeats() -> None:
    env = gym.make(CUBE_ID, scramble=8)
    first, _ = env.reset(seed=42)
    again, _ = env.reset(seed=42)
    other, _ = env.reset(seed=43)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    env.close()


def test_the_scramble_depth_can_be_set_per_episode() -> None:
    env = gym.make(CUBE_ID, scramble=10)
    obs, _ = env.reset(seed=0, options={"scramble": 0})
    assert obs.tolist() == [i // 9 for i in range(54)]
    obs, _ = env.reset(seed=0, options={"scramble": 6})
    assert obs.tolist() != [i // 9 for i in range(54)]
    env.close()


def test_the_sparse_reward_pays_only_for_solving_it() -> None:
    env = gym.make(CUBE_ID, scramble=3, reward="sparse")
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(0)
    assert reward == 0.0
    env.close()


def test_the_one_hot_observation_is_the_array_expanded() -> None:
    env = gym.make(CUBE_ID, scramble=3, observation="onehot")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (54 * 6,)
    assert obs.sum() == 54  # one indicator set per sticker

    plain = gym.make(CUBE_ID, scramble=3, observation="colors")
    colors, _ = plain.reset(seed=0)
    assert np.array_equal(obs.reshape(54, 6).argmax(axis=1), colors)
    env.close()
    plain.close()


def test_an_episode_is_truncated_rather_than_running_for_ever() -> None:
    env = gym.make(CUBE_ID, scramble=30, max_episode_steps=5)
    env.reset(seed=0)
    for i in range(5):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated:
            pytest.skip("solved by accident")
        assert truncated == (i == 4)
    env.close()


def test_render_modes() -> None:
    rgb = gym.make(CUBE_ID, scramble=2, render_mode="rgb_array")
    rgb.reset(seed=0)
    frame = rgb.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
    rgb.close()

    ansi = gym.make(CUBE_ID, scramble=2, render_mode="ansi")
    ansi.reset(seed=0)
    board = ansi.render()
    assert isinstance(board, str)
    assert len(board.splitlines()) == 6
    assert all(len(line) == 9 for line in board.splitlines())
    ansi.close()


def test_a_jumbling_puzzle_is_refused_with_an_explanation() -> None:
    with pytest.raises(ValueError, match="jumbles"):
        tpg.TwistyPuzzleEnv("?shell=I$1&cut=I$1/3")


def test_an_action_out_of_range_is_refused() -> None:
    env = tpg.TwistyPuzzleEnv(scramble=1)
    env.reset(seed=0)
    with pytest.raises(IndexError, match="out of range"):
        env.step(99)


# --------------------------------------------------------------------------- #
#  Many environments                                                           #
# --------------------------------------------------------------------------- #


def test_make_vec_uses_the_batched_implementation_by_default() -> None:
    envs = gym.make_vec(CUBE_ID, num_envs=4, scramble=3)
    assert isinstance(envs, tpg.TwistyPuzzleVectorEnv)
    envs.close()


def test_the_vector_env_batches_the_api() -> None:
    envs = gym.make_vec(CUBE_ID, num_envs=6, scramble=4)
    obs, infos = envs.reset(seed=0)
    assert obs.shape == (6, 54)
    assert envs.observation_space.contains(obs)
    assert set(infos) == {"applied", "is_solved", "action_mask"}
    assert infos["action_mask"].shape == (6, 12)

    obs, rewards, terminations, truncations, infos = envs.step(np.zeros(6, dtype=np.int64))
    assert obs.shape == (6, 54)
    assert rewards.shape == (6,)
    assert rewards.dtype == np.float64
    assert terminations.shape == truncations.shape == (6,)
    envs.close()


def test_a_seeded_vector_reset_repeats() -> None:
    a = gym.make_vec(CUBE_ID, num_envs=4, scramble=6)
    b = gym.make_vec(CUBE_ID, num_envs=4, scramble=6)
    first, _ = a.reset(seed=11)
    second, _ = b.reset(seed=11)
    assert np.array_equal(first, second)
    # And the copies differ from each other, or the batch is one puzzle six times.
    assert len({row.tobytes() for row in first}) > 1
    a.close()
    b.close()


def test_the_batched_env_agrees_with_the_serial_one() -> None:
    """Whatever the vectorizer, the same moves must give the same states."""
    moves = [3, 0, 7, 2, 5, 1]
    batched = gym.make_vec(CUBE_ID, num_envs=2, scramble=0, vectorization_mode="vector_entry_point")
    sync = gym.make_vec(CUBE_ID, num_envs=2, scramble=0, vectorization_mode="sync")
    batched.reset(seed=5)
    sync.reset(seed=5)
    for m in moves:
        a, *_ = batched.step(np.full(2, m, dtype=np.int64))
        b, *_ = sync.step(np.full(2, m, dtype=np.int64))
        assert np.array_equal(a, b)
    batched.close()
    sync.close()


def test_the_wrong_number_of_actions_is_refused() -> None:
    envs = tpg.TwistyPuzzleVectorEnv(3, scramble=2)
    envs.reset(seed=0)
    with pytest.raises(ValueError, match="3 puzzles"):
        envs.step(np.zeros(2, dtype=np.int64))
    envs.close()


@pytest.mark.parametrize("mode", [AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP, AutoresetMode.DISABLED])
def test_every_autoreset_mode_runs(mode: AutoresetMode) -> None:
    envs = tpg.TwistyPuzzleVectorEnv(4, scramble=1, autoreset_mode=mode, max_episode_steps=3)
    assert envs.metadata["autoreset_mode"] is mode
    obs, _ = envs.reset(seed=0)
    assert obs.shape == (4, 54)
    for _ in range(6):
        obs, rewards, terminations, truncations, infos = envs.step(np.zeros(4, dtype=np.int64))
        assert obs.shape == (4, 54)
        assert rewards.shape == (4,)
        if mode is AutoresetMode.SAME_STEP and (terminations | truncations).any():
            assert "final_obs" in infos
        if mode is AutoresetMode.DISABLED:
            # The caller owns the reset, so clear the flags the way one would.
            if (terminations | truncations).any():
                envs.reset(seed=1)
    envs.close()


def test_next_step_autoreset_restarts_the_episode() -> None:
    """The step after a finish ignores its action and hands back a fresh start."""
    envs = tpg.TwistyPuzzleVectorEnv(1, scramble=1, autoreset_mode=AutoresetMode.NEXT_STEP)
    envs.reset(seed=0)
    perms = envs.permutations
    assert perms is not None

    # One move from solved, so the move that undoes it terminates the episode.
    state = envs.states[0]
    solving = next(a for a in range(len(perms)) if np.array_equal(state[perms[a]], envs.goal))
    obs, rewards, terminations, truncations, _ = envs.step(np.array([solving], dtype=np.int64))
    assert terminations[0]
    assert not truncations[0]
    assert rewards[0] == -1.0
    assert np.array_equal(obs[0], envs.goal)

    # The next step restarts rather than acting: no reward, no flags, and a
    # position that is scrambled again.
    obs, rewards, terminations, truncations, _ = envs.step(np.array([0], dtype=np.int64))
    assert rewards[0] == 0.0
    assert not terminations[0]
    assert not truncations[0]
    assert not np.array_equal(obs[0], envs.goal)
    envs.close()


def test_a_step_is_a_gather_not_a_turn() -> None:
    """The fast path has to agree with the geometry, or it is just fast."""
    env = tpg.TwistyPuzzleEnv(scramble=0)
    perms = env.permutations
    assert perms is not None, "a 3x3x3 has fixed permutations"
    assert perms.shape == (12, 54)

    env.reset(seed=0, options={"scramble": 0})
    geometric = tp.Puzzle("Rubik's Cube (3x3x3)")
    rng = np.random.default_rng(0)
    for action in rng.integers(12, size=30):
        env.step(int(action))
        geometric.apply_action(int(action))
        assert env.state.tolist() == geometric.stickers()
    # And the geometry the environment keeps for drawing agrees too, once it is
    # asked for, which is the only time it is caught up.
    assert env.puzzle.stickers() == geometric.stickers()


def test_the_permutation_table_is_derived_once_per_recipe() -> None:
    """A batch of environments must not each pay for the table."""
    from twistypuzzle.gym._common import clear_tables, permutation_table  # ruff: ignore[import-outside-top-level]

    clear_tables()
    puzzle = tp.Puzzle("Rubik's Cube (3x3x3)")
    first = permutation_table(puzzle)
    second = permutation_table(tp.Puzzle("Rubik's Cube (3x3x3)"))
    assert first is second, "the table was derived twice for one recipe"


def _drawing_puzzle() -> tp.Puzzle:
    """A puzzle drawn exactly the way the environments draw theirs."""
    return tp.Puzzle("Rubik's Cube (3x3x3)", yaw=-28.0, pitch=20.0, supersample=2, background=(255, 255, 255, 255))


def test_racing_threads_derive_the_table_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the GIL, eight threads must not each pay for the same table."""
    from concurrent.futures import ThreadPoolExecutor  # ruff: ignore[import-outside-top-level]
    from threading import Lock  # ruff: ignore[import-outside-top-level]

    from twistypuzzle.gym._common import clear_tables, permutation_table  # ruff: ignore[import-outside-top-level]

    derivations = 0
    counting = Lock()
    real = tp.Puzzle.action_permutations

    def tally(puzzle: tp.Puzzle) -> list[list[int]] | None:
        nonlocal derivations
        with counting:
            derivations += 1
        return real(puzzle)

    monkeypatch.setattr(tp.Puzzle, "action_permutations", tally)
    clear_tables()
    with ThreadPoolExecutor(max_workers=8) as pool:
        tables = list(pool.map(lambda _: permutation_table(tp.Puzzle("Rubik's Cube (3x3x3)")), range(8)))

    assert derivations == 1, f"the table was derived {derivations} times"
    assert all(t is tables[0] for t in tables)


def test_replaying_after_a_restore_matches_a_fresh_puzzle() -> None:
    """Both renderers rewind a scratch puzzle and replay into it."""
    used = tp.Puzzle("Rubik's Cube (3x3x3)")
    for action in (3, 7, 1, 9):
        used.apply_action(action)
    assert used.restore(), "a 3x3x3 unwinds"

    fresh = tp.Puzzle("Rubik's Cube (3x3x3)")
    for action in (2, 5, 0):
        used.apply_action(action)
        fresh.apply_action(action)

    assert used.stickers() == fresh.stickers()
    assert np.array_equal(np.asarray(used.render(48, 48)), np.asarray(fresh.render(48, 48)))


def test_a_reset_after_drawing_draws_the_new_position() -> None:
    """A reset leaves the geometry dirty, so the next frame has to catch it up."""
    env = tpg.TwistyPuzzleEnv(scramble=0, render_mode="rgb_array", width=48, height=48)
    env.reset(seed=0, options={"scramble": 0})
    for action in (4, 8, 2):
        env.step(action)
    env.render()  # Dirties the scratch geometry with the first episode.

    env.reset(seed=0, options={"scramble": 0})
    actions = (1, 6)
    for action in actions:
        env.step(action)
    frame = env.render()
    assert isinstance(frame, np.ndarray)

    fresh = _drawing_puzzle()
    for action in actions:
        fresh.apply_action(action)
    assert env.puzzle.stickers() == fresh.stickers()
    assert np.array_equal(frame, np.asarray(fresh.render(48, 48))[:, :, :3])
    env.close()


def test_the_vector_env_draws_the_position_it_restarted_into() -> None:
    """After an autoreset the frame must be the new scramble, not the old one."""
    envs = tpg.TwistyPuzzleVectorEnv(
        1, scramble=1, autoreset_mode=AutoresetMode.NEXT_STEP, render_mode="rgb_array", width=48, height=48
    )
    envs.reset(seed=0)
    perms = envs.permutations
    assert perms is not None
    envs.render()  # Dirties the scratch geometry with the pre-reset position.

    state = envs.states[0]
    solving = next(a for a in range(len(perms)) if np.array_equal(state[perms[a]], envs.goal))
    _, _, terminations, _, _ = envs.step(np.array([solving], dtype=np.int64))
    assert terminations[0]
    envs.step(np.array([0], dtype=np.int64))  # Autoresets rather than acting.

    # One move from solved, so the move it was scrambled with is recoverable.
    restarted = envs.states[0]
    scrambled_with = next(a for a in range(len(perms)) if np.array_equal(envs.goal[perms[a]], restarted))
    frames = envs.render()
    assert frames is not None

    fresh = _drawing_puzzle()
    fresh.apply_action(scrambled_with)
    assert np.array_equal(frames[0], np.asarray(fresh.render(48, 48))[:, :, :3])
    envs.close()


def test_sync_and_async_vectorizers_still_work() -> None:
    s = gym.make_vec(CUBE_ID, num_envs=3, scramble=2, vectorization_mode="sync")
    assert isinstance(s, SyncVectorEnv)
    assert s.reset(seed=3)[0].shape == (3, 54)
    s.close()

    a = gym.make_vec(CUBE_ID, num_envs=3, scramble=2, vectorization_mode="async")
    assert isinstance(a, AsyncVectorEnv)
    assert a.reset(seed=3)[0].shape == (3, 54)
    a.close()


def test_vector_render_modes() -> None:
    envs = tpg.TwistyPuzzleVectorEnv(3, scramble=2, render_mode="rgb_array", width=48, height=48)
    envs.reset(seed=0)
    frames = envs.render()
    assert frames is not None
    assert len(frames) == 3
    assert all(f.shape == (48, 48, 3) for f in frames)
    envs.close()

    ansi = tpg.TwistyPuzzleVectorEnv(2, scramble=2, render_mode="ansi")
    ansi.reset(seed=0)
    boards = ansi.render()
    assert boards is not None
    assert len(boards) == 2
    assert all(len(b.splitlines()) == 6 for b in boards)
    ansi.close()


def test_a_bad_render_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="render_mode"):
        tpg.TwistyPuzzleEnv(render_mode="opengl")
    with pytest.raises(ValueError, match="render_mode"):
        tpg.TwistyPuzzleVectorEnv(2, render_mode="opengl")


def test_a_negative_scramble_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        tpg.TwistyPuzzleEnv(scramble=-1)
    with pytest.raises(ValueError, match="negative"):
        tpg.TwistyPuzzleVectorEnv(2, scramble=-1)


def test_other_puzzles_have_their_own_shapes() -> None:
    env = gym.make("twistypuzzle/Megaminx-v0", scramble=3)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (132,)
    assert env.action_space.n == 24
    env.close()
