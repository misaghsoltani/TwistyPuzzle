"""Many twisty puzzles stepped together, as one Gymnasium vector environment.

Gymnasium's own vectorizers run a list of Python environments: ``SyncVectorEnv``
in a loop, ``AsyncVectorEnv`` in subprocesses paying a pickle round trip per
step. Both work here and both are worth having.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
import numpy as np

from twistypuzzle._native import Puzzle, PuzzleBatch

from ._common import (
    ObsSpaceArray,
    StepArray,
    action_space,
    encode,
    observation_space,
    permutation_table,
    refuse_jumbling,
    text_board,
)
from ._env import DEFAULT_RECIPE, DEFAULT_SCRAMBLE

if TYPE_CHECKING:
    from typing import Any, Final

    from numpy.typing import NDArray

    from twistypuzzle._types import RecipeLike

    from ._common import ObsArray, Observation, Permutations, RewardScheme


__all__: list[str] = ["TwistyPuzzleVectorEnv"]

#: Seeds handed to the fallback batch's scramblers are drawn from this range.
_SEED_LIMIT: Final[int] = 1 << 63


def _metadata() -> dict[str, Any]:
    """What Gymnasium reads off the class before it is instantiated.

    Built by a call rather than written as a literal because Gymnasium declares
    `metadata` as an ordinary attribute that an instance replaces (as this one
    does, to record the autoreset mode it was given).

    Returns:
        The render modes, frame rate and default autoreset mode.
    """
    return {"render_modes": ["rgb_array", "ansi"], "render_fps": 10, "autoreset_mode": AutoresetMode.NEXT_STEP}


class TwistyPuzzleVectorEnv(VectorEnv[ObsSpaceArray, "NDArray[np.int64]", StepArray]):
    """``num_envs`` copies of one puzzle, turned in a single call.

    Reached through ``gymnasium.make_vec(..., vectorization_mode="vector_entry_point")``,
    which is the default for the registered ids because they carry one. The
    ``"sync"`` and ``"async"`` modes still work and wrap
    :class:`~twistypuzzle.gym.TwistyPuzzleEnv` the usual way.

        >>> import gymnasium as gym
        >>> import twistypuzzle.gym  # registers the ids
        >>> envs = gym.make_vec("twistypuzzle/Puzzle-v0", num_envs=4, scramble=3)
        >>> obs, infos = envs.reset(seed=0)
        >>> obs.shape
        (4, 54)
        >>> obs, rewards, terminations, truncations, infos = envs.step([0, 1, 2, 3])
        >>> rewards.shape
        (4,)
        >>> envs.close()

    Args:
        num_envs: How many copies to keep.
        recipe: Catalog name, recipe query string, or ``Recipe``.
        scramble: How many moves from solved a reset leaves each puzzle.
        observation: ``"colors"`` or ``"onehot"``.
        reward: ``"cost"`` charges 1 a move, whereas ``"sparse"`` pays 1 for solving it.
        max_episode_steps: Truncate an episode after this many moves. Registered
            ids pass their own, and ``None`` never truncates.
        autoreset_mode: How a finished episode restarts. All three of
            Gymnasium's modes are implemented.
        render_mode: ``"rgb_array"``, ``"ansi"``, or ``None``.
        width: Frame width for ``"rgb_array"``.
        height: Frame height.
        yaw: Degrees to swing the camera sideways from head-on when drawing.
        pitch: Degrees to raise it.

    Raises:
        ValueError: If the puzzle jumbles, or an argument is out of range.
    """

    metadata: dict[str, Any] = _metadata()

    def __init__(
        self,
        num_envs: int = 1,
        recipe: RecipeLike = DEFAULT_RECIPE,
        *,
        scramble: int = DEFAULT_SCRAMBLE,
        observation: Observation = "colors",
        reward: RewardScheme = "cost",
        max_episode_steps: int | None = None,
        autoreset_mode: AutoresetMode | str = AutoresetMode.NEXT_STEP,
        render_mode: str | None = None,
        width: int = 320,
        height: int = 320,
        yaw: float = -28.0,
        pitch: float = 20.0,
    ) -> None:
        if num_envs < 1:
            msg = f"num_envs must be at least 1, got {num_envs}"
            raise ValueError(msg)
        if scramble < 0:
            msg = f"scramble must not be negative, got {scramble}"
            raise ValueError(msg)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            msg = f"render_mode must be one of {self.metadata['render_modes']}, not {render_mode!r}"
            raise ValueError(msg)

        self._puzzle = Puzzle(recipe, yaw=yaw, pitch=pitch, supersample=2, background=(255, 255, 255, 255))
        refuse_jumbling(self._puzzle.query, str(recipe))

        self.num_envs = num_envs
        self._scramble = scramble
        self._observation: Observation = observation
        self._reward = reward
        self._max_episode_steps = max_episode_steps
        self.autoreset_mode = (
            autoreset_mode if isinstance(autoreset_mode, AutoresetMode) else AutoresetMode(autoreset_mode)
        )
        self.metadata = dict(self.metadata)
        self.metadata["autoreset_mode"] = self.autoreset_mode
        self.render_mode = render_mode
        self._width = width
        self._height = height

        self._solved: ObsArray = np.asarray(self._puzzle.solved_stickers, dtype=np.uint8)
        self._color_count = self._puzzle.color_count
        self._sticker_count = self._puzzle.sticker_count
        self._action_count = self._puzzle.action_count

        self.single_observation_space = observation_space(self._sticker_count, self._color_count, observation)
        self.single_action_space = action_space(self._action_count)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)

        self._table: Permutations | None = permutation_table(self._puzzle)
        self._states: ObsArray = np.tile(self._solved, (num_envs, 1))
        self._rows = np.arange(num_envs, dtype=np.intp)[:, None]
        # One history per copy, so a frame can be drawn by replaying it.
        self._histories: list[list[int]] = [[] for _ in range(num_envs)]
        # Only for a puzzle whose moves are not fixed permutations, which then
        # has to be turned copy by copy through its geometry.
        self._batch: PuzzleBatch | None = None if self._table is not None else PuzzleBatch(recipe, num_envs)

        self._elapsed = np.zeros(num_envs, dtype=np.int64)
        self._pending = np.zeros(num_envs, dtype=np.bool_)

    @property
    def actions(self) -> list[str]:
        """The name of every move, in action-index order."""
        return self._puzzle.actions

    @property
    def states(self) -> ObsArray:
        """Every copy's sticker array, as an ``(n, k)`` block."""
        return self._states.copy()

    @property
    def goal(self) -> ObsArray:
        """The color in each slot when the puzzle is solved."""
        return self._solved.copy()

    @property
    def permutations(self) -> Permutations | None:
        """Every move as a permutation of the slots, or ``None`` if it has none."""
        return self._table

    def _observe(self) -> ObsArray:
        return encode(self._states, self._color_count, self._observation)

    def _solved_flags(self) -> NDArray[np.bool_]:
        return np.all(self._states == self._solved, axis=1)

    def _masks(self) -> NDArray[np.bool_]:
        if self._batch is None:
            # A table is only derived for a puzzle whose layers all turn and
            # keep turning, which the table's own construction checks.
            return np.ones((self.num_envs, self._action_count), dtype=np.bool_)
        flat = np.asarray(self._batch.action_masks(), dtype=np.bool_)
        return flat.reshape(self.num_envs, self._action_count)

    def _infos(self, applied: NDArray[np.bool_], solved: NDArray[np.bool_]) -> dict[str, Any]:
        return {"applied": applied, "is_solved": solved, "action_mask": self._masks()}

    def _sync_from_batch(self) -> None:
        """Copy the fallback batch's states into the array the agent sees."""
        if self._batch is None:
            return
        flat: ObsArray = np.frombuffer(self._batch.observations(), dtype=np.uint8)
        self._states = flat.reshape(self.num_envs, self._sticker_count).copy()

    def _scramble_walk(self, which: NDArray[np.intp], depth: int) -> None:
        """Walk the selected copies `depth` moves away from solved.

        Args:
            which: Indices of the copies to scramble.
            depth: How many moves to make.
        """
        if depth == 0 or self._action_count == 0 or which.size == 0:
            return
        if self._table is None:
            if self._batch is not None:
                for i in which:
                    self._batch.reset_one(int(i), scramble=depth)
                self._sync_from_batch()
            return

        count = self._action_count
        previous = np.full(which.size, -1, dtype=np.int64)
        for _ in range(depth):
            choice = self.np_random.integers(count, size=which.size)
            # Never undo the move just made, or a walk of length k can end up
            # much closer to solved than k. Nudging a clash to the next move
            # keeps the walk exactly `depth` long, which redrawing would not.
            clash = choice == (previous ^ 1)
            if clash.any():
                choice[clash] = (choice[clash] + 2) % count
            for row, env, action in zip(range(which.size), which, choice, strict=True):
                self._states[env] = self._states[env][self._table[action]]
                self._histories[env].append(int(action))
                previous[row] = action

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsArray, dict[str, Any]]:
        """Solve every puzzle and walk each one away from solved again.

        Args:
            seed: Seeds every copy, so the same seed gives the same batch.
            options: ``{"scramble": k}`` overrides the depth for this episode.

        Returns:
            The first batch of observations, and its info.

        Raises:
            ValueError: If ``options["scramble"]`` is negative.
        """
        super().reset(seed=seed)
        if seed is not None and self._batch is not None:
            for i, s in enumerate(self.np_random.integers(_SEED_LIMIT, size=self.num_envs)):
                self._batch.seed(i, int(s))

        depth = self._scramble
        if options is not None and "scramble" in options:
            depth = int(options["scramble"])
            if depth < 0:
                msg = f"scramble must not be negative, got {depth}"
                raise ValueError(msg)

        self._states = np.tile(self._solved, (self.num_envs, 1))
        self._histories = [[] for _ in range(self.num_envs)]
        if self._batch is not None:
            self._batch.reset(scramble=0)
        self._scramble_walk(np.arange(self.num_envs, dtype=np.intp), depth)

        self._elapsed[:] = 0
        self._pending[:] = False
        solved = self._solved_flags()
        applied = np.ones(self.num_envs, dtype=np.bool_)
        return self._observe(), self._infos(applied, solved)

    def step(
        self, actions: NDArray[np.int64] | list[int]
    ) -> tuple[ObsArray, NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], dict[str, Any]]:
        """Turn every puzzle once.

        Args:
            actions: One move index per puzzle.

        Returns:
            The usual five, batched.

        Raises:
            ValueError: If ``actions`` is the wrong length.
            IndexError: If a move index is out of range.
        """
        acts = np.asarray(actions, dtype=np.int64).reshape(-1)
        if acts.size != self.num_envs:
            msg = f"got {acts.size} actions for {self.num_envs} puzzles"
            raise ValueError(msg)
        if acts.size and (np.min(acts) < 0 or np.max(acts) >= self._action_count):
            msg = f"actions must be in range(0, {self._action_count})"
            raise IndexError(msg)

        restarting = (
            self._pending.copy()
            if self.autoreset_mode is AutoresetMode.NEXT_STEP
            else np.zeros(self.num_envs, dtype=np.bool_)
        )

        applied = self._advance(acts)
        self._elapsed += 1
        solved = self._solved_flags()

        terminations = solved.copy()
        truncations = np.zeros(self.num_envs, dtype=np.bool_)
        if self._max_episode_steps is not None:
            truncations = (self._elapsed >= self._max_episode_steps) & ~terminations

        rewards = (
            np.where(solved, 1.0, 0.0) if self._reward == "sparse" else np.full(self.num_envs, -1.0, dtype=np.float64)
        )

        infos: dict[str, Any] = {}
        if self.autoreset_mode is AutoresetMode.NEXT_STEP:
            if restarting.any():
                # The action was not the agent's to take: these finished last
                # step, and this one restarts them.
                self._restart(restarting)
                rewards[restarting] = 0.0
                terminations[restarting] = False
                truncations[restarting] = False
                applied[restarting] = True
                solved = self._solved_flags()
            self._pending = terminations | truncations
        elif self.autoreset_mode is AutoresetMode.SAME_STEP:
            done = terminations | truncations
            if done.any():
                final = self._observe()
                masks = self._masks()
                for i in np.nonzero(done)[0]:
                    infos = self._add_info(
                        infos,
                        {"final_obs": final[i], "final_info": {"is_solved": bool(solved[i]), "action_mask": masks[i]}},
                        int(i),
                    )
                self._restart(done)
                solved = self._solved_flags()

        infos.update(self._infos(applied, solved))
        return self._observe(), rewards, terminations, truncations, infos

    def _advance(self, acts: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Make one move in every copy.

        Args:
            acts: One move index per copy.

        Returns:
            Whether each move could be made at all.
        """
        if self._table is None:
            if self._batch is None:
                return np.zeros(self.num_envs, dtype=np.bool_)
            _, applied_list = self._batch.step(acts.tolist())
            self._sync_from_batch()
            return np.asarray(applied_list, dtype=np.bool_)
        # One gather for the whole batch: `new[e, i] = old[e, perm[a_e][i]]`.
        self._states = self._states[self._rows, self._table[acts]]
        for env, action in enumerate(acts):
            self._histories[env].append(int(action))
        return np.ones(self.num_envs, dtype=np.bool_)

    def _restart(self, which: NDArray[np.bool_]) -> None:
        """Solve and re-scramble the copies `which` selects."""
        chosen = np.nonzero(which)[0].astype(np.intp)
        for i in chosen:
            self._states[i] = self._solved
            self._histories[i] = []
            self._elapsed[i] = 0
        if self._batch is not None:
            for i in chosen:
                self._batch.reset_one(int(i), scramble=0)
        self._scramble_walk(chosen, self._scramble)

    def render(self) -> tuple[NDArray[np.uint8], ...] | tuple[str, ...] | None:
        """Draw every puzzle, however ``render_mode`` asked for.

        ``"rgb_array"`` replays each copy's moves through one scratch puzzle, so
        it costs a turn per move per copy. It is for looking at a batch, not for
        recording one every step, whereas ``"ansi"`` is free.

        Returns:
            One frame or one board per puzzle, or ``None`` with no render mode.
        """
        if self.render_mode == "rgb_array":
            frames: list[NDArray[np.uint8]] = []
            for history in self._histories:
                if not self._puzzle.restore():
                    self._puzzle.reset()
                for action in history:
                    self._puzzle.apply_action(action)
                frame = np.asarray(self._puzzle.render(self._width, self._height))
                frames.append(np.ascontiguousarray(frame[:, :, :3]))
            return tuple(frames)
        if self.render_mode == "ansi":
            return tuple(text_board(self._states[i], self._solved) for i in range(self.num_envs))
        return None

    def close_extras(self, **kwargs: Any) -> None:
        """Nothing to release: the batch owns no window, process or device."""

    def __repr__(self) -> str:
        return f"TwistyPuzzleVectorEnv(num_envs={self.num_envs}, stickers={self._sticker_count})"
