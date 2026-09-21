"""Many twisty puzzles stepped together, as one Gymnasium vector environment.

The whole point of this class is that a vector step is a *single* call into
Rust: every puzzle turns, every state is read, and every finished episode
restarts without the interpreter being entered once per environment. The
states never live in Python at all: they live in one block of memory the
batch owns, and what comes back is a view of it.

Gymnasium's own vectorizers still work: ``SyncVectorEnv`` runs a list of
:class:`~twistypuzzle.gym.TwistyPuzzleEnv` in a loop and ``AsyncVectorEnv``
runs them in subprocesses, paying a pickle round trip per step. This is the
default for the registered ids because it is neither.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
import numpy as np

from twistypuzzle._native import Puzzle, PuzzleBatch

from ._common import (
    DEFAULT_VIEWS,
    ObsSpaceArray,
    StepArray,
    action_space,
    depth_range,
    encode,
    layout_for,
    observation_space,
    refuse_jumbling,
    solved_stickers,
    text_board,
)
from ._env import DEFAULT_RECIPE, DEFAULT_SCRAMBLE

if TYPE_CHECKING:
    from typing import TypeAlias

    from numpy.typing import NDArray

    from twistypuzzle._native import Layout
    from twistypuzzle._types import RecipeLike

    from ._common import Depth, ObsArray, Observation, Permutations, RewardScheme, View


__all__: list[str] = ["TwistyPuzzleVectorEnv", "VectorInfo"]

_SubInfo: TypeAlias = "dict[str, bool | NDArray[np.bool_]]"
_InfoValue: TypeAlias = "NDArray[np.bool_] | NDArray[np.uint32] | list[ObsArray | None] | list[_SubInfo | None]"
VectorInfo: TypeAlias = "dict[str, _InfoValue]"


def _metadata() -> dict[str, list[str] | int | AutoresetMode]:
    """What Gymnasium reads off the class before it is instantiated.

    Built by a call instead of written as a literal because Gymnasium declares
    `metadata` as an ordinary attribute that an instance replaces (as this one
    does, to record the autoreset mode it was given).

    Returns:
        The render modes, frame rate and default autoreset mode.
    """
    return {"render_modes": ["rgb_array", "ansi"], "render_fps": 10, "autoreset_mode": AutoresetMode.NEXT_STEP}


class TwistyPuzzleVectorEnv(VectorEnv[ObsSpaceArray, "NDArray[np.int64]", StepArray]):
    """``num_envs`` copies of one puzzle, turned in a single call.

    Reached through ``gymnasium.make_vec(..., vectorization_mode="vector_entry_point")``,
    which is the default for the registered ids because they carry one.

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
        scramble: How many moves from solved a reset leaves each puzzle, or
            ``(low, high)`` to draw a length for each episode.
        observation: ``"colors"``, ``"ids"``, ``"onehot"``, ``"facelets"`` or
            ``"image"``.
        reward: ``"cost"`` charges 1 a move, whereas ``"sparse"`` pays 1 for solving it.
        max_episode_steps: Truncate an episode after this many moves. Registered
            ids pass their own, and ``None`` never truncates.
        autoreset_mode: How a finished episode restarts. All three of
            Gymnasium's modes are implemented.
        render_mode: ``"rgb_array"``, ``"ansi"``, or ``None``.
        width: Frame width for ``"rgb_array"`` and for an image observation.
        height: Frame height.
        views: Viewpoints an image observation stacks on the channel axis.
        yaw: Degrees to swing the camera sideways from head-on when drawing.
        pitch: Degrees to raise it.

    Raises:
        ValueError: If the puzzle jumbles, or an argument is out of range.
    """

    metadata: dict[str, list[str] | int | AutoresetMode] = _metadata()

    def __init__(
        self,
        num_envs: int = 1,
        recipe: RecipeLike = DEFAULT_RECIPE,
        *,
        scramble: Depth = DEFAULT_SCRAMBLE,
        observation: Observation = "colors",
        reward: RewardScheme = "cost",
        max_episode_steps: int | None = None,
        autoreset_mode: AutoresetMode | str = AutoresetMode.NEXT_STEP,
        render_mode: str | None = None,
        width: int = 320,
        height: int = 320,
        views: tuple[View, ...] = DEFAULT_VIEWS,
        yaw: float = -28.0,
        pitch: float = 20.0,
    ) -> None:
        if num_envs < 1:
            msg = f"num_envs must be at least 1, got {num_envs}"
            raise ValueError(msg)
        self._low, self._high = depth_range(scramble)
        modes = self.metadata["render_modes"]
        if render_mode is not None and isinstance(modes, list) and render_mode not in modes:
            msg = f"render_mode must be one of {modes}, not {render_mode!r}"
            raise ValueError(msg)

        self._batch = PuzzleBatch(
            recipe, num_envs, yaw=yaw, pitch=pitch, supersample=2, background=(255, 255, 255, 255)
        )
        refuse_jumbling(self._batch.query, str(recipe))

        self.num_envs = num_envs
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
        self._views = tuple(views)
        # Only a facelet observation needs one, so it is derived when asked
        # for and not before.
        self._layout: Layout | None = layout_for(self._batch) if observation == "facelets" else None

        self._solved: ObsArray = solved_stickers(self._batch)
        self._action_count = self._batch.action_count
        # An image observation is small by convention, and drawing at the
        # render size would make an observation a hundred times dearer.
        obs_w, obs_h = (32, 32) if observation == "image" else (width, height)
        self._obs_size = (obs_w, obs_h)

        self.single_observation_space = observation_space(
            self._batch, observation, views=len(self._views), width=obs_w, height=obs_h
        )
        self.single_action_space = action_space(self._action_count)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)

        self._elapsed = np.zeros(num_envs, dtype=np.uint32)
        self._pending = np.zeros(num_envs, dtype=np.bool_)
        self._depths = np.zeros(num_envs, dtype=np.uint32)

    @property
    def batch(self) -> PuzzleBatch:
        """The puzzles themselves, for drawing and for looking at."""
        return self._batch

    @property
    def actions(self) -> list[str]:
        """The name of every move, in action-index order."""
        return self._batch.actions

    def puzzle(self, index: int = 0) -> Puzzle:
        """A puzzle in copy ``index``'s state, for visualization and inspection.

        Instantiated on demand and rotated through the action history. The
        environment maintains its own state and is unaffected by mutations to
        this object.

        Args:
            index: Which copy to reproduce.

        Returns:
            A puzzle in that copy's position.
        """
        puzzle = Puzzle(self._batch.query)
        for action in self._batch.history(index):
            puzzle.apply_action(action)
        return puzzle

    @property
    def states(self) -> ObsArray:
        """Every copy's sticker array, as an ``(n, k)`` block."""
        return np.asarray(self._batch.observations())

    @property
    def goal(self) -> ObsArray:
        """The color in each slot when the puzzle is solved."""
        return self._solved.copy()

    @property
    def layout(self) -> Layout:
        """The puzzle's face-by-face layout.

        For a cube, the numbering cubes are usually written in. For anything else,
        the puzzle's own. A puzzle that jumbles has none, and asking raises `ValueError`.
        """
        if self._layout is None:
            self._layout = layout_for(self._batch)
        return self._layout

    @property
    def scrambles(self) -> NDArray[np.uint32]:
        """How many moves from solved each copy's episode started."""
        return self._depths.copy()

    @property
    def permutations(self) -> Permutations | None:
        """Every move as a permutation of the slots, or ``None`` if it has none.

        What the batch itself steps with, read straight off it. For an agent
        that wants to look a move ahead without taking it.
        """
        table = self._batch.permutations
        return None if table is None else np.asarray(table)

    def _observe(self) -> ObsArray:
        return encode(
            self._batch,
            self._observation,
            layout=self._layout,
            views=self._views,
            width=self._obs_size[0],
            height=self._obs_size[1],
        )

    def _draw_depths(self, which: NDArray[np.bool_]) -> list[int]:
        """Pick a scramble length for each selected copy, and none for the rest."""
        depths = np.zeros(self.num_envs, dtype=np.uint32)
        picked = which.sum()
        if picked:
            drawn = (
                np.full(int(picked), self._low, dtype=np.uint32)
                if self._low == self._high
                else self.np_random.integers(self._low, self._high + 1, size=int(picked))
            )
            depths[which] = drawn
            self._depths[which] = drawn
        return [int(d) for d in depths]

    def _restart(self, which: NDArray[np.bool_]) -> None:
        """Solve and re-scramble the copies `which` selects, in one call each."""
        rows = [bool(v) for v in which]
        self._batch.solve(rows)
        self._batch.scramble(self._draw_depths(which))
        self._elapsed[which] = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Depth] | None = None) -> tuple[ObsArray, VectorInfo]:
        """Solve every puzzle and walk each one away from solved again.

        Args:
            seed: Seeds every copy, so the same seed gives the same batch.
            options: ``{"scramble": k}`` or ``{"scramble": (low, high)}``
                overrides the depth for this episode. A negative depth, or a
                range that runs backward, is refused.

        Returns:
            The first batch of observations, and its info.

        """
        super().reset(seed=seed)
        if seed is not None:
            self._batch.seed_all(seed)
        if options is not None and "scramble" in options:
            self._low, self._high = depth_range(options["scramble"])

        everything = np.ones(self.num_envs, dtype=np.bool_)
        self._restart(everything)
        self._pending[:] = False
        solved = np.asarray(self._batch.solved())
        applied = np.ones(self.num_envs, dtype=np.bool_)
        return self._observe(), self._infos(applied, solved)

    def step(
        self, actions: NDArray[np.int64] | list[int]
    ) -> tuple[ObsArray, NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], VectorInfo]:
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

        # Handed over as the block it already is: a list comprehension here
        # would cost one trip through the interpreter per puzzle, every step.
        solved_arr, applied_arr = self._batch.step(acts)
        solved = np.asarray(solved_arr)
        applied = np.asarray(applied_arr)
        self._elapsed += 1

        terminations = solved.copy()
        truncations = np.zeros(self.num_envs, dtype=np.bool_)
        if self._max_episode_steps is not None:
            truncations = (self._elapsed >= self._max_episode_steps) & ~terminations

        rewards = (
            np.where(solved, 1.0, 0.0) if self._reward == "sparse" else np.full(self.num_envs, -1.0, dtype=np.float64)
        )

        infos: VectorInfo = {}
        if self.autoreset_mode is AutoresetMode.NEXT_STEP:
            if restarting.any():
                # The action was not the agent's to take: these finished last
                # step, and this one restarts them.
                self._restart(restarting)
                rewards[restarting] = 0.0
                terminations[restarting] = False
                truncations[restarting] = False
                applied[restarting] = True
                solved = np.asarray(self._batch.solved())
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
                solved = np.asarray(self._batch.solved())

        infos.update(self._infos(applied, solved))
        return self._observe(), rewards, terminations, truncations, infos

    def _masks(self) -> NDArray[np.bool_]:
        return np.asarray(self._batch.action_masks())

    def _infos(self, applied: NDArray[np.bool_], solved: NDArray[np.bool_]) -> VectorInfo:
        return {"applied": applied, "is_solved": solved, "action_mask": self._masks(), "scramble": self._depths.copy()}

    def render(self) -> tuple[NDArray[np.uint8], ...] | tuple[str, ...] | None:
        """Draw every puzzle, however ``render_mode`` asked for.

        ``"rgb_array"`` draws the whole batch in one call, by looking up which
        sticker each pixel shows instead of rasterizing each frame, so a
        frame per puzzle per step is affordable. ``"ansi"`` is free.

        Returns:
            One frame or one board per puzzle, or ``None`` with no render mode.
        """
        if self.render_mode == "rgb_array":
            frames = np.asarray(self._batch.render(self._width, self._height, channels=3))
            return tuple(np.ascontiguousarray(f) for f in frames)
        if self.render_mode == "ansi":
            states = self.states
            return tuple(text_board(states[i], self._solved) for i in range(self.num_envs))
        return None

    def close_extras(self, **kwargs: object) -> None:
        """Nothing to release: the batch owns no window, process or device."""

    def __repr__(self) -> str:
        return f"TwistyPuzzleVectorEnv(num_envs={self.num_envs}, stickers={self._batch.sticker_count})"
