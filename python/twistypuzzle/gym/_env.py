"""A single twisty puzzle as a Gymnasium environment.

A batch of one. Everything the vectorized environment does in Rust (turning,
scrambling, reading a state, drawing a frame), this does too, for one puzzle,
so that the two cannot drift apart in what they mean by a move or a reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from twistypuzzle._native import Puzzle, PuzzleBatch

from ._common import (
    DEFAULT_VIEWS,
    ObsSpaceArray,
    action_space,
    depth_range,
    encode,
    layout_for,
    observation_space,
    refuse_jumbling,
    solved_stickers,
    text_board,
)

if TYPE_CHECKING:
    from typing import Final

    from numpy.typing import NDArray

    from twistypuzzle._native import Layout
    from twistypuzzle._types import RecipeLike

    from ._common import Depth, ObsArray, Observation, Permutations, RewardScheme, View


__all__: list[str] = ["DEFAULT_MAX_EPISODE_STEPS", "DEFAULT_RECIPE", "DEFAULT_SCRAMBLE", "TwistyPuzzleEnv"]

#: The puzzle an unqualified environment means.
DEFAULT_RECIPE: Final[str] = "Rubik's Cube (3x3x3)"

#: How far from solved a reset leaves the puzzle.
DEFAULT_SCRAMBLE: Final[int] = 10

#: How long an episode may run before it is truncated.
DEFAULT_MAX_EPISODE_STEPS: Final[int] = 200

#: What `info` carries. Declared so a type checker can see it, since Gymnasium's
#: own signature is a bare dict.
_Info = dict[str, "bool | int | NDArray[np.bool_]"]


def _metadata() -> dict[str, list[str] | int]:
    """What Gymnasium reads off the class before it is instantiated.

    Built by a call instead of written as a literal because Gymnasium declares
    `metadata` as an ordinary attribute that an instance may replace, so it
    cannot be a `ClassVar`.

    Returns:
        The render modes and frame rate.
    """
    return {"render_modes": ["rgb_array", "ansi", "human"], "render_fps": 10}


class TwistyPuzzleEnv(gym.Env[ObsSpaceArray, np.int64]):
    """One twisty puzzle, scrambled and solved a move at a time.

    Episodes initialize at states generated via bounded random walks from the
    solved configuration. While uniform random states are typically near-maximal
    distance from solved, random walks of length ``k`` provide a curriculum
    solvable in at most ``k`` steps by construction.

    A step is a gather, not a turn. Turning the geometry re-derives the whole
    cut structure, which for a 3x3x3 is about two milliseconds. The moves of a
    puzzle that does not jumble are fixed permutations, so they are derived
    once per puzzle and applied by indexing thereafter, in Rust.

        >>> import gymnasium as gym
        >>> import twistypuzzle.gym  # registers the ids
        >>> env = gym.make("twistypuzzle/Puzzle-v0", scramble=3)
        >>> obs, info = env.reset(seed=0)
        >>> obs.shape
        (54,)
        >>> obs, reward, terminated, truncated, info = env.step(0)
        >>> reward
        -1.0

    Args:
        recipe: Catalog name, recipe query string, or ``Recipe``.
        scramble: How many moves from solved a reset leaves the puzzle, or
            ``(low, high)`` to draw a length for each episode.
        observation: ``"colors"``, ``"ids"``, ``"onehot"``, ``"facelets"`` or
            ``"image"``.
        reward: ``"cost"`` charges 1 a move, whereas ``"sparse"`` pays 1 for solving it.
        render_mode: ``"rgb_array"``, ``"ansi"``, ``"human"``, or ``None``.
        width: Frame width for ``"rgb_array"``.
        height: Frame height.
        views: Viewpoints an image observation stacks on the channel axis.
        yaw: Degrees to swing the camera sideways from head-on when drawing.
        pitch: Degrees to raise it.

    Raises:
        ValueError: If the puzzle jumbles, and so has no sticker array.
    """

    metadata: dict[str, list[str] | int] = _metadata()

    def __init__(
        self,
        recipe: RecipeLike = DEFAULT_RECIPE,
        *,
        scramble: Depth = DEFAULT_SCRAMBLE,
        observation: Observation = "colors",
        reward: RewardScheme = "cost",
        render_mode: str | None = None,
        width: int = 320,
        height: int = 320,
        views: tuple[View, ...] = DEFAULT_VIEWS,
        yaw: float = -28.0,
        pitch: float = 20.0,
    ) -> None:
        self._low, self._high = depth_range(scramble)
        modes = self.metadata["render_modes"]
        if render_mode is not None and isinstance(modes, list) and render_mode not in modes:
            msg = f"render_mode must be one of {modes}, not {render_mode!r}"
            raise ValueError(msg)

        self._batch = PuzzleBatch(recipe, 1, yaw=yaw, pitch=pitch, supersample=2, background=(255, 255, 255, 255))
        refuse_jumbling(self._batch.query, str(recipe))

        self._observation: Observation = observation
        self._reward = reward
        self.render_mode = render_mode
        self._width = width
        self._height = height
        self._views = tuple(views)
        self._layout: Layout | None = layout_for(self._batch) if observation == "facelets" else None
        self._depth = 0
        self._moves = 0

        self._solved: ObsArray = solved_stickers(self._batch)
        self._action_count = self._batch.action_count
        obs_w, obs_h = (32, 32) if observation == "image" else (width, height)
        self._obs_size = (obs_w, obs_h)
        self.observation_space = observation_space(
            self._batch, observation, views=len(self._views), width=obs_w, height=obs_h
        )
        self.action_space = action_space(self._action_count)

    @property
    def puzzle(self) -> Puzzle:
        """A puzzle in this environment's state, for visualization and inspection.

        Instantiated on demand and rotated through the action history.
        The environment maintains its own state and is unaffected by mutations to this object.
        """
        puzzle = Puzzle(self._batch.query)
        for action in self._batch.history(0):
            puzzle.apply_action(action)
        return puzzle

    @property
    def batch(self) -> PuzzleBatch:
        """The batch of one this environment is."""
        return self._batch

    @property
    def state(self) -> ObsArray:
        """The color in each sticker slot, before any encoding."""
        return np.asarray(self._batch.observations())[0]

    @property
    def goal(self) -> ObsArray:
        """The color in each slot when the puzzle is solved."""
        return self._solved.copy()

    @property
    def layout(self) -> Layout:
        """The puzzle's face-by-face layout.

        For a cube, the numbering cubes are usually written in. For anything
        else, the puzzle's own. A puzzle that jumbles has none, and asking
        raises `ValueError`.
        """
        if self._layout is None:
            self._layout = layout_for(self._batch)
        return self._layout

    @property
    def scramble_depth(self) -> int:
        """How many moves from solved this episode started."""
        return self._depth

    @property
    def permutations(self) -> Permutations | None:
        """Every move as a permutation of the slots, or ``None`` if it has none.

        What the batch itself steps with, read straight off it. For an agent
        that wants to look a move ahead without taking it.
        """
        table = self._batch.permutations
        return None if table is None else np.asarray(table)

    @property
    def actions(self) -> list[str]:
        """The name of every move, in action-index order."""
        return self._batch.actions

    def _observe(self) -> ObsArray:
        return encode(
            self._batch,
            self._observation,
            layout=self._layout,
            views=self._views,
            width=self._obs_size[0],
            height=self._obs_size[1],
        )[0]

    def _info(self, *, applied: bool = True) -> _Info:
        return {
            "is_solved": bool(np.asarray(self._batch.solved())[0]),
            "moves": int(self._moves),
            "applied": applied,
            "action_mask": np.asarray(self._batch.action_masks())[0],
            "scramble": self._depth,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Depth] | None = None) -> tuple[ObsArray, _Info]:
        """Solve the puzzle and walk it ``scramble`` moves away again.

        Args:
            seed: Seeds the walk, so the same seed gives the same position.
            options: ``{"scramble": k}`` or ``{"scramble": (low, high)}``
                overrides the depth for this episode. A negative depth, or a
                range that runs backward, is refused.

        Returns:
            The first observation and its info.

        """
        super().reset(seed=seed)
        if seed is not None:
            self._batch.seed_all(seed)
        if options is not None and "scramble" in options:
            self._low, self._high = depth_range(options["scramble"])

        self._depth = self._low if self._low == self._high else int(self.np_random.integers(self._low, self._high + 1))
        self._batch.reset(scramble=self._depth)
        self._moves = 0
        return self._observe(), self._info()

    def step(self, action: np.int64 | int) -> tuple[ObsArray, float, bool, bool, _Info]:
        """Make one move.

        A move naming a layer that cannot turn (which only a puzzle without a
        permutation table can produce) leaves the puzzle alone while incrementing
        the step count, so an agent need not consult ``info["action_mask"]`` to act.

        Args:
            action: Which move to make.

        Returns:
            The usual five: observation, reward, terminated, truncated, info.

        Raises:
            IndexError: If ``action`` is not one of the puzzle's moves.
        """
        index = int(action)
        if not 0 <= index < self._action_count:
            msg = f"action {index} out of range (have {self._action_count})"
            raise IndexError(msg)
        solved_arr, applied_arr = self._batch.step([index])
        solved = bool(np.asarray(solved_arr)[0])
        applied = bool(np.asarray(applied_arr)[0])
        self._moves += 1
        reward = (1.0 if solved else 0.0) if self._reward == "sparse" else -1.0
        # Truncation is the TimeLimit wrapper's job, which `gymnasium.make`
        # applies from the registered `max_episode_steps`.
        return self._observe(), reward, solved, False, self._info(applied=applied)

    def render(self) -> NDArray[np.uint8] | str | None:
        """Draw the puzzle, however ``render_mode`` asked for.

        ``"rgb_array"`` returns an ``(h, w, 3)`` frame from the built-in software
        renderer, which needs no GPU, display server or browser. ``"ansi"``
        returns the board as text, one line per face. ``"human"`` writes that
        text to standard output (for a window to turn the puzzle in, install
        the ``gui`` extra and run ``twistypuzzle-gui``).

        Returns:
            A frame, a string, or ``None`` when there is no render mode.
        """
        if self.render_mode == "rgb_array":
            frame = np.asarray(self._batch.render(self._width, self._height, channels=3))[0]
            return np.ascontiguousarray(frame)
        if self.render_mode in {"ansi", "human"}:
            board = text_board(self.state, self._solved)
            if self.render_mode == "human":
                print(board)
                return None
            return board
        return None

    def close(self) -> None:
        """Nothing to release: the renderer owns no window and no device."""
