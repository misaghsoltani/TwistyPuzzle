"""A single twisty puzzle as a Gymnasium environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from twistypuzzle._native import Puzzle

from ._common import (
    ObsSpaceArray,
    action_space,
    encode,
    observation_space,
    permutation_table,
    refuse_jumbling,
    text_board,
)

if TYPE_CHECKING:
    from typing import Any, Final

    from numpy.typing import NDArray

    from twistypuzzle._types import RecipeLike

    from ._common import ObsArray, Observation, Permutations, RewardScheme


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


def _metadata() -> dict[str, Any]:
    """What Gymnasium reads off the class before it is instantiated.

    Built by a call rather than written as a literal because Gymnasium declares
    `metadata` as an ordinary attribute that an instance may replace, so it
    cannot be a `ClassVar`.

    Returns:
        The render modes and frame rate.
    """
    return {"render_modes": ["rgb_array", "ansi", "human"], "render_fps": 10}


class TwistyPuzzleEnv(gym.Env[ObsSpaceArray, np.int64]):
    """One twisty puzzle, scrambled and solved a move at a time.

    The observation is the sticker array (one categorical value per sticker slot).

    An episode starts a known number of moves away from solved rather than at a
    state drawn uniformly. A uniform state is almost always as far from solved
    as a state can be, which teaches a learner nothing. By contrast, a walk of length ``k``
    from the goal is solvable in at most ``k`` moves by construction.

    A step is a gather, not a turn. Turning the geometry re-derives the whole
    cut structure, which for a 3x3x3 is about two milliseconds. The moves of a
    puzzle that does not jumble are fixed permutations, so they are derived once
    per recipe and applied by indexing thereafter. The geometry is kept for
    drawing and caught up with the state only when a frame is asked for.

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
        scramble: How many moves from solved a reset leaves the puzzle.
        observation: ``"colors"`` or ``"onehot"``.
        reward: ``"cost"`` charges 1 a move, whereas ``"sparse"`` pays 1 for solving it.
        render_mode: ``"rgb_array"``, ``"ansi"``, ``"human"``, or ``None``.
        width: Frame width for ``"rgb_array"``.
        height: Frame height.
        yaw: Degrees to swing the camera sideways from head-on when drawing.
        pitch: Degrees to raise it.

    Raises:
        ValueError: If the puzzle jumbles, and so has no sticker array.
    """

    metadata: dict[str, Any] = _metadata()

    def __init__(
        self,
        recipe: RecipeLike = DEFAULT_RECIPE,
        *,
        scramble: int = DEFAULT_SCRAMBLE,
        observation: Observation = "colors",
        reward: RewardScheme = "cost",
        render_mode: str | None = None,
        width: int = 320,
        height: int = 320,
        yaw: float = -28.0,
        pitch: float = 20.0,
    ) -> None:
        if scramble < 0:
            msg = f"scramble must not be negative, got {scramble}"
            raise ValueError(msg)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            msg = f"render_mode must be one of {self.metadata['render_modes']}, not {render_mode!r}"
            raise ValueError(msg)

        self._puzzle = Puzzle(recipe, yaw=yaw, pitch=pitch, supersample=2, background=(255, 255, 255, 255))
        refuse_jumbling(self._puzzle.query, str(recipe))

        self._scramble = scramble
        self._observation: Observation = observation
        self._reward = reward
        self.render_mode = render_mode
        self._width = width
        self._height = height

        self._solved: ObsArray = np.asarray(self._puzzle.solved_stickers, dtype=np.uint8)
        self._color_count = self._puzzle.color_count
        self._action_count = self._puzzle.action_count
        self.observation_space = observation_space(self._puzzle.sticker_count, self._color_count, observation)
        self.action_space = action_space(self._action_count)

        # `None` for a puzzle whose moves are not fixed permutations, which is
        # then turned move by move.
        self._table: Permutations | None = permutation_table(self._puzzle)
        self._state: ObsArray = self._solved.copy()
        # Moves made since the last reset, so the geometry can be caught up
        # with the state when a frame is wanted.
        self._history: list[int] = []
        self._drawable = True

    @property
    def puzzle(self) -> Puzzle:
        """The puzzle itself, caught up with the environment's state.

        For drawing and for looking at. Turning it does not move the
        environment, which keeps its own state, and the next :meth:`reset` puts the
        two back in step.
        """
        self._catch_up()
        return self._puzzle

    @property
    def state(self) -> ObsArray:
        """The color in each sticker slot, before any encoding."""
        return self._state.copy()

    @property
    def goal(self) -> ObsArray:
        """The color in each slot when the puzzle is solved."""
        return self._solved.copy()

    @property
    def permutations(self) -> Permutations | None:
        """Every move as a permutation of the slots, or ``None`` if it has none."""
        return self._table

    @property
    def actions(self) -> list[str]:
        """The name of every move, in action-index order."""
        return self._puzzle.actions

    def _catch_up(self) -> None:
        """Turn the geometry until it agrees with the state."""
        if self._drawable:
            return
        if not self._puzzle.restore():
            self._puzzle.reset()
        for action in self._history:
            self._puzzle.apply_action(action)
        self._drawable = True

    def _advance(self, action: int) -> bool:
        """Make one move, by table if there is one and by geometry if not.

        Args:
            action: Which move to make.

        Returns:
            Whether it could be made at all.
        """
        if self._table is None:
            self._catch_up()
            applied = self._puzzle.apply_action(action)
            if applied:
                self._state = np.asarray(self._puzzle.stickers(), dtype=np.uint8)
            return applied
        self._state = self._state[self._table[action]]
        self._history.append(action)
        self._drawable = False
        return True

    def _observe(self) -> ObsArray:
        return encode(self._state, self._color_count, self._observation)

    def _is_solved(self) -> bool:
        return bool(np.array_equal(self._state, self._solved))

    def _info(self, *, applied: bool = True) -> _Info:
        if self._table is None:
            mask = np.asarray(self._puzzle.action_mask(), dtype=np.bool_)
            moves = int(self._puzzle.move_count)
        else:
            # A table is only derived for a puzzle whose layers all turn and
            # keep turning, which the table's own construction checks rather
            # than assumes.
            mask = np.ones(self._action_count, dtype=np.bool_)
            moves = len(self._history)
        return {"is_solved": self._is_solved(), "moves": moves, "applied": applied, "action_mask": mask}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsArray, _Info]:
        """Solve the puzzle and walk it ``scramble`` moves away again.

        Args:
            seed: Seeds the walk, so the same seed gives the same position.
            options: ``{"scramble": k}`` overrides the depth for this episode.

        Returns:
            The first observation and its info.

        Raises:
            ValueError: If ``options["scramble"]`` is negative.
        """
        super().reset(seed=seed)
        depth = self._scramble
        if options is not None and "scramble" in options:
            depth = int(options["scramble"])
            if depth < 0:
                msg = f"scramble must not be negative, got {depth}"
                raise ValueError(msg)

        self._state = self._solved.copy()
        self._history = []
        self._drawable = False
        if self._table is None:
            # Undoing the turns is one exact rotation each, whereas rebuilding
            # re-derives the whole geometry. Only a locked layer forces that.
            if not self._puzzle.restore():
                self._puzzle.reset()
            self._drawable = True

        count = self._action_count
        if count > 0 and depth > 0:
            made = 0
            previous: int | None = None
            # Bounded: a puzzle whose layers all lock would otherwise spin here.
            for _ in range(depth * 8):
                if made == depth:
                    break
                action = int(self.np_random.integers(count))
                # Never undo the move just made, or a walk of length k can end
                # up much closer to solved than k.
                if previous is not None and action == previous ^ 1:
                    continue
                if self._advance(action):
                    previous = action
                    made += 1
        return self._observe(), self._info()

    def step(self, action: np.int64 | int) -> tuple[ObsArray, float, bool, bool, _Info]:
        """Make one move.

        A move naming a layer that cannot turn (which only a puzzle without a
        permutation table can produce) leaves the puzzle alone and still costs
        a step, so an agent need not consult ``info["action_mask"]`` to act.

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
        applied = self._advance(index)
        solved = self._is_solved()
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

        A frame costs a catch-up: the geometry is turned through the moves the
        state has taken since the last reset, which is why drawing every step is
        far dearer than taking one.

        Returns:
            A frame, a string, or ``None`` when there is no render mode.
        """
        if self.render_mode == "rgb_array":
            self._catch_up()
            frame = np.asarray(self._puzzle.render(self._width, self._height))
            return np.ascontiguousarray(frame[:, :, :3])
        if self.render_mode in {"ansi", "human"}:
            board = text_board(self._state, self._solved)
            if self.render_mode == "human":
                print(board)
                return None
            return board
        return None

    def close(self) -> None:
        """Nothing to release: the renderer owns no window and no device."""
