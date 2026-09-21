"""Twisty puzzles as Gymnasium environments.

Importing this module registers the ids, so it is all the setup there is::

    >>> import gymnasium as gym
    >>> import twistypuzzle.gym
    >>> env = gym.make("twistypuzzle/Puzzle-v0", scramble=5)
    >>> obs, info = env.reset(seed=0)
    >>> obs.shape
    (54,)

The observation is the sticker array. Four other encodings are available
through ``observation=``: ``"ids"`` indexes the sticker in each slot instead of
its color, distinguishing states that identical colors cannot differentiate. ``"onehot"``
expands colors into indicator vectors for neural network inputs. ``"facelets"`` provides
the canonical face-by-face numbering scheme. And ``"image"`` renders multi-view
representations of the puzzle as normalized ``(6, 32, 32)`` float arrays in ``[0.0, 1.0]``.

An episode starts a known number of moves from solved. ``scramble=(1, 30)``
draws a length for each episode instead of fixing one.

Every id carries a vector entry point, so ``make_vec`` uses a batched
implementation in which a whole vector step (every turn, every state, and every
finished episode restarting) is one call into Rust with the interpreter released::

    >>> envs = gym.make_vec("twistypuzzle/Puzzle-v0", num_envs=64, scramble=5)
    >>> obs, infos = envs.reset(seed=0)
    >>> obs.shape
    (64, 54)
    >>> envs.close()

``vectorization_mode="sync"`` and ``"async"`` still work and wrap
:class:`TwistyPuzzleEnv` the way Gymnasium wraps anything else.

Not every puzzle can be one of these. A puzzle that *jumbles* (such as a Radiolarian,
a jumble prism, or the Big Chop) has legal turns that leave pieces where no piece
sits when it is solved, so it has no fixed set of sticker slots and no state
vector. Thirty-six of the eighty-five cataloged puzzles do not jumble, and
those are the ones with ids. Requesting an unsupported recipe raises an exception
instead of returning an invalid array. :func:`twistypuzzle.non_jumbling_entries` lists them.
"""

from __future__ import annotations

# Gymnasium and NumPy are not dependencies of the package: `twistypuzzle`
# itself installs nothing, and these arrive only with `twistypuzzle[gym]`.
# Importing this module is therefore the first point at which their absence
# can be noticed, and a bare `No module named 'gymnasium'` does not tell
# anyone which extra they are missing.
try:
    from ._common import DEFAULT_VIEWS, Depth, ObsArray, Observation, RewardScheme, View
    from ._env import DEFAULT_MAX_EPISODE_STEPS, DEFAULT_RECIPE, DEFAULT_SCRAMBLE, TwistyPuzzleEnv
    from ._registration import GENERIC_ID, NAMESPACE, env_ids, register_envs
    from ._vector import TwistyPuzzleVectorEnv
except ModuleNotFoundError as exc:
    # Only rewrite the two the extra is responsible for. Anything else failing
    # to import here is a real bug and must not be reported as a missing
    # extra.
    if exc.name not in {"gymnasium", "numpy"}:
        raise
    msg = (
        f"twistypuzzle.gym needs {exc.name}, which the base package does not "
        f"install. The Gymnasium environments are an optional extra:\n"
        f"\n"
        f"    pip install 'twistypuzzle[gym]'\n"
        f"\n"
        f"twistypuzzle itself has no dependencies and is unaffected."
    )
    raise ModuleNotFoundError(msg, name=exc.name) from exc

__all__ = [
    "DEFAULT_MAX_EPISODE_STEPS",
    "DEFAULT_RECIPE",
    "DEFAULT_SCRAMBLE",
    "DEFAULT_VIEWS",
    "GENERIC_ID",
    "NAMESPACE",
    "Depth",
    "ObsArray",
    "Observation",
    "RewardScheme",
    "TwistyPuzzleEnv",
    "TwistyPuzzleVectorEnv",
    "View",
    "env_ids",
    "register_envs",
]

register_envs()
