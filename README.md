# Twisty Puzzle

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![image](https://img.shields.io/pypi/v/twistypuzzle.svg)](https://pypi.python.org/pypi/twistypuzzle)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&label=package%20manager)](https://pixi.sh)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Static Badge](https://img.shields.io/badge/statically%20typed-mypy-039dfc)

<p align="center">
  <img src="https://raw.githubusercontent.com/misaghsoltani/TwistyPuzzle/master/crates/gui/ui/app-icon.svg" alt="Logo" width="20%">
</p>

An exact-arithmetic twisty puzzle simulator and software renderer. Polyhedral geometry, cutting planes, and rotations are evaluated via exact algebraic number arithmetic over $\mathbb{Q}(\theta)$, eliminating floating-point drift.

This codebase is a Rust implementation (with Python bindings) of [hydropyrum/puzzle](https://github.com/hydropyrum/puzzle).

<p align="center">
  <img src="https://raw.githubusercontent.com/misaghsoltani/TwistyPuzzle/master/docs/catalog.png" alt="All 85 cataloged puzzles, grouped by family" width="50%">
</p>

## Contents

- [Twisty Puzzle](#twisty-puzzle)
  - [Contents](#contents)
  - [Install](#install)
  - [Using the Library](#using-the-library)
  - [The state](#the-state)
  - [Many puzzles at once](#many-puzzles-at-once)
  - [Numbered face by face](#numbered-face-by-face)
  - [As narrow as the puzzle allows](#as-narrow-as-the-puzzle-allows)
  - [Gymnasium Environment](#gymnasium-environment)
  - [GUI (Desktop Application)](#gui-desktop-application)
  - [Development](#development)
  - [License](#license)

## Install

```bash
pip install twistypuzzle  # core simulator library
pip install "twistypuzzle[gym]"  # optional Gymnasium environments
pip install "twistypuzzle[gui]"  # optional desktop application
```

The core library has no third-party runtime dependencies. The optional `[gym]` extra installs Gymnasium and NumPy for reinforcement learning environments, while `[gui]` installs the standalone desktop application.

## Using the Library

A twisty puzzle is defined by an outer polyhedral shell intersected by planar cuts. Puzzles can be instantiated by catalog name, recipe query string, or explicit shape objects:

```python
import twistypuzzle as tp

p1 = tp.Puzzle("Rubik's Cube (3x3x3)")
p1.render(512, 512).save("cube_1.png")

p2 = tp.Puzzle("?shell=C$1&cut=C$1/3")
p2.render(512, 512).save("cube_22.png")

p3 = tp.Puzzle(shell=tp.Shape.polyhedron("C"), cuts=tp.Shape.polyhedron("C", "1/3"))
p3.render(512, 512).save("cube_3.png")
```

Cutting plane offsets accept exact algebraic expressions such as `"1/3"`, `"sqrt(5)/5"`, or `tp.Fraction(1, 3)`. Exact representations prevent rounding errors inherent in IEEE 754 floating-point approximations.

```python
p.scramble(20)  # queue 20 pseudo-random turns
p.settle()  # complete active animation and queued moves
p.apply("A B' C2")  # apply notation sequence
p.turn(0, +1)  # rotate grip 0 by one angular stop in positive direction
p.turn_to(0, 2)  # rotate grip 0 directly to stop index 2
p.undo()  # invert the most recent turn
```

Every view setting is both a keyword and a property, and `render` accepts the same keywords for a single frame without mutating persistent simulator settings:

```python
p = tp.Puzzle("Megaminx", background=(255, 255, 255, 255), supersample=2)
p.show_arrows = True
frame = p.render(512, 512, yaw=-28, pitch=20)
```

## The state

Current puzzle state can be retrieved as an integer array:

```python
>>> p = tp.Puzzle("Rubik's Cube (3x3x3)")
>>> p.solved_stickers == [i // 9 for i in range(54)]
True
>>> p.stickers()[:9]
[0, 0, 0, 0, 0, 0, 0, 0, 0]
```

`sticker_ids()` returns the permutation array instead of colors, distinguishing states that identical colors cannot differentiate. `ground_atoms()` returns the corresponding first-order relational atoms `("color", "s0", "c0")`.

Jumbling puzzles do not have a fixed sticker array. A puzzle that jumbles (such as a Radiolarian, a jumble prism, or the Big Chop) has legal turns that leave pieces where no piece sits when solved, precluding a static numbering of sticker slots. Of the 85 cataloged puzzles, 36 do not jumble. `tp.non_jumbling_entries()` lists them.

## Many puzzles at once

`PuzzleBatch` maintains multiple states of one puzzle and updates them concurrently. Batch transitions and rendering execute concurrently in native Rust with the Python GIL released:

```python
b = tp.PuzzleBatch("Rubik's Cube (3x3x3)", 1024, seed=0)
b.reset(scramble=20)  # reset and apply 20-move random walk per instance
solved, applied = b.step(actions)  # apply one action per instance
b.reset_where(solved, scramble=20)  # reset instances that reached the solved state
b.undo()  # invert the most recent move per instance
b.apply("A B2' C")  # apply notation sequence across all instances
b.set_states(states)  # inject externally generated state arrays
```

Batch execution exploits shared geometric precomputation across instances of a common recipe. For non-jumbling puzzles, actions are fixed slot permutations, allowing moves to be evaluated via vectorized array gathers instead of re-evaluating 3D geometry. Similarly, frame rendering evaluates pixel-to-slot mapping tables once and subsequently renders via direct table lookups.

Benchmark measurements on an Apple M-series system (256 instances of a 3x3x3 cube):

| Execution Model | Transition Latency (per move) | Render Latency (64×64 frame) |
| --- | --- | --- |
| Individual geometric simulation | 1310 µs | 250 µs |
| `PuzzleBatch` (permutation gather & lookup table) | 0.36 µs | 6.5 µs |

`cargo run --release --example bench_batch` benchmarks throughput on the local system.

Batch operations return instances of `Array`: contiguous memory buffers that `numpy.asarray` wraps without copying, and that `tolist()` unpacks directly without requiring NumPy.

```python
import numpy as np

np.asarray(b.observations())  # (1024, 54) uint8, the color in each slot
np.asarray(b.sticker_ids())  # (1024, 54) uint8, which sticker is in it
np.asarray(b.one_hot())  # (1024, 324) uint8 indicators
np.asarray(b.action_masks())  # (1024, 12) bool
np.asarray(b.render(64, 64))  # (1024, 64, 64, 4) uint8 frames
b.render(32, 32, channels=3, channels_first=True, dtype="f4")  # (1024, 3, 32, 32) in [0, 1]
```

Puzzles that jumble or feature dynamic layer locking lack static permutation tables. Such instances simulate individual 3D geometry per step. The API remains identical across execution modes, with `b.is_tabular` reporting whether permutation gathering is active.

## Numbered face by face

The core simulator indexes sticker slots according to geometric cut traversal order. For cube puzzles, standard literature and external solvers typically index stickers by *facelet*: six faces in a fixed sequence, each traversed in a grid. The `Layout` class computes this bidirectional mapping from puzzle geometry:

```python
c = tp.Layout()  # the 3x3x3, or any n×n×n cube
c.faces  # ['U', 'D', 'L', 'R', 'B', 'F']
c.move_names[:4]  # ['U-1', 'U1', 'D-1', 'D1']
c.goal_colors.tolist()  # arange(54) // 9
c.goal_ids.tolist()  # arange(54)
c.move_index("R"), c.move_index("R'")  # (7, 6), where "R1" and "R-1" name the same two
```

Faces follow the standard sequence `U, D, L, R, B, F` with `U` at `+y`, `R` at `+x`, and `F` at `+z`. Within a face, facelets are ordered row-major with `ROW × COL` defining the outward surface normal. Move indexing is face-major, with `2k` representing counterclockwise rotation as viewed from the exterior and `2k+1` representing clockwise rotation. `tests/data/facelet_cube.txt` specifies the 12 base permutations and 128 validation move sequences, against which the layout implementation is verified.

Non-cube puzzles are indexed canonically in color order: solved states correspond to contiguous color sequences `0,0,...,1,1,...`. Faces correspond to color classes, facelets to sticker slots, and moves to primitive actions.

```python
m = tp.Layout("Megaminx")
m.is_cube, m.face_count, m.facelet_count  # (False, 12, 132)
m.move_names[:4]  # ['A', "A'", 'B', "B'"]
```

`Layout` operates directly on permutations without 3D geometric modeling, providing a standalone transition engine:

```python
states = c.scrambled(4096, (1, 30), seed=0)  # random walks with length sampled in [1, 30]
states = c.next_states(states, moves)  # batched single-move state transitions
path, moves = c.trajectories(20, 4096, seed=0)  # state trajectories across all random walks
c.one_hot(states)  # (4096, 324) indicators
c.inverse_moves(moves)  # inverse move indices
triples = c.with_moves(macro_sequences)  # composite macro-action sequences
```

The `to_facelets` and `from_facelets` functions translate between slot and face representations. Calling `batch.set_facelets(layout, states)` loads external states into a simulator batch for simulation, inspection, and rendering, while `batch.render_states(states, 32, 32)` renders them directly. Reachability from the solved state is not evaluated (reachability determination is computationally intractable for general permutation puzzles). Unreachable configurations are simulated and rendered faithfully without reaching the solved state.

## As narrow as the puzzle allows

Memory representations dynamically select the narrowest integer width capable of representing puzzle bounds. A 3x3x3 contains 54 slots and 6 colors, allowing states, slot indices, actions, and permutation tables to reside in `uint8` arrays. A 9x9x9 contains 486 slots, promoting slot indices to `uint16` while retaining `uint8` for color indices. `Array.typestr` specifies the memory layout, and `observation_space` mirrors it. No artificial bounds constrain puzzle complexity: geometries with more than 256 faces automatically widen integer representation.

This optimization extends throughout the native backend. Permutation tables and state buffers for puzzles with <= 256 slots remain 8-bit, reducing memory bandwidth by 75% compared to 32-bit representations during large-scale simulation.

## Gymnasium Environment

```python
import gymnasium as gym
import twistypuzzle.gym  # registers the ids

env = gym.make("twistypuzzle/Puzzle-v0", scramble=8)
envs = gym.make_vec("twistypuzzle/Puzzle-v0", num_envs=256, scramble=8)
```

Episodes initialize at states generated via bounded random walks from the solved configuration: random walks of length `k` guarantee solvability in at most `k` steps, providing structured difficulty curricula. `scramble=(1, 30)` samples a random episode depth.

`make_vec` defaults to the native batched environment, executing complete vector transitions, state gathers, and episodic auto-resets in a single call into Rust with GIL release. Standard `"sync"` and `"async"` vector wrappers remain supported.

Five encodings, for one environment or a vector of them:

| `observation=` | shape for a 3x3x3 | Description |
| --- | --- | --- |
| `"colors"` | `(54,)` `uint8` | discrete color index per slot |
| `"ids"` | `(54,)` `uint8` | sticker index permutation, distinguishing states that identical colors cannot differentiate |
| `"onehot"` | `(324,)` `uint8` | binary indicator vector of colors |
| `"facelets"` | `(54,)` `uint8` | facelet color encoding under canonical face-major layout |
| `"image"` | `(6, 32, 32)` `float32` | two opposite corner viewpoints, 3 channels each, normalized to `[0.0, 1.0]` |

`reward="cost"` assigns a step penalty of -1 per move, so the cumulative episode return equals the negative trajectory length. `reward="sparse"` yields a terminal reward of +1 upon reaching the solved state.

## GUI (Desktop Application)

```bash
pip install "twistypuzzle[gui]"

twistypuzzle-gui  # interactive graphical window
twistypuzzle-gui Megaminx --scramble 20 --seed 7  # initialize with scramble
twistypuzzle-gui '?shell=C$1&cut=C$1/3' --theme dark
twistypuzzle-gui --list  # display catalog entries
twistypuzzle-gui --help  # complete command options
```

The wheel installs that executable as the `twistypuzzle-gui` command, and `python -m twistypuzzle_gui` (or `twistypuzzle_gui.launch(...)`) delegates invocations to the executable.

The GUI employs software rasterization without external GPU dependencies, rendering frames on demand when state updates occur. When stationary, no CPU cycles are spent rendering, and animated turns or pointer drag interactions activate a 60 Hz timer. Frame scratch buffers are reused across frames. Rendering at 1280x1280 with 2x supersampling processes 2560x2560 pixels (~50 MB framebuffer), and persistent scratch memory avoids 60 Hz allocation and zeroing cycles.

Benchmark measurements on an Apple M-series system (single 1280x1280 frame of a 3x3x3 cube):

| Execution Configuration | Frame Duration | Allocations per Frame |
| --- | --- | --- |
| Rasterize and resolve, per-frame buffer allocation | 7.2 ms | 56.6 MB |
| Rasterize and resolve, persistent scratch buffer reuse | 6.2 ms | 0 B |

`pixi run bench-frame` benchmarks frame rendering throughput. Environment variables `SIZE`, `SUPERSAMPLE`, `RECIPE`, and `FRAMES` configure benchmark parameters.

Window surfaces utilize zero-copy double-buffering. The rasterizer renders directly into the target surface buffer submitted to the window system, eliminating intermediate blit copies. Alternating between two surface buffers ensures the drawing thread never contends with the surface currently held by the display compositor, reducing surface refresh overhead from 0.88 ms to 0.07 ms.

<p align="center">
  <img src="https://raw.githubusercontent.com/misaghsoltani/TwistyPuzzle/master/docs/gui.png" alt="The desktop interface" width="80%">
</p>

## Development

Use [Pixi](https://pixi.sh) to manage the workspace and toolchains:

```bash
pixi run develop  # compile and install the Python extension
pixi run test  # run all Rust and Python test suites
pixi run gui  # launch the desktop interface
pixi run build-wheel  # build release wheel into target/wheels/
```

Or use the repository Makefile:

```bash
make help  # list available targets
make all  # format checks + lint + Rust/Python tests
make wheel  # build wheel into dist/
make gui  # run desktop interface
make docker-ci  # run container matrix from docker-bake.hcl
```

Running tests directly:

```bash
cargo test --release  # Rust test suite
cargo test --release --features bigint-only  # Rust test suite without small-integer fast path
pytest tests/python  # Python test suite
```

## License

Released under the [MIT License](LICENSE).
