# Twisty Puzzle

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![image](https://img.shields.io/pypi/v/twistypuzzle.svg)](https://pypi.python.org/pypi/twistypuzzle)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&label=package%20manager)](https://pixi.sh)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Static Badge](https://img.shields.io/badge/statically%20typed-mypy-039dfc)

A simulator of twisty puzzles that computes every cut and every turn in exact arithmetic, so the geometry never drifts: a face that should meet another meets it exactly, and a fifth of a turn is a fifth of a turn.

This codebase is a Rust implementation (with Python bindings) of [hydropyrum/puzzle](https://github.com/hydropyrum/puzzle).

<p align="center">
  <img src="https://raw.githubusercontent.com/misaghsoltani/TwistyPuzzle/master/docs/catalog.png" alt="All 85 cataloged puzzles, grouped by family" width="600">
</p>

## Contents

- [Install](#install)
- [Using the Library](#using-the-library)
- [The state](#the-state)
- [Gymnasium Environment](#gymnasium-environment)
- [GUI (Desktop Application)](#gui-desktop-application)
- [Development](#development)
- [License](#license)

## Install

```bash
pip install twistypuzzle           # the library
pip install "twistypuzzle[gym]"    # + the Gymnasium environments
pip install "twistypuzzle[gui]"    # + the GUI desktop application
```

The library itself has no dependencies. However, if you want to install the optional `[gym]` extra, it will install Gymnasium and NumPy. The optional `[gui]` extra installs the desktop application.

## Using the Library

A puzzle is a shell carved by cuts. Name one from the catalog, write it as a recipe, or spell it out:

```python
import twistypuzzle as tp

p1 = tp.Puzzle("Rubik's Cube (3x3x3)")
p1.render(512, 512).save("cube_1.png")

p2 = tp.Puzzle("?shell=C$1&cut=C$1/3")
p2.render(512, 512).save("cube_22.png")

p3 = tp.Puzzle(shell=tp.Shape.polyhedron("C"), cuts=tp.Shape.polyhedron("C", "1/3"))
p3.render(512, 512).save("cube_3.png")
```

Offsets are exact expressions such as `"1/3"`, `"sqrt(5)/5"`, or `tp.Fraction(1, 3)`. A fraction like `1/3` has no exact `double`, and a cube cut at 0.3333333333333333 is a different puzzle from one cut at a third.

```python
p.scramble(20)  # twenty random moves
p.settle()
p.apply("A B' C2")  # a written sequence
p.turn(0, +1)  # one layer, one direction
p.turn_to(0, 2)  # one layer, to a named stop
p.undo()  # take it back
```

Every view setting is both a keyword and a property, and `render` takes the same keywords for a single frame without them sticking:

```python
p = tp.Puzzle("Megaminx", background=(255, 255, 255, 255), supersample=2)
p.show_arrows = True
frame = p.render(512, 512, yaw=-28, pitch=20)
```

## The state

A state reads back as an integer array. For example:

```python
>>> p = tp.Puzzle("Rubik's Cube (3x3x3)")
>>> p.solved_stickers == [i // 9 for i in range(54)]
True
>>> p.stickers()[:9]
[0, 0, 0, 0, 0, 0, 0, 0, 0]
```

`sticker_ids()` gives the permutation instead of the colors, which tells apart states the colors cannot. `ground_atoms()` gives the same thing as logic atoms `("color", "s0", "c0")`.

Note that not every puzzle has one. A puzzle that jumbles (such as a Radiolarian, a jumble prism, or the Big Chop) has legal turns that leave pieces where no piece sits when it is solved, so there is no fixed set of sticker slots to number. Of the 85 cataloged puzzles, 36 do not jumble. `tp.non_jumbling_entries()` lists them.

## Gymnasium Environment

```python
import gymnasium as gym
import twistypuzzle.gym  # registers the ids

env = gym.make("twistypuzzle/Puzzle-v0", scramble=8)
envs = gym.make_vec("twistypuzzle/Puzzle-v0", num_envs=256, scramble=8)
```

An episode starts a known number of moves from solved rather than at a state drawn uniformly. `make_vec` uses a batched implementation by default and works with the `"sync"` and `"async"` vectorizers too.

A step is a gather, not a turn. Turning a puzzle geometrically re-derives its whole cut structure. The moves of a puzzle that does not jumble are fixed permutations of its sticker slots, so each one is derived once and applied by indexing ever after.

## GUI (Desktop Application)

```bash
pip install "twistypuzzle[gui]"

twistypuzzle-gui
```

<p align="center">
  <img src="https://raw.githubusercontent.com/misaghsoltani/TwistyPuzzle/master/docs/gui.png" alt="The desktop interface" width="720">
</p>

## Development

Use [Pixi](https://pixi.sh) to manage the workspace and toolchains:

```bash
pixi run develop       # compile and install the Python extension
pixi run test          # run all Rust and Python test suites
pixi run gui           # launch the desktop interface
pixi run build-wheel   # build release wheel into target/wheels/
```

Or use the repository Makefile:

```bash
make help         # list everything
make all          # format checks + lint + Rust/Python tests
make wheel        # build wheel into dist/
make gui          # run desktop interface
make docker-ci    # run container matrix from docker-bake.hcl
```

Running tests directly:

```bash
cargo test --release                          # the Rust suite
cargo test --release --features bigint-only   # the same, without fast path
pytest tests/python                           # the Python suite
```

## License

Released under the [MIT License](LICENSE).
