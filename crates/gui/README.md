# twistypuzzle-gui

The desktop interface for [`twistypuzzle`](https://pypi.org/project/twistypuzzle/): a graphical interface rendering via the simulator's CPU software rasterizer, independent of external GPU or browser runtimes.

```bash
pip install "twistypuzzle[gui]"
twistypuzzle-gui
```

Drag rotates the camera, clicking an arrow executes a layer turn, and scroll zooms. Keyboard shortcuts: Space scrambles, `R` resets, `U` undoes.

## Command-Line Interface

`twistypuzzle-gui` provides both an interactive graphical interface and command-line execution. It initializes the specified puzzle at the requested configuration, while information flags display data and exit without spawning a window.

```bash
twistypuzzle-gui "Rubik's Cube (3x3x3)"          # a cataloged puzzle, by name
twistypuzzle-gui '?shell=C$1&cut=C$1/3'          # arbitrary puzzle recipe query
twistypuzzle-gui Megaminx --scramble 20 --seed 7 # initialize with scramble
twistypuzzle-gui --moves "A B2' C" --theme dark  # apply move sequence
twistypuzzle-gui --list                          # list all cataloged puzzles
twistypuzzle-gui --help                          # complete command options
```

| Option | Description |
| --- | --- |
| `[PUZZLE]`, `-p`, `--puzzle` | A cataloged puzzle by name, or a recipe query string. Anything beginning with `?` is parsed as a recipe. |
| `-r`, `--recipe` | Explicit recipe query string. |
| `-s`, `--scramble`, `--seed` | Pseudo-random walk scramble with deterministic seed. |
| `-m`, `--moves` | Apply a written sequence, such as `"A B2' C"`. |
| `--depth`, `--no-animate` | Depth for the Scramble button, and optional animation toggle. |
| `--yaw`, `--pitch`, `--distance` | Initial camera orientation and distance. |
| `--no-arrows`, `--no-edges` | Toggle rendering of turn arrows and piece edges. |
| `--theme`, `--size` | `dark`, `light`, or `system`, and window dimensions in logical pixels. |
| `-l`, `--list`, `--polyhedra` | Print the catalog or polyhedron codes, then exit. |
| `-V`, `--version`, `-h`, `--help` | Print version or help information, then exit. |

Invalid command-line invocations return an error message with exit code 2.

## Python Integration

The wheel installs the executable as `twistypuzzle-gui`, and the `twistypuzzle_gui` module forwards arguments to the underlying executable, allowing programmatic invocation from Python scripts or notebooks:

```python
import twistypuzzle_gui

twistypuzzle_gui.launch("Megaminx", "--scramble", "12")  # blocks until closed
twistypuzzle_gui.launch("Megaminx", wait=False)  # asynchronous execution
print(twistypuzzle_gui.capture("--list"))  # capture stdout from command-line query flags
```

`python -m twistypuzzle_gui [OPTIONS] [PUZZLE]` passes through command arguments.

The desktop interface operates as an independent binary or via the optional `[gui]` extra, decoupling windowing toolkit dependencies from headless simulator workflows.
