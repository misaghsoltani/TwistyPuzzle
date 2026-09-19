# twistypuzzle-gui

The desktop interface for [`twistypuzzle`](https://pypi.org/project/twistypuzzle/): a window around the simulator's own software renderer, with no GPU, display server or browser involved.

```bash
pip install "twistypuzzle[gui]"
twistypuzzle-gui
```

Drag to turn the view, click an arrow to turn a layer, scroll to zoom. Space scrambles, `R` resets, `U` undoes.

Installed on its own it is the same application. The `[gui]` extra exists so that the library does not carry a windowing toolkit for users who only want the simulator.
