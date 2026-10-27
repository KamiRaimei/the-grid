The Grid - System Simulation

A system simulation where the objective is a perfect Fibonacci calculation loop system with an automated Master Control Program as control.

Requirements

Linux / macOS:
- Python 3.12 or higher
- numpy

Windows:
- Python 3.12 or higher
- numpy
- windows-curses

How to run

Install dependencies:

```
pip install -r requirements.txt
```

Or install from pyproject.toml:

```
pip install .
```

Run the simulation:

```
python -m grid_sim --no-curses
```

Options:

```
--no-curses          Disable curses interface (fallback mode)
--width N            Set grid width (default from config)
--height N           Set grid height (default from config)
--load-personality   Load a specific personality file
--reset-learning     Reset learned training data
--learning-rate F    Set initial learning rate
--exploration F      Set initial exploration rate
```