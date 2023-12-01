# TheCatIsOnTheTablut

## Overview

This repository contains an implementation of an AI player for the board game Tablut. The AI has been developed as part of a university exam module for [FUNDAMENTALS OF ARTIFICIAL INTELLIGENCE AND KNOWLEDGE REPRESENTATION](https://www.unibo.it/en/teaching/course-unit-catalogue/course-unit/2023/446566).


## Installation

1. Clone the repository:

```bash
git clone https://github.com/NotXia/tablut-player
```

2. Install the dependencies:

```bash
cd tablut-player
pip install -r requirements.txt
```

3. If you want to use the Cython version:
```bash
cd src
python setup.py build_ext --inplace
```

## Genetic algorithm
To start the genetic algorithm to train the weights for the heuristics, run:
```
cd src/ga-training
python train.py                                 \
    --epochs [num epochs]                       \
    --indivs [num individuals per population]   \
    --timeout [timeout for decision]            \
    --output [weights output directory]         \
    --mutation-value [amount of each mutation]  \
    --mutation-prob [probability of mutation]   \
    --gui
```
Check `python train.py --help` for more options.



## Run the player
In the `src` directory, run:
```
python play.py                              \
    --color [WHITE/BLACK]                   \
    --ip [server ip]                        \
    --timeout [seconds]                     \
    --tol [timeout tolerance]               \
    --weights [path to weights]             \
    --tt-size [transposition table size]    \
    --debug
```

To run the server you need to follow the guide on [the tablut server repo](https://github.com/AGalassi/TablutCompetition).


## Team members
- [Valerio Costa](https://github.com/Rda1027)
- [Luca Domeniconi](https://github.com/AjejeBrazorfEU)
- [Claudia Maiolino](https://github.com/jeanclaude8)
- [Riccardo Xia](https://github.com/NotXia)