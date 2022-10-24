
# Environments<!-- omit in toc -->

- [1. Available environments](#1-available-environments)
- [2. Methods](#2-methods)
- [3. Action and observation spaces](#3-action-and-observation-spaces)
- [4. Game ending](#4-game-ending)
- [5. Building more environments](#5-building-more-environments)

## 1. Available environments

There are currently 64 environments provided:

- `simplifiedtetris-binary-{height}x{width}-{piece_size}-v0`: The observation space is a flattened NumPy array containing a binary representation of the grid, plus the current piece's ID. A reward of +1 is given for each line cleared, and 0 otherwise
- `simplifiedtetris-partbinary-{height}x{width}-{piece_size}-v0`: The observation space is a flattened NumPy array containing a binary representation of the grid excluding the top `piece_size` rows, plus the current piece's ID. A reward of +1 is given for each line cleared, and 0 otherwise
- `simplifiedtetris-binary-shaped-{height}x{width}-{piece_size}-v0`: The observation space is a flattened NumPy array containing a binary representation of the grid, plus the current piece's ID. The reward function is a potential-based reward function based on the _holes_ feature
- `simplifiedtetris-partbinary-shaped-{height}x{width}-{piece_size}-v0`: The observation space is a flattened NumPy array containing a binary representation of the grid excluding the top `piece_size` rows, plus the current piece's ID. The reward function is a potential-based shaping reward based on the _holes_ feature

where (height, width) are either (20, 10), (10, 10), (8, 6), or (7, 4), and the piece size is either 1, 2, 3, or 4.

## 2. Methods

The `reset()` method returns a 1D array containing some grid binary representation, plus the current piece's ID.

```python
>>> obs = env.reset()
>>> print(obs)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 4]
```

Each environment's `step(action: int)` method returns four values:

- `observation` (**NumPy array**): a 1D array that contains some binary representation of the grid, plus the current piece's ID
- `reward` (**float**): the amount of reward received from the previous action
- `done` (**bool**): a game termination flag
- `info` (**dict**): only contains the `num_rows_cleared` due to taking the previous action

```python
>>> obs, reward, done, info = env.step(action)
```

The `render(mode: str = 'human')` method defaults to rendering to a display.

```python
>>> env.render()
```

The user has access to the following controls during rendering:

- Pause (*SPACEBAR*)
- Speed up (*RIGHT key*)
- Slow down (*LEFT key*)
- Quit (*ESC*)

The user can close all open windows using:

```python
>>> env.close()
```

## 3. Action and observation spaces

Each environment comes with an `observation_space` that is a `Box` space and an `action_space` that is a `Discrete` space. At each time step, the agent must choose an action, an integer from a particular range. Each action maps to a tuple that specifies the column to drop the piece and its rotation. The number of actions available for each of the pieces is given below:

- Monominos: w
- Dominos: 2w - 1
- Trominoes: 4w - 4
- Tetriminos: 4w - 6,

where w is the grid width. Some actions have the same effect on the grid as others with this action space. When actions are selected uniformly at random, and the current piece is the 'O' Tetrimino, two actions are chosen with a smaller probability than the other actions.

## 4. Game ending

Each game terminates if any of the dropped piece's square blocks enter into the top `piece_size` rows before any full rows are cleared. This condition ensures that scores achieved are lower bounds on the score that the agent could have obtained on a standard game of Tetris, as laid out in Colin Fahey's ['Standard Tetris' specification](https://www.colinfahey.com/tetris/tetris.html#:~:text=5.%20%22Standard%20Tetris%22%20specification).

## 5. Building more environments

The user can implement more custom Gym environments by ensuring that they inherit from `_SimplifiedTetrisBaseEnv` and are registered in a similar way to this:

```python
>>> register_env(
>>>     incomplete_id=f"simplifiedtetris-binary",
>>>     entry_point=f"gym_simplifiedtetris.envs:SimplifiedTetrisBinaryEnv",
>>> )
```