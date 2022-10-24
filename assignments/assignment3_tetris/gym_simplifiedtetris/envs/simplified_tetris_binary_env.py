"""Simplified Tetris binary env.
"""

import numpy as np
from gym import spaces
from gym_simplifiedtetris.envs._simplified_tetris_base_env import (
    _SimplifiedTetrisBaseEnv,
)
from gym_simplifiedtetris.register import register_env


class SimplifiedTetrisBinaryEnv(_SimplifiedTetrisBaseEnv):
    """A custom Gym env for Tetris.

    The observation space is the grid's binary representation, plus the current piece's id.
    """

    @property
    def observation_space(self) -> spaces.Box:
        """Override the superclass property.

        :return: Box obs space.
        """
        low = np.append(np.zeros(self._width_ * self._height_), 0)
        high = np.append(
            np.ones(self._width_ * self._height_),
            self._num_pieces_ - 1,
        )
        return spaces.Box(
            low=low,
            high=high,
            dtype=int,
        )

    def _get_obs(self) -> np.ndarray:
        """Return the grid's binary representation, plus the current piece's id.

        Overrides the superclass method.

        :return: current obs.
        """
        current_grid = self._engine._grid.flatten()
        return np.append(current_grid, self._engine._piece._id)


register_env(
    incomplete_id=f"simplifiedtetris-binary",
    entry_point=f"gym_simplifiedtetris.envs:SimplifiedTetrisBinaryEnv",
)
