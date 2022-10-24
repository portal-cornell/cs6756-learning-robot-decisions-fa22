"""Simplified Tetris env, which has a part-binary obs space.
"""

import numpy as np
from gym import spaces
from gym_simplifiedtetris.envs._simplified_tetris_base_env import (
    _SimplifiedTetrisBaseEnv,
)
from gym_simplifiedtetris.register import register_env


class SimplifiedTetrisPartBinaryEnv(_SimplifiedTetrisBaseEnv):
    """A simplified Tetris environment.

    The observation space is a flattened NumPy array containing the grid's binary representation excluding the top |piece_size| rows, plus the current piece's id.
    """

    @property
    def observation_space(self) -> spaces.Box:
        """Override the superclass property.

        :return: Box obs space.
        """
        low = np.append(
            np.zeros(self._width_ * (self._height_ - self._piece_size_)),
            0,
        )
        high = np.append(
            np.ones(self._width_ * (self._height_ - self._piece_size_)),
            self._num_pieces_ - 1,
        )
        return spaces.Box(
            low=low,
            high=high,
            dtype=int,
        )

    def _get_obs(self) -> np.array:
        """Returns a flattened NumPy array containing the grid's binary representation excluding the top |piece_size| rows, plus the current piece's id.

        Overrides the superclass method.

        :return: current observation.
        """
        current_grid = self._engine._grid[:, self._piece_size_ :].flatten()
        return np.append(current_grid, self._engine._piece._id)


register_env(
    incomplete_id="simplifiedtetris-partbinary",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisPartBinaryEnv",
)
