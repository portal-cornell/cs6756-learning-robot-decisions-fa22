"""Simplified Tetris env with the col heights and piece id as the obs space."""

import numpy as np
from gym import spaces
from gym_simplifiedtetris.envs._simplified_tetris_base_env import (
    _SimplifiedTetrisBaseEnv,
)
from gym_simplifiedtetris.register import register_env


class SimplifiedTetrisHeightsEnv(_SimplifiedTetrisBaseEnv):
    """A simplified Tetris environment.

    The observation space is a NumPy array containing the grid's column heights and the current piece's id."""

    @property
    def observation_space(self) -> spaces.Box:
        """Override the superclass property.

        :return: Box obs space.
        """
        low = np.append(np.zeros(self._width_), 0)
        high = np.append(np.zeros(self._width_) + self._height_, self._num_pieces_ - 1)
        return spaces.Box(
            low=low,
            high=high,
            dtype=np.int,
        )

    def _get_obs(self):
        """Return the current observation.

        :return: a NumPy array containing the column heights and the current piece id.
        """
        col_heights = self._engine.get_col_heights()
        return np.append(col_heights, self._engine._piece._id)


register_env(
    incomplete_id="simplifiedtetris-heights",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisHeightsEnv",
)
