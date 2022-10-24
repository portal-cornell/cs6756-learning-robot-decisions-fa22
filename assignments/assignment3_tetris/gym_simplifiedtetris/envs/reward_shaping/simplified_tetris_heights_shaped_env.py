"""
TODO
"""

from typing import Any

from gym_simplifiedtetris.envs.simplified_tetris_heights_env import (
    SimplifiedTetrisHeightsEnv,
)
from gym_simplifiedtetris.register import register_env

from ._potential_based_shaping_reward import _PotentialBasedShapingReward


class SimplifiedTetrisHeightsShapedEnv(
    _PotentialBasedShapingReward, SimplifiedTetrisHeightsEnv
):
    """A simplified Tetris environment.

    The reward function is a potential-based shaping reward and the observation space is the grid's column heights plus the current piece's id.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the object."""
        super().__init__()
        SimplifiedTetrisHeightsEnv.__init__(self, **kwargs)


register_env(
    incomplete_id="simplifiedtetris-heights-shaped",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisHeightsShapedEnv",
)
