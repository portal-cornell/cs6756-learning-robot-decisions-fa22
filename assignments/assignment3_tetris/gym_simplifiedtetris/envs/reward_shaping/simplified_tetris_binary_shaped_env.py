"""Simplified Tetris env, which has a binary obs space and a shaped reward function.
"""

from typing import Any

from gym_simplifiedtetris.envs.simplified_tetris_binary_env import (
    SimplifiedTetrisBinaryEnv,
)
from gym_simplifiedtetris.register import register_env

from ._potential_based_shaping_reward import _PotentialBasedShapingReward


class SimplifiedTetrisBinaryShapedEnv(
    _PotentialBasedShapingReward, SimplifiedTetrisBinaryEnv
):
    """A simplified Tetris environment.

    The reward function is a potential-based shaping reward and the observation space is the grid's binary representation plus the current piece's id.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the object."""
        super().__init__()
        SimplifiedTetrisBinaryEnv.__init__(self, **kwargs)


register_env(
    incomplete_id="simplifiedtetris-binary-shaped",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisBinaryShapedEnv",
)
