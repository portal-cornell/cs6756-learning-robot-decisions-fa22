"""Simplified Tetris env, which has a part-binary obs space and shaping reward function.
"""

from typing import Any

from gym_simplifiedtetris.envs.reward_shaping._potential_based_shaping_reward import (
    _PotentialBasedShapingReward,
)
from gym_simplifiedtetris.envs.simplified_tetris_part_binary_env import (
    SimplifiedTetrisPartBinaryEnv,
)
from gym_simplifiedtetris.register import register_env


class SimplifiedTetrisPartBinaryShapedEnv(
    _PotentialBasedShapingReward, SimplifiedTetrisPartBinaryEnv
):
    """A simplified Tetris env.

    The reward function is a scaled heuristic score and the obs space is the grid's part binary representation plus the current piece's id.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the object."""
        super().__init__()
        SimplifiedTetrisPartBinaryEnv.__init__(self, **kwargs)


register_env(
    incomplete_id="simplifiedtetris-partbinary-shaped",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisPartBinaryShapedEnv",
)
