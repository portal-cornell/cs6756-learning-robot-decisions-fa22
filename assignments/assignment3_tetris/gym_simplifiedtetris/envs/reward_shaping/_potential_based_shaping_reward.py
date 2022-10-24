"""Potential-based shaping reward.

See #30 for a discussion on reward shaping.
"""

from typing import Tuple

import numpy as np


class _PotentialBasedShapingReward(object):
    """A potential-based shaping reward based on the feature, holes.

    :attr _heuristic_range: min and max heuristic values seen so far.
    :attr _old_potential: previous potential.
    """

    # The number of lines cleared is in the range [0, 4]. Every potential is in
    # the range [0, 1]. Therefore, the difference between the new potential
    # and the old potential is in the range [-1, 1]. Hence, the shaping reward
    # range is [-1, 5].
    reward_range = (-1, 5)

    # The initial potential is 1 because there are no holes at the start of a
    # game.
    _INITIAL_POTENTIAL = 1

    def __init__(self) -> None:
        """Initialise the shaping reward object."""
        # Setting the range in this way ensures that the min and max are definitely updated the first time the method "_update_range" is called.
        self._heuristic_range = {"min": 1e7, "max": -1e7}

        self._old_potential = self._INITIAL_POTENTIAL

    def _get_reward(self) -> Tuple[float, int]:
        """Compute and return the potential-based shaping reward.

        :return: potential-based shaping reward and the number of lines cleared.
        """
        num_lines_cleared = self._engine._clear_rows()

        # I chose the potential function to be a function of the well-known holes feature because the number of holes in a given state is (loosely speaking) inversely proportional to the potential of a state.
        heuristic_value = np.count_nonzero(
            (self._engine._grid).cumsum(axis=1) * ~self._engine._grid
        )

        self._update_range(heuristic_value)

        # I wanted the difference in potentials to be in [-1, 1] to improve the stability of neural network convergence. I also wanted the agent to frequently receive non-zero rewards (since bad-performing agents in the standard game of Tetris rarely receive non-zero rewards). Hence, the value of holes was scaled by using the smallest and largest values of holes seen thus far to obtain a value in [0, 1). The result of this was then subtracted from 1 (to obtain a value in (0, 1]) because a state with a larger value of holes has a smaller potential (generally speaking). The function numpy.clip is redundant here.
        new_potential = np.clip(
            1
            - (heuristic_value - self._heuristic_range["min"])
            / (self._heuristic_range["max"] + 1e-9),
            0,
            1,
        )

        # Notice that gamma was set to 1, which isn't strictly allowed since it should be less than 1 according to Theorem 1 in this paper. I found that the agent rarely received positive rewards using this reward function because the agent was frequently transitioning to states with a lower potential (since it was rarely clearing lines).
        # HACK: Added 0.3.
        shaping_reward = (new_potential - self._old_potential) + num_lines_cleared + 0.3

        self._old_potential = new_potential

        return (shaping_reward, num_lines_cleared)

    def _get_terminal_reward(self) -> float:
        """Compute and return the terminal potential-based shaping reward.

        :return: terminal potential-based shaping reward.
        """
        terminal_shaping_reward = -self._old_potential
        self._old_potential = self._INITIAL_POTENTIAL
        return terminal_shaping_reward

    def _update_range(self, heuristic_value: int) -> None:
        """Update the heuristic range.

        :param heuristic_value: computed heuristic value.
        """

        if heuristic_value > self._heuristic_range["max"]:
            self._heuristic_range["max"] = heuristic_value

        if heuristic_value < self._heuristic_range["min"]:
            self._heuristic_range["min"] = heuristic_value
