"""Pierre Dellacherie's agent.
"""

from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from gym_simplifiedtetris.agents.base import BaseAgent
from gym_simplifiedtetris.envs._simplified_tetris_engine import SimplifiedTetrisEngine


class DellacherieAgent(BaseAgent):
    """An agent that selects actions based on the Dellacherie feature set."""

    WEIGHTS = np.array([-1, 1, -1, -1, -4, -1], dtype="double")

    def predict(self, env: SimplifiedTetrisEngine, **kwargs: Any) -> int:
        """Return the action yielding the largest heuristic score.

        Ties are separated using a priority rating, which is based on the translation and rotation of the current piece.

        :param env: environment that the agent resides in.
        :return: action with the largest rating (where ties are separated based on the priority).
        """
        dell_scores = self._compute_dell_scores(env)
        return np.argmax(dell_scores)

    def _compute_dell_scores(self, env: SimplifiedTetrisEngine) -> np.ndarray:
        """Compute and return the Dellacherie feature set values.

        :param env: environment that the agent resides in.
        :return: Dellacherie feature values.
        """
        dell_scores = np.empty((env.num_actions,), dtype="double")

        available_actions = env._engine._all_available_actions[env._engine._piece._id]

        for action, (translation, rotation) in available_actions.items():
            old_grid = deepcopy(env._engine._grid)
            old_colour_grid = deepcopy(env._engine._colour_grid)
            old_anchor = deepcopy(env._engine._anchor)

            env._engine._rotate_piece(rotation)
            env._engine._anchor = [translation, 0]

            env._engine._hard_drop()
            env._engine._update_grid(True)
            env._engine._clear_rows()

            feature_values = np.array(
                [func(env) for func in self._get_dell_funcs()], dtype="double"
            )
            dell_scores[action] = np.dot(feature_values, self.WEIGHTS)

            env._engine._update_grid(False)

            env._engine._grid = deepcopy(old_grid)
            env._engine._colour_grid = deepcopy(old_colour_grid)
            env._engine._anchor = deepcopy(old_anchor)

        best_actions = np.argwhere(dell_scores == np.amax(dell_scores)).flatten()
        is_a_tie = len(best_actions) > 1

        # Resort to the priorities if there is a tie.
        if is_a_tie:
            return self._get_priorities(
                best_actions=best_actions,
                available_actions=available_actions,
                x_spawn_pos=env._width_ / 2 + 1,
                num_actions=env.num_actions,
            )
        return dell_scores

    def _get_priorities(
        self,
        best_actions: np.ndarray,
        available_actions: Dict[int, Dict[int, Tuple[int, int]]],
        x_spawn_pos: int,
        num_actions: int,
    ) -> np.ndarray:
        """Compute and return the priorities of the available actions.

        :param best_actions: actions with the maximum ratings.
        :param available_actions: actions available to the agent.
        :param x_spawn_pos: x-coordinate of the spawn position.
        :param num_actions: number of actions available to the agent.
        :return: priorities.
        """
        priorities = np.ones((num_actions,), dtype="double") * -np.inf

        for action in best_actions:
            translation, rotation = available_actions[action]
            priorities[action] = (
                (100 * abs(translation - x_spawn_pos))
                - (rotation / 90)
                + (10 * (translation < x_spawn_pos))
            )

        return priorities

    def _get_dell_funcs(self) -> List[Callable[..., int]]:
        """Return the Dellacherie feature functions.

        :return: Dellacherie feature functions.
        """
        return [
            self._get_landing_height,
            self._get_eroded_cells,
            self._get_row_transitions,
            self._get_col_transitions,
            self._get_holes,
            self._get_cum_wells,
        ]

    def _get_landing_height(self, env: SimplifiedTetrisEngine) -> int:
        """Compute the landing height and return it.

        Landing height = the midpoint of the last piece to be placed.

        :param env: environment that the agent resides in.
        :return: landing height.
        """
        return (
            env._engine._last_move_info["landing_height"]
            if "landing_height" in env._engine._last_move_info
            else 0
        )

    def _get_eroded_cells(self, env: SimplifiedTetrisEngine) -> int:
        """Return the eroded cells value.

        Eroded cells = number of rows cleared x number of blocks removed that were added to the grid by the last action.

        :param env: environment that the agent resides in.
        :return: eroded cells.
        """
        return (
            env._engine._last_move_info["num_rows_cleared"]
            * env._engine._last_move_info["eliminated_num_blocks"]
            if "num_rows_cleared" in env._engine._last_move_info
            else 0
        )

    def _get_row_transitions(self, env: SimplifiedTetrisEngine) -> int:
        """Return the row transitions value.

        Row transitions = Number of transitions from empty to full cells (or vice versa), examining each row one at a time.

        Author: Ben Schofield
        Source: https://github.com/Benjscho/gym-mdptetris/blob/1a47edc33330deb638a03275e484c3e26932d802/gym_mdptetris/envs/feature_functions.py#L45

        :param env: environment that the agent resides in.
        :return: row transitions.
        """
        # Adds a column either side.
        grid = np.ones((env._engine._width + 2, env._engine._height), dtype="bool")

        grid[1:-1, :] = env._engine._grid.copy()
        return int(np.diff(grid.T).sum())

    def _get_col_transitions(self, env: SimplifiedTetrisEngine) -> int:
        """Return the column transitions value.

        Column transitions = Number of transitions from empty to full (or vice versa), examining each column one at a time.

        Author: Ben Schofield
        Source: https://github.com/Benjscho/gym-mdptetris/blob/1a47edc33330deb638a03275e484c3e26932d802/gym_mdptetris/envs/feature_functions.py#L60

        :param env: environment that the agent resides in.
        :return: column transitions.
        """
        # Adds a full row to the bottom.
        grid = np.ones((env._engine._width, env._engine._height + 1), dtype="bool")

        grid[:, :-1] = env._engine._grid.copy()
        return int(np.diff(grid).sum())

    def _get_holes(self, env: SimplifiedTetrisEngine) -> int:
        """Compute the number of holes present in the current grid and return it.

        A hole is an empty cell with at least one full cell above it in the same column.

        :param env: environment that the agent resides in.
        :return: value of the feature holes.
        """
        return np.count_nonzero((env._engine._grid).cumsum(axis=1) * ~env._engine._grid)

    def _get_cum_wells(self, env: SimplifiedTetrisEngine) -> int:
        """Compute the cumulative wells value and return it.

        Cumulative wells is defined here:
        https://arxiv.org/abs/1905.01652.  For each well, find the depth of
        the well, d(w), then calculate the sum of i from i=1 to d(w).  Lastly,
        sum the well sums.  A block is part of a well if the cells directly on
        either side are full and the block can be reached from above (i.e., there are no full cells directly above it).

        Attribution: Ben Schofield

        :param env: environment that the agent resides in.
        :return: cumulative wells value.
        """
        grid_ext = np.ones(
            (env._engine._width + 2, env._engine._height + 1), dtype="bool"
        )
        grid_ext[1:-1, 1:] = env._engine._grid[:, : env._engine._height]

        # This includes some cells that cannot be reached from above.
        potential_wells = (
            np.roll(grid_ext, 1, axis=0) & np.roll(grid_ext, -1, axis=0) & ~grid_ext
        )

        col_heights = np.zeros(env._engine._width + 2)
        col_heights[1:-1] = env._engine._height - np.argmax(env._engine._grid, axis=1)
        col_heights = np.where(col_heights == env._engine._height, 0, col_heights)

        x = np.linspace(1, env._engine._width + 2, env._engine._width + 2)
        y = np.linspace(env._engine._height + 1, 1, env._engine._height + 1)
        _, yv = np.meshgrid(x, y)

        # A cell that is part of a well must be above the playfield's outline, which consists of the highest full cells in each column.
        above_outline = (col_heights.reshape(-1, 1) < yv.T).astype(int)

        # Exclude the cells that cannot be reached from above by multiplying by 'above_outline'.
        cumulative_wells = np.sum(
            np.cumsum(potential_wells, axis=1) * above_outline,
        )

        return cumulative_wells
