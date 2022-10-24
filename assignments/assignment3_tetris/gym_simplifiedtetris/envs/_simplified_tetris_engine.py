"""Tetris engine.
"""

import random
import time
from copy import deepcopy
from typing import Dict, Iterable, List, Tuple, Union

import cv2.cv2 as cv
import numpy as np
import torch as th
from PIL import Image

from ..auxiliary import Colours, Polymino


class SimplifiedTetrisEngine(object):
    """A simplified Tetris engine.

    Methods related to rendering:
    > _close
    > _add_statistics
    > _render
    > _draw_boundary
    > _get_grid
    > _resize_grid
    > _draw_separating_lines
    > _add_img_left

    Methods related to the game dynamics:
    > _rotate_piece
    > _get_translation_rotation
    > _generate_id_randomly
    > _init_pieces
    > _reset
    > _update_anchor
    > _get_new_piece
    > _is_illegal
    > _hard_drop
    > _clear_rows
    > _update_grid
    > _get_reward
    > _compute_all_available_actions
    > _compute_available_actions
    """

    CELL_SIZE = 50

    BLOCK_COLOURS = {
        0: Colours.WHITE.value,
        1: Colours.CYAN.value,
        2: Colours.ORANGE.value,
        3: Colours.YELLOW.value,
        4: Colours.PURPLE.value,
        5: Colours.BLUE.value,
        6: Colours.GREEN.value,
        7: Colours.RED.value,
    }

    @staticmethod
    def _close() -> None:
        """Close the open windows."""
        cv.waitKey(1)
        cv.destroyAllWindows()
        cv.waitKey(1)

    @staticmethod
    def _add_statistics(
        img: np.ndarray,
        stats: Dict[str, str],
    ) -> None:
        """Add statistics to the array provided.

        :param img: image.
        :param stats: items to be added to the image.
        """
        for stat_idx, (stat_name, stat_value) in enumerate(stats.items()):
            cv.putText(
                img,
                stat_name,
                (50, 60 * (stat_idx + 1)),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                Colours.WHITE.value,
                2,
                cv.LINE_AA,
            )
            cv.putText(
                img,
                stat_value,
                (300, 60 * (stat_idx + 1)),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                Colours.WHITE.value,
                2,
                cv.LINE_AA,
            )

    def __init__(
        self,
        grid_dims: Union[Tuple[int, int], List[int]],
        piece_size: int,
        num_pieces: int,
        num_actions: int,
        obs_space_shape: tuple,
    ) -> None:
        """Initialise the object.

        :param grid_dims: grid dimensions (height, width).
        :param piece_size: size of each piece in use.
        :param num_pieces: number of pieces in use.
        :param num_actions: number of available actions in each state.
        """
        self._height, self._width = grid_dims
        self._piece_size = piece_size
        self._num_pieces = num_pieces
        self._num_actions = num_actions
        self._obs_space_shape = obs_space_shape

        self._grid = np.zeros((grid_dims[1], grid_dims[0]), dtype="bool")
        self._colour_grid = np.zeros((grid_dims[1], grid_dims[0]), dtype="int")
        self._anchor = [grid_dims[1] / 2 - 1, piece_size - 1]

        self._final_scores = np.empty((0,), dtype="int")
        self._sleep_time = 500
        self._show_agent_playing = True

        self._img = np.empty((0,), dtype="int")
        self._last_move_info: Dict[str, float] = {}

        self._pieces = self._init_pieces()
        self._all_available_actions = self._compute_all_available_actions()
        self._reset()

    def _generate_id_randomly(self) -> int:
        """Generate an id, uniformly at random.

        :return: a randomly generated ID.
        """
        return random.randint(0, self._num_pieces - 1)

    def _init_pieces(self) -> Dict[int, Polymino]:
        """Create a dictionary containing the pieces.

        :return: pieces to be used.
        """
        return {idx: Polymino(self._piece_size, idx) for idx in range(self._num_pieces)}

    def _reset(self) -> None:
        """Reset the score, grids, piece coords, piece id and _anchor."""
        self._score = 0

        self._grid = np.zeros_like(self._grid, dtype="bool")
        self._colour_grid = np.zeros_like(self._colour_grid, dtype="int")

        self._update_anchor()
        self._get_new_piece()

    def _render(self, mode: str = "human") -> np.ndarray:
        """Show an image of the current grid, having dropped the current piece.

        The human has the option to pause (SPACEBAR), speed up (RIGHT key),
        slow down (LEFT key) or quit (ESC) the window.

        :param mode: the render mode.
        :return: the image pixel values.
        """
        if mode not in ["human", "rgb_array"]:
            raise ValueError("Mode should be 'human' or 'rgb_array'.")

        grid = self._get_grid()
        self._resize_grid(grid)
        self._draw_separating_lines()
        self._add_img_left()
        self._draw_boundary()

        if mode == "human" and self._show_agent_playing:
            cv.imshow(f"Simplified Tetris", self._img)
            first_key = cv.waitKey(self._sleep_time)

            if first_key == 3:  # Right arrow has been pressed.
                self._sleep_time -= 100

                if self._sleep_time < 100:
                    self._sleep_time = 1

                time.sleep(self._sleep_time / 1000)
            elif first_key == 2:  # Left arrow has been pressed.
                self._sleep_time += 100
                time.sleep(self._sleep_time / 1000)
            elif first_key == 27:  # Esc has been pressed.
                self._show_agent_playing = False
                self._close()
            elif first_key == 32:  # Spacebar has been pressed.
                while True:
                    second_key = cv.waitKey(30)

                    if second_key == 32:  # Spacebar has been pressed.
                        break
                    elif second_key == 27:  # Esc has been pressed.
                        self._show_agent_playing = False
                        self._close()
                        break
        else:
            return self._img

    def _draw_boundary(self) -> None:
        """Draw a horizontal red line to indicate the cut off point."""
        cut_off_point = self._piece_size * self.CELL_SIZE
        self._img[
            cut_off_point
            - int(self.CELL_SIZE / 40) : cut_off_point
            + int(self.CELL_SIZE / 40)
            + 1,
            400:,
            :,
        ] = Colours.RED.value

    def _get_grid(self) -> np.ndarray:
        """Returns the array of the current grid containing the colour tuples.

        :return: the array of the current grid.
        """
        grid = [
            [
                self.BLOCK_COLOURS[self._colour_grid[x_coord][y_coord]]
                for x_coord in range(self._width)
            ]
            for y_coord in range(self._height)
        ]
        return np.array(grid)

    def _resize_grid(self, grid: np.ndarray) -> None:
        """Reshape the grid, convert it to an Image and resize it.

        :param grid: the grid to be resized.
        """
        self._img = np.repeat(
            np.repeat(grid, self.CELL_SIZE, axis=0), self.CELL_SIZE, axis=1
        )
        self._img = self._img.reshape(
            (self._height * self.CELL_SIZE, self._width * self.CELL_SIZE, 3)
        ).astype(np.uint8)
        self._img = Image.fromarray(self._img, "RGB")
        self._img = np.array(self._img)

    def _draw_separating_lines(self) -> None:
        """Draw the horizontal and vertical _black lines to separate the grid's cells."""
        for j in range(-int(self.CELL_SIZE / 40), int(self.CELL_SIZE / 40) + 1):
            self._img[
                [i * self.CELL_SIZE + j for i in range(self._height)], :, :
            ] = Colours.BLACK.value
            self._img[
                :, [i * self.CELL_SIZE + j for i in range(self._width)], :
            ] = Colours.BLACK.value

    def _add_img_left(self) -> None:
        """Add the image that will appear to the left of the grid."""
        img_array = np.zeros((self._height * self.CELL_SIZE, 400, 3)).astype(np.uint8)
        mean_score = (
            0.0 if len(self._final_scores) == 0 else np.mean(self._final_scores)
        )
        stats = {
            "Height": f"{self._height}",
            "Width": f"{self._width}",
            "": "",
            "Current score": f"{self._score}",
            "Mean score": f"{mean_score:.1f}",
        }

        self._add_statistics(img_array, stats)
        self._img = np.concatenate((img_array, self._img), axis=1)

    def _update_anchor(self) -> None:
        """Update the current piece, and reset the _anchor."""
        self._anchor = [self._width / 2 - 1, self._piece_size - 1]

    def _get_new_piece(self) -> None:
        """Get a new piece."""
        random_id = self._generate_id_randomly()
        self._piece = self._pieces[random_id]

    def _is_illegal(self) -> bool:
        """Check if the piece's current position is illegal by looping over each of its square blocks.

        Author: Andrean Lay
        Source: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/42e11e98573edf0c5270d0cc33f1cf1bae3d9d49/src/engine.py#L23

        :return: whether the piece's current position is illegal.
        """
        # Loop over each of the piece's blocks.
        for x_coord, y_coord in self._piece._coords:
            x_pos, y_pos = self._anchor[0] + x_coord, self._anchor[1] + y_coord

            # Don't check if the move is illegal when the block is too high.
            if y_pos < 0:
                continue

            # Check if the move is illegal.
            block_off_left = x_pos < 0
            block_off_right = x_pos >= self._width
            block_below_bot = y_pos >= self._height

            if block_off_left or block_off_right or block_below_bot:
                return True

            cell_full = self._grid[x_pos, y_pos] > 0

            if cell_full:
                return True

        return False

    def _hard_drop(self) -> None:
        """Find where to place the piece by hard dropping the current piece."""
        while True:
            # Keep going until the current piece occupies a full cell, then backtrack once.
            if not self._is_illegal():
                self._anchor[1] += 1
            else:
                self._anchor[1] -= 1
                break

    def _clear_rows(self) -> int:
        """Remove blocks from every full row.

        Author: Andrean Lay
        Source: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/42e11e98573edf0c5270d0cc33f1cf1bae3d9d49/src/engine.py#L83

        :return: number of rows cleared.
        """
        can_clear = np.all(self._grid, axis=0)
        new_grid = np.zeros_like(self._grid)
        new_colour_grid = np.zeros_like(self._colour_grid)
        col_id = self._height - 1

        self._last_move_info["eliminated_num_blocks"] = 0

        for row_num in range(self._height - 1, -1, -1):

            if not can_clear[row_num]:
                new_grid[:, col_id] = self._grid[:, row_num]
                new_colour_grid[:, col_id] = self._colour_grid[:, row_num]
                col_id -= 1
            else:
                self._last_move_info["eliminated_num_blocks"] += self._last_move_info[
                    "rows_added_to"
                ][row_num]

        self._grid = new_grid
        self._colour_grid = new_colour_grid

        num_rows_cleared = sum(can_clear)
        self._last_move_info["num_rows_cleared"] = num_rows_cleared

        return num_rows_cleared

    def _update_grid(self, set_piece: bool) -> None:
        """Either set the current piece or remove the last piece from the grid.

        :param set_piece: whether to set the piece.
        """
        self._last_move_info["rows_added_to"] = {
            row_num: 0 for row_num in range(self._height)
        }
        # Loop over each block.
        for piece_x_coord, piece_y_coord in self._piece._coords:
            x_coord, y_coord = (
                piece_x_coord + self._anchor[0],
                piece_y_coord + self._anchor[1],
            )
            if y_coord < 0:
                print()

            if set_piece:
                self._last_move_info["rows_added_to"][y_coord] += 1
                self._grid[x_coord, y_coord] = 1
                self._colour_grid[x_coord, y_coord] = self._piece._id + 1
            else:
                self._grid[x_coord, y_coord] = 0
                self._colour_grid[x_coord, y_coord] = 0

        anchor_height = self._height - self._anchor[1]
        max_y_coord = self._piece._max_y_coord[self._piece._rotation]
        min_y_coord = self._piece._min_y_coord[self._piece._rotation]
        self._last_move_info["landing_height"] = anchor_height - 0.5 * (
            min_y_coord + max_y_coord
        )

    def _get_reward(self) -> Tuple[float, int]:
        """Return the reward, which is the number of rows cleared.

        :return: the reward and the number of rows cleared.
        """
        num_rows_cleared = self._clear_rows()
        return float(num_rows_cleared), num_rows_cleared

    def _compute_all_available_actions(self) -> Dict[int, Dict[int, Tuple[int, int]]]:
        """Get the actions available for each of the pieces in use.

        :return: all available actions.
        """
        return {
            idx: self._compute_available_actions(piece)
            for (idx, piece) in self._pieces.items()
        }

    def _compute_available_actions(self, piece: Polymino) -> Dict[int, Tuple[int, int]]:
        """Compute the actions available with the current piece.

        Author: Andrean Lay
        Source: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/42e11e98573edf0c5270d0cc33f1cf1bae3d9d49/src/engine.py#L196

        :return: the available actions.
        """
        available_actions: Dict[int, Tuple[int, int]] = {}
        count = 0
        self._piece = piece

        for rotation in self._piece._all_coords.keys():
            self._rotate_piece(rotation)

            max_x_coord = self._piece._max_x_coord[rotation]
            min_x_coord = self._piece._min_x_coord[rotation]

            for translation in range(abs(min_x_coord), self._width - max_x_coord):

                if count == self._num_actions:
                    return available_actions

                self._anchor = [translation, 0]
                self._hard_drop()

                self._update_grid(True)
                available_actions[count] = (translation, rotation)
                self._update_grid(False)

                count += 1
        return available_actions

    def _rotate_piece(self, rotation: int) -> None:
        """Set the piece's rotation and rotate the current piece.

        :param rotation: piece's rotation.
        """
        self._piece._rotation = rotation
        self._piece._coords = self._piece._all_coords[self._piece._rotation]

    def _get_translation_rotation(self, action: int) -> Tuple[int, int]:
        """Return the translation and rotation corresponding to action.

        :param action: action.
        :return: translation and rotation corresponding to action.
        """
        return self._all_available_actions[self._piece._id][action]

    def get_best_action(
        self,
        agent: object,
        obs_space: str,
    ) -> int:
        """
        Finds the best action out of the available actions by looking ahead to
        future possible states and averaging over their V-values.

        :param agent: the agent being evaluated.
        :param obs_space: which observation space is being used.
        :return: the best action selected.
        """
        available_actions = self._all_available_actions[self._piece._id]

        initial_grid = deepcopy(self._grid)
        initial_colour_grid = deepcopy(self._colour_grid)
        initial_anchor = deepcopy(self._anchor)
        initial_piece = deepcopy(self._piece)

        v_values = np.zeros(len(available_actions))

        # Iterate over next possible actions.
        for idx, (translation, rotation) in enumerate(available_actions.values()):
            self._grid = deepcopy(initial_grid)
            self._anchor = [translation, 0]

            self._piece = deepcopy(initial_piece)
            self._piece._rotation = rotation
            self._piece._coords = self._piece._all_coords[self._piece._rotation]

            self._hard_drop()
            self._update_grid(True)

            n_lines_cleared = self._clear_rows()
            features = self.get_feat_values(obs_space)

            # Initialise the max q values with the immediate reward.
            max_q_values = np.ones(self._num_pieces) * n_lines_cleared

            # Iterate over the next possible observations.
            for piece_id in range(self._num_pieces):
                self._piece = self._pieces[piece_id]

                # Find the max q-value out of the actions available.
                observation = th.Tensor(np.append(features, self._piece._id))
                observation = observation.reshape((-1,) + self._obs_space_shape)
                observation = th.as_tensor(observation)
                with th.no_grad():
                    q_values = (
                        agent.policy.q_net.forward(observation).detach().numpy()[0]
                    )
                max_q_values[piece_id] += np.max(q_values)

            # Find V-value for action.
            v_values[idx] = np.mean(max_q_values)

        self._anchor = deepcopy(initial_anchor)
        self._grid = deepcopy(initial_grid)
        self._piece = deepcopy(initial_piece)
        self._colour_grid = deepcopy(initial_colour_grid)

        # Choose action leading to the highest V value.
        v_value_index = np.argmax(v_values)
        action = list(available_actions.keys())[v_value_index]

        return action

    def get_col_heights(self) -> np.array:
        """Gets the column heights of the current grid.

        :return: a NumPy array containing the column heights of the grid.
        """
        col_heights = self._height - np.argmax(self._grid, axis=1)
        col_heights[col_heights == self._height] = 0
        return col_heights

    def get_feat_values(self, obs_space: str) -> np.array:
        """Gets the feature values according to the obs space.

        :param obs_space: the obs space in use.
        :return: a NumPy array containing the feature values.
        """
        return {
            # 'Dellacherie': self.get_single_score(np.zeros(6, dtype=int)),
            # 'Clipped-Offsets': self.get_column_clipped_offsets(),
            "Heights": self.get_col_heights(),
            # 'Standard': self.get_current_grid_standard(),
            "Binary": np.clip(self._grid.flatten(), 0, 1),
        }[obs_space]
