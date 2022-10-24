"""Base env."""

from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union

import gym
import numpy as np

from ._simplified_tetris_engine import SimplifiedTetrisEngine


class _SimplifiedTetrisBaseEnv(gym.Env):
    """A simplified Tetris base environment.

    This class ensures that all custom envs inherit from gym.Env and implement
    the required methods and spaces.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}
    reward_range = (0, 4)

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self._num_actions_)

    @property
    @abstractmethod
    def observation_space(self):
        raise NotImplementedError()

    @property
    def num_actions(self) -> int:
        return self._num_actions_

    @property
    def num_pieces(self) -> int:
        return self._num_pieces_

    @property
    def piece_size(self):
        """Size of the pieces in use."""
        return self._piece_size_

    @piece_size.setter
    def piece_size(self, value: int):

        if value not in [1, 2, 3, 4]:
            raise ValueError("piece_size should be either 1, 2, 3, or 4.")

        self._piece_size_ = value

    def __init__(
        self,
        *,
        grid_dims: Union[Tuple[int, int], List[int]] = (20, 10),
        piece_size: int = 4,
        seed: int = 8191,
    ) -> None:
        """Initialise the object.

        :param grid_dims: grid dimensions.
        :param piece_size: size of every piece.
        :param seed: rng seed.
        """

        if len(grid_dims) != 2:
            raise IndexError(
                "Inappropriate format provided for grid_dims. It should be a tuple/list of length 2 containing integers."
            )

        if list(grid_dims) not in [[20, 10], [10, 10], [8, 6], [7, 4]]:
            raise ValueError(
                f"Grid dimensions must be one of (20, 10), (10, 10), (8, 6), or (7, 4)."
            )

        self._height_, self._width_ = grid_dims
        self.piece_size = piece_size

        self._num_actions_, self._num_pieces_ = {
            1: (grid_dims[1], 1),
            2: (2 * grid_dims[1] - 1, 1),
            3: (4 * grid_dims[1] - 4, 2),
            4: (4 * grid_dims[1] - 6, 7),
        }[piece_size]

        self._seed(seed)

        self._engine = SimplifiedTetrisEngine(
            grid_dims=grid_dims,
            piece_size=piece_size,
            num_pieces=self._num_pieces_,
            num_actions=self._num_actions_,
            obs_space_shape=self.observation_space.shape,
        )

    def __str__(self) -> str:
        return np.array(self._engine._grid.T, dtype=int).__str__()

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(({self._height_!r}, {self.
        _width_!r}), {self._piece_size_!r})"""

    def reset(self) -> np.ndarray:
        """Reset the env.

        :return: current obs.
        """
        self._engine._reset()
        obs = self._get_obs()
        return obs

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step the env.

        Hard drop the current piece according to the action. Terminate the
        game if the piece cannot fit into the bottom |height-piece_size| rows.
        Otherwise, select a new piece and reset the anchor.

        :param action: action to be taken.
        :return: next observation, reward, game termination indicator, and env info.
        """
        info = {}

        translation, rotation = self._engine._get_translation_rotation(action)

        self._engine._rotate_piece(rotation)
        self._engine._anchor = [translation, self._piece_size_ - 1]
        info["anchor"] = (translation, rotation)

        self._engine._hard_drop()
        self._engine._update_grid(True)

        # Terminate the game when any of the dropped piece's blocks occupies
        # any of the top |piece_size| rows, before any full rows are cleared.
        if np.any(self._engine._grid[:, : self._piece_size_]):
            info["num_rows_cleared"] = 0
            self._engine._final_scores = np.append(
                self._engine._final_scores, self._engine._score
            )
            return self._get_obs(), self._get_terminal_reward(), True, info

        reward, num_rows_cleared = self._get_reward()
        self._engine._score += num_rows_cleared

        self._engine._update_anchor()
        self._engine._get_new_piece()

        info["num_rows_cleared"] = num_rows_cleared

        return self._get_obs(), reward, False, info

    def render(self, mode: str = "human") -> np.ndarray:
        """Render the env.

        :param mode: render mode.
        :return: image pixel values.
        """
        return self._engine._render(mode)

    def close(self) -> None:
        """Close the open windows."""
        return self._engine._close()

    def _seed(self, seed: int = 8191) -> None:
        """Seed the env.

        :param seed: optional seed to seed the rng with.
        """
        self._np_random, _ = gym.utils.seeding.np_random(seed)

    def _get_reward(self) -> Tuple[float, int]:
        """Return the reward.

        :return: reward and the number of lines cleared.
        """
        return self._engine._get_reward()

    @staticmethod
    def _get_terminal_reward() -> float:
        """Return the terminal reward.

        :return: terminal reward.
        """
        return 0.0

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError()
