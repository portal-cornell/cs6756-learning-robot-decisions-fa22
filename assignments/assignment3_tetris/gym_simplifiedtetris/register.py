"""Env registration."""

import itertools
from typing import List

from gym.envs.registration import register as register_env_in_gym

registered_envs: List[str] = []


def register_env(incomplete_id: str, entry_point: str) -> None:
    """Registers the custom environments in Gym.

    :param incomplete_id: part of the official environment ID.
    :param entry_point: Python entry point of the environment.
    """

    if not incomplete_id.startswith("simplifiedtetris-"):
        raise ValueError('Env ID should start with "simplifiedtetris-".')

    if not entry_point.startswith("gym_simplifiedtetris.envs:SimplifiedTetris"):
        raise ValueError(
            'Entry point should start with "gym_simplifiedtetris envs:SimplifiedTetris".'
        )

    if not entry_point.endswith("Env"):
        raise ValueError('Entry point should end with "Env".')

    grid_dims: List[List[int]] = [[20, 10], [10, 10], [8, 6], [7, 4]]
    piece_sizes: List[int] = [4, 3, 2, 1]

    all_combinations: List[tuple[List[int], int]] = list(
        itertools.product(*[grid_dims, piece_sizes])
    )

    for (height, width), piece_size in all_combinations:
        env_id = incomplete_id + f"-{height}x{width}-{piece_size}-v0"

        if env_id in registered_envs:
            raise ValueError(f"Already registered env id: {env_id}")

        register_env_in_gym(
            id=env_id,
            entry_point=entry_point,
            nondeterministic=True,
            kwargs={
                "grid_dims": (height, width),
                "piece_size": piece_size,
            },
        )
        registered_envs.append(env_id)
