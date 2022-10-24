"""Tetris pieces."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np

PieceCoord = List[Tuple[int, int]]
Rotation = int
PieceCoords = Dict[Rotation, PieceCoord]
PieceInfo = Dict[str, Union[PieceCoords, str]]
PieceID = int
PiecesInfo = Dict[PieceID, PieceInfo]
PieceSize = int

_MONOMINOS: PiecesInfo = dict()
_MONOMINOS[0] = {"coords": {0: [(0, 0)]}, "name": "O"}

_DOMINOS: PiecesInfo = dict()
_DOMINOS[0] = {
    "coords": {
        0: [(0, 0), (0, -1)],
        90: [(0, 0), (1, 0)],
    },
    "name": "I",
}

_TROMINOS: PiecesInfo = dict()
_TROMINOS[0] = {
    "coords": {
        0: [(0, 0), (0, -1), (0, -2)],
        90: [(0, 0), (1, 0), (2, 0)],
        180: [(0, 0), (0, 1), (0, 2)],
        270: [(0, 0), (-1, 0), (-2, 0)],
    },
    "name": "I",
}
_TROMINOS[1] = {
    "coords": {
        0: [(0, 0), (1, 0), (0, -1)],
        90: [(0, 0), (0, 1), (1, 0)],
        180: [(0, 0), (-1, 0), (0, 1)],
        270: [(0, 0), (0, -1), (-1, 0)],
    },
    "name": "L",
}

_TETRIMINOS: PiecesInfo = dict()
_TETRIMINOS[0] = {
    "coords": {
        0: [(0, 0), (0, -1), (0, -2), (0, -3)],
        90: [(0, 0), (1, 0), (2, 0), (3, 0)],
        180: [(0, 0), (0, 1), (0, 2), (0, 3)],
        270: [(0, 0), (-1, 0), (-2, 0), (-3, 0)],
    },
    "name": "I",
}
_TETRIMINOS[1] = {
    "coords": {
        0: [(0, 0), (1, 0), (0, -1), (0, -2)],
        90: [(0, 0), (0, 1), (1, 0), (2, 0)],
        180: [(0, 0), (-1, 0), (0, 1), (0, 2)],
        270: [(0, 0), (0, -1), (-1, 0), (-2, 0)],
    },
    "name": "L",
}
_TETRIMINOS[2] = {
    "coords": {
        0: [(0, 0), (0, -1), (-1, 0), (-1, -1)],
        90: [(0, 0), (1, 0), (0, -1), (1, -1)],
        180: [(0, 0), (0, 1), (1, 0), (1, 1)],
        270: [(0, 0), (-1, 0), (0, 1), (-1, 1)],
    },
    "name": "O",
}
_TETRIMINOS[3] = {
    "coords": {
        0: [(0, 0), (-1, 0), (1, 0), (0, 1)],
        90: [(0, 0), (0, -1), (0, 1), (-1, 0)],
        180: [(0, 0), (1, 0), (-1, 0), (0, -1)],
        270: [(0, 0), (0, 1), (0, -1), (1, 0)],
    },
    "name": "T",
}
_TETRIMINOS[4] = {
    "coords": {
        0: [(0, 0), (-1, 0), (0, -1), (0, -2)],
        90: [(0, 0), (0, -1), (1, 0), (2, 0)],
        180: [(0, 0), (1, 0), (0, 1), (0, 2)],
        270: [(0, 0), (0, 1), (-1, 0), (-2, 0)],
    },
    "name": "J",
}
_TETRIMINOS[5] = {
    "coords": {
        0: [(0, 0), (-1, 0), (0, -1), (1, -1)],
        90: [(0, 0), (0, -1), (1, 0), (1, 1)],
        180: [(0, 0), (1, 0), (0, 1), (-1, 1)],
        270: [(0, 0), (0, 1), (-1, 0), (-1, -1)],
    },
    "name": "S",
}
_TETRIMINOS[6] = {
    "coords": {
        0: [(0, 0), (-1, -1), (0, -1), (1, 0)],
        90: [(0, 0), (1, -1), (1, 0), (0, 1)],
        180: [(0, 0), (1, 1), (0, 1), (-1, 0)],
        270: [(0, 0), (-1, 1), (-1, 0), (0, -1)],
    },
    "name": "Z",
}

_PIECES: Dict[PieceSize, PiecesInfo] = dict()
_PIECES[1] = _MONOMINOS
_PIECES[2] = _DOMINOS
_PIECES[3] = _TROMINOS
_PIECES[4] = _TETRIMINOS

MAX_MIN_SETTINGS: Dict[str, Dict[str, Any]] = {
    "max_y_coord": {"func": np.max, "index": 1},
    "min_y_coord": {"func": np.min, "index": 1},
    "max_x_coord": {"func": np.max, "index": 0},
    "min_x_coord": {"func": np.min, "index": 0},
}


def _generate_max_min(coord_string: str, piece_coords: PieceCoords) -> Dict[int, int]:
    """Returns the max or min x or y coordinates for the coordinate string and coordinates provided.

    :param coord_string: string specifying what to calculate.
    :param piece_coords: piece coordinates.
    :return: max or min x or y coordinates for the coordinate string and coordinates provided.
    """
    settings = MAX_MIN_SETTINGS[coord_string]

    func = settings["func"]
    idx = settings["index"]

    max_min_scores = {}
    for rotation, coords in piece_coords.items():
        max_min_scores[rotation] = func([coord[idx] for coord in coords])
    return max_min_scores


@dataclass
class Polymino(object):
    """Represents a Tetris piece.

    :field _size: size of piece.
    :field _id: id of piece.
    :field _rotation: rotation of piece.
    :field _all_coords: all coordinates.
    :field _coords: piece's current coordinates.
    :field _name: name of piece.
    :field _max_y_coord: maximum y-coordinate.
    :field _min_y_coord: minimum y-coordinate.
    :field _max_x_coord: maximum x-coordinate.
    :field _min_x_coord: minimum x-coordinate.
    """

    _size: int
    _id: int
    _rotation: int = 0

    _all_coords: PieceCoords = field(init=False)
    _coords: PieceCoord = field(init=False)
    _name: str = field(init=False)
    _max_y_coord: Dict[int, int] = field(init=False)
    _min_y_coord: Dict[int, int] = field(init=False)
    _max_x_coord: Dict[int, int] = field(init=False)
    _min_x_coord: Dict[int, int] = field(init=False)

    def __post_init__(self):
        self._all_coords = deepcopy(_PIECES[self._size][self._id]["coords"])
        self._coords = self._all_coords[self._rotation]
        self._name = deepcopy(_PIECES[self._size][self._id]["name"])
        self._max_y_coord = _generate_max_min("max_y_coord", self._all_coords)
        self._min_y_coord = _generate_max_min("min_y_coord", self._all_coords)
        self._max_x_coord = _generate_max_min("max_x_coord", self._all_coords)
        self._min_x_coord = _generate_max_min("min_x_coord", self._all_coords)
