from enum import Enum
from typing import Tuple

from matplotlib import colors


def _get_bgr_code(colour: str) -> Tuple[float, float, float]:
    """Return the inverted RGB code corresponding to the colour provided.

    :param colour: colour.
    :return: inverted RGB code.
    """
    return tuple([255 * code for code in colors.to_rgb(colour)][::-1])


class Colours(Enum):
    """Enumerate inverted RGB code."""

    WHITE = _get_bgr_code("white")
    BLACK = _get_bgr_code("black")
    CYAN = _get_bgr_code("cyan")
    ORANGE = _get_bgr_code("orange")
    YELLOW = _get_bgr_code("yellow")
    PURPLE = _get_bgr_code("purple")
    BLUE = _get_bgr_code("blue")
    GREEN = _get_bgr_code("green")
    RED = _get_bgr_code("red")
