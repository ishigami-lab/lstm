"""
description: script containing useful functions
author: Masafumi Endo
"""

import dataclasses
import numpy as np

@dataclasses.dataclass
class Data:
    """
    structure containing data for map

    :param height: terrain height
    """
    height: np.array = None

@dataclasses.dataclass
class Param:
    """
    structure containing parameters for map

    :param n: # of grid in one axis
    :param res: grid resolution [m]
    :param re: roughness exponent for fractal surface (0 < re < 1)
    :param sigma: amplitude gain for fractal surface
    """
    n: int = None
    res: float = None
    re: float = None
    sigma: float = None