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
class CraterProp:
    """
    structure containing crater property

    :param distribution: crater distribution ("random" and "single")
    :param geometry: geometry type from "normal", "mound", "flat", and "concentric"
    :param num_crater: number of crater
    :param min_range: minimum crater range
    :param max_range: maximum crater range
    """
    distribution: str = None
    geometry: str = None
    num_crater: int = None
    min_D: float = None
    max_D: float = None

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
    is_fractal: bool = None
    is_crater: bool = None
    crater_prop: any = None