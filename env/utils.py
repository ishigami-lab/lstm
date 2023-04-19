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
    :param min_D: minimum crater range
    :param max_D: maximum crater range
    :param con_D: constant crater range
    """
    distribution: str = None
    geometry: str = None
    min_D: float = None
    max_D: float = None
    con_D: float = None

@dataclasses.dataclass
class Param:
    """
    structure containing parameters for map

    :param n: # of grid in one axis
    :param res: grid resolution [m]
    :param re: roughness exponent for fractal surface (0 < re < 1)
    :param sigma: amplitude gain for fractal surface
    :param is_fractal: choose to apply fractal surface
    :param is_crater: choose to apply crater shape
    :param crater_prop: crater property denoted as "CraterProp" data structure
    """
    n: int = None
    res: float = None
    re: float = None
    sigma: float = None
    is_fractal: bool = None
    is_crater: bool = None
    crater_prop: any = None