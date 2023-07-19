"""
description: a class to generate lunar surface topographic map. 
author: Masafumi Endo
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import fftpack

from env.utils import Data

class GridMap:

    def __init__(self, param, seed: int = 0):
        """
        __init__:

        :param param: structure containing map parameters
        :param seed: random seed 
        """
        # set given parameters
        self.param = param
        # identify center and lower left positions
        self.c_x = self.param.n * self.param.res / 2.0
        self.c_y = self.param.n * self.param.res / 2.0
        self.lower_left_x = self.c_x - self.param.n / 2.0 * self.param.res
        self.lower_left_y = self.c_y - self.param.n / 2.0 * self.param.res

        # generate data array
        self.param.num_grid = self.param.n**2
        self.data = Data(height=np.zeros(self.param.num_grid))

        # set randomness
        self.seed = seed
        self.set_randomness()

    def set_randomness(self):
        """
        set_randomness: set randomness for reproductivity

        """
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()

    # following functions are used for basic operations
    def get_value_from_xy_id(self, x_id: int, y_id: int, field_name: str = "height"):
        """
        get_value_from_xy_id: get values at specified location described as x- and y-axis indices from data structure

        :param x_id: x index
        :param y_id: y index
        :param field_name: structure's name
        """
        grid_id = self.calc_grid_id_from_xy_id(x_id, y_id)

        if 0 <= grid_id < self.param.num_grid:
            data = getattr(self.data, field_name)
            return data[grid_id]
        else:
            return None

    def get_xy_id_from_xy_pos(self, x_pos: float, y_pos: float):
        """
        get_xy_id_from_xy_pos: get x- and y-axis indices for given positional information

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """
        x_id = self.calc_xy_id_from_pos(x_pos, self.lower_left_x, self.param.n)
        y_id = self.calc_xy_id_from_pos(y_pos, self.lower_left_y, self.param.n)

        return x_id, y_id

    def calc_grid_id_from_xy_id(self, x_id: int, y_id: int):
        """
        calc_grid_id_from_xy_id: calculate one-dimensional grid index from x- and y-axis indices (2D -> 1D transformation)

        :param x_id: x index
        :param y_id: y index
        """
        grid_id = int(y_id * self.param.n + x_id)
        return grid_id

    def calc_xy_id_from_pos(self, pos: float, lower_left: float, max_id: int):
        """
        calc_xy_id_from_pos: calculate x- or y-axis indices for given positional information

        :param pos: x- or y-axis position
        :param lower_left: lower left information
        :param max_id: max length (width or height)
        """
        id = int(np.floor((pos - lower_left) / self.param.res))
        assert 0 <= id <= max_id, 'given position is out of the map!'
        return id

    def set_value_from_xy_pos(self, x_pos: float, y_pos: float, val: float):
        """
        set_value_from_xy_pos: substitute given arbitrary values into data structure at specified x- and y-axis position

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param val: arbitrary spatial information
        """
        x_id, y_id = self.get_xy_id_from_xy_pos(x_pos, y_pos)

        if (not x_id) or (not y_id):
            return False
        flag = self.set_value_from_xy_id(x_id, y_id, val)
        return flag

    def set_value_from_xy_id(self, x_id: int, y_id: int, val: float, field_name: str = "height", is_increment: bool = True):
        """
        set_value_from_xy_id: substitute given arbitrary values into data structure at specified x- and y-axis indices

        :param x_id: x index
        :param y_id: y index
        :param val: arbitrary spatial information
        :param field_name: structure's name
        :param is_increment: increment data if True. Otherwise, simply update value information.
        """
        if (x_id is None) or (y_id is None):
            return False, False
        grid_id = int(y_id * self.param.n + x_id)

        if 0 <= grid_id < self.param.num_grid:
            data = getattr(self.data, field_name)
            if is_increment:
                data[grid_id] += val
            else:
                data[grid_id] = val
                setattr(self.data, field_name, data)
            return True
        else:
            return False

    def extend_data(self, data: np.array, field_name: str):
        """
        extend_data: extend self.data in case additional terrain features are necessary

        :param data: appended data array
        :param field_name: structure's name
        """
        setattr(self.data, field_name, data)

    # following functions are used for visualization objective
    def print_grid_map_info(self):
        """
        print_grid_map_info: show grid map information

        """
        print("range: ", self.param.n * self.param.res, " [m]")
        print("resolution: ", self.param.res, " [m]")
        print("# of data: ", self.param.num_grid)

    def plot_maps(self, figsize: tuple = (10, 4), field_name: str = "height"):
        """
        plot_maps: plot 2D and 2.5D figures with given size

        :param figsize: size of figure
        """
        sns.set()
        sns.set_style('whitegrid')
        fig = plt.figure(figsize=figsize)
        _, ax_3d = self.plot_3d_map(fig=fig, rc=121, field_name=field_name)
        _, ax_2d = self.plot_2d_map(fig=fig, rc=122, field_name=field_name)
        plt.tight_layout()

    def plot_2d_map(self, grid_data: np.ndarray = None, fig: plt.figure = None, rc: int = 111, field_name: str = "height", 
                    cmap: str = "jet", label: str = "height m"):
        """
        plot_2d_map: plot 2D grid map

        :param grid_data: data to visualize
        :param fig: figure
        :param rc: position specification as rows and columns
        :param field_name: name of fields
        :param cmap: color map spec
        :param label: label of color map
        """
        xx, yy = np.meshgrid(np.arange(0.0, self.param.n * self.param.res, self.param.res),
                             np.arange(0.0, self.param.n * self.param.res, self.param.res))

        if grid_data is None:
            data = getattr(self.data, field_name)
            if field_name == "height":
                grid_data = np.reshape(data, (self.param.n, self.param.n))
            elif field_name == "color":
                grid_data = data
            data = getattr(self.data, field_name)
        else:
            data = np.reshape(grid_data, -1)
        if not fig:
            fig = plt.figure()

        ax = fig.add_subplot(rc)
        if field_name == "height":
            hmap = ax.pcolormesh(xx + self.param.res / 2.0, yy + self.param.res / 2.0, grid_data,
                                cmap=cmap, vmin=min(data), vmax=max(data))
            ax.set_xlim(xx.min(), xx.max() + self.param.res)
            ax.set_ylim(yy.min(), yy.max() + self.param.res)
            plt.colorbar(hmap, ax=ax, label=label, orientation='vertical')
        elif field_name == "color":
            hmap = ax.imshow(grid_data, origin='lower')
            ax.grid(False)
        ax.set_xlabel("x-axis m")
        ax.set_ylabel("y-axis m")
        ax.set_aspect("equal")

        return hmap, ax

    def plot_3d_map(self, grid_data: np.ndarray = None, fig: plt.figure = None, rc: int = 111, field_name: str = "height"):
        """
        plot_3d_map: plot 2.5D grid map

        :param grid_data: data to visualize
        :param fig: figure
        :param rc: position specification as rows and columns
        """
        xx, yy = np.meshgrid(np.arange(0.0, self.param.n * self.param.res, self.param.res),
                             np.arange(0.0, self.param.n * self.param.res, self.param.res))

        if grid_data is None:
            grid_data = np.reshape(self.data.height, (self.param.n, self.param.n))
            data = self.data.height
        else:
            data = np.reshape(grid_data, -1)
        if not fig:
            fig = plt.figure()

        ax = fig.add_subplot(rc, projection="3d")
        if field_name == "height":
            hmap = ax.plot_surface(xx + self.param.res / 2.0, yy + self.param.res / 2.0, grid_data, rstride=1, cstride=1,
                                cmap="jet", vmin=min(data), vmax=max(data), linewidth=0, antialiased=False)
        elif field_name == "color":
            hmap = ax.plot_surface(xx + self.param.res / 2.0, yy + self.param.res / 2.0, grid_data, rstride=1, cstride=1,
                                facecolors=self.data.color, linewidth=0, antialiased=False)
        ax.set_xlabel("x-axis m")
        ax.set_ylabel("y-axis m")
        ax.set_zticks(np.arange(xx.min(), xx.max(), 10))
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect((1, 1, 0.35))
        ax.set_xlim(xx.min(), xx.max() + self.param.res)
        ax.set_ylim(yy.min(), yy.max() + self.param.res)
        ax.set_zlim(min(data), xx.max() / 25)

        return hmap, ax