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
        self.crater_prop = self.param.crater_prop
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
        :param i_tf: index of terrain features
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
        :param i_tf: index of terrain features
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

    def set_terrain_env(self):
        """
        set_terrain_env: set planetary terrain environment based on fractal method w/ crater

        """
        if self.param.is_crater:
            if self.crater_prop.distribution == "single":
                c_xy_ = np.array([int(self.c_x), int(self.c_y)]).reshape(1, 2)
                D = self.crater_prop.min_D
                self.set_crater(c_xy=c_xy_, D=D)
            elif self.crater_prop.distribution == "random":
                i = 0
                while i < self.crater_prop.num_crater:
                    c_xy_ = self.rng.integers(self.lower_left_x, (self.param.n - 1) * self.param.res, 2).reshape(1, 2)
                    D = self.rng.integers(self.crater_prop.min_D, self.crater_prop.max_D)
                    if i == 0:
                        self.set_crater(c_xy=c_xy_, angles=self.rng.integers(0, 20), D=D)
                        # init array for checking circle hit
                        c_arr = c_xy_
                        r_arr = np.array([D])
                        i += 1
                    else:
                        is_hit = self.check_circle_hit(c_arr, r_arr, c_xy_, D)
                        if not is_hit:
                            self.set_crater(c_xy=c_xy_, angles=self.rng.integers(0, 20), D=D)
                            c_arr = np.append(c_arr, c_xy_, axis=0)
                            r_arr = np.append(r_arr, D)
                            i += 1
        if self.param.is_fractal:
            self.set_fractal_surf()
        self.data.height = self.set_offset(self.data.height)

    def check_circle_hit(self, c_arr: np.ndarray, r_arr: np.ndarray, c_t: np.ndarray, r_t: np.ndarray):
        """
        check_circle_hit: check whether given craters are overlapped or not

        """
        for c, r, in zip(c_arr, r_arr):
            dist_c = np.sqrt((c[0] - c_t[0, 0])**2 + (c[1] - c_t[0, 1])**2)
            sum_r = r + r_t
            if dist_c < sum_r:
                return True
        return False
        
    def set_fractal_surf(self):
        """
        set_fractal_surf: set fractal surface into data structure

        """
        z = self.generate_fractal_surf()
        # set offset
        z = self.set_offset(np.ravel(z))
        self.data.height += z

    def set_crater(self, c_xy: np.ndarray, D: float):
        """
        set_crater: set arbitrary crater generated with given parameters into 2D map environment

        :param c_xy: center of crater position in x- and y-axis [m]
        :param D: crater inner-rim range
        """
        z = self.generate_crater(D)
        x_c_id, y_c_id = self.get_xy_id_from_xy_pos(c_xy[0, 0], c_xy[0, 1])
        x_len = int(z.shape[0] / 2)
        y_len = int(z.shape[1] / 2)
        for y_id_ in range(y_c_id - y_len, y_c_id + y_len):
            for x_id_ in range(x_c_id - x_len, x_c_id + x_len):
                if self.lower_left_x <= x_id_ < self.param.n and self.lower_left_y <= y_id_ < self.param.n:
                    self.set_value_from_xy_id(x_id_, y_id_, z[x_id_ - (x_c_id - x_len), y_id_ - (y_c_id - y_len)])

    def set_offset(self, z: np.ndarray):
        """
        set_offset: adjust z-axis value starting from zero

        :param z: z-axis information (typically for distance information, such as terrain height)
        """
        z_ = z - min(z)
        z = (z_ + abs(z_)) / 2
        z_ = z - max(z)
        z_ = (z_ - abs(z_)) / 2
        z = z_ - min(z_)
        return z
            
    def generate_fractal_surf(self):
        """
        generate_fractal_surf: generate random height information based on fractional Brownian motion (fBm).

        """
        z = np.zeros((self.param.n, self.param.n), complex)
        for y_id_ in range(int(self.param.n / 2) + 1):
            for x_id_ in range(int(self.param.n / 2) + 1):
                phase = 2 * np.pi * self.rng.random()
                if x_id_ != 0 or y_id_ != 0:
                    rad = 1 / (x_id_**2 + y_id_**2)**((self.param.re + 1) / 2)
                else:
                    rad = 0.0
                z[y_id_, x_id_] = rad * np.exp(1j * phase)
                if x_id_ == 0:
                    x_id_0 = 0
                else:
                    x_id_0 = self.param.n - x_id_
                if y_id_ == 0:
                    y_id_0 = 0
                else:
                    y_id_0 = self.param.n - y_id_
                z[y_id_0, x_id_0] = np.conj(z[y_id_, x_id_])

        z[int(self.param.n / 2), 0] = np.real(z[int(self.param.n / 2), 0])
        z[0, int(self.param.n / 2)] = np.real(z[0, int(self.param.n / 2)])
        z[int(self.param.n / 2), int(self.param.n / 2)] = np.real(z[int(self.param.n / 2), int(self.param.n / 2)])

        for y_id_ in range(1, int(self.param.n / 2)):
            for x_id_ in range(1, int(self.param.n / 2)):
                phase = 2 * np.pi * self.rng.random()
                rad = 1 / (x_id_ ** 2 + y_id_ ** 2) ** ((self.param.re + 1) / 2)
                z[y_id_, self.param.n - x_id_] = rad * np.exp(1j * phase)
                z[self.param.n - y_id_, x_id_] = np.conj(z[y_id_, self.param.n - x_id_])

        z = z * abs(self.param.sigma) * (self.param.n * self.param.res * 1e+3)**(self.param.re + 1 + .5)
        z = np.real(fftpack.ifft2(z)) / (self.param.res * 1e+3)**2
        z = z * 1e-3
        return z

    def generate_crater(self, D: np.ndarray):
        """
        generate_crater: generate crater height information

        :param D: crater inner-rim range
        """
        xx, yy = np.meshgrid(np.arange(0.0, int(2.0 * D * 2.5), self.param.res),
                             np.arange(0.0, int(2.0 * D * 2.5), self.param.res))
        xx, yy = xx / D, yy / D
        c_x = c_y = 2.5
        rr = np.sqrt((xx - c_x)**2 + (yy - c_y)**2)
        zz = np.full((rr.shape[0], rr.shape[1]), np.nan)
        rmax = - np.inf
        hout = np.nan
        for y_id_, x_id_ in np.ndindex(rr.shape):
            r = rr[y_id_, x_id_]
            if self.crater_prop.geometry == "normal":
                h = self.normal_model(r, D)
            elif self.crater_prop.geometry == "mound":
                h = self.central_mound_crater_model(r, D)
            elif self.crater_prop.geometry == "flat":
                h = self.flat_bottomed_crater_model(r, D)
            elif self.crater_prop.geometry == "concentric":
                h = self.concentric_crater_model(r, D)
            zz[y_id_, x_id_] = h
            if rmax < r:
                rmax = r
                hout = h
        zz[zz == np.nan] = hout
        zz -= hout
        zz *= D
        return zz

    def normal_model(self, r, D):
        """
        normal_model: normal crater model

        :param r: input variable
        :param D: radius
        """
        d0 = 0.114 * D**(-0.002)
        hr = 0.02513 * D**(-0.0757)
        alpha = -3.1906
        
        a = -2.8567
        b = 5.8270
        C = d0 * ((np.exp(a) + 1) / (np.exp(b) - 1))
        if 0 <= r <= 1.0:
            h = C * (np.exp(b * r) - np.exp(b)) / (1 + np.exp(a + b * r))
        else:
            h = hr * (r**alpha - 1)
        return h

    def central_mound_crater_model(self, r, D):
        """
        central_mound_crater_model: central mound crater model

        :param r: input variable
        :param D: radius
        """
        d0 = 0.114 * D**(-0.002)
        hr = 0.02513 * D**(-0.0757)
        alpha = -3.1906
        
        rm = 0.293 * D**(-0.086)
        rb = 0.793 * D**(-0.242)
        hm = 0.23 * 10**(-3) * D**(0.64)
        a = -2.6921
        b = 6.1678
        C = d0 * ((np.exp(a) + 1) / (np.exp(b) - 1))
        if 0 <= r <= rm:
            h = (1 - (r / rm)) * hm - d0
        elif rm < r <= rb:
            h = -d0
        elif rb < r <= 1:
            r0 = (r - rb) / (1 - rb)
            h = C * ((np.exp(b * r0) - np.exp(b)) / (1 + np.exp(a + b * r0)))
        else:
            h = hr * (r**alpha - 1)
        return h

    def flat_bottomed_crater_model(self, r, D):
        """
        flat_bottomed_crater_model: flat-bottomed crater model

        :param r: input variable
        :param D: radius
        """
        d0 = 0.114 * D**(-0.002)
        hr = 0.02513 * D**(-0.0757)
        alpha = -3.1906
        
        rb = 0.091 * D**(0.208)
        a = -2.6003
        b = 5.8783
        C = d0 * ((np.exp(a) + 1) / (np.exp(b) - 1))
        if 0 <= r <= rb:
            h = -d0
        elif rb < r <= 1:
            r0 = (r - rb) / (1 - rb)
            h = C * ((np.exp(b * r0) - np.exp(b)) / (1 + np.exp(a + b * r0)))
        else:
            h = hr * (r**alpha - 1)
        return h

    def concentric_crater_model(self, r, D):
        """
        concentric_crater_model: concentric crater model

        :param r: input variable
        :param D: radius
        """
        d0 = 0.114 * D**(-0.002)
        hr = 0.02513 * D**(-0.0757)
        alpha = -3.1906
        
        C1 = 0.1920
        C2 = 0.0100
        C3 = 0.0155 * D**(0.343)
        ri = 0.383 * D**(0.053)
        ro = 0.421 * D**(0.102)
        a = -1.6536
        b = 4.7626
        h1 = np.nan
        h2 = np.nan
        if 0 <= r <= ri:
            h = C1 * r**2 + C2 * r - d0
        elif ri < r <= ro:
            h = C3 * (r - ri) + h1
        elif ro < r <= 1:
            C = - h2 * ((np.exp(a) + 1) / (np.exp(b) - 1))
            r0 = (r - ro) / (1 - ro)
            h = C * ((np.exp(b * r0) - np.exp(b)) / (1 + np.exp(a + b * r0)))
        else:
            h = hr * (r**alpha - 1)
        return h

    def extend_data(self, data: np.array, field_name):
        """
        extend_data: extend self.data in case additional terrain features are necessary

        :param data: appended data array
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

    def plot_maps(self, figsize: tuple = (10, 4)):
        """
        plot_maps: plot 2D and 2.5D figures with given size

        :param figsize: size of figure
        :param is_tf: existence of terrain features
        """
        sns.set()
        sns.set_style('whitegrid')
        fig = plt.figure(figsize=figsize)
        # _, ax_3d = self.plot_3d_map(fig=fig, rc=111)
        _, ax_2d = self.plot_2d_map(fig=fig, rc=111)
        plt.tight_layout()

    def plot_2d_map(self, grid_data: np.ndarray = None, fig: plt.figure = None, rc: int = 111, field_name: str = "height", 
                    title: str = "2D celestial terrain", cmap: str = "jet", label: str = "height m"):
        """
        plot_2d_map: plot 2D grid map

        :param grid_data: data to visualize
        :param fig: figure
        :param rc: position specification as rows and columns
        :param i_tf: index of terrain features
        :param title: title of shown figure
        """
        xx, yy = np.meshgrid(np.arange(0.0, self.param.n * self.param.res, self.param.res),
                             np.arange(0.0, self.param.n * self.param.res, self.param.res))

        if grid_data is None:
            grid_data = np.reshape(getattr(self.data, field_name), (self.param.n, self.param.n))
            data = getattr(self.data, field_name)
        else:
            data = np.reshape(grid_data, -1)
        if not fig:
            fig = plt.figure()

        ax = fig.add_subplot(rc)
        hmap = ax.pcolormesh(xx + self.param.res / 2.0, yy + self.param.res / 2.0, grid_data,
                             cmap=cmap, vmin=min(data), vmax=max(data))
        ax.set_xlabel("x-axis m")
        ax.set_ylabel("y-axis m")
        ax.set_aspect("equal")
        ax.set_xlim(xx.min(), xx.max() + self.param.res)
        ax.set_ylim(yy.min(), yy.max() + self.param.res)

        plt.colorbar(hmap, ax=ax, label=label, orientation='vertical')

        return hmap, ax

    def plot_3d_map(self, grid_data: np.ndarray = None, fig: plt.figure = None, rc: int = 111):
        """
        plot_3d_map: plot 2.5D grid map

        :param grid_data: data to visualize
        :param fig: figure
        :param rc: position specification as rows and columns
        :param is_tf: existence of terrain features
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
        hmap = ax.plot_surface(xx + self.param.res / 2.0, yy + self.param.res / 2.0, grid_data,
                            cmap="jet", vmin=min(data), vmax=max(data), linewidth=0, antialiased=False)
        ax.set_xlabel("x-axis m")
        ax.set_ylabel("y-axis m")
        ax.set_zticks(np.arange(xx.min(), xx.max(), 10))
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect((1, 1, 0.35))
        ax.set_xlim(xx.min(), xx.max() + self.param.res)
        ax.set_ylim(yy.min(), yy.max() + self.param.res)
        ax.set_zlim(min(data), xx.max() / 10)

        return hmap, ax