"""
description: script defining fractal and crater shapes
author: Masafumi Endo
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import fftpack

from env.env import GridMap

class SurfaceModel(GridMap):

    def __init__(self, param, seed: int = 0):
        super().__init__(param, seed)
        self.crater_prop = self.param.crater_prop

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
                if self.crater_prop.con_D is not None:
                    num_crater = self.calculate_num_crater(D=self.crater_prop.con_D)
                else:
                    num_crater = self.calculate_num_crater(D=self.crater_prop.max_D)
                while i < num_crater:
                    c_xy_ = self.rng.integers(self.lower_left_x, (self.param.n - 1) * self.param.res, 2).reshape(1, 2)
                    if self.crater_prop.con_D is not None:
                        D = self.crater_prop.con_D
                    else:
                        D = self.rng.integers(self.crater_prop.min_D, self.crater_prop.max_D)
                    if i == 0:
                        self.set_crater(c_xy=c_xy_, D=D)
                        # init array for checking circle hit
                        c_arr = c_xy_
                        d_arr = np.array([D])
                        i += 1
                    else:
                        is_hit = self.check_circle_hit(c_arr, d_arr, c_xy_, D)
                        if not is_hit:
                            self.set_crater(c_xy=c_xy_, D=D)
                            c_arr = np.append(c_arr, c_xy_, axis=0)
                            d_arr = np.append(d_arr, D)
                            i += 1
        if self.param.is_fractal:
            self.set_fractal_surf()
        self.data.height = self.set_offset(self.data.height)

    def check_circle_hit(self, c_arr: np.ndarray, d_arr: np.ndarray, c_t: np.ndarray, d_t: np.ndarray):
        """
        check_circle_hit: check whether given craters are overlapped or not
        
        :param c_arr: center positions of generated craters so far
        :param d_arr: diameter information of generated creaters so far
        :param c_t: center position of newly generated crater
        :param d_t: diameter information of newly generated crater
        """
        for c, d, in zip(c_arr, d_arr):
            dist_c = np.sqrt((c[0] - c_t[0, 0])**2 + (c[1] - c_t[0, 1])**2)
            sum_d = (d + d_t)
            if dist_c < sum_d:
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
        :param D: diameter
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
        :param D: diameter
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
        :param D: diameter
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
        :param D: diameter
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

    def calculate_num_crater(self, D: float, a: float = 1.54, b: float = - 2):
        """
        calculate_num_crater: calculate number of craters based on analytical model

        :param D: diameter
        :param a: coefficient of the function
        :param b: coefficient of the function
        """
        density = a * D**b
        area_t = (self.param.n * self.param.res)**2
        area_c = (D / 2)**2 * np.pi
        num_crater = int((area_t) * density / area_c)
        return num_crater