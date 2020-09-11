"""
@original_author: andreas.boden
@adapted_by: stafak
"""

from typing import Tuple

import numpy as np

import frcpredict.model as mdl
from .numba_compat import jit, prange


@jit(nopython=True, parallel=True)
def make_random_positions_structure(area_side_um: float,
                                    f_per_um2: float = 1000) -> Tuple[np.ndarray, np.ndarray]:
    N = np.int(area_side_um ** 2 * f_per_um2)

    f_array_x = np.zeros(N)
    f_array_y = np.zeros(N)

    for i in prange(N):
        f_array_x[i] = np.random.uniform(0, area_side_um)
        f_array_y[i] = np.random.uniform(0, area_side_um)

    return f_array_x, f_array_y


@jit(nopython=True)
def make_pool_pairs_structure(area_side_um: float, d: float = 0.1, num_pairs: int = 100,
                              f_per_pool: int = 10,
                              poisson_labelling: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    arr_len = np.int(area_side_um * num_pairs * f_per_pool * 2 * 2)
    f_array_x = np.zeros(arr_len)
    f_array_y = np.zeros(arr_len)

    num_elements = 0
    for _ in range(num_pairs):
        random_angle = np.random.uniform(0, 2 * np.pi)
        random_point_x = np.random.uniform(0, area_side_um)
        random_point_y = np.random.uniform(0, area_side_um)

        f1_x = random_point_x + (d / 2) * np.cos(random_angle)
        f1_y = random_point_y + (d / 2) * np.sin(random_angle)

        f2_x = random_point_x - (d / 2) * np.cos(random_angle)
        f2_y = random_point_y - (d / 2) * np.sin(random_angle)

        if (0 < f1_x < area_side_um and 0 < f1_y < area_side_um and
                0 < f2_x < area_side_um and 0 < f2_y < area_side_um):
            if poisson_labelling:
                p1 = np.random.poisson(f_per_pool)
                p2 = np.random.poisson(f_per_pool)
            else:
                p1 = p2 = f_per_pool

            f_array_x[num_elements:num_elements + p1] = f1_x
            f_array_y[num_elements:num_elements + p1] = f1_y
            num_elements += p1

            f_array_x[num_elements:num_elements + p2] = f2_x
            f_array_y[num_elements:num_elements + p2] = f2_y
            num_elements += p2

    return f_array_x[:num_elements], f_array_y[:num_elements]


@jit(nopython=True)
def make_lines_structure(area_side_um: float, num_lines: int = 100,
                         f_per_um: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    arr_len = np.int(area_side_um * num_lines * f_per_um * 2)
    f_array_x = np.zeros(arr_len)
    f_array_y = np.zeros(arr_len)

    line_length = 2 * np.sqrt(2) * area_side_um
    N = np.int(line_length * f_per_um)

    num_elements = 0
    for _ in range(num_lines):
        random_angle = np.random.uniform(0, 2 * np.pi)
        random_point_x = np.random.uniform(0, area_side_um)
        random_point_y = np.random.uniform(0, area_side_um)

        for _ in range(N):
            random_distance = np.random.uniform(-np.sqrt(2) * area_side_um,
                                                np.sqrt(2) * area_side_um)

            f_pos_x = random_point_x + random_distance * np.cos(random_angle)
            f_pos_y = random_point_y + random_distance * np.sin(random_angle)

            if 0 < f_pos_x < area_side_um and 0 < f_pos_y < area_side_um:
                f_array_x[num_elements] = f_pos_x
                f_array_y[num_elements] = f_pos_y
                num_elements += 1

    return f_array_x[:num_elements], f_array_y[:num_elements]


@jit(nopython=True)
def positions2im(area_side_um: float, px_size_um: float,
                 f_array_x: np.ndarray, f_array_y: np.ndarray) -> np.ndarray:
    tot_fluorophores = len(f_array_x)

    im_side_px = np.int(np.round(area_side_um / px_size_um))

    im = np.zeros((im_side_px, im_side_px))

    for i in range(tot_fluorophores):
        x_px = np.int(f_array_x[i] // px_size_um)
        y_px = np.int(f_array_y[i] // px_size_um)
        try:
            im[y_px, x_px] += 1
        except:
            pass

    return im
