from typing import Tuple

import numpy as np
from astropy.modeling.functional_models import AiryDisk2D, Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma

from .spin_average import spinavej


# Functions
def get_canvas_radius_nm(extend_sides_to_diagonal: bool = False) -> float:
    """ Returns the radius of the canvas, in nanometres. """
    return _canvas_inner_radius_nm if not extend_sides_to_diagonal else _canvas_outer_radius_nm


def get_canvas_dimensions_px(radius_nm: float, pixels_per_nm: float) -> Tuple[int, int]:
    """ Returns the radius and side length of the canvas respectively, in pixels. """
    radius_px = np.int(radius_nm / pixels_per_nm)
    side_length_px = radius_px * 2 - 1
    return radius_px, side_length_px


def radial_profile(data: np.ndarray, fftshift: bool = False) -> np.ndarray:
    """ Calculates the radial profile of a 2D array. """
    if fftshift:
        data = np.fft.fftshift(data)

    return spinavej(data)


def generate_gaussian(*, amplitude: float, fwhm: float,
                      canvas_radius: float, pixels_per_nm: float) -> np.ndarray:
    """ Generates a 2D gaussian pattern. """

    stddev = fwhm / pixels_per_nm * gaussian_fwhm_to_sigma
    model = Gaussian2D(amplitude=amplitude, x_stddev=stddev, y_stddev=stddev)

    x, y = _canvas_meshgrid(canvas_radius, pixels_per_nm)
    result = model(x, y)

    return result


def generate_doughnut(*, periodicity: float,
                      canvas_radius: float, pixels_per_nm: float) -> np.ndarray:
    """ Generates a 2D doughnut pattern. """

    def Doughnut1D(radius: float) -> np.ndarray:
        return np.where(
            radius < periodicity / (2 * pixels_per_nm),
            0.5 - 0.5 * np.cos(2 * np.pi * radius * pixels_per_nm / periodicity),
            1
        )

    return Doughnut1D(_radial_to_2d(canvas_radius, pixels_per_nm))


def generate_airy(*, amplitude: float, fwhm: float,
                  canvas_radius: float, pixels_per_nm: float) -> np.ndarray:
    """ Generates a 2D airy pattern. """

    radius = fwhm * 1.22 / pixels_per_nm
    model = AiryDisk2D(amplitude=amplitude, radius=radius)

    x, y = _canvas_meshgrid(canvas_radius, pixels_per_nm)
    result = model(x, y)

    return result


def generate_digital_pinhole(*, fwhm: float,
                             canvas_radius: float, pixels_per_nm: float) -> np.ndarray:
    """ Generates a 2D digital pinhole pattern. """

    g_base = generate_gaussian(amplitude=1, fwhm=fwhm,
                               canvas_radius=canvas_radius, pixels_per_nm=pixels_per_nm)
    const_base = np.ones_like(g_base)

    b0_vec = g_base.reshape(g_base.size)
    b1_vec = const_base.reshape(const_base.size)

    b_mat = np.array([b0_vec, b1_vec])

    b_inv = np.linalg.pinv(b_mat)

    return b_inv[:, 0].reshape(g_base.shape)


def generate_physical_pinhole(*, radius: float,
                              canvas_radius: float, pixels_per_nm: float) -> np.ndarray:
    """ Generates a 2D physical pinhole pattern. """

    def model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x ** 2 + y ** 2 < (radius / pixels_per_nm) ** 2).astype(float)

    x, y = _canvas_meshgrid(canvas_radius, pixels_per_nm)
    result = model(x, y)

    return result


# Internal functions
def _canvas_meshgrid(radius_nm: float, pixels_per_nm: float) -> np.ndarray:
    radius_px, _ = get_canvas_dimensions_px(radius_nm, pixels_per_nm)
    side = np.linspace(-radius_px + 1, radius_px - 1, radius_px * 2 - 1)
    return np.meshgrid(side, side)


def _radial_to_2d(radius_nm: float, pixels_per_nm: float) -> np.ndarray:
    x, y = _canvas_meshgrid(radius_nm, pixels_per_nm)
    return np.sqrt(x ** 2 + y ** 2)


# Constants
_canvas_inner_radius_nm = 2000.0
_canvas_outer_radius_nm = _canvas_inner_radius_nm * np.sqrt(2)
