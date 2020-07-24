from typing import Tuple

import numpy as np
from astropy.modeling.functional_models import AiryDisk2D, Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma


# Functions
def get_canvas_params(pixels_per_nm: float) -> Tuple[int, int]:
    """ Returns the inner radius and side length of the canvas respectively, in pixels. """

    canvas_inner_radius_px = np.int(_canvas_inner_radius_nm / pixels_per_nm)
    canvas_side_length_px = canvas_inner_radius_px * 2 - 1

    return canvas_inner_radius_px, canvas_side_length_px


def generate_gaussian(amplitude: float, fwhm: float, pixels_per_nm: float) -> np.ndarray:
    """ Generates a 2D gaussian pattern. """

    stddev = fwhm / pixels_per_nm * gaussian_fwhm_to_sigma
    model = Gaussian2D(amplitude=amplitude, x_stddev=stddev, y_stddev=stddev)

    x, y = _canvas_meshgrid(pixels_per_nm)
    result = model(x, y)

    return result


def generate_doughnut(periodicity: float, pixels_per_nm: float) -> np.ndarray:
    """ Generates a 2D doughnut pattern. """

    def Doughnut1D(radius: float) -> np.ndarray:
        return np.where(
            radius < periodicity / (2 * pixels_per_nm),
            0.5 - 0.5 * np.cos(2 * np.pi * radius * pixels_per_nm / periodicity),
            1
        )

    return Doughnut1D(_radial_to_2d(pixels_per_nm))


def generate_airy(amplitude: float, fwhm: float, pixels_per_nm: float) -> np.ndarray:
    """ Generates a 2D airy pattern. """

    radius = fwhm * 1.22 / pixels_per_nm
    model = AiryDisk2D(amplitude=amplitude, radius=radius)

    x, y = _canvas_meshgrid(pixels_per_nm)
    result = model(x, y)

    return result


def generate_digital_pinhole(fwhm: float, pixels_per_nm: float) -> np.ndarray:
    """ Generates a 2D digital pinhole pattern. """

    g_base = generate_gaussian(amplitude=1, fwhm=fwhm, pixels_per_nm=pixels_per_nm)
    const_base = np.ones_like(g_base)

    b0_vec = g_base.reshape(g_base.size)
    b1_vec = const_base.reshape(const_base.size)

    b_mat = np.array([b0_vec, b1_vec])

    b_inv = np.linalg.pinv(b_mat)

    return b_inv[:, 0].reshape(g_base.shape)


def generate_physical_pinhole(radius: float, pixels_per_nm: float) -> np.ndarray:
    """ Generates a 2D physical pinhole pattern. """

    def model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x ** 2 + y ** 2 < (radius / pixels_per_nm) ** 2).astype(float)

    x, y = _canvas_meshgrid(pixels_per_nm)
    result = model(x, y)

    return result


# Internal functions
def _canvas_meshgrid(pixels_per_nm: float) -> np.ndarray:
    canvas_inner_radius_px, _ = get_canvas_params(pixels_per_nm)

    side = np.linspace(
        -canvas_inner_radius_px + 1,
        canvas_inner_radius_px - 1,
        canvas_inner_radius_px * 2 - 1
    )
    return np.meshgrid(side, side)


def _radial_to_2d(pixels_per_nm: float) -> np.ndarray:
    x, y = _canvas_meshgrid(pixels_per_nm)
    return np.sqrt(x ** 2 + y ** 2)


# Constants
_canvas_inner_radius_nm = 2000
