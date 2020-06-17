from astropy.modeling.functional_models import AiryDisk2D, Gaussian1D, Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
from typing import Tuple

# Temp stuff!
# TODO: Make everything better.


_canvas_inner_radius_nm = 2000


def get_canvas_params(pixels_per_nm: float) -> Tuple[int, int]:
    canvas_inner_radius_px = np.int(_canvas_inner_radius_nm / pixels_per_nm)
    canvas_side_length_px = canvas_inner_radius_px * 2 - 1

    return canvas_inner_radius_px, canvas_side_length_px


def gaussian_test1(amplitude: float, fwhm: float, pixels_per_nm: float) -> np.ndarray:
    canvas_inner_radius_px, canvas_side_length_px = get_canvas_params(pixels_per_nm)

    stddev = fwhm / pixels_per_nm * gaussian_fwhm_to_sigma
    model = Gaussian2D(amplitude=amplitude, x_stddev=stddev, y_stddev=stddev)

    result = np.zeros((canvas_side_length_px, canvas_side_length_px))
    x = range(-canvas_inner_radius_px + 1, canvas_inner_radius_px)
    for y in range(-canvas_inner_radius_px + 1, canvas_inner_radius_px):
        result[(y + canvas_inner_radius_px - 1) % canvas_side_length_px] = model(x, y)

    return result


def doughnut_test1(periodicity: float, pixels_per_nm: float) -> np.ndarray:
    canvas_inner_radius_px, _ = get_canvas_params(pixels_per_nm)

    def Doughnut1D(radius: float) -> np.ndarray:
        return np.where(
            radius < periodicity/ (2 * pixels_per_nm),
            0.5 - 0.5 * np.cos(2 * np.pi * radius * pixels_per_nm / periodicity),
            1
        )

    return Doughnut1D(_radial_to_2d(canvas_inner_radius_px))


def airy_test1(amplitude: float, fwhm: float, pixels_per_nm: float) -> np.ndarray:
    canvas_inner_radius_px, canvas_side_length_px = get_canvas_params(pixels_per_nm)

    airy_radius = fwhm/pixels_per_nm * 0.353/(0.61/2)  # TODO: Might not be correct
    model = AiryDisk2D(amplitude=amplitude, radius=airy_radius)
    result = np.zeros((canvas_side_length_px, canvas_side_length_px))
    x = range(-canvas_inner_radius_px + 1, canvas_inner_radius_px)
    for y in range(-canvas_inner_radius_px + 1, canvas_inner_radius_px):
        result[(y + canvas_inner_radius_px - 1) % canvas_side_length_px] = model(x, y)

    return result


def digital_pinhole_test1(fwhm: float, pixels_per_nm: float) -> np.ndarray:
    g_base = gaussian_test1(amplitude=1, fwhm=fwhm, pixels_per_nm=pixels_per_nm)
    const_base = np.ones_like(g_base)

    b0_vec = g_base.reshape(g_base.size)
    b1_vec = const_base.reshape(const_base.size)

    b_mat = np.array([b0_vec, b1_vec])

    b_inv = np.linalg.pinv(b_mat)

    return b_inv[:, 0].reshape(g_base.shape)


def _radial_to_2d(canvas_inner_radius_px: int) -> np.ndarray:
    side = np.linspace(
        -canvas_inner_radius_px,
        canvas_inner_radius_px,
        canvas_inner_radius_px * 2 - 1
    )
    x, y = np.meshgrid(side, side)

    return np.sqrt(x**2 + y**2)
