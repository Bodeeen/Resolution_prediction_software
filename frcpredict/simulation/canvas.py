"""
@original_author: andreas.boden
@adapted_by: stafak
"""

import numpy as np
from scipy.interpolate import interp1d

from frcpredict.util import get_canvas_radius_nm, get_canvas_dimensions_px


def expand_kernels_to_2d(*kernels,
                         canvas_inner_radius_nm: float, px_size_nm: float) -> np.ndarray:
    """
    Expands radial kernels to 2D arrays. The output will be an array containing these 2D arrays.
    """

    canvas_outer_rad_px, _ = get_canvas_dimensions_px(
        get_canvas_radius_nm(canvas_inner_radius_nm, extend_sides_to_diagonal=True), px_size_nm
    )
    half_side = np.int(np.floor((canvas_outer_rad_px - 1) / 2))

    kernels2d = np.array([
        expand_dimensions(
            np.linspace(0, canvas_outer_rad_px - 1, canvas_outer_rad_px),
            kernels[i], half_side
        )
        for i in range(len(kernels))
    ])

    return kernels2d


def expand_dimensions(radialvec: np.ndarray, radialval: np.ndarray, outradius: int) -> np.ndarray:
    """ Function to create a 2D image/array from a 1D radial profile.

    - radialvec: vector containing distances from center
    - radialval: vector containing values at radialvec distance
    - outradius: determines size of output image, outradius is distance from
        image center to closest image edge (not diagonal)
    """
    x = np.linspace(-outradius, outradius, 2 * outradius + 1)

    c, r = np.meshgrid(x, x)
    rad = np.sqrt(c ** 2 + r ** 2)
    f = interp1d(radialvec, radialval)

    return f(rad)
