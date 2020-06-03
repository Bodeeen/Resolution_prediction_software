import numpy as np
from astropy.modeling.functional_models import Gaussian1D, Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma

# Temp stuff!
# TODO: Make everything better.
# TODO: Also, don't compute patterns immediately (takes longer to load the program).

px_size_nm = 20
radius_nm = np.sqrt(2) * 2000
radius_px = np.int(radius_nm / px_size_nm)
width_px = radius_px * 2
height_px = radius_px * 2


def radial_to_2d(size: int):
    x, y = np.meshgrid(range(size), range(size))
    return np.sqrt((x - (size/2) + 1)**2 + (y - (size/2) + 1)**2)


def gaussian_test1(fwhm: float) -> np.ndarray:
    stddev = fwhm * gaussian_fwhm_to_sigma
    model = Gaussian2D(x_mean=width_px / 2, y_mean=height_px / 2, x_stddev=stddev, y_stddev=stddev)

    result = np.zeros((width_px, height_px))
    x = np.arange(width_px)
    for y in range(0, height_px):
        result[y] = model(x, y)

    return result


def gaussian_test2(fwhm: float) -> np.ndarray:
    model = Gaussian1D(mean=radius_px / 2, stddev=fwhm * gaussian_fwhm_to_sigma)
    return model(radial_to_2d(radius_px*2 - 1))


patterns = {
    "Gaussian": gaussian_test1(480*4 / px_size_nm),
    "Doughnut": gaussian_test2(480 / px_size_nm)
}
