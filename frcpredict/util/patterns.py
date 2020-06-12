import numpy as np
from astropy.modeling.functional_models import AiryDisk2D, Gaussian1D, Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma

# Temp stuff!
# TODO: Make everything better.

px_size_nm = 20
radius_nm = np.sqrt(2) * 2000
radius_px = np.int(radius_nm / px_size_nm)
width_px = radius_px * 2 - 1
height_px = radius_px * 2 - 1


def radial_to_2d(size: int):
    x, y = np.meshgrid(range(size), range(size))
    return np.sqrt((x - (size/2) + 1)**2 + (y - (size/2) + 1)**2)


def gaussian_test1(fwhm: float) -> np.ndarray:
    stddev = fwhm * gaussian_fwhm_to_sigma
    model = Gaussian2D(x_mean=(width_px - 1) / 2, y_mean=(height_px - 1) / 2, x_stddev=stddev, y_stddev=stddev)

    result = np.zeros((width_px, height_px))
    x = np.arange(width_px)
    for y in range(0, height_px):
        result[y] = model(x, y)

    return result


def gaussian_test2(fwhm: float) -> np.ndarray:
    model = Gaussian1D(mean=radius_px / 2, stddev=fwhm * gaussian_fwhm_to_sigma)
    return model(radial_to_2d(radius_px*2 - 1))


def airy_test(fwhm: float) -> np.ndarray:
    model = AiryDisk2D(x_0=width_px / 2, y_0=height_px / 2, radius=fwhm)
    result = np.zeros((width_px, height_px))
    x = np.arange(width_px)
    for y in range(0, height_px):
        result[y] = model(x, y)

    return result


patterns = {
    "Gau240": lambda: gaussian_test1(240 / px_size_nm),
    "Gau480": lambda: gaussian_test1(480 / px_size_nm),
    "Dn240": lambda: gaussian_test2(240 / px_size_nm),
    "Dn480": lambda: gaussian_test2(480 / px_size_nm),
    "Airy200": lambda: airy_test(200 / px_size_nm),
    "Airy230": lambda: airy_test(230 / px_size_nm)
}
