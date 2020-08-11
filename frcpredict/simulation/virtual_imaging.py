import numpy as np
from scipy.signal import fftconvolve


def get_expected_image_from_kernels2d(kernels2d: np.ndarray,
                                      sample_image_arr: np.ndarray) -> np.ndarray:
    """ Returns the expected image based on the given sample image and simulated kernels. """

    exp_kernel_result = fftconvolve(sample_image_arr, kernels2d[0], mode="same")
    var_kernel_result = fftconvolve(sample_image_arr, kernels2d[1], mode="same")
    gaussian_noise = np.random.normal(0, np.sqrt(var_kernel_result))

    expected_image = exp_kernel_result + gaussian_noise
    return expected_image
