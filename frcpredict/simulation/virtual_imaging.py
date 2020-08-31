import numpy as np
from scipy.signal import fftconvolve

import frcpredict.model as mdl


def get_expected_image_from_kernels2d(kernels2d: np.ndarray, run_instance: "mdl.RunInstance",
                                      sample_image_arr: np.ndarray) -> np.ndarray:
    """ Returns the expected image based on the given sample image and simulated kernels. """

    exp_kernel_result = fftconvolve(sample_image_arr, kernels2d[0], mode="same")
    var_kernel_result = fftconvolve(sample_image_arr, kernels2d[1], mode="same").clip(0)

    gaussian_noise = np.random.normal(
        0, np.sqrt(var_kernel_result + run_instance.camera_properties.readout_noise ** 2)
    )

    expected_image = exp_kernel_result + gaussian_noise
    return expected_image
