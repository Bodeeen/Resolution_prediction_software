import numpy as np
from scipy.signal import fftconvolve

import frcpredict.model as mdl


def get_expected_image_from_kernels2d(kernels2d: np.ndarray, run_instance: "mdl.RunInstance",
                                      displayable_sample: mdl.DisplayableSample) -> np.ndarray:
    """ Returns the expected image based on the given sample image and simulated kernels. """

    sample_image_arr = displayable_sample.get_image_arr(
        run_instance.imaging_system_settings.scanning_step_size
    )
    exp_kernel_result = fftconvolve(sample_image_arr, kernels2d[0], mode="same")
    var_kernel_result = fftconvolve(sample_image_arr, kernels2d[1], mode="same").clip(0)

    canvas_inner_rad_nm = run_instance.simulation_settings.canvas_inner_radius
    pinhole = run_instance.imaging_system_settings.pinhole_function
    noise_var = run_instance.detector_properties.get_total_readout_noise_var(canvas_inner_rad_nm,
                                                                             pinhole)
    gaussian_noise = np.random.normal(0, np.sqrt(var_kernel_result + noise_var))

    expected_image = exp_kernel_result + gaussian_noise
    return expected_image
