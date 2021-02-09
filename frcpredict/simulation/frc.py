"""
@original_author: andreas.boden
@adapted_by: stafak
"""

from typing import Tuple

import numpy as np

import frcpredict.model as mdl
from frcpredict.util import get_canvas_radius_nm, get_canvas_dimensions_px, radial_profile


def get_frc_curve_from_kernels2d(kernels2d: np.ndarray,
                                 run_instance: "mdl.RunInstance") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a tuple that contains arrays of X and Y values respectively of the resulting FRC curve
    from the given simulated kernels. run_instance must be a RunInstance without any multivalues.
    """

    # Pixel size in nanometres
    px_size_nm = run_instance.imaging_system_settings.scanning_step_size

    # Radius in pixels of all 2D patterns (PSF, illumination patterns, pinholes etc.)
    canvas_inner_rad_nm = run_instance.simulation_settings.canvas_inner_radius
    canvas_outer_rad_nm = get_canvas_radius_nm(canvas_inner_rad_nm, extend_sides_to_diagonal=True)
    canvas_outer_rad_px, _ = get_canvas_dimensions_px(canvas_outer_rad_nm, px_size_nm)

    # Calculate canvas parameters
    half_canvas_side = np.int(np.floor((canvas_outer_rad_px - 1) / 2))
    freq_arr = np.fft.fftfreq(2 * half_canvas_side - 1, px_size_nm)[:half_canvas_side]
    freq_arr_step = freq_arr[1] - freq_arr[0]

    # Retrieve/calculate sample/detector related parameters
    combined_sample_properties = run_instance.sample_properties.get_combined_properties(px_size_nm)
    input_power = combined_sample_properties.input_power
    D_0_0 = combined_sample_properties.D_origin

    pinhole = run_instance.imaging_system_settings.pinhole_function
    noise_var = run_instance.detector_properties.get_total_readout_noise_var(canvas_inner_rad_nm,
                                                                             pinhole)
    # Fourier transform kernels
    ft_kernels2d = np.abs(np.fft.fft2(kernels2d))

    # Calculate the power of signal and power of noise
    Ps = input_power * (np.abs(ft_kernels2d[0]) ** 2)  # Denominator of eq (35) in publication
    Pn = D_0_0 * ft_kernels2d[1, 0, 0] + noise_var  # Numerator of eq (35) in publication
    frc_spectra2d = Ps / (Ps + Pn)  # Eq (35)
    frc_spectra = radial_profile(frc_spectra2d, fftshift=True)

    # Return X and Y values
    return np.arange(0, len(frc_spectra)) * freq_arr_step, frc_spectra
