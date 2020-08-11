"""
@original_author: andreas.boden
@adapted_by: stafak
"""

from typing import Tuple

import numpy as np

import frcpredict.model as mdl
from frcpredict.util import radial_profile


def get_frc_curve_from_kernels2d(kernels2d: np.ndarray,
                                 run_instance: "mdl.RunInstance") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a tuple that contains arrays of X and Y values respectively of the resulting FRC curve
    from the given simulated kernels.
    """

    # Pixel size in nanometres
    px_size_nm = run_instance.imaging_system_settings.scanning_step_size

    # Inner radius in nm of all 2D patterns (PSF, illumination patterns, pinholes etc.)
    psf_kernel_rad_nm = 2000

    # Inner radius in pixels of all 2D patterns (PSF, illumination patterns, pinholes etc.)
    psf_kernel_rad_px = np.int(psf_kernel_rad_nm / px_size_nm)

    # Calculate canvas parameters
    half_canvas_side = np.int(np.floor((psf_kernel_rad_px - 1) / 2))
    freq_arr = np.fft.fftfreq(2 * half_canvas_side - 1, px_size_nm)[:half_canvas_side]
    freq_arr_step = freq_arr[1] - freq_arr[0]

    # Retrieve/calculate sample/camera related parameters
    labelling_density = run_instance.sample_properties.labelling_density
    sample_spectral_power = run_instance.sample_properties.spectral_power
    input_power = labelling_density ** 2 * sample_spectral_power
    K_0_0 = run_instance.sample_properties.K_origin
    D_0_0 = labelling_density * K_0_0

    ro_var = run_instance.camera_properties.readout_noise ** 2

    # Fourier transform kernels
    ft_kernels2d = np.abs(np.fft.fft2(kernels2d))

    # Calculate the power of signal and power of noise
    Ps = input_power * (np.abs(ft_kernels2d[0]) ** 2)  # Denominator of eq (35) in publication
    Pn = D_0_0 * ft_kernels2d[1, 0, 0] + ro_var  # Numerator of eq (35) in publication
    frc_spectra2d = Ps / (Ps + Pn)  # Eq (35)
    frc_spectra = radial_profile(frc_spectra2d, fftshift=True)

    # Return X and Y values
    return np.arange(0, len(frc_spectra)) * freq_arr_step, frc_spectra
