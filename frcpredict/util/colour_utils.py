from typing import Tuple

import colour


def wavelength_to_rgb(wavelength: int, gamma: float = 2.2) -> Tuple[int, int, int]:
    """
    Converts a wavelength (in nanometres) to a gamma corrected RGB tuple with values [0, 255].
    Returns white if the wavelength is outside the visible spectrum or any other error occurs.
    """

    try:
        xyz = colour.wavelength_to_XYZ(wavelength)
        srgb = colour.XYZ_to_sRGB(xyz).clip(0, 1)
        gamma_corrected_rgb = 255 * srgb ** (1 / gamma)
        return tuple(gamma_corrected_rgb)
    except ValueError:
        return 255, 255, 255
