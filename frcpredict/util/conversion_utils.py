"""
@original_author: andreas.boden
@adapted_by: stafak
"""

import numpy as np


def int_to_flux(intensity: float, wavelength: float) -> float:
    """ Takes intensity as W/cm^2 and wavelength in nm and returns photons/(nm^2*ms) """

    E = 1000 * 1.24 / wavelength

    intm2 = intensity * 10000
    fluxm2s = intm2 / (1.60218e-19 * E)

    fluxnm2s = fluxm2s * 1e-18
    fluxnm2ms = fluxnm2s * 1e-3

    return fluxnm2ms


def na_to_collection_efficiency(na: float, refractive_index: float) -> float:
    alpha = np.arcsin(na / refractive_index)
    collection_efficiency = (1 - np.cos(alpha)) / 2
    return collection_efficiency
