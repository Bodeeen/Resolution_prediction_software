# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:46:08 2019

@author: andreas.boden
"""

import numpy as np

def Int2Flux(intensity, wavelength):
    """Takes intensity as W/cm^2 and wavelength in nm and returns photons/(nm^2*ms)"""
    
    E = 1000 * 1.24 / wavelength
    
    intm2 = intensity * 10000
    fluxm2s = intm2 / (1.60218e-19 * E)
    
    fluxnm2s = fluxm2s * 1e-18
    fluxnm2ms = fluxnm2s * 1e-3
    
    return fluxnm2ms


def NA2CollEff(NA, refInd):
    alpha = np.arcsin(NA/refInd)
    
    capA = 2*np.pi*(1-np.cos(alpha))
    sphereA = 4*np.pi
    ratio = capA / sphereA
    
    return ratio