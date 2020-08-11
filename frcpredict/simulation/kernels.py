"""
@original_author: andreas.boden
@adapted_by: stafak
"""

from typing import Optional, Tuple

import numpy as np
from PySignal import Signal
from scipy.signal import fftconvolve

import frcpredict.model as mdl
from frcpredict.util import (
    get_paths_of_multivalues, expand_multivalues, int_to_flux, na_to_collection_efficiency
)
from .telegraph import make_random_telegraph_data


def expected_ONtime(P_on, Ron, Roff, T_obs):  # Eq. 6
    """ Funcition calculating the expected time a fluorophore is ON
    during an observation time T_obs

    - P_on: Probability of fluorophore being ON at start
    - Ron: Rate of ON-switching (events/ms)
    - Roff: Rate of OFF-switching (events/ms)
    - T_obs: Observation time

    """
    k = Ron + Roff

    t1 = T_obs * Ron / k
    fac1 = P_on - Ron / k
    fac2 = (1 - np.exp(-k * T_obs)) / k

    exp_ONt = t1 + fac1 * fac2

    return exp_ONt


def expected_Pon(P_pre, Ron, Roff, T_exp):  # Eq. 5
    """ Function for calculating the probability of a fluorophore
    being in the ON-state after some observation time.

    - P_pre: Probability of fluorophore being ON at start
    - Ron: Rate of ON-switching (events/ms)
    - Roff: Rate of OFF-switching (events/ms)
    - T_exp: Observation time
    - P_post: Probability of fluorophore being ON at end

    """

    k = Ron + Roff + np.finfo(float).eps

    t1 = Ron / k
    fac1 = P_pre - t1
    fac2 = np.exp(-k * T_exp)

    P_post = t1 + fac1 * fac2

    return P_post


def variance_Detection(N, Ron, Roff, alpha, T_exp, Pon):
    """ Function to calculate/estimate the variance of emitted photons
    for a single fluorophore observer for a certain observation time. Ron, Roff
    and alpha are given as arrays of "paired" values. Outputs are also arrays.

    - N: number of times to simulate fluorophore in order to get good statistics
    - Ron: Rate of ON-switching (events/ms)
    - Roff: Rate of OFF-switching (events/ms)
    - alpha: expected number of photons emitted/detected per ms that the
        fluorophore is in the ON-state
    - T_exp: Observation time
    - Pon: Probability of fluorophore being ON at start
    """
    assert Ron.shape == Roff.shape == alpha.shape == Pon.shape

    v = np.zeros(Ron.shape)
    m = np.zeros(Ron.shape)
    Ns = np.zeros(Ron.shape)

    for idx, val in np.ndenumerate(Ron):
        ON_times, N_switches = make_random_telegraph_data(N,
                                                          t_on=1 / Roff[idx],
                                                          t_off=1 / Ron[idx],
                                                          t_bleach=1e10,
                                                          t_exp=T_exp,
                                                          p_on=Pon[idx])

        # Eq (2) in publication
        variance = alpha[idx] * ON_times.mean() + alpha[idx] ** 2 * ON_times.var()

        v[idx] = variance
        m[idx] = alpha[idx] * ON_times.mean()
        Ns[idx] = N_switches.mean()

    return v, m, Ns


def make_kernels_detection(N, QE, det_eff, PonStart, E_p_RO, RO_ill, Ponswitch, Poffswitch, Pfl,
                           T_obs):
    """ Function to create the expeced emission and variance of emission
    "kernels".

    - N: number of times to simulate fluorophore in order to get good statistics
    - QE: Quantum efficiency of camera
    - det_eff: Detection efficiency through optical system (including collection
        angle of objective)
    - PonStart: Probability of fluorophore being ON at start
    - E_p_RO: Expected maximum number of photons arriving from read-out
        illumination (at relative illumination = 1)
    - RO_ill: Relative read-out illumination intensity
    - Ponswitch: Probability of photon causing an ON-switch event
    - Poffswitch: Probability of photon causing an OFF-switch event
    - Pfl: Probability of photon causing a fluoreschent emission event
    """
    assert PonStart.shape == RO_ill.shape, 'Not the same shapes'

    alpha = QE * det_eff * E_p_RO * RO_ill * Pfl
    expONt = expected_ONtime(PonStart, E_p_RO * RO_ill * Ponswitch, E_p_RO * RO_ill * Poffswitch,
                             T_obs)
    expDet = np.multiply(alpha, expONt)  # Comparable to eq (1) i publication
    varDet, mean, N_switches = variance_Detection(
        N, E_p_RO * RO_ill * Ponswitch, E_p_RO * RO_ill * Poffswitch, alpha, T_obs, PonStart)

    return expDet, varDet, mean, N_switches  # mean is just for confirmation


def simulate(run_instance: "mdl.RunInstance", *,
             precache_frc_curves: bool = True,
             abort_signal: Optional[Signal] = None,
             preprocessing_finished_callback: Optional[Signal] = None,
             progress_updated_callback: Optional[Signal] = None) -> Optional["mdl.SimulationResults"]:
    """ Simulates kernels based on the given run instance. """

    # Input validation
    if len(run_instance.fluorophore_settings.responses) < 1:
        raise Exception("You have not set any fluorophore responses!")

    if len(run_instance.pulse_scheme.pulses) < 1:
        raise Exception("The pulse scheme is empty!")

    # Set up abort handling
    aborting = False

    if abort_signal is not None:
        def abort_handler():
            nonlocal aborting
            aborting = True

        abort_signal.connect(abort_handler)

    if progress_updated_callback is not None:
        progress_updated_callback.emit(0)

    # Expand run instances
    multivalue_paths, num_iterations = get_paths_of_multivalues(run_instance)
    expanded_run_instances = expand_multivalues(run_instance, multivalue_paths)

    if preprocessing_finished_callback is not None:
        preprocessing_finished_callback.emit(num_iterations)

    # Run simulations
    completed_simulations = 0

    def dynamic_simulate_single(data):
        if aborting:
            return None

        multivalue_values, run_instance_single = data
        exp_kernel, var_kernel = _simulate_single(run_instance_single)

        result = mdl.KernelSimulationResult(
            multivalue_values=multivalue_values,
            exp_kernel=exp_kernel,
            var_kernel=var_kernel
        )

        if precache_frc_curves:
            result.cache_frc_curve(run_instance_single)

        nonlocal completed_simulations
        completed_simulations += 1

        if progress_updated_callback is not None:
            progress_updated_callback.emit(completed_simulations / num_iterations)

        return result

    kernel_results = np.frompyfunc(dynamic_simulate_single, 1, 1)(expanded_run_instances)

    # Return results, or None if aborted
    if not aborting:
        return mdl.SimulationResults(
            run_instance=run_instance,
            multivalue_paths=multivalue_paths,
            kernel_results=kernel_results
        )
    else:
        return None


def _simulate_single(run_instance: "mdl.RunInstance") -> Tuple[np.ndarray, np.ndarray]:
    """ Main function to run the simulations """
    px_size_nm = run_instance.imaging_system_settings.scanning_step_size  # Pixel size nanometers

    # Radius in nm of all 2D patterns (PSF, illumination patterns, pinholes etc.) (inside the square)
    psf_kernel_rad_nm = 2000
    # Radius in pixels of all 2D patterns (PSF, illumination patterns, pinholes etc.) (inside the square)
    psf_kernel_rad_px = np.int(psf_kernel_rad_nm / px_size_nm)

    """ Calculate G(x,y), the convolution of the detection PSF and the pinhole fcn 
    This section is messy beacuse we played around with some different ways of 
    generating the pinhole function P. """
    PSF = run_instance.imaging_system_settings.optical_psf.get_numpy_array(px_size_nm)
    P = run_instance.imaging_system_settings.pinhole_function.get_numpy_array(px_size_nm)
    G2D = fftconvolve(P, PSF, mode="same")
    G_rad = np.zeros(psf_kernel_rad_px)
    G_rad_temp = G2D[psf_kernel_rad_px - 1][psf_kernel_rad_px - 1:]
    G_rad[0:len(G_rad_temp)] = G_rad_temp

    if isinstance(run_instance.imaging_system_settings.optical_psf.pattern_data, mdl.AiryNAPatternData):
        collection_efficiency = na_to_collection_efficiency(
            run_instance.imaging_system_settings.optical_psf.pattern_data.na,
            run_instance.imaging_system_settings.refractive_index
        )
    else:
        collection_efficiency = 0.3  # TODO: This is a temporary fallback

    """ Calculate ON-state probabilities after ON-switching illumination"""
    P_on = np.zeros(psf_kernel_rad_px)
    for pulse_index, pulse in enumerate(run_instance.pulse_scheme.pulses):
        # TODO: This does currently not create radial profile from an average
        illumination_pattern_rad = pulse.illumination_pattern.get_numpy_array(
            px_size_nm
        )[psf_kernel_rad_px - 1][psf_kernel_rad_px - 1:]

        response = run_instance.fluorophore_settings.get_response(pulse.wavelength)
        expected_photons = int_to_flux(pulse.max_intensity * 1000,
                                       pulse.wavelength) * px_size_nm ** 2

        # The following comes from the rate being defined per photon falling in a 20x20 nm area
        # TODO: Change
        temp_scaling_factor = (20 / px_size_nm) ** 2

        if pulse_index < len(run_instance.pulse_scheme.pulses) - 1:
            P_on = expected_Pon(
                P_on,
                expected_photons * response.cross_section_off_to_on * temp_scaling_factor * illumination_pattern_rad,
                expected_photons * response.cross_section_on_to_off * temp_scaling_factor * illumination_pattern_rad,
                pulse.duration
            )
        else:  # Last pulse (readout pulse)
            exp_kernel, var_kernel, m, N_switches = make_kernels_detection(
                500000,
                run_instance.camera_properties.quantum_efficiency,
                collection_efficiency,
                P_on,
                expected_photons,
                illumination_pattern_rad,
                response.cross_section_off_to_on * temp_scaling_factor,
                response.cross_section_on_to_off * temp_scaling_factor,
                response.cross_section_emission * temp_scaling_factor,
                pulse.duration
            )

            # As described in eq (17) and (18) in publication
            exp_kernel *= G_rad
            var_kernel *= np.abs(G_rad)

            return exp_kernel, var_kernel

    raise Exception("Run instance didn't have a readout pulse!")
