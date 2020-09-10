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
    get_paths_of_multivalues, expand_multivalues,
    int_to_flux, na_to_collection_efficiency,
    get_canvas_radius_nm, get_canvas_dimensions_px, radial_profile
)
from .telegraph import make_random_telegraph_data


def expected_on_time(*, P_on: np.ndarray, R_on: np.ndarray, R_off: np.ndarray,
                     T_obs: float) -> np.ndarray:
    """
    Calculates the expected time a fluorophore is ON during some observation time.
    (Eq (6) in publication.)

    - P_on: Probability of fluorophore being ON at start
    - Ron: Rate of ON-switching (events/ms)
    - Roff: Rate of OFF-switching (events/ms)
    - T_obs: Observation time
    """

    k = R_on + R_off

    t1 = T_obs * R_on / k
    fac1 = P_on - R_on / k
    fac2 = (1 - np.exp(-k * T_obs)) / k

    exp_ONt = t1 + fac1 * fac2

    return exp_ONt


def expected_P_on(*, P_pre: np.ndarray, R_on: np.ndarray, R_off: np.ndarray,
                  T_exp: float) -> np.ndarray:
    """
    Calculates the probability of a fluorophore being in the ON-state after some observation time.
    (Eq (5) in publication.)

    - P_pre: Probability of fluorophore being ON at start
    - R_on: Rate of ON-switching (events/ms)
    - R_off: Rate of OFF-switching (events/ms)
    - T_exp: Observation time
    - P_post: Probability of fluorophore being ON at end
    """

    k = R_on + R_off + np.finfo(float).eps

    t1 = R_on / k
    fac1 = P_pre - t1
    fac2 = np.exp(-k * T_exp)

    P_post = t1 + fac1 * fac2

    return P_post


def variance_detection(*, num_fluorophore_simulations: int, R_on: np.ndarray, R_off: np.ndarray,
                       alpha: np.ndarray, T_exp: float, P_on: np.ndarray) -> np.ndarray:
    """
    Estimates the variance of emitted photons for a single fluorophore observer for a certain
    observation time. R_on, R_off and alpha are given as arrays of "paired" values. Outputs are also
    arrays.

    - num_fluorophore_simulations: Number of times to simulate fluorophore
    - R_on: Rate of ON-switching (events/ms)
    - R_off: Rate of OFF-switching (events/ms)
    - alpha: Expected number of photons emitted/detected per ms that the
        fluorophore is in the ON-state
    - T_exp: Observation time
    - P_on: Probability of fluorophore being ON at start
    """

    assert R_on.shape == R_off.shape == alpha.shape == P_on.shape

    variances = np.zeros(R_on.shape)

    for idx, val in np.ndenumerate(R_on):
        ON_times = make_random_telegraph_data(num_fluorophore_simulations,
                                              t_on=1 / R_off[idx],
                                              t_off=1 / R_on[idx],
                                              t_bleach=1e10,
                                              t_exp=T_exp,
                                              P_on=P_on[idx])

        # Eq (2) in publication
        variances[idx] = alpha[idx] * ON_times.mean() + alpha[idx] ** 2 * ON_times.var()

    return variances


def make_kernels_detection(*, num_fluorophore_simulations: int,
                           quantum_efficiency: float, collection_efficiency: float,
                           readout_expected_photons: float, relative_readout_intensity: np.ndarray,
                           P_pre: np.ndarray, P_on_switch: float, P_off_switch: float,
                           P_fluorescent: float, T_obs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates the expected emission and variance of emission "kernels".

    - num_fluorophore_simulations: Number of times to simulate fluorophore
    - quantum_efficiency: Quantum efficiency of detector
    - collection_efficiency: Detection efficiency through optical system (including collection
        angle of objective)
    - readout_expected_photons: Expected maximum number of photons arriving from read-out
        illumination (at relative illumination = 1)
    - relative_readout_intensity: Relative read-out illumination intensity
    - P_pre: Probability of fluorophore being ON at start
    - P_on_switch: Probability of photon causing an ON-switch event
    - P_off_switch: Probability of photon causing an OFF-switch event
    - P_fluorescent: Probability of photon causing a fluorescent emission event
    - T_obs: Observation time
    """

    assert P_pre.shape == relative_readout_intensity.shape, "Not the same shapes"

    photon_illumination = readout_expected_photons * relative_readout_intensity

    alpha = quantum_efficiency * collection_efficiency * photon_illumination * P_fluorescent
    expected_on_at_obs = expected_on_time(P_on=P_pre, R_on=photon_illumination * P_on_switch,
                                          R_off=photon_illumination * P_off_switch, T_obs=T_obs)

    exp_det = np.multiply(alpha, expected_on_at_obs)  # Comparable to eq (1) in publication
    var_det = variance_detection(
        num_fluorophore_simulations=num_fluorophore_simulations,
        R_on=photon_illumination * P_on_switch, R_off=photon_illumination * P_off_switch,
        alpha=alpha, T_exp=T_obs, P_on=P_pre
    )

    return exp_det, var_det


def simulate(run_instance: "mdl.RunInstance", *,
             cache_kernels2d: bool = True,
             precache_frc_curves: bool = True,
             abort_signal: Optional[Signal] = None,
             preprocessing_finished_callback: Optional[Signal] = None,
             progress_updated_callback: Optional[Signal] = None) -> Optional["mdl.SimulationResults"]:
    """ Simulates kernels based on the given run instance. Aborts if abort_signal is emitted. """

    # Input validation
    if len(run_instance.fluorophore_settings.responses) < 1:
        raise ValueError("Input didn't contain any fluorophore responses!")

    if len(run_instance.pulse_scheme.pulses) < 1:
        raise ValueError("The pulse scheme is empty!")

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
            result.cache_frc_curve(run_instance_single, cache_kernels2d=cache_kernels2d)

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
    """
    Main function to run the simulations. run_instance must be a RunInstance without any
    multivalues.
    """

    px_size_nm = run_instance.imaging_system_settings.scanning_step_size  # Pixel size nanometers

    # Radius in pixels of all 2D patterns (PSF, illumination patterns, pinholes etc.)
    canvas_inner_rad_nm = run_instance.simulation_settings.canvas_inner_radius
    canvas_outer_rad_nm = get_canvas_radius_nm(canvas_inner_rad_nm, extend_sides_to_diagonal=True)
    canvas_outer_rad_px, _ = get_canvas_dimensions_px(canvas_outer_rad_nm, px_size_nm)

    # Calculate G(x,y), the convolution of the detection PSF and the pinhole function
    psf = run_instance.imaging_system_settings.optical_psf
    pinhole = run_instance.imaging_system_settings.pinhole_function

    radial_psf_and_pinhole = (isinstance(psf.pattern_data, mdl.RadialPatternData) and
                              isinstance(pinhole.pattern_data, mdl.RadialPatternData))

    psf_arr = psf.get_numpy_array(canvas_inner_rad_nm, px_size_nm,
                                  extend_sides_to_diagonal=radial_psf_and_pinhole)
    pinhole_arr = pinhole.get_numpy_array(canvas_inner_rad_nm, px_size_nm,
                                          extend_sides_to_diagonal=radial_psf_and_pinhole)

    G_2D = fftconvolve(pinhole_arr, psf_arr, mode="same")
    G_rad = np.zeros(canvas_outer_rad_px)
    if radial_psf_and_pinhole:
        G_rad[0:canvas_outer_rad_px] = G_2D[canvas_outer_rad_px - 1, canvas_outer_rad_px - 1:]
    else:
        radial_profile_result = radial_profile(G_2D)
        G_rad[0:len(radial_profile_result)] = radial_profile_result

    # Calculate collection efficiency
    if isinstance(psf.pattern_data, mdl.AiryNAPatternData):
        collection_efficiency = na_to_collection_efficiency(
            run_instance.imaging_system_settings.optical_psf.pattern_data.na,
            run_instance.imaging_system_settings.refractive_index
        )
    else:
        raise ValueError("Unsupported optical PSF type (only Airy from NA is currently supported)")

    # Calculate ON-state probabilities after illumination
    P_on = np.zeros(canvas_outer_rad_px)
    for pulse_index, pulse in enumerate(run_instance.pulse_scheme.pulses):
        illumination_pattern_rad = pulse.illumination_pattern.get_radial_profile(
            canvas_inner_rad_nm, px_size_nm
        )
        expected_photons = int_to_flux(pulse.max_intensity * 1000, pulse.wavelength) * px_size_nm ** 2
        response = run_instance.fluorophore_settings.get_response(pulse.wavelength)

        # The following comes from the rate being defined per photon falling in a 20x20 nm area
        temp_scaling_factor = (20 / px_size_nm) ** 2  # TODO: Change

        if pulse_index < len(run_instance.pulse_scheme.pulses) - 1:
            P_on = expected_P_on(
                P_pre=P_on,
                R_on=expected_photons * response.cross_section_off_to_on * temp_scaling_factor * illumination_pattern_rad,
                R_off=expected_photons * response.cross_section_on_to_off * temp_scaling_factor * illumination_pattern_rad,
                T_exp=pulse.duration
            )
        else:  # Last pulse (readout pulse)
            exp_kernel, var_kernel = make_kernels_detection(
                num_fluorophore_simulations=run_instance.simulation_settings.num_kernel_detection_iterations,
                quantum_efficiency=run_instance.detector_properties.quantum_efficiency,
                collection_efficiency=collection_efficiency,
                readout_expected_photons=expected_photons,
                relative_readout_intensity=illumination_pattern_rad,
                P_pre=P_on,
                P_on_switch=response.cross_section_off_to_on * temp_scaling_factor,
                P_off_switch=response.cross_section_on_to_off * temp_scaling_factor,
                P_fluorescent=response.cross_section_emission * temp_scaling_factor,
                T_obs=pulse.duration
            )

            # As described in eq (17) and (18) in publication
            exp_kernel *= G_rad
            var_kernel *= G_rad ** 2

            return exp_kernel, var_kernel

    raise ValueError("Input didn't contain any pulses!")
