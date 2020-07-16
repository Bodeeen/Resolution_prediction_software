"""
Created on Tue Jun 11 13:38:07 2019

@original_author: andreas.boden
@adapted_by: stafak
"""
from typing import Optional

from PySignal import Signal

from Old_scripts.conversions import Int2Flux
import Old_scripts.telegraph as telegraph
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

import frcpredict.model
from frcpredict.util import get_paths_of_multivalues, expand_multivalues


def expected_ONtime(P_on, Ron, Roff, T_obs):  # Eq. 6
    """ Funcition calculating the expected time a fluorophore is ON 
    during an observation time T_obs

    - P_on: Probability of fluorophore being ON at start
    - Ron: Rate of ON-switching (events/ms)
    - Roff: Rate of OFF-switching (events/ms)
    - T_obs: Observation time

    """
    Ron = np.asarray(Ron)
    Roff = np.asarray(Roff)

    k = np.add(Ron, Roff)

    t1 = np.multiply(T_obs, np.divide(Ron, k))
    fac1 = np.subtract(P_on, np.divide(Ron, k))
    fac2 = np.divide(np.subtract(1, np.exp(-np.multiply(k, T_obs))), k)

    exp_ONt = np.add(t1, np.multiply(fac1, fac2))

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

    Ron = np.asarray(Ron)
    Roff = np.asarray(Roff)

    k = np.add(Ron, Roff) + np.finfo(float).eps

    t1 = np.divide(Ron, k)
    fac1 = np.subtract(P_pre, np.divide(Ron, k))
    fac2 = np.exp(-np.multiply(k, T_exp))

    P_post = np.add(t1, np.multiply(fac1, fac2))

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
    Ron = np.asarray(Ron)
    Roff = np.asarray(Roff)
    alpha = np.asarray(alpha)
    Pon = np.asarray(Pon)

    assert Ron.shape == Roff.shape == alpha.shape == Pon.shape

    v = np.zeros(Ron.shape)
    m = np.zeros(Ron.shape)
    Ns = np.zeros(Ron.shape)

    for idx, val in np.ndenumerate(Ron):
        ON_times, N_switches = telegraph.make_random_telegraph_data(N,
                                                                    t_on=1/Roff[idx],
                                                                    t_off=1/Ron[idx],
                                                                    t_bleach=1e10,
                                                                    t_exp=T_exp,
                                                                    p_on=Pon[idx])

        variance = alpha[idx]*ON_times.mean() + alpha[idx]**2 * \
            ON_times.var()  # Eq (2) in publication
        v[idx] = variance
        m[idx] = alpha[idx]*ON_times.mean()
        Ns[idx] = N_switches.mean()

    return v, m, Ns


def make_kernels_detection(N, QE, det_eff, PonStart, E_p_RO, RO_ill, Ponswitch, Poffswitch, Pfl, T_obs):
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

    alpha = QE*det_eff*E_p_RO*RO_ill*Pfl
    expONt = expected_ONtime(PonStart, E_p_RO*RO_ill*Ponswitch, E_p_RO*RO_ill*Poffswitch, T_obs)
    expDet = np.multiply(alpha, expONt)  # Comparable to eq (1) i publication
    varDet, mean, N_switches = variance_Detection(
        N, E_p_RO*RO_ill*Ponswitch, E_p_RO*RO_ill*Poffswitch, alpha, T_obs, PonStart)

    return np.array([expDet, varDet]), mean, N_switches  # mean is just for confirmation


def expandDimensions(radialvec, radialval, outradius):
    """ Function to create a 2D image/array from a 1D radial profile.

    - radialvec: vector containing distances from center
    - radialval: vector containing values at radialvec distance
    - outradius: determines size of output image, outradius is distance from
        image center to closest image edge (not diagonal)
    """
    x = np.linspace(-outradius, outradius, 2*outradius+1)

    c, r = np.meshgrid(x, x)
    rad = np.sqrt(np.power(c, 2) + np.power(r, 2))
    f = interp1d(radialvec, radialval)

    out = f(rad)

    return out


def radial_profile_of_raw_fft(data, center):
    """Function to calculate the radial profile of a fourier tranform image.
    The functino expects the input fft to not be fftshifted.

    Note: This function is a bit messy...

    - fft: input fft
    - center: specifies the center pixel of the fft (zero-frequency)
    """

    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = np.fft.fftshift(r.astype(np.int))

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def radial_profile(data, center):
    """ Is actually same as functio above right now. May be removed later 
    if all depending on it are changed"""

    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def simulate(run_instance, abort_signal: Optional[Signal] = None,
             preprocessing_finished_callback: Optional[Signal] = None,
             progress_updated_callback: Optional[Signal] = None):
    aborting = False
    completed_simulations = 0

    # Set up abort handling
    if abort_signal is not None:
        def abort_handler():
            nonlocal aborting
            aborting = True

        abort_signal.connect(abort_handler)

    if progress_updated_callback is not None:
        progress_updated_callback.emit(0)

    # Expand run instances
    print(run_instance)
    multivalue_paths, num_simulations = get_paths_of_multivalues(run_instance)
    print(f"{num_simulations} simulations")
    expanded_run_instances = expand_multivalues(run_instance, multivalue_paths)

    if preprocessing_finished_callback is not None:
        preprocessing_finished_callback.emit(num_simulations)

    # Run simulations
    def dynamic_simulate_single(data):
        if aborting:
            return None

        result = _simulate_single(data)

        nonlocal completed_simulations
        completed_simulations += 1

        if progress_updated_callback is not None:
            progress_updated_callback.emit(completed_simulations / num_simulations)
        
        return result

    frc_curves = np.frompyfunc(dynamic_simulate_single, 1, 1)(expanded_run_instances)

    # Return results, or None if aborted
    if not aborting:
        return frcpredict.model.FrcSimulationResults(
            run_instance=run_instance,
            multivalue_paths=multivalue_paths,
            frc_curves=frc_curves
        )
    else:
        return None


def _simulate_single(data):
    """ Main function to run the simulations """
    multivalue_values, run_instance = data
    print(run_instance)

    px_size_nm = run_instance.imaging_system_settings.scanning_step_size  # Pixel size nanometers

    # Radius in nm of all 2D patterns (PSF, illumination patterns, pinholes etc.) (inside the square)
    psf_kernel_rad_nm = 2000
    # Radius in pixels of all 2D patterns (PSF, illumination patterns, pinholes etc.) (inside the square)
    psf_kernel_rad_px = np.int(psf_kernel_rad_nm / px_size_nm)

    half_side = np.int(np.floor((psf_kernel_rad_px-1) / 2))
    freq_arr = np.fft.fftfreq(2*half_side-1, px_size_nm)[:half_side]
    df = freq_arr[1]-freq_arr[0]

    """ Set some camera parameters """
    ro_var = run_instance.camera_properties.readout_noise**2

    """ Calculate G(x,y), the convolution of the detection PSF and the pinhole fcn 
    This section is messy beacuse we played around with some different ways of 
    generating the pinhole function P. """
    PSF = run_instance.imaging_system_settings.optical_psf.get_numpy_array(px_size_nm)
    P = run_instance.imaging_system_settings.pinhole_function.get_numpy_array(px_size_nm)
    G2D = fftconvolve(P, PSF, 'same')
    G_rad = np.zeros(psf_kernel_rad_px)
    G_rad_temp = G2D[psf_kernel_rad_px - 1][psf_kernel_rad_px - 1:]
    G_rad[0:len(G_rad_temp)] = G_rad_temp
    # print(G_rad)

    """ Set sample properties """
    labelling_factor = run_instance.sample_properties.labelling_density
    sample_power = run_instance.sample_properties.spectral_power
    input_power = labelling_factor**2*sample_power
    K_0_0 = run_instance.sample_properties.K_origin
    D_0_0 = labelling_factor*K_0_0

    """ Calculate ON-state probabilities after ON-switching illumination"""
    P_on = np.zeros(psf_kernel_rad_px)
    for pulse_index, pulse in enumerate(run_instance.pulse_scheme.pulses):
        illumination_pattern_rad = pulse.illumination_pattern.get_numpy_array(
            px_size_nm
        )[psf_kernel_rad_px - 1][psf_kernel_rad_px - 1:]  # TODO: This currently only extracts the radial profile

        response = run_instance.fluorophore_settings.get_response(pulse.wavelength)
        expected_photons = Int2Flux(pulse.max_intensity * 1000, pulse.wavelength) * px_size_nm**2

        if pulse_index < len(run_instance.pulse_scheme.pulses) - 1:
            P_on = expected_Pon(
                P_on,
                expected_photons * response.cross_section_off_to_on * illumination_pattern_rad,
                expected_photons * response.cross_section_on_to_off * illumination_pattern_rad,
                pulse.duration
            )
        else:
            kernels, m, N_switches = make_kernels_detection(
                500000,
                run_instance.camera_properties.quantum_efficiency,
                0.3,  # TODO: This is a temporary value
                P_on,
                expected_photons,
                illumination_pattern_rad,
                response.cross_section_off_to_on,
                response.cross_section_on_to_off,
                response.cross_section_emission,
                pulse.duration
            )

            # As described in eq (17) and (18) in publication
            kernels[0] = np.multiply(kernels[0], G_rad)
            kernels[1] = np.multiply(kernels[1], np.abs(G_rad))
            """Expand kernels from radial function to symmetric 2D funtion in order to allow
            taking the 2D Fourier transform """
            kernels2d = np.array([expandDimensions(np.linspace(
                0, psf_kernel_rad_px-1, psf_kernel_rad_px), kernels[i], half_side) for i in range(len(kernels))])

            """ Fourier transform kernels """
            ft_kernels2d = np.abs(np.fft.fft2(kernels2d))

            """Calculate the power of signal and power of noise """
            Ps = input_power * np.power(np.abs(ft_kernels2d[0]), 2)  # Denominator of eq (35) in publication
            Pn = D_0_0*ft_kernels2d[1, 0, 0] + ro_var  # NUmerator of eq (35) in publication
            frc_spectra2d = np.divide(Ps, np.add(Ps, Pn))  # Eq (35)

            """ Below is to again calculate the radial profile of the different spectra """
            center = np.floor(np.divide(frc_spectra2d.shape, 2)
                              )  # gives index of zero-frequencu component in unshifted fft spectra
            frc_spectra = radial_profile_of_raw_fft(frc_spectra2d, center)

    return frcpredict.model.FrcCurve(
        multivalue_values=multivalue_values,
        x=np.arange(0, len(frc_spectra)) * df,
        y=frc_spectra
    )
