# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:38:07 2019

@author: andreas.boden
"""

import telegraph
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.filters import convolve
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter
from scipy.ndimage import minimum_filter
import sys
import os
if not os.environ['PY_UTILS_PATH'] in sys.path:
    sys.path.append(os.environ['PY_UTILS_PATH'])
import patterns_creator
import DataIO_tools
import conversions
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

def expected_ONtime(P_on, Ron, Roff, Tobs):
    """ Funcition calculating the expected time a fluorophore is ON 
    during an observation time Tobs
    
    - P_on: Probability of fluorophore being ON at start
    - Ron: Rate of ON-switching (events/ms)
    - Roff: Rate of OFF-switching (events/ms)
    - Tobs: Observation time
    
    """
    Ron = np.asarray(Ron)
    Roff = np.asarray(Roff)
    
    k = np.add(Ron, Roff)
    
    t1 = np.multiply(Tobs, np.divide(Ron, k))
    fac1 = np.subtract(P_on, np.divide(Ron, k))
    fac2 = np.divide(np.subtract(1, np.exp(-np.multiply(k, Tobs))), k)
    
    exp_ONt = np.add(t1, np.multiply(fac1, fac2))
    
    return exp_ONt


def expected_Pon(P_pre, Ron, Roff, Tobs):
    """ Function for calculating the probability of a fluorophore 
    being in the ON-state after some observation time.
    
    - P_pre: Probability of fluorophore being ON at start
    - Ron: Rate of ON-switching (events/ms)
    - Roff: Rate of OFF-switching (events/ms)
    - Tobs: Observation time
    - P_post: Probability of fluorophore being ON at end
    
    """
    
    Ron = np.asarray(Ron)
    Roff = np.asarray(Roff)
    
    k = np.add(Ron, Roff) + np.finfo(float).eps
    
    t1 = np.divide(Ron, k)
    fac1 = np.subtract(P_pre, np.divide(Ron, k))
    fac2 = np.exp(-np.multiply(k, Tobs))
    
    P_post = np.add(t1, np.multiply(fac1, fac2))
    
    return P_post


def variance_Detection(N, Ron, Roff, alpha, Tobs, Pon):
    """ Function to calculate/estimate the variance of emitted photons
    for a single fluorophore observer for a certain observation time. Ron, Roff
    and alpha are given as arrays of "paired" values. Outputs are also arrays.
    
    - N: number of times to simulate fluorophore in order to get good statistics
    - Ron: Rate of ON-switching (events/ms)
    - Roff: Rate of OFF-switching (events/ms)
    - alpha: expected number of photons emitted/detected per ms that the
        fluorophore is in the ON-state
    - Tobs: Observation time
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
                                                        t_exp=Tobs,
                                                        p_on=Pon[idx])
        
        variance = alpha[idx]*ON_times.mean() + alpha[idx]**2*ON_times.var() # Eq (2) in publication
        v[idx] = variance
        m[idx] = alpha[idx]*ON_times.mean()
        Ns[idx] = N_switches.mean()
    
    return v, m, Ns

def make_kernels_detection(N, QE, det_eff, PonStart, E_p_RO, RO_ill, Ponswitch, Poffswitch, Pfl, Tobs):
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
    expONt= expected_ONtime(PonStart, E_p_RO*RO_ill*Ponswitch, E_p_RO*RO_ill*Poffswitch, Tobs)
    expDet = np.multiply(alpha, expONt) # Comparable to eq (1) i publication
    varDet, mean, N_switches = variance_Detection(N, E_p_RO*RO_ill*Ponswitch, E_p_RO*RO_ill*Poffswitch, alpha, Tobs, PonStart)
   
    return np.array([expDet, varDet]), mean, N_switches #mean is just for confirmation


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

def analytic_state_curve(Ron, Roff, t_max, dt):
    """Function to calculate the analytic curve describing the probability of
    a fluorophore being ON at a certain time.
    
    - Ron: Rate of ON-switching (events/ms)
    - Roff: Rate of OFF-switching (events/ms)
    - t_max: function evaluated from 0-t_max
    - dt: time step
    """
    t_points = np.linspace(0, t_max, 1 + t_max/dt)
        
    asc = Ron/(Ron+Roff) + (1 - Ron/(Ron+Roff))*np.exp(-t_points*(Ron + Roff))
    
    return t_points, asc
    

def radial_profile_of_raw_fft(data, center_ind):
    """Function to calculate the radial profile of a fourier tranform image.
    The functino expects the input fft to not be fftshifted.
    
    Note: This function is a bit messy...
    
    - fft: input fft
    - center: specifies the center pixel of the fft (zero-frequency)
    """
    
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center_ind[0])**2 + (y - center_ind[1])**2)
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

def simulate(OFF_sat_arr):
    """ Main function to run the simulations """
#    OFF_sat = 1
#    RO_sat = 100
    """Probabilities of photon inducing the different switches"""
    Pon_405 = 6.6e-7
    Poff_405 = 0
    Pfl_405 = 5e-8
        
#    """rsEGFP2"""
    Pon_488 = 3.3e-08
    Poff_488 = 7.3e-07
    Pfl_488 = 7e-06

    """rsEGFP(N205S)"""
#    Pon_488 = 4.52e-09
#    Poff_488 = 1e-07
#    Pfl_488 = 7e-06
    
    """Expected incident photons to simulate"""
    E_p_ON = 15656087
#    E_p_OFF = OFF_sat / Poff_488
    E_p_RO = 1849617
    
    T_on_switch = 0.1 #Time for applied ON-switching light
    T_off_switch = 4 #Time for applied OFF-switching light
    
    px_size_nm = 20 # Pixel size nanometers
    
    psf_kernel_rad_nm = np.sqrt(2)*2000 #Radius in nm of all 2D patterns (PSF, illumination patterns, pinholes etc.)
    psf_kernel_rad_px = np.int(psf_kernel_rad_nm / px_size_nm) #Radius in pixels of all 2D patterns (PSF, illumination patterns, pinholes etc.)
    
    
    """ Creating the different illumination patterns"""
    wf = np.ones(psf_kernel_rad_px) # "Widefield" pattern (constant)
    empty = np.zeros(psf_kernel_rad_px) # Empty pattern (zeros)
    airy190 = patterns_creator.RadialOf2DAiry(psf_kernel_rad_px, 200/px_size_nm) # "Airy" pattern with FWHM 190 nm (basically a sinc^2 function)
    airy220 = patterns_creator.RadialOf2DAiry(psf_kernel_rad_px, 230/px_size_nm) # "Airy" pattern with FWHM 220 nm (basically a sinc^2 function)
    doughnut = wf - patterns_creator.RadialOf2DGauss(psf_kernel_rad_px, 480/px_size_nm) # "Doughnut" patterns with 480 nm ""FWHM"" (created as widefield minus gaussian)
    
    x = np.linspace(0, psf_kernel_rad_nm, psf_kernel_rad_px)
    
    
    """  """
    periodicity = 510#360*sqrt(2)  Periodicity of OFF-swithching "grid"-pattern
    # Another way of creating a "Doughnut"-like pattern where the center is a sine center.
    sinecenter = 0.5 - 0.5*np.cos(2*np.pi*x/periodicity) 
    sinecenter[periodicity//(2*px_size_nm)::] = 1
    
    
    """ Assigning light patterns to illuminatio variables """
    ONSwitch_ill = airy190
    zero_percent = 0
    zero_ratio = zero_percent/100
    OFFSwitch_ill = zero_ratio + np.multiply(sinecenter, 1-zero_ratio) #Create a small backgrund/non-zero center of OFF-switching illumination.
    RO_ill = airy220
    
    """ Calculate ON-state probabilities after ON-switching illumination"""
    Pon1 = expected_Pon(empty, E_p_ON*ONSwitch_ill*Pon_405, E_p_ON*ONSwitch_ill*Poff_405, T_on_switch)
    
    
    """ Create array of obseration times (here termed expodure times T-exp/s to evaluate on """
#    tps = 30
#    ln_start_t = -4
#    ln_end_t = 5
#    T_exps = np.exp(np.linspace(ln_start_t, ln_end_t, tps))
    
    tps = 20
    min_t = 0.4
    max_t = 1.2
    T_exps = np.linspace(min_t, max_t, tps)
    
    """ Set some camera parameters """
    QE = 0.82
    ro_rms = 0
    pixels_used = 2*np.pi*(250/(65*2.355))**2
#    ro_var = pixels_used*ro_rms**2
    ro_var = pixels_used*ro_rms**2
    
    """Calculate the analytics state curve for some parameters """
    tps_ac, ac = analytic_state_curve(E_p_RO*Pon_488, E_p_RO*Poff_488, max_t, 0.1)
    
    
    """ Calculate some values used later (not super important)"""
    half_side = np.int(np.floor((psf_kernel_rad_px-1) / np.sqrt(2)))
    freq_arr = np.fft.fftfreq(2*half_side-1, px_size_nm)[:half_side]
    df = freq_arr[1]-freq_arr[0]
    res_arr = np.divide(1, freq_arr)
    full_side = 2*half_side + 1
#    i = np.linspace(0, full_side - 1, full_side)
#    freq_arr = np.divide(i, full_side)
#    half_freq_arr = freq_arr[0:half_side+1]
    
    """ Calculate G(x,y), the convolution of the detection PSF and the pinhole fcn 
    This section is messy beacuse we played around with some different ways of 
    generating the pinhole function P. """
#    det_psf = patterns_creator.GaussIm([full_side, full_side], 250/px_size_nm)
#    pinhole_fcn = patterns_creator.GaussIm([full_side, full_side], 250/px_size_nm)
    fwhm_Det = 250
    fwhm_Pin = 220
#    G_2D = fftconvolve(det_psf, pinhole_fcn, 'same')
    fwhmG = np.sqrt(fwhm_Det**2+fwhm_Pin**2)
    G_rad = patterns_creator.RadialOf2DGauss(psf_kernel_rad_px, fwhmG/px_size_nm)
    
#    Gauss = patterns_creator.GaussIm(PSF.shape, 250/px_size_nm)
#    Gauss = Gauss/Gauss.sum()
#    C = np.ones_like(Gauss)
#    Gvec = Gauss.reshape(Gauss.size)
#    Cvec = C.reshape(C.size)
#    bmat = np.append(Gvec, Cvec, 1)
##    P = np.linalg.pinv(np.transpose()
#    P = P - P.mean()
#    G2D = fftconvolve(P, PSF, 'same')
#    G_rad = radial_profile(G2D, [100,100])
#    
    im_shape = [750//px_size_nm+1, 750//px_size_nm+1]
    
    #    PSF = DataIO_tools.load_data('PSF_RW_1.4NA_20nmPxSize.tif')
    PSF = patterns_creator.GaussIm(im_shape, fwhm_Pin/px_size_nm)
    
    g_base = patterns_creator.GaussIm(im_shape, fwhm_Det/px_size_nm)
    const_base = np.ones_like(g_base)
    
    b0_vec = g_base.reshape(g_base.size)
    b1_vec = const_base.reshape(const_base.size)
    
    b_mat = np.array([b0_vec, b1_vec])
    
    b_inv = np.linalg.pinv(b_mat)
    
    P = b_inv[:,0].reshape(im_shape)
    G2D = fftconvolve(P, PSF, 'same')
    G_rad = np.zeros(psf_kernel_rad_px)
    G_rad_temp = radial_profile(G2D, np.floor(np.divide(im_shape, 2)) + 1)
    G_rad[0:len(G_rad_temp)] = G_rad_temp
    
    
    
    """Allocate some variables for storing results"""
    snr_spectra = []
    frc_spectra = []
    Ps_spectra = []
    exp_emissions = []
    exp_variances = []
    mean_switches = []
    Pon = []
    
    """ Set sample properties """
    labelling_factor = 5
    input_power = labelling_factor**2*6.1
    K = labelling_factor*3.19
    NA = 1.4
    n = 1.51
    """Objective collection efficiency, in math of publication, this is
    incorporated into the PSF"""
    obj_ce = conversions.NA2CollEff(NA, n)
    
    """Below, OS is for OFF_Saturation, determining how much power is in the 
    illumination pattern that switches the fluorophores OFF"""
    for OS in OFF_sat_arr:
        E_p_OFF = OS / Poff_488 # OS is translated into maximum expected photons of OFF-light
        """Calculate Pon2 representing the probability of fluorophores being ON 
        after applying the OFF illumination"""
        Pon2 = expected_Pon(Pon1, E_p_OFF*Pon_488*OFFSwitch_ill, E_p_OFF*Poff_488*OFFSwitch_ill, T_off_switch)
        """Save the 2D ON-probabilities in Pon array """
        Pon.append(expandDimensions(np.linspace(0, psf_kernel_rad_px-1, psf_kernel_rad_px), Pon2, half_side))
        """ Iterate over different exposure/observation times"""
        for T in T_exps:    
            print(T)
            """ Calculate the kernels (expectation value and variance) for these specific parameters,
            "kernels" here correspond to E[P] and Var[P] in publication"""
            #N_switches below still contains spatial informations i.e. exp nr of switches as radial coordinate
            kernels, m, N_switches = make_kernels_detection(500000, QE, obj_ce, Pon2, E_p_RO, RO_ill, Pon_488, Poff_488, Pfl_488, T)
            
            
            kernels[0] = np.multiply(kernels[0], G_rad) # As described in eq (17) and (18) in publication
            kernels[1] = np.multiply(kernels[1], np.abs(G_rad))
            """Expand kernels from radial function to symmetric 2D funtion in order to allow
            taking the 2D Fourier transform """
            kernels2d = np.array([expandDimensions(np.linspace(0, psf_kernel_rad_px-1, psf_kernel_rad_px), kernels[i], half_side) for i in range(len(kernels))])
            
            """ Fourier transform kernels """
            ft_kernels2d = np.abs(np.fft.fft2(kernels2d))
            
            """ SNR-spectra is not really relevant now"""
            snr_spectra2d = np.divide(ft_kernels2d[0], np.sqrt(ft_kernels2d[1,0,0]+2*ro_var))
            
            """Calculate the power of signal and power of noise """
            Ps = input_power*np.power(np.abs(ft_kernels2d[0]), 2) # Denominator of eq (35) in publication
            Pn = K*ft_kernels2d[1,0,0] + ro_var # NUmerator of eq (35) in publication
            frc_spectra2d = np.divide(Ps, np.add(Ps, Pn)) # Eq (35)
            
            
            """ Below is to again calculate the radial profile of the different spectra """
            center_ind = int(np.ceil(np.divide(np.subtract(snr_spectra2d.shape, 1), 2))) #gives index of zero-frequencu component in unshifted fft spectra
            rad_spectra = radial_profile_of_raw_fft(snr_spectra2d, center_ind)
            rad_frc = radial_profile_of_raw_fft(frc_spectra2d, center_ind)
            rad_Ps = radial_profile_of_raw_fft(Ps, center_ind)
            snr_spectra.append(rad_spectra)
            frc_spectra.append(rad_frc)
            Ps_spectra.append(rad_Ps)
            exp_emissions.append(kernels2d[0])
            exp_variances.append(kernels2d[1])
            print(N_switches.shape)
            mean_switches.append(N_switches.mean(axis=0))
            
    Pon = np.asarray(Pon)
    snr_spectra = np.asarray(snr_spectra).reshape([len(OFF_sat_arr), len(T_exps), len(rad_spectra)])
    frc_spectra = np.asarray(frc_spectra).reshape([len(OFF_sat_arr), len(T_exps), len(rad_spectra)])
    Ps_spectra = np.asarray(Ps_spectra).reshape([len(OFF_sat_arr), len(T_exps), len(rad_spectra)])
    exp_emissions = np.asarray(exp_emissions).reshape([len(OFF_sat_arr), len(T_exps), full_side, full_side])
    exp_variances = np.asarray(exp_variances).reshape([len(OFF_sat_arr), len(T_exps), full_side, full_side])
    mean_switches = np.asarray(mean_switches)
    
    per = px_size_nm / freq_arr
    
    return locals()
    

def analyse_spectra(spectra=None, path=None):
    
    if spectra is None:
        spectra = DataIO_tools.load_data(path=path)
    
    opt_t_exp = []
    opt_OFFs = []
    max_snr = []
    for i in range(spectra.shape[2]):
        s = spectra[:,:,i]
        cmax = np.mod(s.argmax(), s.shape[1])
        rmax = s.argmax() // s.shape[1]
    
        max_snr.append(s[rmax, cmax])
        if s[rmax, cmax] < 0.01:
            opt_t_exp.append(np.nan)
            opt_OFFs.append(np.nan)
        else:
            opt_t_exp.append(cmax)
            opt_OFFs.append(rmax)
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(range(spectra.shape[2]), opt_t_exp, opt_OFFs, marker='o')

    return max_snr, opt_t_exp, opt_OFFs


def show_isosurface(spectra, level):
    
    verts, faces, normals, values = measure.marching_cubes_lewiner(snr_spectra, level)#, spacing=(0.05, 0.05, 0.05))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                cmap='Spectral', lw=1)
    
    return fig, ax

def make_animation(spectra, level_range, frames):
    current_frame = 0
    for i in range(frames):
        level = level_range[0] - i*(level_range[0]-level_range[1])/(frames-1)
        current_frame += 1
        fig, ax = show_isosurface(spectra, level)
        ax.view_init(45, -45)
#        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
#        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
#        ax.set_xticklabels(['']*len(ax.get_xticklabels()))
#        ax.set_yticklabels(['']*len(ax.get_yticklabels()))
#        ax.set_zticklabels(['']*len(ax.get_zticklabels()))
#        ax.set_aspect(1)
        print("Frame:", current_frame, 'level:', level)
        plt.savefig('./Animation/animation_1_frame_%06i.png'%current_frame,
                    bbox_inches='tight', dpi=500)
        plt.close(fig)

def make_row_plot_anim(im, crop_sides_px=0):
    
    for i in range(len(im)):
        fig = plt.figure()
        plt.plot(im[i, crop_sides_px:-crop_sides_px])
        plt.ylim(0)
        plt.savefig('./Animation/animation_frame_%03i.jpg'%(i+1),
                    bbox_inches='tight', dpi=500)
        plt.close(fig)

def make_im_stack_anim(stack, crop_frame_px=0):
    
    for i in range(len(stack)):
        fig = plt.figure()
        plt.imshow(stack[i, crop_frame_px:-crop_frame_px, crop_frame_px:-crop_frame_px])
        plt.savefig('./Animation/animation_frame_%03i.jpg'%(i+1),
                    bbox_inches='tight', dpi=500)
        plt.close(fig)
        

def get_max_freqs(spectra_slice, df, contrast, interp_fac=False):
    
    max_freq = []
    for i in range(spectra_slice.shape[0]):
        spectra_vec = spectra_slice[i]
        if interp_fac:
            f_orig = df*np.linspace(0, (len(spectra_vec)-1), len(spectra_vec))
            interp_spectra = interp1d(f_orig, spectra_vec)
            freqs = df*np.linspace(0, (len(spectra_vec)-1), interp_fac*len(spectra_vec))
            for i, f in enumerate(freqs):
                if interp_spectra(f) < contrast:
                    max_freq.append(freqs[i-1])
                    break
        else:        
            for i, f in enumerate(f_orig):
                if spectra_vec(i) < contrast:
                    max_freq.append(f_orig[i-1])
                    break
    
    return max_freq, np.divide(1, max_freq)

if __name__ == '__main__':
    
    """Run analysis and save relevant variables"""
#    OSs = [5.1*2, 1.75*5.1, 1.5*5.1, 1.25*5.1, 5.1, 5.1/2, 5.1/4, 5.1/8, 5.1/16, 5.1/32]#np.array([5.1])#np.linspace(0, 4, 10)
    OSs = [5.1]
    locals_from_analysis = simulate(OSs)
    snr_spectra = locals_from_analysis['snr_spectra']
    frc_spectra = locals_from_analysis['frc_spectra']
    exp_em = locals_from_analysis['exp_emissions']
    exp_em_norm = []
    for i in range(len(exp_em)):
        exp_em_norm.append(exp_em[i]/exp_em[i].max())
    exp_em_norm = np.asarray(exp_em_norm)
    
    exp_var = locals_from_analysis['exp_variances']
    ac = locals_from_analysis['ac']
    tps_ac = locals_from_analysis['tps_ac']
    mean_switches = locals_from_analysis['mean_switches']
    kernels2d = locals_from_analysis['kernels2d']
    ft_kernels2d = locals_from_analysis['ft_kernels2d']
    hE = kernels2d[0]
    hVar = kernels2d[1]
    Pn_last = locals_from_analysis['Pn']
    Ps_last = locals_from_analysis['Ps']
#    with open('./Results/locals_from_sim.pickle', 'wb') as variablefile:
#        pickle.dump(locals_from_analysis, variablefile)
    
    locals().update(locals_from_analysis)
    cutoff_nm = 100
    cutoff_freq = px_size_nm / cutoff_nm
    px_index = cutoff_freq / df
    mf = []
    mr = []
    for i in range(frc_spectra.shape[0]):
        f, r = get_max_freqs(frc_spectra[i], df, 0.143, interp_fac=200)
        mf.append(f)
        mr.append(r)
#    plt.imshow(snr_spectra, vmin=0, vmax=2)
#    plt.axvline(x=px_index, color='r', linestyle='--')
    
    DataIO_tools.save_data(snr_spectra.astype(np.float32), './Results/SNR_spectra.tif')
    DataIO_tools.save_data(exp_em.astype(np.float32), './Results/Exp_Em.tif')
    DataIO_tools.save_data(exp_em_norm.astype(np.float32), './Results/Exp_Em_norm.tif')
    DataIO_tools.save_data(exp_var.astype(np.float32), './Results/Exp_Var.tif')
    
    
    
    