# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:54:20 2017

@author: sajid

Based on the MATLAB code by Michael Wojcik

M. van Heela, and M. Schatzb, "Fourier shell correlation threshold
criteria," Journal of Structural Biology 151, 250-262 (2005)

"""

#importing required libraries

import numpy as np
import spin_average as sa
import matplotlib.pyplot as plt

def FRC(i1,i2,disp=0,thresh=0.143):
    '''
    Check whether the inputs dimensions match and the images are square
    '''
    if ( np.shape(i1) != np.shape(i2) ) :
        print('input images must have the same dimensions')
    if ( np.shape(i1)[0] != np.shape(i1)[1]) :
        print('input images must be squares')
    I1 = np.fft.fftshift(np.fft.fft2(i1))
    I2 = np.fft.fftshift(np.fft.fft2(i2))
    '''
    I1 and I2 store the DFT of the images to be used in the calcuation for the FSC
    '''
    C  = sa.spinavej(np.multiply(I1,np.conj(I2)))
    C1 = sa.spinavej(np.multiply(I1,np.conj(I1)))
    C2 = sa.spinavej(np.multiply(I2,np.conj(I2)))
    
    FRC = abs(C)/np.sqrt(abs(np.multiply(C1,C2)))
    
    '''
    T is the SNR threshold calculated accoring to the input SNRt, if nothing is given
    a default value of 0.1 is used.
    
    x2 contains the normalized spatial frequencies
    '''
    r = np.arange(1+np.shape(i1)[0]/2)
#    n = 2*np.pi*r
#    n[0] = 1
#    eps = np.finfo(float).eps
#    t1 = np.divide(np.ones(np.shape(n)),n+eps)
#    t2 = SNRt + 2*np.sqrt(SNRt)*t1 + np.divide(np.ones(np.shape(n)),np.sqrt(n))
#    t3 = SNRt + 2*np.sqrt(SNRt)*t1 + 1
#    T = np.divide(t2,t3)
    x1 = np.arange(np.shape(C)[0])/(np.shape(i1)[0])
    x2 = r/(np.shape(i1)[0])   
    
    cFRC = FRC[0:len(x2)-1]
    plt.plot(cFRC)
    above = np.argwhere(cFRC > thresh)[:,0]
    below = np.argwhere(cFRC < thresh)[:,0]
    crossing_reg = np.array([x2[below[0]], x2[above[-1]]])
    
    '''
    If the disp input is set to 1, an output plot is generated. 
    '''
    if disp != 0 :
        plt.plot(x1,FRC,label = 'FRC')
        plt.plot(x2,thresh*np.ones_like(x2),'--',label = 'Threshold SNR = '+str(thresh))
        plt.xlim(0,0.5)
        plt.legend()
        plt.xlabel('Spatial Frequency/Nyquist')
        plt.show()

    return FRC, crossing_reg
    
    
    
    
    
    
    
    
    
    