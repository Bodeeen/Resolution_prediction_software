# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:59:05 2020

@author: andreas.boden
"""

import numpy as np
import matplotlib.pyplot as plt
import spin_average
""" Make coordinates of fluorophores on filamentous structure"""

"""Make line"""


def make_random_positions(area_side, f_per_um2=1000):
    f_array_x = []
    f_array_y = []
    
    N = area_side**2*f_per_um2
    
    for n in range(N):
        f_array_x.append(np.random.uniform(0, area_side))
        f_array_y.append(np.random.uniform(0, area_side))
        
    return [f_array_x, f_array_y]

def make_fluorophore_pool_pairs_structure(area_side, d=0.1, N_pairs=100, f_per_pool=10, poisson_labelling=False):
    
    f_array_x = []
    f_array_y = []
    
    for n in range(N_pairs):
        
        random_angle = np.random.uniform(0, 2*np.pi)
        random_point = [np.random.uniform(0, area_side), np.random.uniform(0, area_side)]
        
        f1_x = random_point[0] + (d/2)*np.cos(random_angle)
        f1_y = random_point[1] + (d/2)*np.sin(random_angle)
        
        f2_x = random_point[0] - (d/2)*np.cos(random_angle)
        f2_y = random_point[1] - (d/2)*np.sin(random_angle)
        
        if 0 < f1_x < area_side and 0 < f1_y < area_side and 0 < f2_x < area_side and 0 < f2_y < area_side:
            if poisson_labelling:
                p1 = np.random.poisson(f_per_pool)
                p2 = np.random.poisson(f_per_pool)
            else:
                p1 = p2 = f_per_pool
                
            
            f_array_x += p1*[f1_x] + p2*[f2_x]
            f_array_y += p1*[f1_y] + p2*[f2_y]

                
    return [f_array_x, f_array_y]
        
def make_line_structure(area_side, lines=100, f_per_um=100):
    f_array_x = []
    f_array_y = []

    line_length = 2*np.sqrt(2)*area_side
    N = np.int(line_length*f_per_um)
    
    for L in range(lines):
        random_angle = np.random.uniform(0, 2*np.pi)
        
        random_point = [np.random.uniform(0, area_side), np.random.uniform(0, area_side)]
    
        
        for i in range(N):
            random_distance = np.random.uniform(-np.sqrt(2)*area_side, np.sqrt(2)*area_side)
            
            f_pos = [random_point[0] + random_distance*np.cos(random_angle),
                     random_point[1] + random_distance*np.sin(random_angle)]
            
            if 0 < f_pos[0] < area_side and 0 < f_pos[1] < area_side:
                f_array_x.append(f_pos[0])
                f_array_y.append(f_pos[1])
            
    return [f_array_x, f_array_y]
            


def positions2im(area_side, px_size, f_array_x, f_array_y):
    
    tot_fluorophores = len(f_array_x)

    im_side_px = np.int(np.ceil(area_side/px_size))
    
    im = np.zeros([im_side_px, im_side_px])
    
    for f in range(tot_fluorophores):
        x_px = np.int(f_array_x[f]//px_size)
        y_px = np.int(f_array_y[f]//px_size)
        try:
            im[y_px, x_px] += 1
        except:
            pass
        
    return im

if __name__ == '__main__':
    area_side = 5 #um
    f_arrays = make_line_structure(area_side, 100, 500)
    # f_arrays = make_fluorophore_pool_pairs_structure(area_side, 0.5, 500, 10, True)
    # f_arrays = make_random_positions(area_side, 10000)
    im = positions2im(area_side, 0.02, f_arrays[0], f_arrays[1])
    
    plt.figure()                  
    plt.subplot(2,3,1)
    plt.imshow(im)
    ft_im = np.fft.fftshift(np.fft.fft2(im))
    plt.subplot(2,3,2)
    plt.imshow(np.log(np.abs(ft_im)))
    power = np.power(ft_im, 2)
    plt.subplot(2,3,3)
    plt.imshow(np.log(np.abs(power)))
    radial_average = spin_average.spinavej(np.abs(power))
    plt.subplot(2,3,4)
    plt.plot(radial_average)
    plt.subplot(2,3,5)
    plt.plot(radial_average[1:])
    
    