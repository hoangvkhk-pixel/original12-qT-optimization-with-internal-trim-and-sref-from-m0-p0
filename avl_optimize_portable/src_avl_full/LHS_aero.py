#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:40:30 2022

@author: lab104
"""

import numpy as np
from smt.sampling_methods import LHS

des_var = np.array([[4., 15.],            # Forward f_w aspect ratio          0
                    [1.,  3.],            # Forward f_w taper ratio           1
                    [4., 15.],            # Aftward a_w AR                    2
                    [1., 3.],             # Aftward a_w taper                 3
                    [0.2, 0.5],           # Aftward a_w S_rel                 4
                    [4., 7.],             # Aftward a_w x_loc                 5
                    [0., 15.],            # Sweep of main wing                6
                    [0.,  10.],           # Dihedral of main wing             7
                    [30.,  50.],          # Dihedral_2                        8
                    [2., 4.],             # Vertical v_w aspect ratio         9
                    [0.1, 0.3],           # Vertical v_w S_rel                10               
                    [0., 0.],             # Number scheme base on fuse        11
                    [1., 1.],             # Number scheme base on vertical    12 
                    [30., 90.],           # Velocity                          13
                    [5., 110.],           # Wing loading                      14
                    [500., 3000.]])       # Initial mass                      15       


sampling = LHS(xlimits=des_var)
variable = des_var[:,0].size
const = 2
init_pop = 10 * (variable-const)
x = sampling(init_pop)

print(x.shape)
print(x.size) 
np.save(f'initial_population_{init_pop}', x)
