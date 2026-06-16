#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:40:30 2022

@author: lab104
"""

import numpy as np
from smt.sampling_methods import LHS

des_var = np.array([[4, 15],            # Forward f_w aspect ratio          0
                    [0, 50],            # Forward f_w sweep ratio           1
                    [1,  3],            # Forward f_w taper ratio           2
                    [-4, 4],            # Forward f_w y_rot                 3
                    [4, 15],            # Aftward a_w AR                    4
                    [-50, 50],          # Aftward a_w sweep                 5
                    [1, 3],             # Aftward a_w taper                 6
                    [-4, 4],            # Aftward a_w y_rot                 7
                    [0.1, 0.9],         # Aftward a_w S_rel                 8
                    [3, 6],             # Aftward a_w x_loc                 9
                    [-5,  0],           # Twist                             10
                    [0,  30],           # Dihedral                          11
                    [30,  50],          # Dihedral_2                        12
                    [-0.5,  0.5],       # z_location                        13
                    [3, 5],             # Vertical v_w aspect ratio         14
                    [0, 30],            # Vertical v_w sweep ratio          15
                    [1,  3],            # Vertical v_w taper ratio          16
                    [0.02, 0.15],       # Vertical v_w S_rel                17
                    [3, 6],             # Vertical v_w x_loc                18
                    [10, 60],           # Area reference                    19
                    [30, 90],           # Velocity                          20
                    [-10, 10],          # Angle attack                      21
                    [-5, 5],            # Angle sideslip                    22
                    [0, 1],             # Scheme base on fuse               23
                    [0, 1]])            # Scheme base on vertical           24               

                          

sampling = LHS(xlimits=des_var)
variable = des_var[:,0].size
const = 0
init_pop = 1000 * (variable-const)
x = sampling(init_pop)

print(x.shape)
print(x.size) 
np.save(f'initial_population_{init_pop}', x)
