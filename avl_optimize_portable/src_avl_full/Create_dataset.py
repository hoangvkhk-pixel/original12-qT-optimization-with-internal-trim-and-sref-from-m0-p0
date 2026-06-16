# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:25:59 2023

@author: HUNG HOANG
"""

# Shade Optimzator
import numpy as np
import geometry_UAV as cac
import time
import os
import shutil

usr_path = os.path.expanduser('~')
gen_path = os.path.join(usr_path, 'Creat_png_AVL_0/gen/')
data_path = os.path.join(gen_path, 'Data/')

cores = 10
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

variable = des_var[:,0].size
const = 0
init_pop = 1000 * (variable-const)
start = time.time()

#%% Load initial LHS
Pg = np.load(f'initial_population_{init_pop}.npy')

#%% Calculate aerodynamica coefficients

calc = cac.multijob(Pg, cores, init_pop, usr_path)
info_to_FreeCAD = cac.vector_info_FreeCAD(calc)
info_to_FreeCAD = cac.save_output_DataFrame_FreeCAD(info_to_FreeCAD)
info_to_FreeCAD.to_excel(gen_path+'info_to_FreeCAD' + '.xlsx')

info_geometry = cac.vector_info_geometry(calc)
info_geometry = cac.save_geomtry_DataFrame(info_geometry)
info_geometry.to_excel(gen_path+'info_geometry' + '.xlsx')
for i in range(init_pop-1): 
    shutil.rmtree(usr_path + f'/Creat_png_AVL_{i+1}')
    
end = time.time()


print('\n--------------------------------------------------------------------------------')
print('Total calculation time: {}'.format(end-start))    
print('--------------------------------------------------------------------------------')

