# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:13:07 2024

@author: Aspirant2
"""
import os
import shade as sd
import numpy as np
import time
from pathlib import Path

start = time.time()
MODULE_DIR = Path(__file__).resolve().parent
AUTO_FULL_ROOT = Path(os.environ.get("AUTO_FULL_ROOT", str(MODULE_DIR.parent / "runs")))
GEN_ROOT = AUTO_FULL_ROOT / "Auto_full_0" / "gen"
AUTO_FULL_ROOT.mkdir(parents=True, exist_ok=True)
GEN_ROOT.mkdir(parents=True, exist_ok=True)

des_var = np.array([[4., 15.],            # Forward f_w aspect ratio          0
                    [1.,  3.],            # Forward f_w taper ratio           1
                    [4., 15.],            # Aftward a_w AR                    2
                    [1., 3.],             # Aftward a_w taper                 3
                    [0.2, 0.8],           # Aftward a_w S_rel                 4
                    [4., 7.],             # Aftward a_w x_loc                 5
                    [0., 15.],            # Sweep of main wing                6
                    [0.,  10.],           # Dihedral of main wing             7
                    [30.,  50.],          # Dihedral_2                        8
                    [2., 4.],             # Vertical v_w aspect ratio         9
                    [0.1, 0.3],           # Vertical v_w S_rel                10               
                    [0., 1.],             # Number scheme base on fuse        11
                    [0., 1.],             # Number scheme base on vertical    12 
                    [30., 90.],           # Velocity                          13
                    [5., 110.],           # Wing loading                      14
                    [500., 3000.]])       # Initial mass                      15 
cores = int(os.environ.get("OPT_CORES", "10"))
const_1 = 0
const_2 = 2
epsilon_1 = 0.3
epsilon_2 = 0.0031
gen_path_0 = str(GEN_ROOT) + os.sep

g = sd.SHADE_algorithm(des_var, const_1, epsilon_1, gen_path_0, cores)

n_1_1_normal, n_1_2_normal, n_1_3_normal,\
    n_2_1_normal, n_2_2_normal, n_2_3_normal,\
        n_3_1_normal, n_3_2_normal, n_3_3_normal,\
            n_1_x_duck, n_2_x_duck, n_3_x_duck = sd.create_folder(g, gen_path_0)
         
if n_1_1_normal > 0:
    gen_path_1 = os.path.join(gen_path_0 + 'normal_1_1/')
    des_var[4,0] = 0.2
    des_var[4,1] = 0.5

    des_var[11,0] = 0.
    des_var[11,1] = 0.

    des_var[12,0] = 0.
    des_var[12,1] = 0.
    g_1_1_normal = sd.SHADE_algorithm(des_var, const_2, epsilon_2, gen_path_1, cores)

if n_1_2_normal > 0:
    gen_path_2 = os.path.join(gen_path_0 + 'normal_1_2/')
    des_var[4,0] = 0.2
    des_var[4,1] = 0.5

    des_var[11,0] = 0.
    des_var[11,1] = 0.

    des_var[12,0] = 0.5
    des_var[12,1] = 0.5
    g_1_2_normal = sd.SHADE_algorithm(des_var, const_2, epsilon_2, gen_path_2, cores)
    
if n_1_3_normal > 0:
    gen_path_3 = os.path.join(gen_path_0 + 'normal_1_3/')
    des_var[4,0] = 0.2
    des_var[4,1] = 0.5

    des_var[11,0] = 0.
    des_var[11,1] = 0.

    des_var[12,0] = 1.
    des_var[12,1] = 1.
    g_1_3_normal = sd.SHADE_algorithm(des_var, const_2, epsilon_2, gen_path_3, cores)
    
if n_2_1_normal > 0:
    gen_path_4 = os.path.join(gen_path_0 + 'normal_2_1/')
    des_var[4,0] = 0.2
    des_var[4,1] = 0.5

    des_var[11,0] = 0.5
    des_var[11,1] = 0.5

    des_var[12,0] = 0.
    des_var[12,1] = 0.
    g_2_1_normal = sd.SHADE_algorithm(des_var, const_2, epsilon_2, gen_path_4, cores)   
    
if n_2_2_normal > 0:
    gen_path_5 = os.path.join(gen_path_0 + 'normal_2_2/')
    des_var[4,0] = 0.2
    des_var[4,1] = 0.5

    des_var[11,0] = 0.5
    des_var[11,1] = 0.5

    des_var[12,0] = 0.5
    des_var[12,1] = 0.5
    g_2_2_normal = sd.SHADE_algorithm(des_var, const_2, epsilon_2, gen_path_5, cores)    
    
if n_2_3_normal > 0:
    gen_path_6 = os.path.join(gen_path_0 + 'normal_2_3/')
    des_var[4,0] = 0.2
    des_var[4,1] = 0.5

    des_var[11,0] = 0.5
    des_var[11,1] = 0.5

    des_var[12,0] = 1.
    des_var[12,1] = 1.
    g_2_3_normal = sd.SHADE_algorithm(des_var, const_2, epsilon_2, gen_path_6, cores)
    
if n_3_1_normal > 0:
    gen_path_7 = os.path.join(gen_path_0 + 'normal_3_1/')
    des_var[4,0] = 0.2
    des_var[4,1] = 0.5

    des_var[11,0] = 1.
    des_var[11,1] = 1.

    des_var[12,0] = 0.
    des_var[12,1] = 0.
    g_3_1_normal = sd.SHADE_algorithm(des_var, const_2, epsilon_2, gen_path_7, cores)
    
if n_3_2_normal > 0:
    gen_path_8 = os.path.join(gen_path_0 + 'normal_3_2/')
    des_var[4,0] = 0.2
    des_var[4,1] = 0.5

    des_var[11,0] = 1.
    des_var[11,1] = 1.

    des_var[12,0] = 0.5
    des_var[12,1] = 0.5
    g_3_2_normal = sd.SHADE_algorithm(des_var, const_2, epsilon_2, gen_path_8, cores)   
  
if n_3_3_normal > 0:
    gen_path_9 = os.path.join(gen_path_0 + 'normal_3_3/')
    des_var[4,0] = 0.2
    des_var[4,1] = 0.5

    des_var[11,0] = 1.
    des_var[11,1] = 1.

    des_var[12,0] = 1.
    des_var[12,1] = 1.
    g_3_3_normal = sd.SHADE_algorithm(des_var, const_2, epsilon_2, gen_path_9, cores)   
  
if n_1_x_duck > 0:
    gen_path_10 = os.path.join(gen_path_0 + 'duck_1_x/')
    des_var[4,0] = 0.5
    des_var[4,1] = 0.8

    des_var[11,0] = 0.
    des_var[11,1] = 0.

    des_var[12,0] = 0.5
    des_var[12,1] = 0.5
    g_1_x_duck = sd.SHADE_algorithm(des_var, const_2, epsilon_2, gen_path_10, cores)     
  
if n_2_x_duck > 0:
    gen_path_11 = os.path.join(gen_path_0 + 'duck_2_x/')
    des_var[4,0] = 0.5
    des_var[4,1] = 0.8

    des_var[11,0] = 0.5
    des_var[11,1] = 0.5

    des_var[12,0] = 0.5
    des_var[12,1] = 0.5
    g_2_x_duck = sd.SHADE_algorithm(des_var, const_2, epsilon_2, gen_path_11, cores)     
  
if n_3_x_duck > 0:
    gen_path_12 = os.path.join(gen_path_0 + 'duck_3_x/')
    des_var[4,0] = 0.5
    des_var[4,1] = 0.8

    des_var[11,0] = 1.
    des_var[11,1] = 1.

    des_var[12,0] = 0.5
    des_var[12,1] = 0.5
    g_3_x_duck = sd.SHADE_algorithm(des_var, const_2, epsilon_2, gen_path_12, cores)      
  
end = time.time()  
print('\n--------------------------------------------------------------------------------')
print('\n Xin chÃºc má»«ng anh HÆ°ng Ä‘áº¹p trai Ä‘Ã£ hoÃ n thÃ nh chÆ°Æ¡ng trÃ¬nh tá»‘i Æ°u hÃ³a')
print('Total calculation time: {}'.format(end-start))    
print('--------------------------------------------------------------------------------')  
  
    
  
    
  
    
  
    
  
    
  
    
