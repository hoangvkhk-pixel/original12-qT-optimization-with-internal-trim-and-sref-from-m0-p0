# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:03:39 2023

@author: HUNG HOANG
"""
import AeroCoeff_AVL as ac
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed

def calc_coeff(des_par, kk):
    
    f_aspect = des_par[kk][0]
    f_sweep = des_par[kk][1]
    f_taper = des_par[kk][2]
    f_delta = des_par[kk][3]
       
    a_aspect = des_par[kk][4]
    a_sweep = des_par[kk][5]
    a_taper = des_par[kk][6]
    a_delta = des_par[kk][7]
    a_S = des_par[kk][8]
    a_L = des_par[kk][9]      
    
    if a_S <= 0.5:
        f_twist = des_par[kk][10]
        f_dihedral = des_par[kk][11]
        a_twist = 0
        a_dihedral = 0
    else:
        a_twist = des_par[kk][10]
        a_dihedral = des_par[kk][11]
        f_twist = 0
        f_dihedral = 0
        
    v_aspect = 3.5
    v_sweep = 30
    v_taper = 1.5
    v_twist = 0
    v_dihedral = 90
    v_delta = 0
    v_S = des_par[kk][12]
    v_L = des_par[kk][13]
    
    S_ref =  des_par[kk][14]
    V = des_par[kk][15]
    alpha = des_par[kk][16]
    beta = des_par[kk][17]
    
    a_w_area = a_S*S_ref
    f_w_area = S_ref - a_w_area
    v_w_area = v_S*S_ref
    f_w_geo_char = ac.input_lift_surface_data(f_aspect, f_sweep, f_taper, f_twist, f_dihedral, f_delta, f_w_area, 0)
    a_w_geo_char = ac.input_lift_surface_data(a_aspect, a_sweep, a_taper, a_twist, a_dihedral, a_delta, a_w_area, a_L)
    v_w_geo_char = ac.input_lift_surface_data(v_aspect, v_sweep, v_taper, v_twist, v_dihedral, v_delta, v_w_area, v_L)
    
    if f_w_area >= a_w_area:
        mac = ac.ref_dim_lift_surface(f_w_geo_char)  
    else:
        mac = ac.ref_dim_lift_surface(a_w_geo_char)
    a_w_loc = a_L*mac
    v_w_loc = v_L*mac
    f_span, f_root_chord, f_tip_chord = ac.lift_surface_def(f_w_geo_char)
    a_span, a_root_chord, a_tip_chord = ac.lift_surface_def(a_w_geo_char)
    v_span, v_root_chord, v_tip_chord = ac.lift_surface_def(v_w_geo_char)

    fuse_AR = 10
    if a_w_loc >= v_w_loc:
        l_cylinder = a_w_loc
        l_nose = 0.3*l_cylinder
        l_tail = a_root_chord + 0.1*l_cylinder
        l_fuse = l_nose + l_cylinder + l_tail
        d_fuse = l_fuse/fuse_AR
        AR_nose = l_nose/d_fuse
        AR_C = l_cylinder/d_fuse
        AR_tail = l_tail/d_fuse
    else:
        l_cylinder = v_w_loc
        l_nose = 0.3*l_cylinder
        l_tail = v_root_chord + 0.1*l_cylinder
        l_fuse = l_nose + l_cylinder + l_tail
        d_fuse = l_fuse/fuse_AR
        AR_nose = l_nose/d_fuse
        AR_C = l_cylinder/d_fuse
        AR_tail = l_tail/d_fuse

    fuse_geo_char = ac.input_body_data(AR_nose, AR_C, AR_tail, d_fuse, d_fuse, 30, 9, -l_nose)
    ac.avl_run(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, kk)
    
    return  0
   

def save_Dataset(Dataset):
    name_columns = ['f_w_aspect', 'f_w_sweep', 'f_w_taper', 'f_w_delta',
                    'a_w_aspect', 'a_w_sweep', 'a_w_taper', 'a_w_delta', 'a_S', 'a_L',
                    'twist', 'dihedral',
                    'v_S', 'v_L',
                    'S_ref','V', 'alpha', 'beta', 
                    'cx','cy', 'mx', 'my', 'mz']
    data_frame = pd.DataFrame(Dataset, columns = name_columns)
    return data_frame

def vector_info(calc):
    v_info = np.array([calc[0], calc[1], calc[2], calc[3], calc[4]])
    # cx, cy, mx, my, mz
    v_info = v_info.transpose()
    return v_info

def multijob(des_par, cores, init_pop, usr_path):
    for i in range(init_pop):  
        if os.path.exists(usr_path + f'/Creat_png_AVL_{i}'):
            pass
        else:
            os.mkdir(usr_path + f'/Creat_png_AVL_{i}')
            os.mkdir(usr_path + f'/Creat_png_AVL_{i}/tmp/')
            
    Parallel(n_jobs=cores)(delayed(calc_coeff)(des_par,kk) for kk in range(init_pop))
       
    return 0











