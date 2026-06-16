# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:03:39 2023

@author: HUNG HOANG
"""

import AeroCoeff_AVL as ac
import numpy as np
import math
import pandas as pd
import os
from joblib import Parallel, delayed

def geometry(des_par, kk):
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
        a_dihedral = des_par[kk][12]
        f_z_loc = des_par[kk][13]
        a_z_loc = 0
    else:
        a_twist = des_par[kk][10]
        a_dihedral = des_par[kk][11]
        f_twist = 0
        f_dihedral = 0
        a_z_loc = des_par[kk][13]
        f_z_loc = 0
    if a_S <= 0.5 and a_dihedral >= 40:
        v_aspect = 0
        v_sweep = 0
        v_taper = 0
        v_twist = 0
        v_dihedral = 0
        v_delta = 0
        v_S = 0
        v_L = 0
    else:
        v_aspect = des_par[kk][14]
        v_sweep = des_par[kk][15]
        v_taper = des_par[kk][16]
        v_twist = 0
        v_dihedral = 90
        v_delta = 0
        v_S = des_par[kk][17]
        v_L = des_par[kk][18]
    
    nose_f_aspect = des_par[kk][19]
    center_f_aspect = des_par[kk][20]
    tail_f_aspect = des_par[kk][21]
    f_diameter = des_par[kk][22]
    l_nose = nose_f_aspect*f_diameter
    
    S_ref =  des_par[kk][23]
    V = des_par[kk][24]
    alpha = des_par[kk][25]
    beta = des_par[kk][26]
    
    a_w_area = a_S*S_ref
    f_w_area = S_ref - a_w_area
    v_w_area = v_S*S_ref
    f_w_geo_char = ac.input_lift_surface_data(f_aspect, f_sweep, f_taper, f_twist, f_dihedral, f_delta, f_w_area, 0, f_z_loc)
    a_w_geo_char = ac.input_lift_surface_data(a_aspect, a_sweep, a_taper, a_twist, a_dihedral, a_delta, a_w_area, a_L, a_z_loc)
    v_w_geo_char = ac.input_lift_surface_data(v_aspect, v_sweep, v_taper, v_twist, v_dihedral, v_delta, v_w_area, v_L)
    fuse_geo_char = ac.input_body_data(nose_f_aspect, center_f_aspect, tail_f_aspect, f_diameter, f_diameter, 30, 9, -l_nose)
    
    d_e_body, l_body, S_body, Sxq_body = ac.body_def(fuse_geo_char)
    if f_w_area >= a_w_area:
        mac = ac.ref_dim_lift_surface(f_w_geo_char)  
    else:
        mac = ac.ref_dim_lift_surface(a_w_geo_char)
    a_w_loc = a_L*mac
    v_w_loc = v_L*mac
    f_z_loc = f_w_geo_char[8]*d_e_body
    a_z_loc = a_w_geo_char[8]*d_e_body
    
    f_span, f_root_chord, f_tip_chord = ac.lift_surface_def(f_w_geo_char)
    f_x, f_y, f_z, f_chord, f_Ain = ac.lift_surface_sec(f_w_geo_char)
    f_x_loc_root_chord = f_x[0] 
    f_y_loc_root_chord = f_y[0]
    f_z_loc_root_chord = f_z[0] + f_z_loc
    
    f_x_loc_tip_chord = f_x[1]
    f_y_loc_tip_chord = f_y[1]
    f_z_loc_tip_chord = f_z[1] + f_z_loc
    f_w_delta = f_delta
    f_w_twist = f_Ain[1]

    a_span, a_root_chord, a_tip_chord = ac.lift_surface_def(a_w_geo_char)
    a_x, a_y, a_z, a_chord, a_Ain = ac.lift_surface_sec(a_w_geo_char)
    a_x_loc_root_chord = a_x[0] + a_w_loc
    a_y_loc_root_chord = a_y[0]
    a_z_loc_root_chord = a_z[0] + a_z_loc
    
    a_x_loc_tip_chord = a_x[1] + a_w_loc
    a_y_loc_tip_chord = a_y[1]
    a_z_loc_tip_chord = a_z[1] + a_z_loc
    
    a_w_delta = a_delta
    a_w_twist = a_Ain[1]
    
    v_span, v_root_chord, v_tip_chord = ac.lift_surface_def(v_w_geo_char)
    v_x, v_y, v_z, v_chord, v_Ain = ac.lift_surface_sec(v_w_geo_char)
    v_x_loc_root_chord = v_x[0] + v_w_loc
    v_y_loc_root_chord = v_y[0]
    v_z_loc_root_chord = v_z[0]
    
    v_x_loc_tip_chord = v_x[1] + v_w_loc
    v_y_loc_tip_chord = v_y[1]
    v_z_loc_tip_chord = v_z[1]  
                                                      
    
    return f_w_delta, f_w_twist, f_root_chord, f_tip_chord,f_x_loc_root_chord,f_y_loc_root_chord,f_z_loc_root_chord,f_x_loc_tip_chord,f_y_loc_tip_chord,f_z_loc_tip_chord,\
        a_w_delta, a_w_twist, a_root_chord, a_tip_chord,a_x_loc_root_chord,a_y_loc_root_chord,a_z_loc_root_chord,a_x_loc_tip_chord,a_y_loc_tip_chord,a_z_loc_tip_chord,\
            v_root_chord, v_tip_chord,v_x_loc_root_chord,v_y_loc_root_chord,v_z_loc_root_chord,v_x_loc_tip_chord,v_y_loc_tip_chord,v_z_loc_tip_chord

def save_output_DataFrame_FreeCAD(plane):
    name_columns = ['f_w_delta', 'f_w_twist', 'f_root_chord', 'f_tip_chord','f_x_loc_root_chord','f_y_loc_root_chord','f_z_loc_root_chord','f_x_loc_tip_chord','f_y_loc_tip_chord','f_z_loc_tip_chord',
        'a_w_delta', 'a_w_twist', 'a_root_chord', 'a_tip_chord','a_x_loc_root_chord','a_y_loc_root_chord','a_z_loc_root_chord','a_x_loc_tip_chord','a_y_loc_tip_chord','a_z_loc_tip_chord',
            'v_root_chord', 'v_tip_chord','v_x_loc_root_chord','v_y_loc_root_chord','v_z_loc_root_chord','v_x_loc_tip_chord','v_y_loc_tip_chord','v_z_loc_tip_chord']
    data_frame = pd.DataFrame(plane, columns = name_columns)
    return data_frame

def vector_info_FreeCAD(calc):
    v_info = np.array([calc[0], calc[1], calc[2], calc[3], calc[4], calc[5], calc[6], calc[7], calc[8], calc[9], 
                       calc[10], calc[11], calc[12], calc[13], calc[14], calc[15], calc[16], calc[17], calc[18], calc[19],
                       calc[20], calc[21], calc[22], calc[23], calc[24], calc[25], calc[26], calc[27]])
    #f_w_delta, f_w_twist, f_root_chord, f_tip_chord,f_x_loc_root_chord,f_y_loc_root_chord,f_z_loc_root_chord,f_x_loc_tip_chord,f_y_loc_tip_chord,f_z_loc_tip_chord,\
    #a_w_delta, a_w_twist, a_root_chord, a_tip_chord,a_x_loc_root_chord,a_y_loc_root_chord,a_z_loc_root_chord,a_x_loc_tip_chord,a_y_loc_tip_chord,a_z_loc_tip_chord,\
    #v_root_chord, v_tip_chord,v_x_loc_root_chord,v_y_loc_root_chord,v_z_loc_root_chord,v_x_loc_tip_chord,v_y_loc_tip_chord,v_z_loc_tip_chord
    v_info = v_info.transpose()
    return v_info


def multijob(des_par, cores, init_pop, usr_path):
    for i in range(init_pop):
        if os.path.exists(usr_path + f'/Creat_png_AVL_{i}'):
            pass
        else:
            os.mkdir(usr_path + f'/Creat_png_AVL_{i}')
            os.mkdir(usr_path + f'/Creat_png_AVL_{i}/tmp/')
    
    t = Parallel(n_jobs=cores)(delayed(geometry(des_par, kk)) for kk in range(init_pop))
    
    f_w_delta = np.array([])
    f_w_twist = np.array([])
    f_root_chord = np.array([])
    f_tip_chord = np.array([])
    f_x_loc_root_chord = np.array([])
    f_y_loc_root_chord = np.array([])
    f_z_loc_root_chord = np.array([])
    f_x_loc_tip_chord = np.array([])
    f_y_loc_tip_chord = np.array([])
    f_z_loc_tip_chord = np.array([])
    
    a_w_delta = np.array([])
    a_w_twist = np.array([])
    a_root_chord = np.array([])
    a_tip_chord = np.array([])
    a_x_loc_root_chord = np.array([])
    a_y_loc_root_chord = np.array([])
    a_z_loc_root_chord = np.array([])
    a_x_loc_tip_chord = np.array([])
    a_y_loc_tip_chord = np.array([])
    a_z_loc_tip_chord = np.array([])
    
    v_root_chord = np.array([])
    v_tip_chord = np.array([])
    v_x_loc_root_chord = np.array([])
    v_y_loc_root_chord = np.array([])
    v_z_loc_root_chord = np.array([])
    v_x_loc_tip_chord = np.array([])
    v_y_loc_tip_chord = np.array([])
    v_z_loc_tip_chord = np.array([])

    for kk in range(init_pop):
        
        f_w_delta = np.append(f_w_delta, t[kk][0])
        f_w_twist = np.append(f_w_twist, t[kk][1])
        f_root_chord = np.append(f_root_chord, t[kk][2])
        f_tip_chord = np.append(f_tip_chord, t[kk][3])
        f_x_loc_root_chord = np.append(f_x_loc_root_chord, t[kk][4])
        f_y_loc_root_chord = np.append(f_y_loc_root_chord, t[kk][5])
        f_z_loc_root_chord = np.append(f_z_loc_root_chord, t[kk][6])
        f_x_loc_tip_chord = np.append(f_x_loc_tip_chord, t[kk][7])
        f_y_loc_tip_chord = np.append(f_y_loc_tip_chord, t[kk][8])
        f_z_loc_tip_chord = np.append(f_z_loc_tip_chord, t[kk][9])
        
        a_w_delta = np.append(a_w_delta, t[kk][10])
        a_w_twist = np.append(a_w_twist, t[kk][11])
        a_root_chord = np.append(a_root_chord, t[kk][12])
        a_tip_chord = np.append(a_tip_chord, t[kk][13])
        a_x_loc_root_chord = np.append(a_x_loc_root_chord, t[kk][14])
        a_y_loc_root_chord = np.append(a_y_loc_root_chord, t[kk][15])
        a_z_loc_root_chord = np.append(a_z_loc_root_chord, t[kk][16])
        a_x_loc_tip_chord = np.append(a_x_loc_tip_chord, t[kk][17])
        a_y_loc_tip_chord = np.append(a_y_loc_tip_chord, t[kk][18])
        a_z_loc_tip_chord = np.append(a_z_loc_tip_chord, t[kk][19])
        
        v_root_chord = np.append(v_root_chord, t[kk][20])
        v_tip_chord = np.append(v_tip_chord, t[kk][21])
        v_x_loc_root_chord = np.append(v_x_loc_root_chord, t[kk][22])
        v_y_loc_root_chord = np.append(v_y_loc_root_chord, t[kk][23])
        v_z_loc_root_chord = np.append(v_z_loc_root_chord, t[kk][24])
        v_x_loc_tip_chord = np.append(v_x_loc_tip_chord, t[kk][25])
        v_y_loc_tip_chord = np.append(v_y_loc_tip_chord, t[kk][26])
        v_z_loc_tip_chord = np.append(v_z_loc_tip_chord, t[kk][27])

    return f_w_delta, f_w_twist, f_root_chord, f_tip_chord,f_x_loc_root_chord,f_y_loc_root_chord,f_z_loc_root_chord,f_x_loc_tip_chord,f_y_loc_tip_chord,f_z_loc_tip_chord,\
        a_w_delta, a_w_twist, a_root_chord, a_tip_chord,a_x_loc_root_chord,a_y_loc_root_chord,a_z_loc_root_chord,a_x_loc_tip_chord,a_y_loc_tip_chord,a_z_loc_tip_chord,\
            v_root_chord, v_tip_chord,v_x_loc_root_chord,v_y_loc_root_chord,v_z_loc_root_chord,v_x_loc_tip_chord,v_y_loc_tip_chord,v_z_loc_tip_chord
