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
    
    a_taper = des_par[kk][6]
    a_delta = des_par[kk][7]
    a_S = des_par[kk][8]
    a_L = des_par[kk][9]      
    f_S = 1 - a_S
    f_L = 0
    
    scheme_fuse = des_par[23]
    if scheme_fuse <= 1/3:
        scheme_fuse = 1
    elif scheme_fuse > 1/3 and scheme_fuse <= 2/3:
        scheme_fuse = 2
    else:
        scheme_fuse = 3
    scheme_vertical = des_par[24]
    if scheme_vertical <= 1/3:
        scheme_vertical = 1
    elif scheme_vertical > 1/3 and scheme_vertical <= 2/3:
        scheme_vertical = 2
    else:
        scheme_vertical = 3
        
    if a_S <= 0.5:
        f_twist = des_par[kk][10]
        f_dihedral = des_par[kk][11]
        a_twist = 0
        
        if scheme_vertical == 1:
            a_dihedral = 0
        elif scheme_vertical == 2:
            a_dihedral = des_par[kk][12]
        else:
            a_dihedral = -des_par[kk][12]
        
        if a_dihedral == 0:
            a_sweep = des_par[kk][5]
        else:
            a_sweep = 0
        f_z_loc = des_par[kk][13]
        a_z_loc = 0
    else:
        a_sweep = des_par[kk][5]
        a_twist = des_par[kk][10]
        a_dihedral = des_par[kk][11]
        f_twist = 0
        f_dihedral = 0
        a_z_loc = des_par[kk][13]
        f_z_loc = 0
        
    if a_S <= 0.5 and a_dihedral != 0:
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
    
    S_ref =  des_par[kk][19] 
    V = des_par[kk][20]
    alpha = des_par[kk][21]
    beta = des_par[kk][22]
    
    a_area = a_S*S_ref
    f_area = S_ref - a_area
    v_area = v_S*S_ref
    f_geo_char = ac.input_lift_surface_data(f_aspect, f_sweep, f_taper, f_twist, f_dihedral, f_delta, f_area, 0, f_z_loc)
    a_geo_char = ac.input_lift_surface_data(a_aspect, a_sweep, a_taper, a_twist, a_dihedral, a_delta, a_area, a_L, a_z_loc)
    v_geo_char = ac.input_lift_surface_data(v_aspect, v_sweep, v_taper, v_twist, v_dihedral, v_delta, v_area, v_L)
    
    if f_area >= a_area:
        mac = ac.ref_dim_lift_surface(f_geo_char)
    else:
        mac = ac.ref_dim_lift_surface(a_geo_char)

        
    v_span, v_root_chord, v_tip_chord = ac.lift_surface_def(v_geo_char)
    a_span, a_root_chord, a_tip_chord = ac.lift_surface_def(a_geo_char)
    f_span, f_root_chord, f_tip_chord = ac.lift_surface_def(f_geo_char)
    distance_two_fuse = min(a_span, f_span)
    a_loc = a_L*mac
    v_loc = v_L*mac
    
    fuse_diameter = 0.95
    nose_f_aspect = 2

    if scheme_fuse == 1:
        l_center = max(a_loc, v_loc)
        center_f_aspect = l_center/fuse_diameter
        tg1 = a_loc + a_root_chord
        tg2 = v_loc + v_root_chord
        l_tail = max(tg1, tg2) - l_center
        tail_f_aspect = l_tail/fuse_diameter
        n_fuse = 1
        n_vertical = 1
    elif scheme_fuse == 2:
        n_fuse = 1
        l_center = max(a_root_chord, f_root_chord)
        center_f_aspect = l_center/fuse_diameter
        tail_f_aspect = 2
        n_vertical = 2
    else:
        n_fuse = 2
        n_vertical = 2
        l_center = max(a_loc, v_loc)
        center_f_aspect = l_center/fuse_diameter
        chord_i = (a_tip_chord-a_root_chord)*distance_two_fuse/a_span + a_root_chord
        tg1 = a_loc + a_root_chord
        tg2 = v_loc + v_root_chord
        tg3 = a_loc + chord_i + distance_two_fuse*(math.tan(math.radians(a_geo_char[1])))/2
        l_tail = max(tg1, tg2, tg3) - l_center
        tail_f_aspect = l_tail/fuse_diameter
    
    l_nose = nose_f_aspect*fuse_diameter
    l_center = center_f_aspect*fuse_diameter
    l_tail = tail_f_aspect*fuse_diameter
    l_nose_center = l_center + l_nose
    
    v_geo_char = ac.input_lift_surface_data(v_aspect, v_sweep, v_taper, v_twist, v_dihedral, v_delta, v_area, v_L, 0, n_vertical)
    
    fuse_geo_char = ac.input_body_data(nose_f_aspect, center_f_aspect, tail_f_aspect, fuse_diameter, fuse_diameter)
    d_e_body, l_body, S_body, Sxq_body = ac.body_def(fuse_geo_char)
    
    if a_S >= 0.5 and l_nose_center < a_loc: #Xem lai dieu kien nay(hinh 2)
        fuse_x_loc = a_loc - l_nose
    else:
        fuse_x_loc = -l_nose
    
    if f_area >= a_area:
        if (distance_two_fuse/2*np.tan(math.radians(f_geo_char[4])) + f_z_loc) > (d_e_body/2):
            f_geo_char[4] = math.atan(d_e_body/distance_two_fuse)
    else:
        if (distance_two_fuse/2*np.tan(math.radians(a_geo_char[4])) + a_z_loc) > (d_e_body/2):
            a_geo_char[4] = math.atan(d_e_body/distance_two_fuse)
      
    fuse_geo_char = ac.input_body_data(nose_f_aspect, center_f_aspect, tail_f_aspect, fuse_diameter, fuse_diameter, fuse_x_loc, n_fuse)

    f_z_loc = f_geo_char[8]*d_e_body
    a_z_loc = a_geo_char[8]*d_e_body

    v_span, v_root_chord, v_tip_chord = ac.lift_surface_def(v_geo_char)
    v_z_loc = 0
    v_x, v_y, v_z, v_chord, v_Ain = ac.lift_v_tail_sec(v_geo_char)
    if n_vertical == 1:
        v_x_loc_root_chord = v_x[0] + v_loc
        v_y_loc_root_chord = v_y[0]
        v_z_loc_root_chord = v_z[0]
    
        v_x_loc_tip_chord = v_x[1] + v_loc
        v_y_loc_tip_chord = v_y[1]
        v_z_loc_tip_chord = v_z[1]
    else:
        v_x_loc_root_chord = v_x[0] + v_loc
        v_y_loc_root_chord = v_y[0] + distance_two_fuse/2
        v_z_loc_root_chord = v_z[0]
    
        v_x_loc_tip_chord = v_x[1] + v_loc
        v_y_loc_tip_chord = v_y[1] + distance_two_fuse/2
        v_z_loc_tip_chord = v_z[1]
        
    if f_area >= a_area:
        f_span, f_root_chord, f_tip_chord = ac.lift_surface_def(f_geo_char)
        f_x, f_y, f_z, f_chord, f_Ain = ac.lift_wing_sec(f_geo_char)
        f_x_loc_root_chord = f_x[0] 
        f_y_loc_root_chord = f_y[0]
        f_z_loc_root_chord = f_z[0] + f_z_loc
    
        f_x_loc_tip_chord = f_x[1]
        f_y_loc_tip_chord = f_y[1]
        f_z_loc_tip_chord = f_z[1] + f_z_loc

        a_span, a_root_chord, a_tip_chord = ac.lift_surface_def(a_geo_char)
        a_x, a_y, a_z, a_chord, a_Ain = ac.lift_h_tail_sec(f_geo_char, a_geo_char, v_geo_char, fuse_geo_char)
        a_x_loc_root_chord = a_x[0] + a_loc
        a_y_loc_root_chord = a_y[0]
        a_z_loc_root_chord = a_z[0] + a_z_loc
    
        a_x_loc_tip_chord = a_x[1] + a_loc
        a_y_loc_tip_chord = a_y[1]
        a_z_loc_tip_chord = a_z[1] + a_z_loc
    else:
        f_span, f_root_chord, f_tip_chord = ac.lift_surface_def(f_geo_char)
        f_x, f_y, f_z, f_chord, f_Ain = ac.lift_h_tail_sec(a_geo_char, f_geo_char, v_geo_char, fuse_geo_char)
        f_x_loc_root_chord = f_x[0] 
        f_y_loc_root_chord = f_y[0]
        f_z_loc_root_chord = f_z[0] + f_z_loc
    
        f_x_loc_tip_chord = f_x[1]
        f_y_loc_tip_chord = f_y[1]
        f_z_loc_tip_chord = f_z[1] + f_z_loc

        a_span, a_root_chord, a_tip_chord = ac.lift_surface_def(a_geo_char)
        a_x, a_y, a_z, a_chord, a_Ain = ac.lift_wing_sec(a_geo_char)
        a_x_loc_root_chord = a_x[0] + a_loc
        a_y_loc_root_chord = a_y[0]
        a_z_loc_root_chord = a_z[0] + a_z_loc
    
        a_x_loc_tip_chord = a_x[1] + a_loc
        a_y_loc_tip_chord = a_y[1]
        a_z_loc_tip_chord = a_z[1] + a_z_loc
    
    
      
                                                      
    
    return f_delta, f_twist, f_root_chord, f_tip_chord,f_x_loc_root_chord,f_y_loc_root_chord,f_z_loc_root_chord,f_x_loc_tip_chord,f_y_loc_tip_chord,f_z_loc_tip_chord,\
        a_delta, a_twist, a_root_chord, a_tip_chord,a_x_loc_root_chord,a_y_loc_root_chord,a_z_loc_root_chord,a_x_loc_tip_chord,a_y_loc_tip_chord,a_z_loc_tip_chord,\
            v_root_chord, v_tip_chord,v_x_loc_root_chord,v_y_loc_root_chord,v_z_loc_root_chord,v_x_loc_tip_chord,v_y_loc_tip_chord,v_z_loc_tip_chord,\
            f_aspect, f_sweep, f_taper, f_twist, f_dihedral, f_delta, f_S, f_L, f_z_loc,\
                a_aspect, a_sweep, a_taper, a_twist, a_dihedral, a_delta, a_S, a_L, a_z_loc,\
                    v_aspect, v_sweep, v_taper, v_twist, v_dihedral, v_delta, v_S, v_L, v_z_loc,\
                        nose_f_aspect,center_f_aspect,tail_f_aspect,fuse_diameter, fuse_x_loc, n_vertical, n_fuse, distance_two_fuse
                        
                
def save_output_DataFrame_FreeCAD(plane):
    name_columns = ['f_w_delta', 'f_w_twist', 'f_root_chord', 'f_tip_chord','f_x_loc_root_chord','f_y_loc_root_chord','f_z_loc_root_chord','f_x_loc_tip_chord','f_y_loc_tip_chord','f_z_loc_tip_chord',
        'a_w_delta', 'a_w_twist', 'a_root_chord', 'a_tip_chord','a_x_loc_root_chord','a_y_loc_root_chord','a_z_loc_root_chord','a_x_loc_tip_chord','a_y_loc_tip_chord','a_z_loc_tip_chord',
            'v_root_chord', 'v_tip_chord','v_x_loc_root_chord','v_y_loc_root_chord','v_z_loc_root_chord','v_x_loc_tip_chord','v_y_loc_tip_chord','v_z_loc_tip_chord',
            'nose_f_aspect','center_f_aspect','tail_f_aspect','fuse_diameter', 'fuse_x_loc', 'n_vertical', 'n_fuse', 'distance_two_fuse']
    data_frame = pd.DataFrame(plane, columns = name_columns)
    return data_frame

def save_geomtry_DataFrame(plane):
    name_columns = ['f_aspect', 'f_sweep', 'f_taper', 'f_twist', 'f_dihedral', 'f_delta',  'f_S', 'f_L','f_z_loc',
                    'a_aspect', 'a_sweep', 'a_taper', 'a_twist', 'a_dihedral', 'a_delta',  'a_S', 'a_L','a_z_loc',
                    'v_aspect', 'v_sweep', 'v_taper', 'v_twist', 'v_dihedral', 'v_delta',  'v_S', 'v_L','v_z_loc',
                    'fuse_AR_nose','fuse_AR_center','fuse_AR_tail','fuse_diameter', 'n_vertical', 'n_fuse']
    data_frame = pd.DataFrame(plane, columns = name_columns)
    return data_frame

def vector_info_FreeCAD(calc):
    v_info = np.array([calc[0], calc[1], calc[2], calc[3], calc[4], calc[5], calc[6], calc[7], calc[8], calc[9], 
                       calc[10], calc[11], calc[12], calc[13], calc[14], calc[15], calc[16], calc[17], calc[18], calc[19],
                       calc[20], calc[21], calc[22], calc[23], calc[24], calc[25], calc[26], calc[27],
                       calc[55], calc[56], calc[57], calc[58], calc[59], calc[60], calc[61], calc[62]])
    #f_w_delta, f_w_twist, f_root_chord, f_tip_chord,f_x_loc_root_chord,f_y_loc_root_chord,f_z_loc_root_chord,f_x_loc_tip_chord,f_y_loc_tip_chord,f_z_loc_tip_chord,\
    #a_w_delta, a_w_twist, a_root_chord, a_tip_chord,a_x_loc_root_chord,a_y_loc_root_chord,a_z_loc_root_chord,a_x_loc_tip_chord,a_y_loc_tip_chord,a_z_loc_tip_chord,\
    #v_root_chord, v_tip_chord,v_x_loc_root_chord,v_y_loc_root_chord,v_z_loc_root_chord,v_x_loc_tip_chord,v_y_loc_tip_chord,v_z_loc_tip_chord
    # 'fuse_AR_nose','fuse_AR_center','fuse_AR_tail','fuse_diameter', fuse_x_loc, n_vertical, n_fuse, distance_two_fuse
    v_info = v_info.transpose()
    return v_info

def vector_info_geometry(calc):
    v_info = np.array([calc[28], calc[29], calc[30], calc[31], calc[32], calc[33], calc[34], calc[35], calc[36], 
                       calc[37], calc[38], calc[39], calc[40], calc[41], calc[42], calc[43], calc[44], calc[45],
                       calc[46], calc[47], calc[48], calc[49], calc[50], calc[51], calc[52], calc[53], calc[54],
                       calc[55], calc[56], calc[57], calc[58], calc[60], calc[61]])
    #'f_aspect', 'f_sweep', 'f_taper', 'f_twist', 'f_dihedral', 'f_delta',  'f_S', 'f_L','f_z_loc',
    #'a_aspect', 'a_sweep', 'a_taper', 'a_twist', 'a_dihedral', 'a_delta',  'a_S', 'a_L','a_z_loc',
    #'v_aspect', 'v_sweep', 'v_taper', 'v_twist', 'v_dihedral', 'v_delta',  'v_S', 'v_L','v_z_loc',
    #'fuse_AR_nose','fuse_AR_center','fuse_AR_tail','fuse_diameter', n_vertical, n_fuse
    v_info = v_info.transpose()
    return v_info

def multijob(des_par, cores, init_pop, usr_path):
    for i in range(init_pop):
        if os.path.exists(usr_path + f'/Creat_png_AVL_{i}'):
            pass
        else:
            os.mkdir(usr_path + f'/Creat_png_AVL_{i}')
            os.mkdir(usr_path + f'/Creat_png_AVL_{i}/tmp/')
    
    t = Parallel(n_jobs=cores)(delayed(geometry)(des_par, kk) for kk in range(init_pop))
    
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
    
    f_aspect = np.array([])
    f_sweep = np.array([])
    f_taper = np.array([])
    f_twist = np.array([])
    f_dihedral = np.array([])
    f_delta = np.array([])
    f_S = np.array([])
    f_L = np.array([])
    f_z_loc = np.array([])
    
    a_aspect = np.array([])
    a_sweep = np.array([])
    a_taper = np.array([])
    a_twist = np.array([])
    a_dihedral = np.array([])
    a_delta = np.array([])
    a_S = np.array([])
    a_L = np.array([])
    a_z_loc = np.array([])
    
    v_aspect = np.array([])
    v_sweep = np.array([])
    v_taper = np.array([])
    v_twist = np.array([])
    v_dihedral = np.array([])
    v_delta = np.array([])
    v_S = np.array([])
    v_L = np.array([])
    v_z_loc = np.array([])
    
    nose_f_aspect = np.array([])
    center_f_aspect = np.array([])
    tail_f_aspect = np.array([])
    fuse_diameter = np.array([])
    fuse_x_loc = np.array([])
    n_vertical = np.array([])
    n_fuse = np.array([])
    distance_two_fuse = np.array([])

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
        
        f_aspect = np.append(f_aspect, t[kk][28])
        f_sweep = np.append(f_sweep, t[kk][29])
        f_taper = np.append(f_taper, t[kk][30])
        f_twist = np.append(f_twist, t[kk][31])
        f_dihedral = np.append(f_dihedral, t[kk][32])
        f_delta = np.append(f_delta, t[kk][33])
        f_S = np.append(f_S, t[kk][34])
        f_L = np.append(f_L, t[kk][35])
        f_z_loc = np.append(f_z_loc, t[kk][36])
        
        a_aspect = np.append(a_aspect, t[kk][37])
        a_sweep = np.append(a_sweep, t[kk][38])
        a_taper = np.append(a_taper, t[kk][39])
        a_twist = np.append(a_twist, t[kk][40])
        a_dihedral = np.append(a_dihedral, t[kk][41])
        a_delta = np.append(a_delta, t[kk][42])
        a_S = np.append(a_S, t[kk][43])
        a_L = np.append(a_L, t[kk][44])
        a_z_loc = np.append(a_z_loc, t[kk][45])
        
        v_aspect = np.append(v_aspect, t[kk][46])
        v_sweep = np.append(v_sweep, t[kk][47])
        v_taper = np.append(v_taper, t[kk][48])
        v_twist = np.append(v_twist, t[kk][49])
        v_dihedral = np.append(v_dihedral, t[kk][50])
        v_delta = np.append(v_delta, t[kk][51])
        v_S = np.append(v_S, t[kk][52])
        v_L = np.append(v_L, t[kk][53])
        v_z_loc = np.append(v_z_loc, t[kk][54])
        
        nose_f_aspect = np.append(nose_f_aspect, t[kk][55])
        center_f_aspect = np.append(center_f_aspect, t[kk][56])
        tail_f_aspect = np.append(tail_f_aspect, t[kk][57])
        fuse_diameter = np.append(fuse_diameter, t[kk][58])
        fuse_x_loc = np.append(fuse_x_loc, t[kk][59])
        
        n_vertical = np.append(n_vertical, t[kk][60])
        n_fuse = np.append(n_fuse, t[kk][61])
        distance_two_fuse = np.append(distance_two_fuse, t[kk][62])
    return f_w_delta, f_w_twist, f_root_chord, f_tip_chord,f_x_loc_root_chord,f_y_loc_root_chord,f_z_loc_root_chord,f_x_loc_tip_chord,f_y_loc_tip_chord,f_z_loc_tip_chord,\
        a_w_delta, a_w_twist, a_root_chord, a_tip_chord,a_x_loc_root_chord,a_y_loc_root_chord,a_z_loc_root_chord,a_x_loc_tip_chord,a_y_loc_tip_chord,a_z_loc_tip_chord,\
            v_root_chord, v_tip_chord,v_x_loc_root_chord,v_y_loc_root_chord,v_z_loc_root_chord,v_x_loc_tip_chord,v_y_loc_tip_chord,v_z_loc_tip_chord,\
                f_aspect, f_sweep, f_taper, f_twist, f_dihedral, f_delta, f_S, f_L, f_z_loc,\
                    a_aspect, a_sweep, a_taper, a_twist, a_dihedral, a_delta, a_S, a_L, a_z_loc,\
                        v_aspect, v_sweep, v_taper, v_twist, v_dihedral, v_delta, v_S, v_L, v_z_loc,\
                            nose_f_aspect,center_f_aspect,tail_f_aspect,fuse_diameter, fuse_x_loc, n_vertical, n_fuse, distance_two_fuse
