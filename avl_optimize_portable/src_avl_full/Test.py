# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:36:52 2024

@author: Aspirant2
"""

import AeroCoeff_AVL as ac
import math
import os
import subprocess
import numpy as np
import m0_calc  as m0
import balance as bl
import sizing as sz



des_par = [19.88,	7.22,	2.77,	0,	
           6.75,	0, 1,	0.257,	3.95,	
           0,	0, -40, 0,
           0,	0,	0,	0,	0,	
           0, 0.1, 0.8, 47, 73.18, 1020]

mpay = 600
w_rpm = 5800
t = 22
margin = -0.1
type_power = 'DBC'
gamma = 0.87
Ce1 = 0.285
Ce2 = 0.27
'''
f_aspect = des_par[0]
f_sweep = des_par[1]
f_taper = des_par[2]
f_delta = des_par[3]
   
a_aspect = des_par[4]
a_taper = des_par[6]
a_S = des_par[7]
a_L = des_par[8]  

f_S = 1 - a_S
f_L = 0
scheme_fuse = des_par[19]
if scheme_fuse <= 1/3:
    scheme_fuse = 1
elif scheme_fuse > 1/3 and scheme_fuse <= 2/3:
    scheme_fuse = 2
else:
    scheme_fuse = 3
scheme_vertical = des_par[20]
if scheme_vertical <= 1/3:
    scheme_vertical = 1
elif scheme_vertical > 1/3 and scheme_vertical <= 2/3:
    scheme_vertical = 2
else:
    scheme_vertical = 3 
    
if a_S <= 0.5:
    f_twist = des_par[9]
    f_dihedral = des_par[10]
    a_twist = 0
    
    if scheme_vertical == 1:
        a_dihedral = 0
    elif scheme_vertical == 2:
        a_dihedral = des_par[11]
    else:
        a_dihedral = -des_par[11]
    
    if a_dihedral == 0:
        a_sweep = des_par[5]
    else:
        a_sweep = 0
    f_z_loc = des_par[12]
    a_z_loc = 0
else:
    a_sweep = des_par[5]
    a_twist = des_par[9]
    a_dihedral = des_par[10]
    f_twist = 0
    f_dihedral = 0
    a_z_loc = des_par[12]
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
    v_aspect = des_par[13]
    v_sweep = des_par[14]
    v_taper = des_par[15]
    v_twist = 0
    v_dihedral = 90
    v_delta = 0
    v_S = des_par[16]
    v_L = des_par[17]
beta = des_par[18]
V = des_par[21]
p0 = des_par[22] 
m0 = des_par[23] 

t_cr = t
eff = [0.76, 0.7, 0.7] # Prop, motor, batt
theta = [5.0, 0, -30]   

S_ref = m0/p0
a_area = a_S*S_ref
f_area = S_ref - a_area
v_area = v_S*S_ref
f_geo_char = ac.input_lift_surface_data(f_aspect, f_sweep, f_taper, f_twist, f_dihedral, f_delta, f_area, 0, f_z_loc)
a_geo_char = ac.input_lift_surface_data(a_aspect, a_sweep, a_taper, a_twist, a_dihedral, -2, a_area, a_L, a_z_loc)
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
    l_center = max(a_root_chord, f_root_chord)
    center_f_aspect = l_center/fuse_diameter
    tail_f_aspect = 2
    n_fuse = 1
    n_vertical = 2
else:
    l_center = max(a_loc, v_loc)
    center_f_aspect = l_center/fuse_diameter
    chord_i = (a_tip_chord-a_root_chord)*distance_two_fuse/a_span + a_root_chord
    tg1 = a_loc + a_root_chord
    tg2 = v_loc + v_root_chord
    tg3 = a_loc + chord_i + distance_two_fuse*(math.tan(math.radians(a_geo_char[1])))/2
    l_tail = max(tg1, tg2, tg3) - l_center
    tail_f_aspect = l_tail/fuse_diameter
    n_fuse = 2
    n_vertical = 2
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



alpha_bal, delta_bal, center_mass, cx0, cx_bal, cy_bal, mx_bal, my_bal, mz_bal, K_bal = bl.balanc(f_geo_char, a_geo_char, v_geo_char, fuse_geo_char,
                                                                                                  m0, V, beta, theta[0], margin)
'''
for i in range(6):
    mtow_out, m_V, D_V, m_pu, m_fue, p2w, N, m_const_f_w, m_const_a_w, m_const_v_w, m_F, m_SS, cx0, cx_bal, cy_bal, mx_bal, my_bal, mz_bal, K_bal, center_mass, alpha_bal, delta_bal, A = m0.m0_calc(des_par, mpay, w_rpm, t, margin, type_power, gamma, Ce1, Ce2)
    des_par[23] = mtow_out