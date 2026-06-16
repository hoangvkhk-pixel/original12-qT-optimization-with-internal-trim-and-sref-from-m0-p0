# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:03:39 2023

@author: HUNG HOANG
"""
import sizing as sz
import balance as bl
import AeroCoeff_AVL as ac
import numpy as np
import math
import pandas as pd

def m_segment2(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse,
               V, m0, eff, t, margin, theta, w_rpm, type_power, gamma, Ce, kk, H=0):
   
    alpha_bal, delta_bal, center_mass, cx_bal, cy_bal, mz_bal, mx_beta, my_beta, K_bal = bl.balanc(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, 
                                                                                 scheme_fuse, m0, V, theta, margin, kk, H)
    A, L = ac.vol_coeff(f_w_geo_char, a_w_geo_char, center_mass)
    rho0 = 1.225
    rho = rho0*(1-H/44300)**4.256
    X_bal = cx_bal*rho*V**2*(f_w_geo_char[6]+a_w_geo_char[6])/2
    T = (X_bal+m0*9.81*np.sin(np.deg2rad(theta)))/np.cos(np.deg2rad(alpha_bal))
    p2w = sz.pwr_to_weight(V, K_bal, alpha_bal, eff[0], theta)
    m_pow = sz.m_power(p2w, gamma)
    if (type_power == 'eltr'):
        m_fue = sz.m_batery(0.2, p2w, t, eff[1], eff[2])
    else:
        m_fue = sz.m_fuel_cl(p2w, t, Ce)
    
    m_V_DBC, D_V_DBC = sz.mvinta_DBC(T, w_rpm, H)
    return  m_V_DBC, D_V_DBC, m_pow, m_fue, p2w, cx_bal, cy_bal, mz_bal, mx_beta, my_beta, K_bal, center_mass, alpha_bal, delta_bal, T, L, A

def m_segment3(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse, 
               V, m0, eff, t, margin, theta, w_rpm, type_power, gamma, Ce, kk, H = 0):
   
    aoa_bal, aero_center, cy_bal = bl.alpha_search(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char,
                                                   scheme_fuse, m0, V, theta, kk, H)
    center_mass = aero_center + margin
    flight_cond = ac.input_flight_cond(V, aoa_bal, H)
    cx, cy, mx, my, mz = ac.aero_calc(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse, flight_cond, kk, center_mass)
    K = cy/cx
    rho0 = 1.225
    rho = rho0*(1-H/44300)**4.256
    X = cx*rho*V**2*(f_w_geo_char[6]+a_w_geo_char[6])/2
    T = (X+m0*9.81*np.sin(np.deg2rad(theta)))/np.cos(np.deg2rad(aoa_bal))
    p2w = sz.pwr_to_weight(V, K, aoa_bal, eff[0], theta)
    
    m_pow = sz.m_power(p2w, gamma)
    if (type_power == 'eltr'):
        m_fue = sz.m_batery(0.2, p2w, t, eff[1], eff[2]) 
    else:
        m_fue = sz.m_fuel_cl(p2w, t, Ce)
            
    m_V_DBC, D_V_DBC = sz.mvinta_DBC(T, w_rpm, H)
    return m_V_DBC, D_V_DBC, m_pow, m_fue, p2w, T

def m0_calc(des_par, mpay, w_rpm, t, margin, type_power, gamma, Ce1, Ce2, kk, H=0):
    f_aspect = des_par[kk][0]
    f_taper = des_par[kk][1]
    
    a_aspect = des_par[kk][2]
    a_taper = des_par[kk][3]
    a_S = des_par[kk][4]
    a_L = des_par[kk][5]  

    scheme_fuse = des_par[kk][11]
    if scheme_fuse <= 1/3:
        scheme_fuse = 1
    elif scheme_fuse >= 2/3:
        scheme_fuse = 3
    else:
        scheme_fuse = 2
        
    scheme_vertical = des_par[kk][12]
    if scheme_vertical <= 1/3:
        scheme_vertical = 1
    elif scheme_vertical >= 2/3:
        scheme_vertical = 3
    else:
        scheme_vertical = 2 
        
    if a_S <= 0.5:
        f_sweep = des_par[kk][6]
        f_delta = 2.5
        f_twist = 0
        f_dihedral = des_par[kk][7]
        
        a_sweep = 0
        a_twist = 0
        a_delta = -2
        if scheme_vertical == 2:
            a_dihedral = 0
        elif scheme_vertical == 1:
            a_dihedral = des_par[kk][8]
        else:
            a_dihedral = -des_par[kk][8]
    else:
        f_sweep = 0
        f_twist = 0
        f_dihedral = 0
        f_delta = 2
        
        a_sweep = des_par[kk][6]
        a_delta = 0
        a_twist = 0
        a_dihedral = des_par[kk][7]
        
    
    if a_S <= 0.5 and a_dihedral != 0:
        v_aspect = 0
        v_sweep = 0
        v_taper = 0
        v_twist = 0
        v_dihedral = 0
        v_delta = 0
        v_S = 0
        v_L = 0
        if a_aspect > 7:
            a_aspect = 7
    else:
        v_aspect = des_par[kk][9]
        v_sweep = 15
        v_taper = 2
        v_twist = 0
        v_dihedral = 90
        v_delta = 0
        v_S = des_par[kk][10]
        v_L = a_L
    V = des_par[kk][13]
    p0 = des_par[kk][14] 
    m0 = des_par[kk][15] 

    t_cr = t
    eff = [0.76, 0.7, 0.7] # Prop, motor, batt
    theta = [5.0, 0, -30]   
    
    S_ref = m0/p0
    a_area = a_S*S_ref
    f_area = S_ref - a_area
    v_area = v_S*S_ref
    f_geo_char = ac.input_lift_surface_data(f_aspect, f_sweep, f_taper, f_twist, f_dihedral, f_delta, f_area)
    a_geo_char = ac.input_lift_surface_data(a_aspect, a_sweep, a_taper, a_twist, a_dihedral, a_delta, a_area, a_L)
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
    if scheme_fuse == 1:
        fuse_diameter = 0.76/2.5
    elif scheme_fuse == 2:
        fuse_diameter = 0.76/1.5
    else:
        fuse_diameter = 0.76
    nose_f_aspect = 2
    if scheme_fuse == 2:
        l_center = max(a_loc, v_loc)
        center_f_aspect = l_center/fuse_diameter
        tg1 = a_loc + a_root_chord
        tg2 = v_loc + v_root_chord
        l_tail = max(tg1, tg2) - l_center
        tail_f_aspect = l_tail/fuse_diameter
        n_fuse = 1
        n_vertical = 1
    elif scheme_fuse == 3:
        l_center = max(a_root_chord, f_root_chord)
        center_f_aspect = l_center/fuse_diameter
        tail_f_aspect = 2
        n_fuse = 1
        n_vertical = 1
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
        n_vertical = 1
    l_nose = nose_f_aspect*fuse_diameter
    l_center = center_f_aspect*fuse_diameter
    l_tail = tail_f_aspect*fuse_diameter
    l_nose_center = l_center + l_nose
    if a_S > 0.5 and scheme_fuse == 3 : #Xem lai dieu kien nay(hinh 2)
        fuse_x_loc = a_loc - l_nose
    else:
        fuse_x_loc = -l_nose

    fuse_geo_char = ac.input_body_data(nose_f_aspect, center_f_aspect, tail_f_aspect, fuse_diameter, fuse_diameter, fuse_x_loc)
    
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
    
    f_z_loc = 0
    a_z_loc = 0
    if a_S <= 0.5:
        f_span, f_root_chord, f_tip_chord = ac.lift_surface_def(f_geo_char)
        f_x, f_y, f_z, f_chord, f_Ain = ac.lift_wing_sec(f_geo_char)
        f_x_loc_root_chord = f_x[0] 
        f_y_loc_root_chord = f_y[0]
        f_z_loc_root_chord = f_z[0] + f_z_loc
    
        f_x_loc_tip_chord = f_x[1]
        f_y_loc_tip_chord = f_y[1]
        f_z_loc_tip_chord = f_z[1] + f_z_loc

        a_span, a_root_chord, a_tip_chord = ac.lift_surface_def(a_geo_char)
        a_x, a_y, a_z, a_chord, a_Ain = ac.lift_h_tail_sec(a_geo_char, scheme_fuse)
        a_x_loc_root_chord = a_x[0] + a_loc
        a_y_loc_root_chord = a_y[0]
        a_z_loc_root_chord = a_z[0] + a_z_loc
    
        a_x_loc_tip_chord = a_x[1] + a_loc
        a_y_loc_tip_chord = a_y[1]
        a_z_loc_tip_chord = a_z[1] + a_z_loc
    else:
        f_span, f_root_chord, f_tip_chord = ac.lift_surface_def(f_geo_char)
        f_x, f_y, f_z, f_chord, f_Ain = ac.lift_h_tail_sec(f_geo_char, scheme_fuse)
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
        
    # Cruise
    m_V_DBC_cr, D_V_DBC_cr, m_pow_cr, m_fue_cr, p2w_cr, cx_bal_cr, cy_bal_cr, mz_bal_cr, mx_beta, my_beta, K_bal_cr,\
        center_mass_cr, alpha_bal_cr, delta_bal_cr, T_cr, L, A = m_segment2(f_geo_char, a_geo_char, v_geo_char, fuse_geo_char, scheme_fuse,
                                                                            V, m0, eff, t_cr, margin, theta[1],  
                                                                            w_rpm, type_power, gamma, Ce2, kk, H)
    mV = m_V_DBC_cr
    m_pu = m_pow_cr
    m_fue = m_fue_cr
    p2w = p2w_cr
    D_V = D_V_DBC_cr
    cx = cx_bal_cr
    cy = cy_bal_cr
    mz = mz_bal_cr
    K = K_bal_cr
    center_mass = center_mass_cr
    alpha_bal = alpha_bal_cr
    delta_bal = delta_bal_cr
    
    # Climb
    m_V_DBC_cl, D_V_DBC_cl, m_pow_cl, m_fue_cl, p2w_cl, T_cl = m_segment3(f_geo_char, a_geo_char, v_geo_char, fuse_geo_char, scheme_fuse,
                                                                          0.9*V, m0, eff, 0.05*t_cr, margin, theta[0],
                                                                          w_rpm, type_power, gamma, Ce1, kk, H)  
    if m_fue_cl > 0:   
        m_fue = m_fue + m_fue_cl
    if m_V_DBC_cl > mV:
        D_V = D_V_DBC_cl
        mV = m_V_DBC_cl
    if m_pow_cl > m_pu:
        m_pu = m_pow_cl
    if p2w_cl > p2w:
        p2w = p2w_cl

    # Declimb
    m_V_DBC_dcl, D_V_DBC_dcl, m_pow_dcl, m_fue_dcl, p2w_dcl, T_dcl = m_segment3(f_geo_char, a_geo_char, v_geo_char, fuse_geo_char, scheme_fuse,
                                                                                         0.9*V,m0,eff, 0.1*t_cr,margin, theta[2],
                                                                                         w_rpm, type_power, gamma, Ce1, kk, H)

    if m_fue_dcl > 0:   
        m_fue = m_fue + m_fue_dcl
    if m_V_DBC_dcl > mV:
        D_V = D_V_DBC_dcl
        mV = m_V_DBC_dcl
    if m_pow_dcl > m_pu:
        m_pu = m_pow_dcl
    if p2w_dcl > p2w:
        p2w = p2w_dcl
    

    N = p2w*m0  
    if v_aspect == 0:
        m_const_v_w = 0
    else:
        m_const_v_w = sz.m_constr_surface(v_geo_char, m0)
    mSS = sz.m_SS(m0)
    mF = sz.m_constr_fueslage(fuse_geo_char, V, scheme_fuse)
    m_equip = 0.08
    m_const_f_w = sz.m_constr_surface(f_geo_char, m0)
    m_const_a_w = sz.m_constr_surface(a_geo_char, m0)
    
    mtow_out = (mpay + mV + mSS + mF) / (1 - m_fue - m_const_f_w - m_const_a_w - m_const_v_w - m_equip - m_pu)

    if mtow_out <=0:
        mtow_out = 100000
    
    m_V = mV/mtow_out
    m_SS = mSS/mtow_out
    m_F = mF/mtow_out
    
    if a_geo_char[6] <= f_geo_char[6]:
        a_delta = delta_bal
    else:
        f_delta = delta_bal

    return mtow_out, m_V, D_V, m_pu, m_fue, p2w, N, m_const_f_w, m_const_a_w, m_const_v_w, m_F, m_SS, cx, cy, mz, mx_beta, my_beta, K, center_mass, alpha_bal, delta_bal, A,\
        f_delta, f_twist, f_root_chord, f_tip_chord,f_x_loc_root_chord,f_y_loc_root_chord,f_z_loc_root_chord,f_x_loc_tip_chord,f_y_loc_tip_chord,f_z_loc_tip_chord,\
            a_delta, a_twist, a_root_chord, a_tip_chord,a_x_loc_root_chord,a_y_loc_root_chord,a_z_loc_root_chord,a_x_loc_tip_chord,a_y_loc_tip_chord,a_z_loc_tip_chord,\
                v_root_chord, v_tip_chord,v_x_loc_root_chord,v_y_loc_root_chord,v_z_loc_root_chord,v_x_loc_tip_chord,v_y_loc_tip_chord,v_z_loc_tip_chord,\
                    nose_f_aspect,center_f_aspect,tail_f_aspect,fuse_diameter, fuse_x_loc, n_vertical, n_fuse, distance_two_fuse

def save_input_DataFrame(plane):
    name_columns = ['f_aspect', 'f_taper',
                    'a_aspect', 'a_taper', 'a_S','a_x_loc', 
                    'sweep','dihedral', 'dihedral_2',
                    'v_aspect', 'v_S',
                    'scheme_fuse', 'scheme_vertical', 'V','p0','m0_in']
    data_frame = pd.DataFrame(plane, columns = name_columns)
    return data_frame

def save_output_DataFrame(info_aircraft):
    name_columns = ['mtow_out', 'm_V', 'D_V','m_pu', 'm_fue', 'p2w', 'N', 
                    'm_const_f_w', 'm_const_a_w', 'm_const_v_w', 'm_F', 'm_SS',
                    'cx', 'cy', 'mz', 'mx_beta', 'my_beta', 'K', 'center_mass', 'alpha_bal', 'delta_bal', 'A']
    
    data_frame = pd.DataFrame(info_aircraft, columns = name_columns)
    return data_frame


def save_output_DataFrame_FreeCAD(plane):
    name_columns = ['f_delta', 'f_twist','f_root_chord', 'f_tip_chord','f_x_loc_root_chord','f_y_loc_root_chord', 'f_z_loc_root_chord','f_x_loc_tip_chord','f_y_loc_tip_chord', 'f_z_loc_tip_chord',
                    'a_delta', 'a_twist', 'a_root_chord', 'a_tip_chord','a_x_loc_root_chord','a_y_loc_root_chord', 'a_z_loc_root_chord','a_x_loc_tip_chord','a_y_loc_tip_chord', 'a_z_loc_tip_chord',
                    'v_root_chord', 'v_tip_chord', 'v_x_loc_root_chord', 'v_y_loc_root_chord', 'v_z_loc_root_chord', 'v_x_loc_tip_chord', 'v_y_loc_tip_chord', 'v_z_loc_tip_chord',
                    'nose_f_aspect', 'center_f_aspect', 'tail_f_aspect', 'fuse_diameter', 'fuse_x_loc', 'n_vertical', 'n_fuse', 'distance_two_fuse']
    data_frame = pd.DataFrame(plane, columns = name_columns)
    return data_frame