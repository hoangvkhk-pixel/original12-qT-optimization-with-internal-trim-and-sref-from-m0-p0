# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:14:08 2023

@author: HUNG HOANG
"""
import numpy as np
import AeroCoeff_AVL as ac

def cy_needed(p0, q, theta = 0, n = 1):
    return 9.81 *n* p0 * np.cos(np.deg2rad(theta))/q

def alpha_search(f_w_geo_char, a_w_geo_char,v_w_geo_char,fuse_geo_char, scheme_fuse,
                 m0, V, theta, kk, H=0):       
    q = ac.inputValuesAtmospheric(V, H)  
    p0 = m0/(f_w_geo_char[6] + a_w_geo_char[6])
    cy_bal = np.around(cy_needed(p0,q,theta), decimals=4)
    aoa_range = np.arange(-10,11,20)
    cy_array = []
    mz_array = []
    for aoa in aoa_range:
        flight_cond = ac.input_flight_cond(V, aoa, H)
        cx, cy, mx, my, mz = ac.aero_calc(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse, flight_cond, kk)
        cy_array.append(cy)
        mz_array.append(mz)
    aoa_bal = np.interp(cy_bal, cy_array, aoa_range)
    mz_cy = -(mz_array[-1] - mz_array[0]) / (cy_array[-1] - cy_array[0])
    aero_center = mz_cy
    return aoa_bal, aero_center, cy_bal

def balanc(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse,
           m0, V, theta, margin, kk, H=0):
    delta = np.arange(-5,6,10)
    alpha = np.arange(-20,21,40)
    cx = np.zeros([len(delta),len(alpha)])
    cy = np.zeros([len(delta),len(alpha)])
    mz = np.zeros([len(delta),len(alpha)])
    mz0 = np.zeros(len(delta))
    mz_CG = np.zeros([len(delta),len(alpha)])
    K = np.zeros([len(delta),len(alpha)])
    aero_center = np.zeros(len(delta))
    center_mass_new = np.zeros(len(delta))
    i = 0
    for i in range(len(delta)):
        j = 0
        for j in range(len(alpha)):
            if a_w_geo_char[6] <= f_w_geo_char[6]:
                a_w_geo_char[5] = delta[i]
            else:
                f_w_geo_char[5] = delta[i]
            
            flight_cond = ac.input_flight_cond(V, alpha[j], H)
            c_x, c_y, m_x, m_y, m_z= ac.aero_calc(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse, flight_cond, kk)
            K_ = c_y/c_x
            cx[i,j] = c_x
            cy[i,j] = c_y
            mz[i,j] = m_z
            K[i,j] = K_  
            j+=1
        aero_center[i] = float(-(mz[i,-1] - mz[i,0]) / (cy[i,-1] - cy[i,0]))
        center_mass_new[i] = float(aero_center[i] + margin)
        mz0[i] = float(round(np.interp(0, cy[i], mz[i]), 5))
        i+=1
        
    i = 0
    for i in range(len(delta)):
        j = 0
        for j in range(len(alpha)):
            mz_CG[i,j] = float((mz[i,j] - mz0[i]) * (-margin) / aero_center[i] + mz0[i])
    q = ac.inputValuesAtmospheric(V, H)
    p0 = m0/(f_w_geo_char[6] + a_w_geo_char[6])
    cy_bal = cy_needed(p0, q, theta) 
    coeff_mz = polyfit2D(alpha,delta,mz_CG)
    coeff_cy = polyfit2D(alpha,delta,cy)
    #Solve balanc
    #Equation for mz = 0: coeff_mz[1]*alpha + coeff_mz[2]*delta = - coeff_mz[0]
    #Equation for cy = cy_bal: coeff_cy[1]*alpha + coeff_cy[2]*delta = - coeff_cy[0] + cy_bal
    HS = np.array([[coeff_mz[1],coeff_mz[2]],[coeff_cy[1],coeff_cy[2]]])
    KQ = np.array([-coeff_mz[0],-coeff_cy[0]+cy_bal])
    alpha_bal, delta_bal = np.linalg.solve(HS, KQ)
    if alpha_bal > 50:
        alpha_bal = 50
    if alpha_bal < -50:
        alpha_bal = -50
    
    if a_w_geo_char[6] <= f_w_geo_char[6]:
        a_w_geo_char[5] = delta_bal
    else:
        f_w_geo_char[5] = delta_bal
    
    center_mass = float(center_mass_new[0])
    flight_cond = ac.input_flight_cond(V, alpha_bal, H)
    cx_bal, cy_bal, mx, my, mz_bal = ac.aero_calc(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse,
                                       flight_cond, kk, center_mass)
    cy_bal = float(coeff_cy[0] + coeff_cy[1]*alpha_bal + coeff_cy[2]*delta_bal)
    mz_bal = float(coeff_mz[0] + coeff_mz[1]*alpha_bal + coeff_mz[2]*delta_bal)
    K_bal = cy_bal/cx_bal
    
    beta = np.arange(0,6,5)
    mx = np.zeros(len(beta))
    my = np.zeros(len(beta))
    i = 0
    for i in range(len(beta)):
        if a_w_geo_char[6] <= f_w_geo_char[6]:
            a_w_geo_char[5] = delta_bal
        else:
            f_w_geo_char[5] = delta_bal
        flight_cond = ac.input_flight_cond(V, alpha_bal, H, beta[i])
        c_x, c_y, m_x, m_y, m_z = ac.aero_calc(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse, 
                                               flight_cond, kk, center_mass)
        mx[i] = float(m_x)
        my[i] = float(m_y)
        i += 1
    mx_beta = float((mx[-1] - mx[0]) / (beta[-1] - beta[0]))
    my_beta = float((my[-1] - my[0]) / (beta[-1] - beta[0]))
    return alpha_bal, delta_bal, center_mass, cx_bal, cy_bal, mz_bal, mx_beta, my_beta, K_bal
'''
def mx_my_beta(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char,
               m0, V, theta, margin, kk, H=0):
    beta = np.arange(0,2,1)
    mx = np.zeros([len(beta),1])
    my = np.zeros([len(beta),1])
    alpha_bal, delta_bal, center_mass, cx0, cx_bal, cy_bal, mx_bal, my_bal, mz_bal, K_bal = balanc(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char,
                                                                                                   m0, V, theta, margin, kk, H)
    i = 0
    for i in range(len(beta)):
        a_w_geo_char[5] = delta_bal
        flight_cond = ac.input_flight_cond(V, alpha_bal, H, beta[i])
        c_xi, c_y, m_x, m_y, m_z = ac.aero_calc(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, flight_cond, kk, center_mass)
        mx[i] = m_x
        my[i] = m_y
        i += 1
    mx_beta = (mx[-1] - mx[0]) / (beta[-1] - beta[0])
    my_beta = (my[-1] - my[0]) / (beta[-1] - beta[0])
    return mx_beta, my_beta
'''
def polyfit2D(alpha,delta,mz_CG):
    #Grid coords
    alpha_new, delta_new = np.meshgrid(alpha, delta)
    alpha_new = alpha_new.flatten()
    delta_new = delta_new.flatten()
    #Solve array
    A = np.array([alpha_new*0+1,alpha_new,delta_new]).T
    mz_CG_new = mz_CG.flatten()
    coeff, residuals, rank, s = np.linalg.lstsq(A, mz_CG_new,rcond=None)
    return coeff
