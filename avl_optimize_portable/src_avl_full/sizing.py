# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:01:04 2023

@author: HUNG HOANG
"""
import numpy as np
import math
import AeroCoeff_AVL as ac

def m_batery(E, p2w, t, eta_motor, SoC_op=0.7):
    SEC = p2w * t # Specific energy consumption
    return 9.81 * SEC / E / eta_motor/SoC_op / 10  

def m_constr_surface(lift_surface_geo_char, m0, thickness = 0.12):
    '''
    k_mex = 0.95
    k_constr = 0.95
    k_MT = 0.85
    teta = 0.9
    phi = 0.93
    n_p = 5.5
    coef = 1.1*10**(-4)*k_mex*k_constr*k_MT*phi*n_p
    lamda = lift_surface_geo_char[0]
    khi = lift_surface_geo_char[1]
    S = lift_surface_geo_char[3]
    eta = lift_surface_geo_char[2]
    m_constr = coef*lamda*np.sqrt(S)*(eta+4)/((np.cos(np.deg2rad(khi)))**1.5)/np.sqrt(teta)/np.sqrt(thickness)/(eta+1)
    '''
    n_p = 3.75
    span, root_chord, tip_chord = ac.lift_surface_def(lift_surface_geo_char)
    S = lift_surface_geo_char[6]
    lamda = lift_surface_geo_char[0]
    m_constr = 1-(m0-(S*span/100/thickness+2.9*S+5))/m0/(1+0.85*10**(-5)*n_p*span*(lamda/thickness+17))                                     
    
    return m_constr

def m_constr_HT(lift_surface_geo_char, V):
    m = 7.2*(lift_surface_geo_char[6])**1.2*(0.4+(V+113)/935)
    return m

def m_constr_VT(lift_surface_geo_char, V):
    m = 6.8*(lift_surface_geo_char[6])**1.2*(0.4+(V+113)/1100)
    return m

def m_constr_fueslage(body_geo_char, V, n_fuse):
    d_e_body, l_body, S_body, Sxq_body = ac.body_def(body_geo_char)
    m = 0.23*(Sxq_body**1.2)*np.sqrt(V*l_body/(body_geo_char[3]+body_geo_char[4]))
    if n_fuse == 3 or n_fuse == 2:
        m = m
    else:
        m = 2*m
    return m 

def m_SS(m0):
    k = 1.08
    A1 = 11.3
    B1 = 0
    C1 = 0.0024
    D1 = 0
    m1 = k*(A1+B1*m0**(3/4)+C1*m0+D1*m0**(3/2))
    A2 = 9.1
    B2 = 0.082
    C2 = 0.019
    D2 = 0
    m2 = k*(A2+B2*m0**(3/4)+C2*m0+D2*m0**(3/2))
    m = m1+m2
    return m

def m_power(p2w, gamma):
    """
    @param V: flight speed, m/s
    @param K: aerodynamic efficiency
    @param specific_pu_w: specific power unit weight, dN/kW
    @param n_prop: propeller efficiency
    @param n_moto: elecmotor efficiency
    
    @return specific power unit's mass
    """
    m_cy = gamma * p2w
    
    return m_cy

def m_control_equip(m_pu):
    return 0.7*m_pu

def m_fuel_cl(p2w, t_flight, Ce = 0.285):
    '''
    Parameters
    ----------
    p2w : power-to-weight ratio, kW/dN
    Ce : Specific fuel consumption rate
    t_flight : time flight
    Returns
    -------
    m_f : mass fuel
    '''
    m_f = 1.05*p2w*Ce*t_flight
    return m_f
def m_fuel_cr(L, V, K, Ce = 0.285):
    m_f = 1 - math.e**(-L*Ce/K/V)
    return m_f
    
def pwr_to_weight(V, K, alpha, efficiency = 0.75, theta = 0):
    return (np.cos(np.deg2rad(theta)) + np.sin(np.deg2rad(theta)) * K) / (K * np.cos(np.deg2rad(alpha)) + np.sin(np.deg2rad(alpha))) * V / efficiency / 100

def mvinta_DBC(T, w_rpm, H=0):
    k_m_V = 4
    k_m_V_DBC = 0.1
    rho0 = 1.225
    rho = rho0*(1-H/44300)**4.256
    w_rps = w_rpm/60
    D_V_DBC = (T/2/k_m_V_DBC/rho/(w_rps**2))**(1/4)
    m_v_DBC = k_m_V*D_V_DBC
    return m_v_DBC, D_V_DBC
