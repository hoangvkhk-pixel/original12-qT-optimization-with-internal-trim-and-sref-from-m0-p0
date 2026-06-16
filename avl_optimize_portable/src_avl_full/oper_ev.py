#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:53:15 2022

@author: lab104
"""

import numpy as np
from joblib import Parallel, delayed
import AeroCoeff_AVL as ac
import os
import m0_calc as m0

AUTO_FULL_ROOT = os.environ.get(
    "AUTO_FULL_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"),
)


def _auto_full_dir(index):
    return os.path.join(AUTO_FULL_ROOT, f"Auto_full_{index}")

def vector_info(calc):
    v_info = np.array([calc[0], 
                       calc[1], calc[2], calc[3], calc[4], calc[5], calc[6], 
                       calc[7], calc[8], calc[9], calc[10], calc[11], 
                       calc[12], calc[13], calc[14], calc[15], calc[16], calc[17], calc[18], 
                       calc[19], calc[20], calc[21]])
    # 'mtow_out',
    # 'm_V', 'D_V', 'm_pu', 'm_fue', 'p2w', 'N',
    # 'm_const_f_w', 'm_const_a_w', 'm_const_v_w', 'm_F', 'm_SS',
    # 'cx', 'cy', 'mz', 'mx_beta', 'my_beta', 'K', 'center_mass', 
    # 'alpha_bal', 'delta_bal', 'A'

    v_info = v_info.transpose()
    return v_info

def vector_info_FreeCAD(calc):
    v_info = np.array([calc[22], calc[23], calc[24], calc[25], calc[26], calc[27], calc[28], calc[29], 
                       calc[30], calc[31], calc[32], calc[33], calc[34], calc[35], calc[36], calc[37], calc[38], calc[39], 
                       calc[40], calc[41], calc[42], calc[43], calc[44], calc[45], calc[46], calc[47], 
                       calc[48], calc[49], calc[50], calc[51], calc[52], calc[53], calc[54], calc[55], calc[56], calc[57]])
    # 'f_delta', 'f_twist','f_root_chord', 'f_tip_chord','f_x_loc_root_chord','f_y_loc_root_chord', 'f_z_loc_root_chord','f_x_loc_tip_chord','f_y_loc_tip_chord', 'f_z_loc_tip_chord',
    # 'a_delta', 'a_twist', 'a_root_chord', 'a_tip_chord','a_x_loc_root_chord','a_y_loc_root_chord', 'a_z_loc_root_chord','a_x_loc_tip_chord','a_y_loc_tip_chord', 'a_z_loc_tip_chord',
    # 'v_root_chord', 'v_tip_chord', 'v_x_loc_root_chord', 'v_y_loc_root_chord', 'v_z_loc_root_chord', 'v_x_loc_tip_chord', 'v_y_loc_tip_chord', 'v_z_loc_tip_chord',
    # 'nose_f_aspect', 'center_f_aspect', 'tail_f_aspect', 'fuse_diameter', 'fuse_x_loc', 'n_vertical', 'n_fuse', 'distance_two_fuse'

    v_info = v_info.transpose()
    return v_info

def multijob(des_par, mpay, w_rpm, t, margin, type_power, gamma, Ce1, Ce2, cores, init_pop):
    os.makedirs(AUTO_FULL_ROOT, exist_ok=True)
    for i in range(init_pop):
        tmp_dir = os.path.join(_auto_full_dir(i), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

    hung = Parallel(n_jobs=cores)(delayed(m0.m0_calc)(des_par, mpay, w_rpm, t, margin, type_power, gamma, Ce1, Ce2, kk) for kk in range(init_pop)) 
    
    mtow_out = np.array([])
    
    m_V = np.array([])
    D_V = np.array([])
    m_pu = np.array([])
    m_fue = np.array([])
    p2w = np.array([])
    N = np.array([])
    m_const_f_w = np.array([])
    m_const_a_w = np.array([])
    m_const_v_w = np.array([])
    m_F = np.array([])
    m_SS = np.array([])

    cx = np.array([])
    cy = np.array([])
    mz = np.array([])
    mx_beta = np.array([])
    my_beta = np.array([])
    K = np.array([])
    center_mass = np.array([])
    
    alpha_bal = np.array([])
    delta_bal = np.array([])
    A = np.array([])
    
    f_delta = np.array([])
    f_twist = np.array([])
    f_root_chord = np.array([])
    f_tip_chord = np.array([])
    f_x_loc_root_chord = np.array([])
    f_y_loc_root_chord = np.array([])
    f_z_loc_root_chord = np.array([])
    f_x_loc_tip_chord = np.array([])
    f_y_loc_tip_chord = np.array([])
    f_z_loc_tip_chord = np.array([])
    
    a_delta = np.array([])
    a_twist = np.array([])
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
    
    nose_f_aspect = np.array([])
    center_f_aspect = np.array([])
    tail_f_aspect = np.array([])
    fuse_diameter = np.array([])
    fuse_x_loc = np.array([])
    n_vertical = np.array([])
    n_fuse = np.array([])
    distance_two_fuse = np.array([])
    
    for kk in range(init_pop):
        mtow_out = np.append(mtow_out, hung[kk][0])
        
        m_V = np.append(m_V, hung[kk][1])
        D_V = np.append(D_V, hung[kk][2])
        m_pu = np.append(m_pu, hung[kk][3])
        m_fue = np.append(m_fue, hung[kk][4])
        p2w = np.append(p2w, hung[kk][5])
        N = np.append(N, hung[kk][6])

        m_const_f_w = np.append(m_const_f_w, hung[kk][7])
        m_const_a_w = np.append(m_const_a_w, hung[kk][8])
        m_const_v_w = np.append(m_const_v_w, hung[kk][9])
        m_F = np.append(m_F, hung[kk][10])
        m_SS = np.append(m_SS, hung[kk][11])

        cx = np.append(cx, hung[kk][12])
        cy = np.append(cy, hung[kk][13])
        mz = np.append(mz, hung[kk][14])
        mx_beta = np.append(mx_beta, hung[kk][15])
        my_beta = np.append(my_beta, hung[kk][16])
        K = np.append(K, hung[kk][17])
        center_mass = np.append(center_mass, hung[kk][18])
                
        alpha_bal = np.append(alpha_bal, hung[kk][19])
        delta_bal = np.append(delta_bal, hung[kk][20])
        A = np.append(A, hung[kk][21])
        
        f_delta = np.append(f_delta, hung[kk][22])
        f_twist = np.append(f_twist, hung[kk][23])
        f_root_chord = np.append(f_root_chord, hung[kk][24])
        f_tip_chord = np.append(f_tip_chord, hung[kk][25])
        f_x_loc_root_chord = np.append(f_x_loc_root_chord, hung[kk][26])
        f_y_loc_root_chord = np.append(f_y_loc_root_chord, hung[kk][27])
        f_z_loc_root_chord = np.append(f_z_loc_root_chord, hung[kk][28])
        f_x_loc_tip_chord = np.append(f_x_loc_tip_chord, hung[kk][29])
        f_y_loc_tip_chord = np.append(f_y_loc_tip_chord, hung[kk][30])
        f_z_loc_tip_chord = np.append(f_z_loc_tip_chord, hung[kk][31])
        
        a_delta = np.append(a_delta, hung[kk][32])
        a_twist = np.append(a_twist, hung[kk][33])
        a_root_chord = np.append(a_root_chord, hung[kk][34])
        a_tip_chord = np.append(a_tip_chord, hung[kk][35])
        a_x_loc_root_chord = np.append(a_x_loc_root_chord, hung[kk][36])
        a_y_loc_root_chord = np.append(a_y_loc_root_chord, hung[kk][37])
        a_z_loc_root_chord = np.append(a_z_loc_root_chord, hung[kk][38])
        a_x_loc_tip_chord = np.append(a_x_loc_tip_chord, hung[kk][39])
        a_y_loc_tip_chord = np.append(a_y_loc_tip_chord, hung[kk][40])
        a_z_loc_tip_chord = np.append(a_z_loc_tip_chord, hung[kk][41])
        
        v_root_chord = np.append(v_root_chord, hung[kk][42])
        v_tip_chord = np.append(v_tip_chord, hung[kk][43])
        v_x_loc_root_chord = np.append(v_x_loc_root_chord, hung[kk][44])
        v_y_loc_root_chord = np.append(v_y_loc_root_chord, hung[kk][45])
        v_z_loc_root_chord = np.append(v_z_loc_root_chord, hung[kk][46])
        v_x_loc_tip_chord = np.append(v_x_loc_tip_chord, hung[kk][47])
        v_y_loc_tip_chord = np.append(v_y_loc_tip_chord, hung[kk][48])
        v_z_loc_tip_chord = np.append(v_z_loc_tip_chord, hung[kk][49])
        
        nose_f_aspect = np.append(nose_f_aspect, hung[kk][50])
        center_f_aspect = np.append(center_f_aspect, hung[kk][51])
        tail_f_aspect = np.append(tail_f_aspect, hung[kk][52])
        fuse_diameter = np.append(fuse_diameter, hung[kk][53])
        fuse_x_loc = np.append(fuse_x_loc, hung[kk][54])
        n_vertical = np.append(n_vertical, hung[kk][55])
        n_fuse = np.append(n_fuse, hung[kk][56])
        distance_two_fuse = np.append(distance_two_fuse, hung[kk][57])
        
    return mtow_out, m_V, D_V, m_pu, m_fue, p2w, N, m_const_f_w, m_const_a_w, m_const_v_w, m_F, m_SS, cx, cy, mz, mx_beta, my_beta, K, center_mass, alpha_bal, delta_bal, A,\
        f_delta, f_twist, f_root_chord, f_tip_chord,f_x_loc_root_chord,f_y_loc_root_chord,f_z_loc_root_chord,f_x_loc_tip_chord,f_y_loc_tip_chord,f_z_loc_tip_chord,\
            a_delta, a_twist, a_root_chord, a_tip_chord,a_x_loc_root_chord,a_y_loc_root_chord,a_z_loc_root_chord,a_x_loc_tip_chord,a_y_loc_tip_chord,a_z_loc_tip_chord,\
                v_root_chord, v_tip_chord,v_x_loc_root_chord,v_y_loc_root_chord,v_z_loc_root_chord,v_x_loc_tip_chord,v_y_loc_tip_chord,v_z_loc_tip_chord,\
                    nose_f_aspect,center_f_aspect,tail_f_aspect,fuse_diameter, fuse_x_loc, n_vertical, n_fuse, distance_two_fuse
       
def pen_fun(mz, mx_beta, my_beta, cy, delta, alpha, A, biaz_mz, max_cy, min_delta, max_delta, min_alpha, max_alpha, min_A, max_A):
     
     pop_size = len(mz)

     psi = np.array([])
     
     for i in range(pop_size):
         ep_sum = 0
         ep_sum += max([0, (mz[i] - biaz_mz)])
         ep_sum += max([0, (mx_beta[i])])
         ep_sum += max([0, (my_beta[i])])
         ep_sum += max([0, (cy[i] - max_cy)])
         ep_sum += max([0, (min_delta - delta[i])])
         ep_sum += max([0, (delta[i] - max_delta)])
         ep_sum += max([0, (min_alpha - alpha[i])])
         ep_sum += max([0, (alpha[i] - max_alpha)])
         ep_sum += max([0, (min_A - A[i])])
         ep_sum += max([0, (A[i] - max_A)])
         psi = np.append(psi, ep_sum)
     
     return psi
    
    
def fit_fun(obj, pen, max_obj, R=100):
    
    pop_size = len(obj)

    L_x = np.array([])
    
    for i in range(pop_size):
        if pen[i] == 0:
            L_x = np.append(L_x, obj[i])
        else:
            if obj[i] <= max_obj:
                L_x = np.append(L_x, R*pen[i]+max_obj)
            else:
                L_x = np.append(L_x, R*pen[i]+obj[i])
        
    return L_x

def best_indiv(NPp, Pg, Fx, var_num):
    Pgp = np.zeros((NPp, var_num))
    
    while Fx.size > Pg.size:
        Fx = np.delete(Fx,np.argmax(Fx))
    
    for i in range(NPp):
        best_ind = np.argmin(Fx)
        Pgp[i] = Pg[best_ind]
        Pg = np.delete(Pg, best_ind, axis=0)
        Fx = np.delete(Fx, best_ind)
        
    return Pgp

def pBestSelect(Lx, NP, p):
    Np = round(NP*p)
    indX = np.arange(Lx.size, dtype='int')
    Xcheck = Lx.copy()
    pBest = np.array([], dtype='int')
    for i in range(Np):
        iBest = np.argmin(Xcheck)
        pBest = np.append(pBest, indX[iBest])
        Xcheck = np.delete(Xcheck, iBest)
        indX = np.delete(indX, iBest)
    return pBest

def oper_CR(H, MF):
    from scipy.stats import cauchy
    rand_unif_int = np.random.randint(0, H)
    
    while 1==1:
        Fe = cauchy.rvs(MF[rand_unif_int], 0.1)
        if Fe>1:
            Fi = 1
            break
        elif Fe<=0:
            pass
        else:
            Fi = Fe
            break
    
    return Fi

def operators(H, MF, MCR):
    from scipy.stats import cauchy
    rand_unif_int = np.random.randint(0, H)   #Tạo ra một số ngẫu nhiên từ 0 đến H  
    while 1==1:
        Fe = cauchy.rvs(MF[rand_unif_int], 0.1) #Tạo ra một số ngẫu nhiên theo phân phối Cauchy
        if Fe>1:
            Fi = 1
            break
        elif Fe<=0:
            pass
        else:
            Fi = Fe
            break
    if MCR[rand_unif_int]==-1:
        CRi = 0
    else:
        CRi = np.random.normal(MCR[rand_unif_int],0.1) #Tạo ra một số ngẫu nhiên theo phân phối chuẩn Gause
    return Fi, CRi

def mut_oper(best_ind, new_pop, Pg, A, i, Fi, num_var, des_var):
    if A.size == 0:
        xi = np.arange(new_pop)
        xi = np.delete(xi, i)
        
        rs = np.random.choice(xi, 2 , replace=False) #Lựa chọn ngẫu nhiên 1 list trong xi (ở đây kích cỡ là 2)
        xr = Pg[rs]
    else:
        xr = np.zeros((2,num_var))
        Pg_A = np.concatenate((Pg, A), axis=0) #Nối 2 mảng dọc theo 1 trục hiện có
        new_Pg_A = np.size(Pg_A, 0)
        
        x0 = np.arange(new_pop)
        x0 = np.delete(x0, i)
        r0 = np.random.choice(x0)
        x1 = np.arange(new_Pg_A)
        x1 = np.delete(x1, [i, r0])
        r1 = np.random.choice(x1)
        
        xr[0] = Pg[r0]
        xr[1] = Pg_A[r1]
    # Operador de mutacion
    vi = Pg[i] + Fi * (best_ind - Pg[i]) + Fi * (xr[0] - xr[1])
    return vi

def cross_oper(Pgi_cross, Pgi_mut, Pgi, CRi, variable):
    for j in range(variable):
        if np.random.rand()<=CRi or j==np.random.randint(0,variable):
            Pgi_cross[j] = Pgi_mut[j]
        else:
            Pgi_cross[j] = Pgi[j]
    return Pgi_cross

def bounds_handling(Pgi_cross, variable, des_var):
    for j in range(variable):
        if Pgi_cross[j] < des_var[j,0]:
            Pgi_cross[j] = des_var[j,0]
        if Pgi_cross[j] > des_var[j,1]:
            Pgi_cross[j] = des_var[j,1]
    return Pgi_cross
    
def Lehmer_weight_average(SF, diff):
    S1 = 0
    S2 = 0
    for i in range(SF.size):
        S2 += diff[i] * SF[i]**2 / sum(diff)
        S1 += diff[i] * SF[i] / sum(diff)
    if S1==0:
        mS = 0
    else:
        mS = S2/S1
    
    return mS

def population_A(new_A_pop, A):
    actual_pop = np.size(A,0)
    if actual_pop> new_A_pop:
        r = np.random.choice(np.arange(actual_pop), new_A_pop, replace=False)
        A_new = A[r]
    else:
        A_new = A
    
    return A_new

def pop_reduction_lineal(max_eval_f, init_pop, min_pop, num_eval_f):
    return round((min_pop - init_pop) * num_eval_f / max_eval_f + init_pop)

def pop_reduction_exponential(max_eval_f, init_pop, min_pop, num_eval_f):
    return round(init_pop * (min_pop / init_pop) ** (num_eval_f / max_eval_f))

def new_generation(new_pop, Pg_new, Lx_new, info_wing_new, psi_new, Fx_new, info_to_FreeCAD_new):
    dx = Lx_new.size - new_pop
    for i in range(dx):
        del_index = np.argmax(Lx_new)
        Pg_new = np.delete(Pg_new, del_index, axis=0)
        info_wing_new = np.delete(info_wing_new, del_index, axis=0)
        Lx_new = np.delete(Lx_new, del_index)
        psi_new = np.delete(psi_new, del_index)
        Fx_new = np.delete(Fx_new, del_index)
        info_to_FreeCAD_new = np.delete(info_to_FreeCAD_new, del_index, axis=0)

    return Pg_new, Lx_new, info_wing_new, psi_new, Fx_new, info_to_FreeCAD_new

def U_obj(pen, Fx, num_pop, U_obj_last):
    Fxf = np.array([])
    for i in range(num_pop):
        if pen[i] == 0:
            Fxf = np.append(Fxf, Fx[i])
        
    if Fxf.size == 0:
        U_obj = U_obj_last
    else:
        U_obj = min(Fxf)
        
    return U_obj


def obj_max(pen, Fx, num_pop, U_obj_last):
    Fxf = np.array([])
    for i in range(num_pop):
        if pen[i] == 0:
            Fxf = np.append(Fxf, Fx[i])
        
    if Fxf.size == 0:
        U_obj = U_obj_last
    else:
        U_obj = max(Fxf)
        
    return U_obj
    
        
    



















