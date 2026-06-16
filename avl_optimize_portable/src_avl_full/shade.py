# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:25:59 2023

@author: HUNG HOANG
"""

# Shade Optimzator
import numpy as np
import m0_calc as m0
import time
import oper_ev as ev
import os
import shutil
import pandas as pd
from smt.sampling_methods import LHS

AUTO_FULL_ROOT = os.environ.get(
    "AUTO_FULL_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"),
)


def _run_root_from_gen_path(gen_path):
    return os.path.dirname(os.path.normpath(gen_path))

def SHADE_algorithm(des_var, const, epsilon, gen_path, cores):
    run_root = _run_root_from_gen_path(gen_path)
    os.makedirs(run_root, exist_ok=True)
    os.makedirs(gen_path, exist_ok=True)
    
    sampling = LHS(xlimits=des_var)
    variable = des_var[:,0].size
    init_pop = 10 * (variable-const)
    x = sampling(init_pop)

    print(x.shape)
    print(x.size) 
    init_pop_path = os.path.join(run_root, f'initial_population_{init_pop}.npy')
    np.save(init_pop_path, x)
    
    #%% SHADE algorithm variables
    t = 25
    margin = -0.1
    mpay = 600

    w_rpm = 5800
    gamma = 0.87
    Ce1 = 0.285
    Ce2 = 0.27
    type_power = 'DBC'

    NP = init_pop
    max_eval = (variable-const) * 1000
    min_pop = variable-const

    H = 200
    p = 1
    rA = 5

    #%% Restrictions
    bias_mz = 0.001
    U_mtow = mpay*100
    min_alpha = -10
    max_alpha = 10
    min_delta = -5
    max_delta = 5
    max_cy = 0.6
    min_A = 0.5
    max_A = 1.2

    #%% First generation
    error = np.array([])
    Lmin = np.array([])
    Lmax = np.array([])
    Lavg = np.array([])

    Pg = np.zeros((NP, variable))

    Pg = np.load(init_pop_path)

    Pg = np.around(Pg, decimals=3)

    start = time.time()
    g = 0

    print('--------------------------------------------------------------------------------')
    print('Generation {} '.format(g))
    print('Progress: ', end='')    

    calc= ev.multijob(Pg, mpay, w_rpm, t, margin, type_power, gamma, Ce1, Ce2, cores, NP)
    info_aircraft = ev.vector_info(calc)
    info_to_FreeCAD = ev.vector_info_FreeCAD(calc)

    m0_val = calc[0]
    m0cns = np.ones(init_pop)*500
    cy_val = calc[13]
    mz_val = abs(calc[14])
    mx_beta_val = (calc[15])
    my_beta_val = (calc[16])
    alpha_val = calc[19]
    delta_val = calc[20]
    A_val = calc[21]

    psi = ev.pen_fun(mz_val, mx_beta_val, my_beta_val, cy_val, delta_val, alpha_val, A_val, bias_mz, max_cy, min_delta, max_delta, min_alpha, max_alpha, min_A, max_A)
    U_mtow = ev.obj_max(psi, m0_val, NP, U_mtow)
    des_var[15,0] = ev.U_obj(psi, m0cns, NP, des_var[15,0])
    des_var[15,1] = ev.obj_max(psi, m0_val, NP, des_var[15,1])
    Lx = ev.fit_fun(m0_val, psi, U_mtow)
    Fx_m0 = m0_val

    #%%
    error = np.append(error, abs(max(Lx)-min(Lx))/max(Lx))
    Lmin = np.append(Lmin, min(Lx))
    Lmax = np.append(Lmax, max(Lx))
    Lavg = np.append(Lavg, np.mean(Lx))
    print(f'\nError: {np.around(error[-1],decimals=4)} \tLmax: {np.around(Lmax[-1],decimals=4)} \tLmin: {np.around(Lmin[-1], decimals=4)} \tU*: {np.around(U_mtow, decimals=4)}')  
    index_opt = np.argmin(Lx)
    confi_opt = Pg[index_opt]
    
    print('Optimum index: ', index_opt)    
    print('Optimum desing parameters vector: ', confi_opt)    
    print('--------------------------------------------------------------------------------')
    Pg_0 = m0.save_input_DataFrame(Pg)
    Pg_0.to_excel(gen_path+'Px_0.xlsx')
    info_aircraft0 = m0.save_output_DataFrame(info_aircraft)
    info_aircraft0.to_excel(gen_path+'info_aircraft_0.xlsx')
    np.save(gen_path+'Lx0.npy', Lx)

    #%% New generation
    MF = 0.5 * np.ones(H)
    MCR = 0.5 * np.ones(H)
    A = np.empty([0, variable])
    new_admin_pop = round(rA * NP)
    k = 0
    num_eval_f = NP
    new_pop = NP
    g = 1

    while new_pop >= min_pop:
    
        new_pop_per = round(p * new_pop)
        Pg_best = ev.best_indiv(new_pop_per, Pg, Lx, variable)
        Dif_fit_mut = np.array([]) # Difference between muted and non-muted
        P_mut = np.zeros((new_pop,variable))
        P_cross = np.zeros((new_pop,variable))
        SF = np.array([])
        SCR = np.array([])    
        Pg_new = np.zeros((new_pop, variable))
        Lx_new = np.zeros(new_pop)
        info_aircraft_new = np.zeros((new_pop, 22))
        info_to_FreeCAD_new = np.zeros((new_pop, 36))
        Fx_m0_new = np.zeros(new_pop)
        psi_new = np.zeros(new_pop)
    
        Fg = np.array([])
        CRg = np.array([])
    
        print('Generation {}'.format(g))
        print('Progress: ', end='')
    
        # Mutation operation
        for i in range(new_pop):
            Fi, CRi = ev.operators(H, MF, MCR)
            Fg = np.append(Fg, Fi)
            CRg = np.append(CRg, CRi)
            best_ind = Pg_best[np.random.choice(new_pop_per)]
            P_mut[i] = ev.mut_oper(best_ind, new_pop, Pg, A, i, Fi, variable, des_var)
            P_cross[i] = ev.cross_oper(P_cross[i], P_mut[i], Pg[i], CRi, variable)
            P_cross[i] = np.clip(P_cross[i],des_var[:,0],des_var[:,1])
            # P_cross[i] = ev.bounds_handling(P_cross[i], Pg[i], variable, des_var)
        #Pcross = m0.save_input_DataFrame(P_cross)
        #Pcross.to_excel(gen_path+'Pcross' + str(g) + '.xlsx')
        calc = ev.multijob(P_cross, mpay, w_rpm, t, margin, type_power, gamma, Ce1, Ce2, cores, new_pop)
        info_aircraft_cross = ev.vector_info(calc)
        info_to_FreeCAD_cross = ev.vector_info_FreeCAD(calc)
            
        m0_mut = calc[0]
        cy_mut = calc[13]
        mz_mut = abs(calc[14])
        mx_beta_mut = calc[15]
        my_beta_mut = calc[16]
        alpha_mut = calc[19]
        delta_mut = calc[20]
        A_mut = calc[21]
    
        print('#',end='')
  
        psi_mut = ev.pen_fun(mz_mut, mx_beta_mut, my_beta_mut, cy_mut, delta_mut, alpha_mut, A_mut, bias_mz, max_cy, min_delta, max_delta, min_alpha, max_alpha, min_A, max_A)
        U_mtow = ev.obj_max(psi_mut, m0_mut, new_pop, U_mtow)
        Lx_mut = ev.fit_fun(m0_mut, psi_mut, U_mtow)              
    
        # Selection operation
        for i in range(new_pop):

            if Lx_mut[i] < Lx[i]:
                A = np.row_stack((A, Pg[i])) #Xếp các mảng theo thứ tự theo chiều dọc (theo hàng).
                SF = np.append(SF, Fg[i])
                SCR = np.append(SCR, CRg[i])
                Dif_fit_mut = np.append(Dif_fit_mut, abs(Lx_mut[i] - Lx[i])) 
                info_aircraft_new[i] = info_aircraft_cross[i]
                info_to_FreeCAD_new[i] = info_to_FreeCAD_cross[i]
                Pg_new[i] = P_cross[i]
                Lx_new[i] = Lx_mut[i]
                Fx_m0_new[i] = m0_mut[i]
                psi_new[i] = psi_mut[i]
            else:
                Pg_new[i] = Pg[i]
                info_aircraft_new[i] = info_aircraft[i]
                info_to_FreeCAD_new[i] = info_to_FreeCAD[i]
                Lx_new[i] = Lx[i]
                Fx_m0_new[i] = Fx_m0[i]
                psi_new[i] = psi[i]
    
        if SF.size!=0:
            mSF = ev.Lehmer_weight_average(SF, Dif_fit_mut)
            mSCR = ev.Lehmer_weight_average(SCR, Dif_fit_mut)
            MF[k] = mSF 
            if MCR[k]==-1 or max(SCR)==0:
                MCR[k] = -1
            else:
                MCR[k] = mSCR
            if k>=H-1:
                k=0
            else:
                k+=1
    
        # m0 range update 
        des_var[15,0] = ev.U_obj(psi_new, Fx_m0_new, new_pop, des_var[15,0])
        des_var[15,1] = ev.obj_max(psi_new, Fx_m0_new, new_pop, des_var[15,1])
    
        A = ev.population_A(new_admin_pop, A)
        num_eval_f += new_pop
        #new_pop = ev.pop_reduction_lineal(max_eval, init_pop, min_pop, num_eval_f)
        new_pop = ev.pop_reduction_exponential(max_eval, NP, min_pop, num_eval_f)
    
        #    
        error = np.append(error, abs(max(Lx_new)-min(Lx_new))/max(Lx_new))
        Lmin = np.append(Lmin, min(Lx_new))
        Lmax = np.append(Lmax, max(Lx_new))
        '''
        Lavg = np.append(Lavg, max(Lx_new))
        #
        if Lavg.size >= 3:
            delta_g = (Lavg[-1] - Lavg[-2])/Lavg[-1]
            delta_g_1 = (Lavg[-2] - Lavg[-3])/Lavg[-2]
            if 0 < delta_g/delta_g_1 and delta_g/delta_g_1 < 1:
                new_pop = round(new_pop*((delta_g/delta_g_1)**(1/100)))

            if new_pop < min_pop:
                new_pop = min_pop
        #%%
        # act_path = os.path.join(gen_path, '{}/'.format(g))
        '''
    
        np.save(gen_path+'Lx'+str(g)+'.npy', Lx_new)
        np.save(gen_path+'MF'+str(g)+'.npy', MF)
        np.save(gen_path+'MCR'+str(g)+'.npy', MCR)
        np.save(gen_path+'A'+str(g)+'.npy', A)
    
        print('\nError: {} \tLmax: {} \tLmin: {} \tU*: {}'.format(np.around(error[-1],decimals=4),
                                                      np.around(Lmax[-1],decimals=4),
                                                      np.around(Lmin[-1], decimals=4),
                                                      np.around(U_mtow, decimals=4)))

        index_opt = np.argmin(Lx_new)
        confi_opt = Pg_new[index_opt]

        print('Optimum index: ', index_opt)    
        print('Optimum desing parameters vector: ', confi_opt)    
        print('--------------------------------------------------------------------------------')
        Pgnew = m0.save_input_DataFrame(Pg_new)
        Pgnew.to_excel(gen_path+'Px_' + str(g) + '.xlsx')
        info_aircraftnew = m0.save_output_DataFrame(info_aircraft_new)
        info_aircraftnew.to_excel(gen_path+'info_aircraft_' + str(g) + '.xlsx')
    
        if ((max(Lx_new) - min(Lx_new))/max(Lx_new)) <= epsilon:
            break
    
        Pg, Lx, info_aircraft, psi, Fx_m0, info_to_FreeCAD = ev.new_generation(new_pop, Pg_new, Lx_new,
                                                      info_aircraft_new, psi_new,
                                                      Fx_m0_new, info_to_FreeCAD_new)
    
        g+=1
    info_to_FreeCAD = m0.save_output_DataFrame_FreeCAD(info_to_FreeCAD)
    info_to_FreeCAD.to_excel(gen_path+'info_to_FreeCAD' + '.xlsx')
    end = time.time()
    for i in range(1, init_pop):
        shutil.rmtree(os.path.join(AUTO_FULL_ROOT, f'Auto_full_{i}'), ignore_errors=True)

    print('\n--------------------------------------------------------------------------------')
    print('Total calculation time: {}'.format(end-start))    
    print('--------------------------------------------------------------------------------')
    return g

def create_folder(g, excel_path):
    file_path = os.path.join(excel_path, f'Px_{g}.xlsx')
    df = pd.read_excel(file_path)
    length = len(df)
    
    n_1_1_normal = 0
    n_1_2_normal = 0
    n_1_3_normal = 0

    n_2_1_normal = 0
    n_2_2_normal = 0
    n_2_3_normal = 0

    n_3_1_normal = 0
    n_3_2_normal = 0
    n_3_3_normal = 0

    n_1_x_duck = 0

    n_2_x_duck = 0

    n_3_x_duck = 0

    for i in range(length):
        if df['a_S'][i] <= 0.5:
            if df['scheme_fuse'][i] < 1/3:
                if df['scheme_vertical'][i] < 1/3:
                    n_1_1_normal += 1
                elif df['scheme_vertical'][i] > 2/3:
                    n_1_3_normal += 1
                else:
                    n_1_2_normal += 1
            elif df['scheme_fuse'][i] > 2/3:
                if df['scheme_vertical'][i] < 1/3:
                    n_3_1_normal += 1
                elif df['scheme_vertical'][i] > 2/3:
                    n_3_3_normal += 1
                else:
                    n_3_2_normal += 1
            else:
                if df['scheme_vertical'][i] < 1/3:
                    n_2_1_normal += 1
                elif df['scheme_vertical'][i] > 2/3:
                    n_2_3_normal += 1
                else:
                    n_2_2_normal += 1
        else:
            if df['scheme_fuse'][i] < 1/3:
                n_1_x_duck += 1
            elif df['scheme_fuse'][i] > 2/3:
                n_3_x_duck += 1
            else:
                n_2_x_duck += 1
    if n_1_1_normal > 0:
        if os.path.exists(excel_path + 'normal_1_1'):
            pass
        else:
            os.mkdir(excel_path + 'normal_1_1')
    if n_1_2_normal > 0:
        if os.path.exists(excel_path + 'normal_1_2'):
            pass
        else:
            os.mkdir(excel_path + 'normal_1_2')
    if n_1_3_normal > 0:
        if os.path.exists(excel_path + 'normal_1_3'):
            pass
        else:
            os.mkdir(excel_path + 'normal_1_3')
    if n_2_1_normal > 0:
        if os.path.exists(excel_path + 'normal_2_1'):
            pass
        else:
            os.mkdir(excel_path + 'normal_2_1')
    if n_2_2_normal > 0:
        if os.path.exists(excel_path + 'normal_2_2'):
            pass
        else:
            os.mkdir(excel_path + 'normal_2_2')
    if n_2_3_normal > 0:
        if os.path.exists(excel_path + 'normal_2_3'):
            pass
        else:
            os.mkdir(excel_path + 'normal_2_3')
    if n_3_1_normal > 0:
        if os.path.exists(excel_path + 'normal_3_1'):
            pass
        else:
            os.mkdir(excel_path + 'normal_3_1')
    if n_3_2_normal > 0:
        if os.path.exists(excel_path + 'normal_3_2'):
            pass
        else:
            os.mkdir(excel_path + 'normal_3_2')
    if n_3_3_normal > 0:
        if os.path.exists(excel_path + 'normal_3_3'):
            pass
        else:
            os.mkdir(excel_path + 'normal_3_3')
    if n_1_x_duck > 0:
        if os.path.exists(excel_path + 'duck_1_x'):
            pass
        else:
            os.mkdir(excel_path + 'duck_1_x')
    if n_2_x_duck > 0:
        if os.path.exists(excel_path + 'duck_2_x'):
            pass
        else:
            os.mkdir(excel_path + 'duck_2_x')
    if n_3_x_duck > 0:
        if os.path.exists(excel_path + 'duck_3_x'):
            pass
        else:
            os.mkdir(excel_path + 'duck_3_x')
    return n_1_1_normal, n_1_2_normal, n_1_3_normal,\
        n_2_1_normal, n_2_2_normal, n_2_3_normal,\
            n_3_1_normal, n_3_2_normal, n_3_3_normal,\
                n_1_x_duck, n_2_x_duck, n_3_x_duck
