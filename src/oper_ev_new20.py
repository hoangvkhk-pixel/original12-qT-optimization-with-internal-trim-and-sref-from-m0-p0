from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

import m0_calc_new20 as m0
from new20_sizing_eval import FREECAD_COLS, INFO_COLS


INFO_COUNT = len(INFO_COLS)
FREECAD_COUNT = len(FREECAD_COLS)
TOTAL_COUNT = INFO_COUNT + FREECAD_COUNT


def vector_info(calc):
    return np.array([calc[i] for i in range(INFO_COUNT)]).transpose()


def vector_info_FreeCAD(calc):
    return np.array([calc[i] for i in range(INFO_COUNT, TOTAL_COUNT)]).transpose()


def multijob(des_par, mpay, w_rpm, t, margin, type_power, gamma, Ce1, Ce2, cores, init_pop, model_dir=None):
    rows = Parallel(n_jobs=cores)(
        delayed(m0.m0_calc)(des_par, mpay, w_rpm, t, margin, type_power, gamma, Ce1, Ce2, kk, model_dir=model_dir)
        for kk in range(init_pop)
    )
    out = [[] for _ in range(TOTAL_COUNT)]
    for row in rows:
        for j in range(TOTAL_COUNT):
            out[j].append(row[j])
    return tuple(np.asarray(col, dtype=float) for col in out)


def pen_fun(mz, mx_beta, my_beta, cy, delta, alpha, A, biaz_mz, max_cy,
            min_delta, max_delta, min_alpha, max_alpha, min_A, max_A):
    penalize_mz = str(__import__("os").environ.get("NEW20_PENALIZE_MZ", "0")).strip() not in {"", "0", "false", "False"}
    psi = []
    for i in range(len(cy)):
        ep_sum = 0.0
        if penalize_mz:
            ep_sum += max(0.0, abs(float(mz[i])) - biaz_mz)
        ep_sum += max(0.0, float(mx_beta[i]))
        ep_sum += max(0.0, float(my_beta[i]))
        ep_sum += max(0.0, float(cy[i]) - max_cy)
        ep_sum += max(0.0, min_delta - float(delta[i]))
        ep_sum += max(0.0, float(delta[i]) - max_delta)
        ep_sum += max(0.0, min_alpha - float(alpha[i]))
        ep_sum += max(0.0, float(alpha[i]) - max_alpha)
        ep_sum += max(0.0, min_A - float(A[i]))
        ep_sum += max(0.0, float(A[i]) - max_A)
        psi.append(ep_sum)
    return np.asarray(psi, dtype=float)


def fit_fun(obj, pen, max_obj, R=100):
    out = []
    for i in range(len(obj)):
        if pen[i] == 0:
            out.append(obj[i])
        elif obj[i] <= max_obj:
            out.append(R * pen[i] + max_obj)
        else:
            out.append(R * pen[i] + obj[i])
    return np.asarray(out, dtype=float)


def obj_max(pen, obj, pop_size, current):
    feasible = [obj[i] for i in range(pop_size) if pen[i] == 0]
    return current if not feasible else max(current, max(feasible))


def best_indiv(NPp, Pg, Fx, var_num):
    Pg_work = Pg.copy()
    Fx_work = Fx.copy()
    out = np.zeros((NPp, var_num))
    while Fx_work.size > Pg_work.shape[0]:
        Fx_work = np.delete(Fx_work, np.argmax(Fx_work))
    for i in range(NPp):
        best = int(np.argmin(Fx_work))
        out[i] = Pg_work[best]
        Pg_work = np.delete(Pg_work, best, axis=0)
        Fx_work = np.delete(Fx_work, best)
    return out


def operators(H, MF, MCR):
    from scipy.stats import cauchy

    idx = np.random.randint(0, H)
    while True:
        Fi = cauchy.rvs(MF[idx], 0.1)
        if Fi > 1:
            Fi = 1.0
            break
        if Fi > 0:
            break
    CRi = 0.0 if MCR[idx] == -1 else float(np.random.normal(MCR[idx], 0.1))
    return float(Fi), CRi


def mut_oper(best_ind, new_pop, Pg, A, i, Fi, num_var, des_var):
    if A.size == 0:
        xi = np.delete(np.arange(new_pop), i)
        xr = Pg[np.random.choice(xi, 2, replace=False)]
    else:
        pool = np.concatenate((Pg, A), axis=0)
        r0 = np.random.choice(np.delete(np.arange(new_pop), i))
        r1 = np.random.choice(np.delete(np.arange(pool.shape[0]), [i, r0]))
        xr = np.zeros((2, num_var))
        xr[0] = Pg[r0]
        xr[1] = pool[r1]
    return Pg[i] + Fi * (best_ind - Pg[i]) + Fi * (xr[0] - xr[1])


def cross_oper(Pgi_cross, Pgi_mut, Pgi, CRi, variable):
    forced = np.random.randint(0, variable)
    for j in range(variable):
        Pgi_cross[j] = Pgi_mut[j] if np.random.rand() <= CRi or j == forced else Pgi[j]
    return Pgi_cross


def Lehmer_weight_average(SF, diff):
    denom = float(np.sum(diff))
    if denom == 0:
        return 0.0
    s1 = np.sum(diff * SF / denom)
    s2 = np.sum(diff * SF ** 2 / denom)
    return 0.0 if s1 == 0 else float(s2 / s1)


def population_A(new_A_pop, A):
    if A.shape[0] > new_A_pop:
        return A[np.random.choice(np.arange(A.shape[0]), new_A_pop, replace=False)]
    return A


def pop_reduction_exponential(max_eval_f, init_pop, min_pop, num_eval_f):
    return round(init_pop * (min_pop / init_pop) ** (num_eval_f / max_eval_f))


def new_generation(new_pop, Pg_new, Lx_new, info_wing_new, psi_new, Fx_new, info_to_FreeCAD_new):
    while Lx_new.size > new_pop:
        idx = int(np.argmax(Lx_new))
        Pg_new = np.delete(Pg_new, idx, axis=0)
        info_wing_new = np.delete(info_wing_new, idx, axis=0)
        Lx_new = np.delete(Lx_new, idx)
        psi_new = np.delete(psi_new, idx)
        Fx_new = np.delete(Fx_new, idx)
        info_to_FreeCAD_new = np.delete(info_to_FreeCAD_new, idx, axis=0)
    return Pg_new, Lx_new, info_wing_new, psi_new, Fx_new, info_to_FreeCAD_new
