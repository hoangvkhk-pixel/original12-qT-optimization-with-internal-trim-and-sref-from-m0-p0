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


def pen_components(mz, mx_beta, my_beta, cy, delta, alpha, A, biaz_mz, max_cy,
                   min_delta, max_delta, min_alpha, max_alpha, min_A, max_A,
                   mz_omegaz=None):
    penalize_mz = str(__import__("os").environ.get("NEW20_PENALIZE_MZ", "0")).strip() not in {"", "0", "false", "False"}
    env = __import__("os").environ
    penalize_damp = env.get("NEW20_PENALIZE_MZ_OMEGAZ", "0").strip().lower() in {"1", "true", "yes", "on"}
    min_mz_omegaz = float(env.get("NEW20_MIN_MZ_OMEGAZ", "-5"))
    if mz_omegaz is None:
        mz_omegaz = np.full(len(cy), -1.0e9, dtype=float)
    psi = []
    pen_mz_arr = []
    pen_beta_arr = []
    pen_cy_arr = []
    pen_alphadelta_arr = []
    pen_A_arr = []
    pen_damp_arr = []
    for i in range(len(cy)):
        pen_mz = 0.0
        if penalize_mz:
            pen_mz = max(0.0, abs(float(mz[i])) - biaz_mz)
        pen_beta = max(0.0, float(mx_beta[i])) + max(0.0, float(my_beta[i]))
        pen_cy = max(0.0, float(cy[i]) - max_cy)
        pen_alphadelta = 0.0
        pen_alphadelta += max(0.0, min_delta - float(delta[i]))
        pen_alphadelta += max(0.0, float(delta[i]) - max_delta)
        pen_alphadelta += max(0.0, min_alpha - float(alpha[i]))
        pen_alphadelta += max(0.0, float(alpha[i]) - max_alpha)
        pen_A = max(0.0, min_A - float(A[i])) + max(0.0, float(A[i]) - max_A)
        pen_damp = max(0.0, float(mz_omegaz[i]) - min_mz_omegaz) if penalize_damp else 0.0
        ep_sum = pen_mz + pen_beta + pen_cy + pen_alphadelta + pen_A + pen_damp
        psi.append(ep_sum)
        pen_mz_arr.append(pen_mz)
        pen_beta_arr.append(pen_beta)
        pen_cy_arr.append(pen_cy)
        pen_alphadelta_arr.append(pen_alphadelta)
        pen_A_arr.append(pen_A)
        pen_damp_arr.append(pen_damp)
    return (
        np.asarray(psi, dtype=float),
        np.asarray(pen_mz_arr, dtype=float),
        np.asarray(pen_beta_arr, dtype=float),
        np.asarray(pen_cy_arr, dtype=float),
        np.asarray(pen_alphadelta_arr, dtype=float),
        np.asarray(pen_A_arr, dtype=float),
        np.asarray(pen_damp_arr, dtype=float),
    )


def pen_fun(mz, mx_beta, my_beta, cy, delta, alpha, A, biaz_mz, max_cy,
            min_delta, max_delta, min_alpha, max_alpha, min_A, max_A, mz_omegaz=None):
    return pen_components(
        mz, mx_beta, my_beta, cy, delta, alpha, A, biaz_mz, max_cy,
        min_delta, max_delta, min_alpha, max_alpha, min_A, max_A, mz_omegaz
    )[0]


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


def new_generation(new_pop, Pg_new, Lx_new, info_wing_new, psi_new, Fx_new, info_to_FreeCAD_new, *extra_arrays):
    extras = [np.asarray(arr) for arr in extra_arrays]
    while Lx_new.size > new_pop:
        idx = int(np.argmax(Lx_new))
        Pg_new = np.delete(Pg_new, idx, axis=0)
        info_wing_new = np.delete(info_wing_new, idx, axis=0)
        Lx_new = np.delete(Lx_new, idx)
        psi_new = np.delete(psi_new, idx)
        Fx_new = np.delete(Fx_new, idx)
        info_to_FreeCAD_new = np.delete(info_to_FreeCAD_new, idx, axis=0)
        extras = [np.delete(arr, idx, axis=0) for arr in extras]
    return (Pg_new, Lx_new, info_wing_new, psi_new, Fx_new, info_to_FreeCAD_new, *extras)
