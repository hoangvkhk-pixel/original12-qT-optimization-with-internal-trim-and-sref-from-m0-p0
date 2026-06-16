from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

import m0_calc_new20 as m0
import oper_ev_new20 as ev
from problem_v2_spec import INPUT_COLS_V2


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_generation(gen_path: Path, g: int, Pg, info_aircraft, Lx) -> None:
    m0.save_input_DataFrame(Pg).to_excel(gen_path / f"Px_{g}.xlsx", index=False)
    m0.save_output_DataFrame(info_aircraft).to_excel(gen_path / f"info_aircraft_{g}.xlsx", index=False)
    np.save(gen_path / f"Lx{g}.npy", Lx)


def _sync_population_cyreq(Pg: np.ndarray, info_aircraft: np.ndarray) -> np.ndarray:
    return np.asarray(Pg, dtype=float)


def _m0_feedback_enabled() -> bool:
    return os.environ.get("NEW20_ENABLE_M0_FEEDBACK", "1").strip().lower() not in {"0", "false", "no", "off"}


def _sync_population_m0(Pg: np.ndarray, info_aircraft: np.ndarray, des_var: np.ndarray) -> np.ndarray:
    if not _m0_feedback_enabled():
        return np.asarray(Pg, dtype=float)
    relax = float(os.environ.get("NEW20_M0_SYNC_RELAX", "0.5"))
    relax = float(np.clip(relax, 0.0, 1.0))
    idx_m0 = INPUT_COLS_V2.index("m0")
    lo = float(des_var[idx_m0, 0])
    hi = float(des_var[idx_m0, 1])
    out = np.asarray(Pg, dtype=float).copy()
    mtow = np.asarray(info_aircraft[:, 0], dtype=float)
    current = np.asarray(out[:, idx_m0], dtype=float)
    blended = relax * mtow + (1.0 - relax) * current
    out[:, idx_m0] = np.clip(blended, lo, hi)
    return np.around(out, decimals=6)


def _adapt_m0_bounds(des_var: np.ndarray, info_aircraft: np.ndarray, psi: np.ndarray, base_bounds: np.ndarray) -> np.ndarray:
    if not _m0_feedback_enabled():
        return np.asarray(des_var, dtype=float)
    feasible = np.asarray(psi, dtype=float) <= 1e-12
    if not np.any(feasible):
        return np.asarray(des_var, dtype=float)
    idx_m0 = INPUT_COLS_V2.index("m0")
    out = np.asarray(des_var, dtype=float).copy()
    mtow_feasible = np.asarray(info_aircraft[feasible, 0], dtype=float)
    lo_base = float(base_bounds[idx_m0, 0])
    hi_base = float(base_bounds[idx_m0, 1])
    lo_new = max(lo_base, float(np.min(mtow_feasible)))
    hi_new = min(hi_base, float(np.max(mtow_feasible)))
    if hi_new >= lo_new:
        out[idx_m0, 0] = lo_new
        out[idx_m0, 1] = hi_new
    return out


def _objective(calc) -> tuple[np.ndarray, str]:
    objective = os.environ.get("NEW20_OBJECTIVE", "q").strip().lower()
    if objective in {"mtow", "mtow_out", "m0"}:
        return calc[0], "mtow_out"
    if objective in {"mfue", "m_fue", "m_fue_kg", "fuel"}:
        return calc[23], "M_fue_kg"
    if objective in {"q", "q_fuel"}:
        return calc[31], "q_fuel"
    if objective in {"q_g", "q_g_per_ton_km"}:
        return calc[32], "q_g_per_ton_km"
    raise ValueError(f"Unsupported NEW20_OBJECTIVE={objective!r}")


def _initial_population(des_var: np.ndarray, init_pop: int, gen_path: Path) -> np.ndarray:
    initial_path = os.environ.get("SHADE_INITIAL_POP", "").strip()
    if initial_path:
        x = np.asarray(np.load(initial_path), dtype=float)
        if x.shape != (init_pop, des_var.shape[0]):
            raise ValueError(f"Initial population shape {x.shape} does not match {(init_pop, des_var.shape[0])}")
    else:
        sampler = qmc.LatinHypercube(d=des_var.shape[0], seed=int(os.environ.get("SHADE_LHS_SEED", "42")))
        u = sampler.random(init_pop)
        lo = des_var[:, 0].astype(float)
        hi = des_var[:, 1].astype(float)
        x = lo + u * (hi - lo)
    np.save(gen_path / f"initial_population_{init_pop}.npy", x)
    return np.around(x, decimals=3)


def SHADE_algorithm(des_var, const, epsilon, gen_path, cores, model_dir=None):
    gen_path = _ensure(Path(gen_path))
    des_var = np.asarray(des_var, dtype=float).copy()
    base_bounds = des_var.copy()
    variable = des_var.shape[0]
    active_variable = variable - const
    init_pop = int(os.environ.get("SHADE_INIT_POP_OVERRIDE", str(10 * active_variable)))
    max_eval_factor = int(os.environ.get("SHADE_MAX_EVAL_FACTOR", "1000"))
    max_eval = max(init_pop, active_variable * max_eval_factor)
    min_pop = min(active_variable, init_pop)

    t = 25
    margin = -0.1
    mpay = 600
    w_rpm = 5800
    gamma = 0.87
    Ce1 = 0.285
    Ce2 = 0.27
    type_power = "DBC"

    bias_mz = float(os.environ.get("NEW20_BIAS_MZ", "0.001"))
    U_obj = mpay * 100
    min_alpha = float(os.environ.get("NEW20_MIN_ALPHA", "-10"))
    max_alpha = float(os.environ.get("NEW20_MAX_ALPHA", "10"))
    min_delta = float(os.environ.get("NEW20_MIN_DELTA", "-5"))
    max_delta = float(os.environ.get("NEW20_MAX_DELTA", "5"))
    max_cy = float(os.environ.get("NEW20_MAX_CY", "0.6"))
    min_A = float(os.environ.get("NEW20_MIN_A", "0.5"))
    max_A = float(os.environ.get("NEW20_MAX_A", "1.2"))
    mission_l_km = float(os.environ.get("NEW20_MISSION_L_KM", "3000"))

    NP = init_pop
    H = 200
    p = 1
    rA = 5
    Pg = _initial_population(des_var, init_pop, gen_path)
    start = time.time()

    calc = ev.multijob(Pg, mpay, w_rpm, t, margin, type_power, gamma, Ce1, Ce2, cores, NP, model_dir=model_dir)
    info_aircraft = ev.vector_info(calc)
    info_to_FreeCAD = ev.vector_info_FreeCAD(calc)
    Pg = _sync_population_cyreq(Pg, info_aircraft)
    obj, obj_name = _objective(calc)
    psi = ev.pen_fun(abs(calc[14]), calc[15], calc[16], calc[13], calc[20], calc[19], calc[21],
                     bias_mz, max_cy, min_delta, max_delta, min_alpha, max_alpha, min_A, max_A)
    des_var = _adapt_m0_bounds(des_var, info_aircraft, psi, base_bounds)
    U_obj = ev.obj_max(psi, obj, NP, U_obj)
    Lx = ev.fit_fun(obj, psi, U_obj)
    Fx_m0 = obj
    print(f"[SHADE] objective={obj_name}, mission_L_km={mission_l_km}", flush=True)
    _save_generation(gen_path, 0, Pg, info_aircraft, Lx)

    MF = 0.5 * np.ones(H)
    MCR = 0.5 * np.ones(H)
    A = np.empty((0, variable))
    new_admin_pop = round(rA * NP)
    k = 0
    num_eval_f = NP
    new_pop = NP
    g = 1
    last_g = 0
    max_generations = int(os.environ.get("SHADE_MAX_GENERATIONS", "0"))

    while new_pop >= min_pop:
        new_pop_per = max(1, round(p * new_pop))
        Pg_best = ev.best_indiv(new_pop_per, Pg, Lx, variable)
        P_mut = np.zeros((new_pop, variable))
        P_cross = np.zeros((new_pop, variable))
        SF = np.array([])
        SCR = np.array([])
        Fg = np.array([])
        CRg = np.array([])

        for i in range(new_pop):
            Fi, CRi = ev.operators(H, MF, MCR)
            CRi = float(np.clip(CRi, 0.0, 1.0))
            Fg = np.append(Fg, Fi)
            CRg = np.append(CRg, CRi)
            best_ind = Pg_best[np.random.choice(new_pop_per)]
            P_mut[i] = ev.mut_oper(best_ind, new_pop, Pg, A, i, Fi, variable, des_var)
            P_cross[i] = ev.cross_oper(P_cross[i], P_mut[i], Pg[i], CRi, variable)
            P_cross[i] = np.clip(P_cross[i], des_var[:, 0], des_var[:, 1])

        calc = ev.multijob(P_cross, mpay, w_rpm, t, margin, type_power, gamma, Ce1, Ce2, cores, new_pop, model_dir=model_dir)
        info_cross = ev.vector_info(calc)
        freecad_cross = ev.vector_info_FreeCAD(calc)
        obj_mut, _ = _objective(calc)
        psi_mut = ev.pen_fun(abs(calc[14]), calc[15], calc[16], calc[13], calc[20], calc[19], calc[21],
                             bias_mz, max_cy, min_delta, max_delta, min_alpha, max_alpha, min_A, max_A)
        U_obj = ev.obj_max(psi_mut, obj_mut, new_pop, U_obj)
        Lx_mut = ev.fit_fun(obj_mut, psi_mut, U_obj)

        Pg_new = np.zeros((new_pop, variable))
        info_new = np.zeros((new_pop, ev.INFO_COUNT))
        freecad_new = np.zeros((new_pop, 36))
        Lx_new = np.zeros(new_pop)
        obj_new = np.zeros(new_pop)
        psi_new = np.zeros(new_pop)
        diff = np.array([])

        for i in range(new_pop):
            if Lx_mut[i] < Lx[i]:
                A = np.row_stack((A, Pg[i]))
                SF = np.append(SF, Fg[i])
                SCR = np.append(SCR, CRg[i])
                diff = np.append(diff, abs(Lx_mut[i] - Lx[i]))
                Pg_new[i] = P_cross[i]
                info_new[i] = info_cross[i]
                freecad_new[i] = freecad_cross[i]
                Lx_new[i] = Lx_mut[i]
                obj_new[i] = obj_mut[i]
                psi_new[i] = psi_mut[i]
            else:
                Pg_new[i] = Pg[i]
                info_new[i] = info_aircraft[i]
                freecad_new[i] = info_to_FreeCAD[i]
                Lx_new[i] = Lx[i]
                obj_new[i] = Fx_m0[i]
                psi_new[i] = psi[i]

        if SF.size:
            MF[k] = ev.Lehmer_weight_average(SF, diff)
            MCR[k] = -1 if MCR[k] == -1 or max(SCR) == 0 else ev.Lehmer_weight_average(SCR, diff)
            k = (k + 1) % H

        A = ev.population_A(new_admin_pop, A)
        num_eval_f += new_pop
        reduced_pop = ev.pop_reduction_exponential(max_eval, NP, min_pop, num_eval_f)
        des_var = _adapt_m0_bounds(des_var, info_new, psi_new, base_bounds)
        Pg_new = _sync_population_m0(Pg_new, info_new, des_var)
        Pg_new = _sync_population_cyreq(Pg_new, info_new)
        np.save(gen_path / f"MF{g}.npy", MF)
        np.save(gen_path / f"MCR{g}.npy", MCR)
        np.save(gen_path / f"A{g}.npy", A)
        _save_generation(gen_path, g, Pg_new, info_new, Lx_new)
        last_g = g

        spread = 0.0 if abs(np.max(Lx_new)) < 1e-12 else abs(np.max(Lx_new) - np.min(Lx_new)) / abs(np.max(Lx_new))
        if spread <= epsilon:
            info_to_FreeCAD = freecad_new
            break
        if max_generations and g >= max_generations:
            info_to_FreeCAD = freecad_new
            break

        new_pop = reduced_pop
        Pg, Lx, info_aircraft, psi, Fx_m0, info_to_FreeCAD = ev.new_generation(
            new_pop, Pg_new, Lx_new, info_new, psi_new, obj_new, freecad_new
        )
        g += 1

    m0.save_output_DataFrame_FreeCAD(info_to_FreeCAD).to_excel(gen_path / "info_to_FreeCAD.xlsx", index=False)
    print(f"Finished {gen_path} in {time.time() - start:.2f}s")
    return last_g
