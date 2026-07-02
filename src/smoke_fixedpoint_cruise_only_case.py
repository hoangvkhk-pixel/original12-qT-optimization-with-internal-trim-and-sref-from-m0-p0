from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from new20_sizing_eval import (
    AvlBackend,
    MlpBackend,
    ac,
    build_geom_from_row,
    cruise_trim,
    alpha_search_phase,
    mission_phase_profile,
    phase_sizing,
    sz,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASE_PATH = PROJECT_ROOT / "analysis_optimizer_300k_hard40k_avl_recheck.csv"
OUT_PATH = PROJECT_ROOT / "analysis_fixedpoint_cruise_only_normal_3_1.json"
NORMAL_MLP_LATEST = PROJECT_ROOT / "models" / "aero_mlp_original12_normal_qkhead_300k_hard40k"


def _avl_eval_task(case: dict[str, float], alpha: float, delta: float, kk: int, center_mass: float | None = None, beta: float = 0.0):
    backend = AvlBackend()
    return backend.eval(case, alpha, delta, kk, center_mass=center_mass, beta=beta)


def select_case(branch: str = "normal_3_1") -> dict[str, float]:
    df = pd.read_csv(CASE_PATH)
    row = df.loc[df["branch"] == branch].iloc[0].to_dict()
    keys = [
        "f_aspect", "f_sweep", "f_taper", "f_twist",
        "a_aspect", "a_sweep", "a_taper", "a_twist",
        "a_x_loc", "a_S_rel", "v_aspect", "v_S_rel",
        "scheme_fuse", "scheme_vertical", "a_dihedral_mag",
        "margin", "S_ref", "V", "H", "cy_req",
    ]
    return {k: float(row[k]) for k in keys}


def evaluate_current(case: dict[str, float], backend: AvlBackend, kk: int) -> dict[str, float]:
    mission_l_km = float(3000.0)
    q_cr = ac.inputValuesAtmospheric(case["V"], case["H"])
    m0 = max(1.0, case["cy_req"] * q_cr * case["S_ref"] / 9.81)
    mission = mission_phase_profile(case, mission_l_km)

    cruise = cruise_trim(case, backend, kk)
    climb = alpha_search_phase(case, backend, kk + 2000, 0.9 * case["V"], 5.0)
    declimb = alpha_search_phase(case, backend, kk + 3000, 0.9 * case["V"], -30.0)

    cr_sz = phase_sizing(case, cruise, m0, mission["t_cruise_h"], 0.0, 5800.0, 0.87, 0.27, "DBC")
    cl_sz = phase_sizing(case, climb, m0, mission["t_climb_h"], 5.0, 5800.0, 0.87, 0.285, "DBC")
    dc_sz = phase_sizing(case, declimb, m0, mission["t_declimb_h"], -30.0, 5800.0, 0.87, 0.285, "DBC")

    f_geo, a_geo, v_geo, fuse_geo, scheme_fuse = build_geom_from_row(case, ac)
    m_v = max(cr_sz["m_V"], cl_sz["m_V"], dc_sz["m_V"])
    m_pu = max(cr_sz["m_pow"], cl_sz["m_pow"], dc_sz["m_pow"])
    m_fue = cr_sz["m_fue"] + max(0.0, cl_sz["m_fue"]) + max(0.0, dc_sz["m_fue"])
    m_const_v = 0.0 if v_geo[6] <= 0 else float(sz.m_constr_surface(v_geo, m0))
    m_ss = float(sz.m_SS(m0))
    m_f = float(sz.m_constr_fueslage(fuse_geo, case["V"], scheme_fuse))
    m_const_f = float(sz.m_constr_surface(f_geo, m0))
    m_const_a = float(sz.m_constr_surface(a_geo, m0))
    denom = 1.0 - m_fue - m_const_f - m_const_a - m_const_v - 0.08 - m_pu
    mtow_out = 100000.0 if denom <= 0 or not np.isfinite(denom) else (600.0 + m_v + m_ss + m_f) / denom
    return {
        "m0_init": float(m0),
        "q_cr": float(q_cr),
        "cruise": cruise,
        "climb": climb,
        "declimb": declimb,
        "mtow_out": float(mtow_out),
    }


def evaluate_fixedpoint_cruise_only(case: dict[str, float], backend: AvlBackend, kk: int) -> dict[str, object]:
    mission = mission_phase_profile(case, 3000.0)
    q_cr = ac.inputValuesAtmospheric(case["V"], case["H"])
    m0_guess = max(1.0, case["cy_req"] * q_cr * case["S_ref"] / 9.81)
    f_geo, a_geo, v_geo, fuse_geo, scheme_fuse = build_geom_from_row(case, ac)

    history: list[dict[str, float]] = []
    relax = 0.5
    tol_rel = 1e-3
    tol_abs = 1.0
    max_iter = 8
    cruise = None
    cr_sz = None
    for i in range(max_iter):
        cy_need = 9.81 * (m0_guess / case["S_ref"]) / q_cr
        local = dict(case)
        local["cy_req"] = float(cy_need)
        cruise = cruise_trim(local, backend, kk + i * 10000)
        cr_sz = phase_sizing(local, cruise, m0_guess, mission["t_cruise_h"], 0.0, 5800.0, 0.87, 0.27, "DBC")

        m_v = float(cr_sz["m_V"])
        m_pu = float(cr_sz["m_pow"])
        m_fue = float(cr_sz["m_fue"])
        m_const_v = 0.0 if v_geo[6] <= 0 else float(sz.m_constr_surface(v_geo, m0_guess))
        m_ss = float(sz.m_SS(m0_guess))
        m_f = float(sz.m_constr_fueslage(fuse_geo, case["V"], scheme_fuse))
        m_const_f = float(sz.m_constr_surface(f_geo, m0_guess))
        m_const_a = float(sz.m_constr_surface(a_geo, m0_guess))
        denom = 1.0 - m_fue - m_const_f - m_const_a - m_const_v - 0.08 - m_pu
        mtow_cruise_only = 100000.0 if denom <= 0 or not np.isfinite(denom) else (600.0 + m_v + m_ss + m_f) / denom
        diff_abs = abs(mtow_cruise_only - m0_guess)
        diff_rel = diff_abs / max(abs(mtow_cruise_only), 1.0)
        history.append(
            {
                "iter": float(i + 1),
                "m0_guess": float(m0_guess),
                "cy_need": float(cy_need),
                "alpha": float(cruise["alpha"]),
                "delta": float(cruise["delta"]),
                "cx": float(cruise["cx"]),
                "cy": float(cruise["cy"]),
                "mtow_cruise_only": float(mtow_cruise_only),
                "diff_abs": float(diff_abs),
                "diff_rel": float(diff_rel),
            }
        )
        if diff_abs <= tol_abs or diff_rel <= tol_rel:
            m0_guess = float(mtow_cruise_only)
            break
        m0_guess = float(relax * mtow_cruise_only + (1.0 - relax) * m0_guess)

    assert cruise is not None and cr_sz is not None
    final_case = dict(case)
    final_case["cy_req"] = float(9.81 * (m0_guess / case["S_ref"]) / q_cr)
    climb = alpha_search_phase(final_case, backend, kk + 900000, 0.9 * case["V"], 5.0)
    declimb = alpha_search_phase(final_case, backend, kk + 910000, 0.9 * case["V"], -30.0)

    cl_sz = phase_sizing(final_case, climb, m0_guess, mission["t_climb_h"], 5.0, 5800.0, 0.87, 0.285, "DBC")
    dc_sz = phase_sizing(final_case, declimb, m0_guess, mission["t_declimb_h"], -30.0, 5800.0, 0.87, 0.285, "DBC")
    m_v = max(cr_sz["m_V"], cl_sz["m_V"], dc_sz["m_V"])
    m_pu = max(cr_sz["m_pow"], cl_sz["m_pow"], dc_sz["m_pow"])
    m_fue = cr_sz["m_fue"] + max(0.0, cl_sz["m_fue"]) + max(0.0, dc_sz["m_fue"])
    m_const_v = 0.0 if v_geo[6] <= 0 else float(sz.m_constr_surface(v_geo, m0_guess))
    m_ss = float(sz.m_SS(m0_guess))
    m_f = float(sz.m_constr_fueslage(fuse_geo, case["V"], scheme_fuse))
    m_const_f = float(sz.m_constr_surface(f_geo, m0_guess))
    m_const_a = float(sz.m_constr_surface(a_geo, m0_guess))
    denom = 1.0 - m_fue - m_const_f - m_const_a - m_const_v - 0.08 - m_pu
    mtow_full_final = 100000.0 if denom <= 0 or not np.isfinite(denom) else (600.0 + m_v + m_ss + m_f) / denom

    return {
        "q_cr": float(q_cr),
        "iters": len(history),
        "history": history,
        "m0_final_for_trim": float(m0_guess),
        "cy_req_final": float(final_case["cy_req"]),
        "cruise_final": cruise,
        "climb_final": climb,
        "declimb_final": declimb,
        "mtow_full_final": float(mtow_full_final),
    }


def evaluate_fixedpoint_5cruise_1full(case: dict[str, float], backend: AvlBackend, kk: int) -> dict[str, object]:
    mission = mission_phase_profile(case, 3000.0)
    q_cr = ac.inputValuesAtmospheric(case["V"], case["H"])
    m0_guess = max(1.0, case["cy_req"] * q_cr * case["S_ref"] / 9.81)
    f_geo, a_geo, v_geo, fuse_geo, scheme_fuse = build_geom_from_row(case, ac)

    history: list[dict[str, float | str]] = []
    relax = 0.5
    tol_rel = 1e-3
    tol_abs = 1.0
    cruise_block = 5
    max_blocks = 3
    cruise = None
    climb = None
    declimb = None

    for block in range(max_blocks):
        for i in range(cruise_block):
            iter_idx = block * cruise_block + i + 1
            cy_need = 9.81 * (m0_guess / case["S_ref"]) / q_cr
            local = dict(case)
            local["cy_req"] = float(cy_need)
            cruise = cruise_trim(local, backend, kk + iter_idx * 10000)
            cr_sz = phase_sizing(local, cruise, m0_guess, mission["t_cruise_h"], 0.0, 5800.0, 0.87, 0.27, "DBC")

            m_v = float(cr_sz["m_V"])
            m_pu = float(cr_sz["m_pow"])
            m_fue = float(cr_sz["m_fue"])
            m_const_v = 0.0 if v_geo[6] <= 0 else float(sz.m_constr_surface(v_geo, m0_guess))
            m_ss = float(sz.m_SS(m0_guess))
            m_f = float(sz.m_constr_fueslage(fuse_geo, case["V"], scheme_fuse))
            m_const_f = float(sz.m_constr_surface(f_geo, m0_guess))
            m_const_a = float(sz.m_constr_surface(a_geo, m0_guess))
            denom = 1.0 - m_fue - m_const_f - m_const_a - m_const_v - 0.08 - m_pu
            mtow_cruise_only = 100000.0 if denom <= 0 or not np.isfinite(denom) else (600.0 + m_v + m_ss + m_f) / denom
            diff_abs = abs(mtow_cruise_only - m0_guess)
            diff_rel = diff_abs / max(abs(mtow_cruise_only), 1.0)
            history.append(
                {
                    "stage": "cruise",
                    "iter": float(iter_idx),
                    "m0_guess": float(m0_guess),
                    "cy_need": float(cy_need),
                    "alpha": float(cruise["alpha"]),
                    "delta": float(cruise["delta"]),
                    "cx": float(cruise["cx"]),
                    "cy": float(cruise["cy"]),
                    "mtow_est": float(mtow_cruise_only),
                    "diff_abs": float(diff_abs),
                    "diff_rel": float(diff_rel),
                }
            )
            m0_guess = float(relax * mtow_cruise_only + (1.0 - relax) * m0_guess)

        cy_need = 9.81 * (m0_guess / case["S_ref"]) / q_cr
        local = dict(case)
        local["cy_req"] = float(cy_need)
        cruise = cruise_trim(local, backend, kk + 500000 + block * 10000)
        climb = alpha_search_phase(local, backend, kk + 700000 + block * 10000, 0.9 * case["V"], 5.0)
        declimb = alpha_search_phase(local, backend, kk + 800000 + block * 10000, 0.9 * case["V"], -30.0)
        cr_sz = phase_sizing(local, cruise, m0_guess, mission["t_cruise_h"], 0.0, 5800.0, 0.87, 0.27, "DBC")
        cl_sz = phase_sizing(local, climb, m0_guess, mission["t_climb_h"], 5.0, 5800.0, 0.87, 0.285, "DBC")
        dc_sz = phase_sizing(local, declimb, m0_guess, mission["t_declimb_h"], -30.0, 5800.0, 0.87, 0.285, "DBC")

        m_v = max(cr_sz["m_V"], cl_sz["m_V"], dc_sz["m_V"])
        m_pu = max(cr_sz["m_pow"], cl_sz["m_pow"], dc_sz["m_pow"])
        m_fue = cr_sz["m_fue"] + max(0.0, cl_sz["m_fue"]) + max(0.0, dc_sz["m_fue"])
        m_const_v = 0.0 if v_geo[6] <= 0 else float(sz.m_constr_surface(v_geo, m0_guess))
        m_ss = float(sz.m_SS(m0_guess))
        m_f = float(sz.m_constr_fueslage(fuse_geo, case["V"], scheme_fuse))
        m_const_f = float(sz.m_constr_surface(f_geo, m0_guess))
        m_const_a = float(sz.m_constr_surface(a_geo, m0_guess))
        denom = 1.0 - m_fue - m_const_f - m_const_a - m_const_v - 0.08 - m_pu
        mtow_full = 100000.0 if denom <= 0 or not np.isfinite(denom) else (600.0 + m_v + m_ss + m_f) / denom
        diff_abs = abs(mtow_full - m0_guess)
        diff_rel = diff_abs / max(abs(mtow_full), 1.0)
        history.append(
            {
                "stage": "full",
                "iter": float((block + 1) * cruise_block),
                "m0_guess": float(m0_guess),
                "cy_need": float(cy_need),
                "alpha": float(cruise["alpha"]),
                "delta": float(cruise["delta"]),
                "cx": float(cruise["cx"]),
                "cy": float(cruise["cy"]),
                "mtow_est": float(mtow_full),
                "diff_abs": float(diff_abs),
                "diff_rel": float(diff_rel),
            }
        )
        if diff_abs <= tol_abs or diff_rel <= tol_rel:
            m0_guess = float(mtow_full)
            break
        m0_guess = float(relax * mtow_full + (1.0 - relax) * m0_guess)

    assert cruise is not None and climb is not None and declimb is not None
    final_case = dict(case)
    final_case["cy_req"] = float(9.81 * (m0_guess / case["S_ref"]) / q_cr)
    return {
        "q_cr": float(q_cr),
        "history": history,
        "m0_final_for_trim": float(m0_guess),
        "cy_req_final": float(final_case["cy_req"]),
        "cruise_final": cruise,
        "climb_final": climb,
        "declimb_final": declimb,
    }


def evaluate_fixedpoint_full_every_iter(case: dict[str, float], backend: AvlBackend, kk: int) -> dict[str, object]:
    mission = mission_phase_profile(case, 3000.0)
    q_cr = ac.inputValuesAtmospheric(case["V"], case["H"])
    m0_guess = max(1.0, case["cy_req"] * q_cr * case["S_ref"] / 9.81)
    f_geo, a_geo, v_geo, fuse_geo, scheme_fuse = build_geom_from_row(case, ac)

    history: list[dict[str, float]] = []
    relax = 0.5
    tol_rel = 1e-3
    tol_abs = 1.0
    max_iter = 10
    cruise = None
    climb = None
    declimb = None

    for i in range(max_iter):
        cy_need = 9.81 * (m0_guess / case["S_ref"]) / q_cr
        local = dict(case)
        local["cy_req"] = float(cy_need)
        cruise = cruise_trim(local, backend, kk + i * 10000)
        climb = alpha_search_phase(local, backend, kk + 2000 + i * 10000, 0.9 * case["V"], 5.0)
        declimb = alpha_search_phase(local, backend, kk + 3000 + i * 10000, 0.9 * case["V"], -30.0)

        cr_sz = phase_sizing(local, cruise, m0_guess, mission["t_cruise_h"], 0.0, 5800.0, 0.87, 0.27, "DBC")
        cl_sz = phase_sizing(local, climb, m0_guess, mission["t_climb_h"], 5.0, 5800.0, 0.87, 0.285, "DBC")
        dc_sz = phase_sizing(local, declimb, m0_guess, mission["t_declimb_h"], -30.0, 5800.0, 0.87, 0.285, "DBC")

        m_v = max(cr_sz["m_V"], cl_sz["m_V"], dc_sz["m_V"])
        m_pu = max(cr_sz["m_pow"], cl_sz["m_pow"], dc_sz["m_pow"])
        m_fue = cr_sz["m_fue"] + max(0.0, cl_sz["m_fue"]) + max(0.0, dc_sz["m_fue"])
        m_const_v = 0.0 if v_geo[6] <= 0 else float(sz.m_constr_surface(v_geo, m0_guess))
        m_ss = float(sz.m_SS(m0_guess))
        m_f = float(sz.m_constr_fueslage(fuse_geo, case["V"], scheme_fuse))
        m_const_f = float(sz.m_constr_surface(f_geo, m0_guess))
        m_const_a = float(sz.m_constr_surface(a_geo, m0_guess))
        denom = 1.0 - m_fue - m_const_f - m_const_a - m_const_v - 0.08 - m_pu
        mtow_full = 100000.0 if denom <= 0 or not np.isfinite(denom) else (600.0 + m_v + m_ss + m_f) / denom
        diff_abs = abs(mtow_full - m0_guess)
        diff_rel = diff_abs / max(abs(mtow_full), 1.0)
        history.append(
            {
                "iter": float(i + 1),
                "m0_guess": float(m0_guess),
                "cy_need": float(cy_need),
                "alpha": float(cruise["alpha"]),
                "delta": float(cruise["delta"]),
                "cx": float(cruise["cx"]),
                "cy": float(cruise["cy"]),
                "mtow_full": float(mtow_full),
                "diff_abs": float(diff_abs),
                "diff_rel": float(diff_rel),
            }
        )
        if diff_abs <= tol_abs or diff_rel <= tol_rel:
            m0_guess = float(mtow_full)
            break
        m0_guess = float(relax * mtow_full + (1.0 - relax) * m0_guess)

    assert cruise is not None and climb is not None and declimb is not None
    final_case = dict(case)
    final_case["cy_req"] = float(9.81 * (m0_guess / case["S_ref"]) / q_cr)
    return {
        "q_cr": float(q_cr),
        "history": history,
        "m0_final_for_trim": float(m0_guess),
        "cy_req_final": float(final_case["cy_req"]),
        "cruise_final": cruise,
        "climb_final": climb,
        "declimb_final": declimb,
    }


def cruise_trim_parallel(case: dict[str, float], pool: ProcessPoolExecutor, kk: int) -> dict[str, float]:
    alpha_vals = np.array([-5.0, 5.0], dtype=float)
    delta_vals = np.array([-5.0, 5.0], dtype=float)
    cy = np.zeros((2, 2))
    mz_ref = np.zeros((2, 2))
    aero_center = np.zeros(2)
    mz0 = np.zeros(2)
    mz_cg = np.zeros((2, 2))

    fut_map = {}
    for i, delta in enumerate(delta_vals):
        for j, alpha in enumerate(alpha_vals):
            fut = pool.submit(_avl_eval_task, case, float(alpha), float(delta), kk + 10 * i + j, None, 0.0)
            fut_map[fut] = (i, j)
    for fut, (i, j) in fut_map.items():
        _, cy_ij, mz_ij, _, _ = fut.result()
        cy[i, j] = cy_ij
        mz_ref[i, j] = mz_ij

    for i in range(2):
        coeff = np.polyfit(cy[i], mz_ref[i], deg=1)
        aero_center[i] = -float(coeff[0])
        mz0[i] = float(coeff[1])
    for i in range(2):
        ac_i = aero_center[i]
        if abs(ac_i) < 1e-12:
            mz_cg[i, :] = mz_ref[i, :]
        else:
            mz_cg[i, :] = (mz_ref[i, :] - mz0[i]) * (-case["margin"]) / ac_i + mz0[i]

    coeff_mz = np.polyfit(
        np.column_stack([np.ones(4), np.array([-5.0, 5.0, -5.0, 5.0]), np.array([-5.0, -5.0, 5.0, 5.0])]),
        np.array([]),
    ) if False else None
    from new20_sizing_eval import polyfit2d, backend_direct_k

    coeff_mz = polyfit2d(alpha_vals, delta_vals, mz_cg)
    coeff_cy = polyfit2d(alpha_vals, delta_vals, cy)
    hs = np.array([[coeff_mz[1], coeff_mz[2]], [coeff_cy[1], coeff_cy[2]]], dtype=float)
    rhs = np.array([-coeff_mz[0], case["cy_req"] - coeff_cy[0]], dtype=float)
    alpha_bal, delta_bal = np.linalg.solve(hs, rhs)
    aero_center_est = float(np.interp(delta_bal, delta_vals, aero_center))
    center_mass = float(aero_center_est + case["margin"])
    cx, cy_final, mz_final, _, _ = _avl_eval_task(case, float(alpha_bal), float(delta_bal), kk + 500, center_mass, 0.0)

    fut0 = pool.submit(_avl_eval_task, case, float(alpha_bal), float(delta_bal), kk + 700, center_mass, 0.0)
    fut5 = pool.submit(_avl_eval_task, case, float(alpha_bal), float(delta_bal), kk + 701, center_mass, 5.0)
    _, _, _, mx0, my0 = fut0.result()
    _, _, _, mx5, my5 = fut5.result()
    mx_beta = (mx5 - mx0) / 5.0
    my_beta = (my5 - my0) / 5.0
    k_val = backend_direct_k(case, AvlBackend(), float(alpha_bal), float(delta_bal), kk + 500, center_mass, float(cx), float(cy_final))
    return {
        "alpha": float(alpha_bal),
        "delta": float(delta_bal),
        "aero_center": aero_center_est,
        "dmz_dcy": float(-aero_center_est),
        "center_mass": center_mass,
        "cx": float(cx),
        "cy": float(cy_final),
        "mz": float(mz_final),
        "mx_beta": float(mx_beta),
        "my_beta": float(my_beta),
        "K": float(k_val),
    }


def alpha_search_phase_parallel(case: dict[str, float], pool: ProcessPoolExecutor, kk: int, v_phase: float, theta: float) -> dict[str, float]:
    local = dict(case)
    local["V"] = float(v_phase)
    q_cr = ac.inputValuesAtmospheric(case["V"], case["H"])
    q_phase = ac.inputValuesAtmospheric(v_phase, case["H"])
    target = float(case["cy_req"] if abs(q_phase) < 1e-12 else case["cy_req"] * q_cr * np.cos(np.deg2rad(theta)) / q_phase)
    alpha_vals = np.array([-10.0, 10.0], dtype=float)
    futs = [pool.submit(_avl_eval_task, local, float(alpha), 0.0, kk + i, None, 0.0) for i, alpha in enumerate(alpha_vals)]
    vals = [f.result() for f in futs]
    cy_vals = [v[1] for v in vals]
    mz_vals = [v[2] for v in vals]
    alpha_bal = float(np.interp(target, cy_vals, alpha_vals))
    slope = (mz_vals[-1] - mz_vals[0]) / (cy_vals[-1] - cy_vals[0]) if abs(cy_vals[-1] - cy_vals[0]) > 1e-12 else 0.0
    center_mass = float(-slope + case["margin"])
    cx, cy, mz, _, _ = _avl_eval_task(local, alpha_bal, 0.0, kk + 20, center_mass, 0.0)
    return {
        "alpha": alpha_bal,
        "delta": 0.0,
        "center_mass": center_mass,
        "cx": float(cx),
        "cy": float(cy),
        "mz": float(mz),
        "K": float(cy / cx) if abs(cx) > 1e-12 else 0.0,
        "cy_target": float(target),
    }


def evaluate_fixedpoint_full_every_iter_parallel(case: dict[str, float], kk: int, workers: int = 6) -> dict[str, object]:
    mission = mission_phase_profile(case, 3000.0)
    q_cr = ac.inputValuesAtmospheric(case["V"], case["H"])
    m0_guess = max(1.0, case["cy_req"] * q_cr * case["S_ref"] / 9.81)
    f_geo, a_geo, v_geo, fuse_geo, scheme_fuse = build_geom_from_row(case, ac)

    history: list[dict[str, float]] = []
    relax = 0.5
    tol_rel = 1e-3
    tol_abs = 1.0
    max_iter = 10
    cruise = None
    climb = None
    declimb = None

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i in range(max_iter):
            cy_need = 9.81 * (m0_guess / case["S_ref"]) / q_cr
            local = dict(case)
            local["cy_req"] = float(cy_need)
            cruise = cruise_trim_parallel(local, pool, kk + i * 10000)
            climb = alpha_search_phase_parallel(local, pool, kk + 2000 + i * 10000, 0.9 * case["V"], 5.0)
            declimb = alpha_search_phase_parallel(local, pool, kk + 3000 + i * 10000, 0.9 * case["V"], -30.0)

            cr_sz = phase_sizing(local, cruise, m0_guess, mission["t_cruise_h"], 0.0, 5800.0, 0.87, 0.27, "DBC")
            cl_sz = phase_sizing(local, climb, m0_guess, mission["t_climb_h"], 5.0, 5800.0, 0.87, 0.285, "DBC")
            dc_sz = phase_sizing(local, declimb, m0_guess, mission["t_declimb_h"], -30.0, 5800.0, 0.87, 0.285, "DBC")

            m_v = max(cr_sz["m_V"], cl_sz["m_V"], dc_sz["m_V"])
            m_pu = max(cr_sz["m_pow"], cl_sz["m_pow"], dc_sz["m_pow"])
            m_fue = cr_sz["m_fue"] + max(0.0, cl_sz["m_fue"]) + max(0.0, dc_sz["m_fue"])
            m_const_v = 0.0 if v_geo[6] <= 0 else float(sz.m_constr_surface(v_geo, m0_guess))
            m_ss = float(sz.m_SS(m0_guess))
            m_f = float(sz.m_constr_fueslage(fuse_geo, case["V"], scheme_fuse))
            m_const_f = float(sz.m_constr_surface(f_geo, m0_guess))
            m_const_a = float(sz.m_constr_surface(a_geo, m0_guess))
            denom = 1.0 - m_fue - m_const_f - m_const_a - m_const_v - 0.08 - m_pu
            mtow_full = 100000.0 if denom <= 0 or not np.isfinite(denom) else (600.0 + m_v + m_ss + m_f) / denom
            diff_abs = abs(mtow_full - m0_guess)
            diff_rel = diff_abs / max(abs(mtow_full), 1.0)
            history.append(
                {
                    "iter": float(i + 1),
                    "m0_guess": float(m0_guess),
                    "cy_need": float(cy_need),
                    "alpha": float(cruise["alpha"]),
                    "delta": float(cruise["delta"]),
                    "cx": float(cruise["cx"]),
                    "cy": float(cruise["cy"]),
                    "mtow_full": float(mtow_full),
                    "diff_abs": float(diff_abs),
                    "diff_rel": float(diff_rel),
                }
            )
            if diff_abs <= tol_abs or diff_rel <= tol_rel:
                m0_guess = float(mtow_full)
                break
            m0_guess = float(relax * mtow_full + (1.0 - relax) * m0_guess)

    assert cruise is not None and climb is not None and declimb is not None
    final_case = dict(case)
    final_case["cy_req"] = float(9.81 * (m0_guess / case["S_ref"]) / q_cr)
    return {
        "q_cr": float(q_cr),
        "history": history,
        "m0_final_for_trim": float(m0_guess),
        "cy_req_final": float(final_case["cy_req"]),
        "cruise_final": cruise,
        "climb_final": climb,
        "declimb_final": declimb,
    }


def main() -> None:
    backend = AvlBackend()
    case = select_case("normal_3_1")

    t0 = time.perf_counter()
    current = evaluate_current(case, backend, 7000)
    t1 = time.perf_counter()
    fixed = evaluate_fixedpoint_cruise_only(case, backend, 8000)
    t2 = time.perf_counter()
    mixed = evaluate_fixedpoint_5cruise_1full(case, backend, 9000)
    t3 = time.perf_counter()
    full_each = evaluate_fixedpoint_full_every_iter(case, backend, 10000)
    t4 = time.perf_counter()
    full_each_p6 = evaluate_fixedpoint_full_every_iter_parallel(case, 11000, workers=6)
    t5 = time.perf_counter()
    os.environ["NEW20_USE_DIRECT_K"] = "1"
    mlp_backend = MlpBackend(NORMAL_MLP_LATEST)
    mlp_full_each = evaluate_fixedpoint_full_every_iter(case, mlp_backend, 12000)
    t6 = time.perf_counter()

    out = {
        "case": case,
        "timing_sec": {
            "current_single_eval": t1 - t0,
            "fixedpoint_cruise_only_then_final_full": t2 - t1,
            "ratio_vs_current": (t2 - t1) / max(t1 - t0, 1e-9),
            "fixedpoint_5cruise_1full": t3 - t2,
            "ratio_5cruise_1full_vs_current": (t3 - t2) / max(t1 - t0, 1e-9),
            "fixedpoint_full_every_iter": t4 - t3,
            "ratio_full_every_iter_vs_current": (t4 - t3) / max(t1 - t0, 1e-9),
            "fixedpoint_full_every_iter_parallel6": t5 - t4,
            "ratio_full_every_iter_parallel6_vs_current": (t5 - t4) / max(t1 - t0, 1e-9),
            "speedup_parallel6_vs_serial_full_every_iter": (t4 - t3) / max(t5 - t4, 1e-9),
            "fixedpoint_full_every_iter_mlp_latest": t6 - t5,
            "ratio_full_every_iter_mlp_latest_vs_current": (t6 - t5) / max(t1 - t0, 1e-9),
        },
        "current": current,
        "fixedpoint": fixed,
        "fixedpoint_5cruise_1full": mixed,
        "fixedpoint_full_every_iter": full_each,
        "fixedpoint_full_every_iter_parallel6": full_each_p6,
        "fixedpoint_full_every_iter_mlp_latest": mlp_full_each,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
