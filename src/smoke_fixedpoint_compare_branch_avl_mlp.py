from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from new20_sizing_eval import (
    AvlBackend,
    MlpBackend,
    ac,
    build_geom_from_row,
    mission_phase_profile,
    phase_sizing,
    polyfit2d,
    sz,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASE_PATH = PROJECT_ROOT / "analysis_optimizer_300k_hard40k_avl_recheck.csv"
NORMAL_MLP = PROJECT_ROOT / "models" / "aero_mlp_original12_normal_qkhead_300k_hard40k"
DUCK_MLP = PROJECT_ROOT / "models" / "aero_mlp_original12_duck_qkhead_300k_hard40k"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", type=str, default="duck_2_x")
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args()


def select_case(branch: str) -> dict[str, float]:
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


def select_mlp(case: dict[str, float]) -> MlpBackend:
    model_dir = DUCK_MLP if case["a_S_rel"] > 0.5 else NORMAL_MLP
    return MlpBackend(model_dir)


def _avl_eval_task(case: dict[str, float], alpha: float, delta: float, kk: int, center_mass: float | None = None, beta: float = 0.0):
    backend = AvlBackend()
    return backend.eval(case, alpha, delta, kk, center_mass=center_mass, beta=beta)


def cruise_trim_avl_parallel(case: dict[str, float], pool: ProcessPoolExecutor, kk: int) -> dict[str, float]:
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
    return {
        "alpha": float(alpha_bal),
        "delta": float(delta_bal),
        "aero_center": aero_center_est,
        "dmz_dcy": float(-aero_center_est),
        "center_mass": center_mass,
        "cx": float(cx),
        "cy": float(cy_final),
        "mz": float(mz_final),
        "mx_beta": float((mx5 - mx0) / 5.0),
        "my_beta": float((my5 - my0) / 5.0),
        "K": float(cy_final / cx) if abs(cx) > 1e-12 else 0.0,
    }


def alpha_search_phase_avl_parallel(case: dict[str, float], pool: ProcessPoolExecutor, kk: int, v_phase: float, theta: float) -> dict[str, float]:
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


def cruise_trim_generic(case: dict[str, float], backend, kk: int) -> dict[str, float]:
    alpha_vals = np.array([-5.0, 5.0], dtype=float)
    delta_vals = np.array([-5.0, 5.0], dtype=float)
    cy = np.zeros((2, 2))
    mz_ref = np.zeros((2, 2))
    aero_center = np.zeros(2)
    mz0 = np.zeros(2)
    mz_cg = np.zeros((2, 2))

    for i, delta in enumerate(delta_vals):
        for j, alpha in enumerate(alpha_vals):
            _, cy_ij, mz_ij, _, _ = backend.eval(case, float(alpha), float(delta), kk + 10 * i + j)
            cy[i, j] = cy_ij
            mz_ref[i, j] = mz_ij
        coeff = np.polyfit(cy[i], mz_ref[i], deg=1)
        aero_center[i] = -float(coeff[0])
        mz0[i] = float(coeff[1])
    for i in range(2):
        ac_i = aero_center[i]
        if abs(ac_i) < 1e-12:
            mz_cg[i, :] = mz_ref[i, :]
        else:
            mz_cg[i, :] = (mz_ref[i, :] - mz0[i]) * (-case["margin"]) / ac_i + mz0[i]

    coeff_mz = polyfit2d(alpha_vals, delta_vals, mz_cg)
    coeff_cy = polyfit2d(alpha_vals, delta_vals, cy)
    hs = np.array([[coeff_mz[1], coeff_mz[2]], [coeff_cy[1], coeff_cy[2]]], dtype=float)
    rhs = np.array([-coeff_mz[0], case["cy_req"] - coeff_cy[0]], dtype=float)
    alpha_bal, delta_bal = np.linalg.solve(hs, rhs)
    aero_center_est = float(np.interp(delta_bal, delta_vals, aero_center))
    center_mass = float(aero_center_est + case["margin"])
    cx, cy_final, mz_final, mx_beta, my_beta = backend.eval(case, float(alpha_bal), float(delta_bal), kk + 500, center_mass=center_mass)
    pred = backend.predict_outputs(case, float(alpha_bal), float(delta_bal))
    return {
        "alpha": float(alpha_bal),
        "delta": float(delta_bal),
        "aero_center": aero_center_est,
        "dmz_dcy": float(-aero_center_est),
        "center_mass": center_mass,
        "cx": float(cx),
        "cy": float(cy_final),
        "mz": float(coeff_mz[0] + coeff_mz[1] * alpha_bal + coeff_mz[2] * delta_bal),
        "mx_beta": float(mx_beta),
        "my_beta": float(my_beta),
        "K": float(pred["K"]) if "K" in pred else (float(cy_final / cx) if abs(cx) > 1e-12 else 0.0),
    }


def alpha_search_phase_generic(case: dict[str, float], backend, kk: int, v_phase: float, theta: float) -> dict[str, float]:
    local = dict(case)
    local["V"] = float(v_phase)
    q_cr = ac.inputValuesAtmospheric(case["V"], case["H"])
    q_phase = ac.inputValuesAtmospheric(v_phase, case["H"])
    target = float(case["cy_req"] if abs(q_phase) < 1e-12 else case["cy_req"] * q_cr * np.cos(np.deg2rad(theta)) / q_phase)
    alpha_vals = np.array([-10.0, 10.0], dtype=float)
    cy_vals = []
    mz_vals = []
    for i, alpha in enumerate(alpha_vals):
        _, cy_i, mz_i, _, _ = backend.eval(local, float(alpha), 0.0, kk + i)
        cy_vals.append(cy_i)
        mz_vals.append(mz_i)
    alpha_bal = float(np.interp(target, cy_vals, alpha_vals))
    slope = (mz_vals[-1] - mz_vals[0]) / (cy_vals[-1] - cy_vals[0]) if abs(cy_vals[-1] - cy_vals[0]) > 1e-12 else 0.0
    center_mass = float(-slope + case["margin"])
    cx, cy, mz, _, _ = backend.eval(local, alpha_bal, 0.0, kk + 20, center_mass=center_mass)
    pred = backend.predict_outputs(local, alpha_bal, 0.0)
    return {
        "alpha": alpha_bal,
        "delta": 0.0,
        "center_mass": center_mass,
        "cx": float(cx),
        "cy": float(cy),
        "mz": float(mz),
        "K": float(pred["K"]) if "K" in pred else (float(cy / cx) if abs(cx) > 1e-12 else 0.0),
        "cy_target": float(target),
    }


def full_fixedpoint_avl_parallel(case: dict[str, float], kk: int, workers: int) -> dict[str, object]:
    mission = mission_phase_profile(case, 3000.0)
    q_cr = ac.inputValuesAtmospheric(case["V"], case["H"])
    m0_guess = max(1.0, case["cy_req"] * q_cr * case["S_ref"] / 9.81)
    f_geo, a_geo, v_geo, fuse_geo, scheme_fuse = build_geom_from_row(case, ac)
    history = []
    relax = 0.5
    cruise = climb = declimb = None

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i in range(10):
            cy_need = 9.81 * (m0_guess / case["S_ref"]) / q_cr
            local = dict(case)
            local["cy_req"] = float(cy_need)
            cruise = cruise_trim_avl_parallel(local, pool, kk + i * 10000)
            climb = alpha_search_phase_avl_parallel(local, pool, kk + 2000 + i * 10000, 0.9 * case["V"], 5.0)
            declimb = alpha_search_phase_avl_parallel(local, pool, kk + 3000 + i * 10000, 0.9 * case["V"], -30.0)
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
            history.append({"iter": i + 1, "m0_guess": float(m0_guess), "cy_need": float(cy_need), "mtow_full": float(mtow_full), "diff_abs": float(diff_abs), "diff_rel": float(diff_rel)})
            if diff_abs <= 1.0 or diff_rel <= 1e-3:
                m0_guess = float(mtow_full)
                break
            m0_guess = float(relax * mtow_full + (1.0 - relax) * m0_guess)

    return {
        "history": history,
        "m0_final_for_trim": float(m0_guess),
        "cy_req_final": float(9.81 * (m0_guess / case["S_ref"]) / q_cr),
        "cruise_final": cruise,
        "climb_final": climb,
        "declimb_final": declimb,
    }


def full_fixedpoint_mlp(case: dict[str, float], backend: MlpBackend, kk: int) -> dict[str, object]:
    os.environ["NEW20_USE_DIRECT_K"] = "1"
    mission = mission_phase_profile(case, 3000.0)
    q_cr = ac.inputValuesAtmospheric(case["V"], case["H"])
    m0_guess = max(1.0, case["cy_req"] * q_cr * case["S_ref"] / 9.81)
    f_geo, a_geo, v_geo, fuse_geo, scheme_fuse = build_geom_from_row(case, ac)
    history = []
    relax = 0.5
    cruise = climb = declimb = None

    for i in range(10):
        cy_need = 9.81 * (m0_guess / case["S_ref"]) / q_cr
        local = dict(case)
        local["cy_req"] = float(cy_need)
        cruise = cruise_trim_generic(local, backend, kk + i * 10000)
        climb = alpha_search_phase_generic(local, backend, kk + 2000 + i * 10000, 0.9 * case["V"], 5.0)
        declimb = alpha_search_phase_generic(local, backend, kk + 3000 + i * 10000, 0.9 * case["V"], -30.0)
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
        history.append({"iter": i + 1, "m0_guess": float(m0_guess), "cy_need": float(cy_need), "mtow_full": float(mtow_full), "diff_abs": float(diff_abs), "diff_rel": float(diff_rel)})
        if diff_abs <= 1.0 or diff_rel <= 1e-3:
            m0_guess = float(mtow_full)
            break
        m0_guess = float(relax * mtow_full + (1.0 - relax) * m0_guess)

    return {
        "history": history,
        "m0_final_for_trim": float(m0_guess),
        "cy_req_final": float(9.81 * (m0_guess / case["S_ref"]) / q_cr),
        "cruise_final": cruise,
        "climb_final": climb,
        "declimb_final": declimb,
    }


def main() -> None:
    args = parse_args()
    case = select_case(args.branch)
    out_path = PROJECT_ROOT / f"analysis_fixedpoint_compare_{args.branch}_avl_mlp.json"

    t0 = time.perf_counter()
    avl = full_fixedpoint_avl_parallel(case, 20000, args.workers)
    t1 = time.perf_counter()
    os.environ["OMP_NUM_THREADS"] = str(args.workers)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.workers)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(args.workers)
    mlp = full_fixedpoint_mlp(case, select_mlp(case), 30000)
    t2 = time.perf_counter()

    out = {
        "branch": args.branch,
        "workers": args.workers,
        "mlp_threads": args.workers,
        "case": case,
        "timing_sec": {
            "avl_parallel": t1 - t0,
            "mlp": t2 - t1,
        },
        "avl_parallel": avl,
        "mlp": mlp,
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
