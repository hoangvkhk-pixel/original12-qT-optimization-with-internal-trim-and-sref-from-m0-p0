from __future__ import annotations

import json
import math
import time
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AVL_SRC = PROJECT_ROOT / "avl_optimize_portable" / "src_avl_full"
PORTABLE_SRC = PROJECT_ROOT / "avl_optimize_portable" / "src"
if str(AVL_SRC) not in sys.path:
    sys.path.insert(0, str(AVL_SRC))
if str(PORTABLE_SRC) not in sys.path:
    sys.path.insert(0, str(PORTABLE_SRC))

import AeroCoeff_AVL as ac  # noqa: E402


CASE = {
    "f_aspect": 8.5,
    "f_sweep": 12.0,
    "f_taper": 2.0,
    "f_twist": -1.0,
    "a_aspect": 7.5,
    "a_sweep": -6.0,
    "a_taper": 1.8,
    "a_twist": 1.0,
    "a_x_loc": 3.8,
    "a_S_rel": 0.34,
    "v_aspect": 2.8,
    "v_S_rel": 0.14,
    "S_ref": 12.0,
    "V": 55.0,
    "H": 0.0,
    "margin": -0.1,
    "cy_req": 0.58,
}


def build_geom(row: dict[str, float]):
    f_geo = ac.input_lift_surface_data(
        row["f_aspect"],
        row["f_sweep"],
        row["f_taper"],
        row["f_twist"],
        0.0,
        0.0,
        row["S_ref"] * (1.0 - row["a_S_rel"]),
    )
    a_geo = ac.input_lift_surface_data(
        row["a_aspect"],
        row["a_sweep"],
        row["a_taper"],
        row["a_twist"],
        0.0,
        0.0,
        row["S_ref"] * row["a_S_rel"],
        row["a_x_loc"],
    )
    v_geo = ac.input_lift_surface_data(
        row["v_aspect"],
        15.0,
        2.0,
        0.0,
        90.0,
        0.0,
        row["S_ref"] * row["v_S_rel"],
        row["a_x_loc"],
    )
    fuse_geo = ac.input_body_data(2.0, 4.0, 2.0, 0.76, 0.76, -2.0 * 0.76)
    return np.asarray(f_geo), np.asarray(a_geo), np.asarray(v_geo), np.asarray(fuse_geo), 1


def polyfit2d(alpha_vals: np.ndarray, delta_vals: np.ndarray, values: np.ndarray) -> np.ndarray:
    alpha_grid, delta_grid = np.meshgrid(alpha_vals, delta_vals)
    a = np.array([np.ones(alpha_grid.size), alpha_grid.flatten(), delta_grid.flatten()]).T
    coeff, _, _, _ = np.linalg.lstsq(a, values.flatten(), rcond=None)
    return coeff


class EvalCtx:
    def __init__(self, case: dict[str, float]):
        self.case = case
        self.f_geo_base, self.a_geo_base, self.v_geo, self.fuse_geo, self.scheme_fuse = build_geom(case)
        self.call_count = 0
        self.kk = 100000

    def aero_eval(self, alpha: float, delta: float, center_mass: float | None = None, beta: float = 0.0):
        self.call_count += 1
        self.kk += 1
        f_geo = self.f_geo_base.copy()
        a_geo = self.a_geo_base.copy()
        if a_geo[6] <= f_geo[6]:
            a_geo[5] = delta
        else:
            f_geo[5] = delta
        flight = ac.input_flight_cond(self.case["V"], alpha, self.case["H"], beta)
        cx, cy, mx, my, mz = ac.aero_calc(
            f_geo, a_geo, self.v_geo, self.fuse_geo, self.scheme_fuse, flight, self.kk, center_mass
        )
        return {
            "cx": float(cx),
            "cy": float(cy),
            "mx": float(mx),
            "my": float(my),
            "mz": float(mz),
            "f_geo": f_geo,
            "a_geo": a_geo,
        }


def infer_aero_center(ctx: EvalCtx, alpha: float, delta: float, alpha_eps: float = 1.0) -> float:
    lo = ctx.aero_eval(alpha - alpha_eps, delta, center_mass=0.0)
    hi = ctx.aero_eval(alpha + alpha_eps, delta, center_mass=0.0)
    dcy = hi["cy"] - lo["cy"]
    if abs(dcy) < 1e-12:
        return 0.0
    return -((hi["mz"] - lo["mz"]) / dcy)


def finalize(ctx: EvalCtx, alpha: float, delta: float, center_mass: float):
    base = ctx.aero_eval(alpha, delta, center_mass=center_mass)
    beta5 = ctx.aero_eval(alpha, delta, center_mass=center_mass, beta=5.0)
    alpha_lo = ctx.aero_eval(alpha - 1.0, delta, center_mass=center_mass)
    alpha_hi = ctx.aero_eval(alpha + 1.0, delta, center_mass=center_mass)
    dcy = alpha_hi["cy"] - alpha_lo["cy"]
    dmz_dcy = math.nan if abs(dcy) < 1e-12 else (alpha_hi["mz"] - alpha_lo["mz"]) / dcy
    return {
        "cx": base["cx"],
        "cy": base["cy"],
        "mz": base["mz"],
        "mx_beta": (beta5["mx"] - base["mx"]) / 5.0,
        "my_beta": (beta5["my"] - base["my"]) / 5.0,
        "dmz_dcy_local": dmz_dcy,
    }


def solve_old_2x2(case: dict[str, float]) -> dict[str, float]:
    ctx = EvalCtx(case)
    alpha_vals = np.array([-5.0, 5.0], dtype=float)
    delta_vals = np.array([-5.0, 5.0], dtype=float)
    cy = np.zeros((2, 2))
    mz_ref = np.zeros((2, 2))
    aero_center = np.zeros(2)
    mz0 = np.zeros(2)
    mz_cg = np.zeros((2, 2))

    for i, delta in enumerate(delta_vals):
        for j, alpha in enumerate(alpha_vals):
            out = ctx.aero_eval(alpha, delta, center_mass=0.0)
            cy[i, j] = out["cy"]
            mz_ref[i, j] = out["mz"]
        coeff_cy_to_mz = np.polyfit(cy[i], mz_ref[i], deg=1)
        aero_center[i] = -float(coeff_cy_to_mz[0])
        mz0[i] = float(coeff_cy_to_mz[1])

    for i in range(2):
        ac_i = float(aero_center[i])
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
    center_mass = aero_center_est + case["margin"]
    final = finalize(ctx, float(alpha_bal), float(delta_bal), float(center_mass))
    return {
        "solver": "old_2x2",
        "alpha_bal_deg": float(alpha_bal),
        "delta_bal_deg": float(delta_bal),
        "aero_center": aero_center_est,
        "center_mass": float(center_mass),
        "cy_error": float(final["cy"] - case["cy_req"]),
        "avl_calls": ctx.call_count,
        **final,
    }


def solve_newton(case: dict[str, float], tol: float = 1e-4, max_iter: int = 10, damp: float = 0.7) -> dict[str, float]:
    ctx = EvalCtx(case)
    alpha = 5.0
    delta = 0.0
    h_alpha = 0.5
    h_delta = 0.25
    last = None

    for it in range(max_iter):
        aero_center = infer_aero_center(ctx, alpha, delta)
        center_mass = aero_center + case["margin"]
        base = ctx.aero_eval(alpha, delta, center_mass=center_mass)
        r = np.array([base["cy"] - case["cy_req"], base["mz"]], dtype=float)
        last = (it + 1, alpha, delta, aero_center, center_mass, base, r)
        if abs(r[0]) <= tol and abs(r[1]) <= tol:
            final = finalize(ctx, alpha, delta, center_mass)
            return {
                "solver": "newton",
                "iterations": it + 1,
                "alpha_bal_deg": float(alpha),
                "delta_bal_deg": float(delta),
                "aero_center": float(aero_center),
                "center_mass": float(center_mass),
                "cy_error": float(final["cy"] - case["cy_req"]),
                "avl_calls": ctx.call_count,
                **final,
            }

        plus_a = ctx.aero_eval(alpha + h_alpha, delta, center_mass=center_mass)
        minus_a = ctx.aero_eval(alpha - h_alpha, delta, center_mass=center_mass)
        plus_d = ctx.aero_eval(alpha, delta + h_delta, center_mass=center_mass)
        minus_d = ctx.aero_eval(alpha, delta - h_delta, center_mass=center_mass)

        jac = np.array(
            [
                [
                    (plus_a["cy"] - minus_a["cy"]) / (2.0 * h_alpha),
                    (plus_d["cy"] - minus_d["cy"]) / (2.0 * h_delta),
                ],
                [
                    (plus_a["mz"] - minus_a["mz"]) / (2.0 * h_alpha),
                    (plus_d["mz"] - minus_d["mz"]) / (2.0 * h_delta),
                ],
            ],
            dtype=float,
        )
        step = np.linalg.solve(jac, -r)
        alpha += damp * float(step[0])
        delta += damp * float(step[1])

    assert last is not None
    _, alpha, delta, aero_center, center_mass, _, _ = last
    final = finalize(ctx, alpha, delta, center_mass)
    return {
        "solver": "newton",
        "iterations": max_iter,
        "alpha_bal_deg": float(alpha),
        "delta_bal_deg": float(delta),
        "aero_center": float(aero_center),
        "center_mass": float(center_mass),
        "cy_error": float(final["cy"] - case["cy_req"]),
        "avl_calls": ctx.call_count,
        **final,
    }


def benchmark(fn, repeats: int = 3):
    samples = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(CASE)
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    assert result is not None
    return {
        "result": result,
        "timings_sec": samples,
        "mean_sec": float(np.mean(samples)),
        "min_sec": float(np.min(samples)),
        "max_sec": float(np.max(samples)),
    }


def main():
    out = {
        "case": CASE,
        "old_2x2": benchmark(solve_old_2x2),
        "newton": benchmark(solve_newton),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
