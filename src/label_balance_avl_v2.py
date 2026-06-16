from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from problem_v2_spec import INPUT_COLS_V2, OUTPUT_BALANCE_COLS


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_AVL_SRC = PROJECT_ROOT / "avl_optimize_portable" / "src_avl_full"
AVL_SRC = os.environ.get("AVL_SRC", str(DEFAULT_AVL_SRC))
AUTO_FULL_ROOT = Path(os.environ.get("AUTO_FULL_ROOT", str(PROJECT_ROOT / "runs")))
if AVL_SRC not in sys.path:
    sys.path.insert(0, AVL_SRC)

import AeroCoeff_AVL as ac  # type: ignore  # noqa: E402


def cleanup_kk(kk: int) -> None:
    root = AUTO_FULL_ROOT / f"Auto_full_{kk}"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


def ensure_kk_tmp(kk: int) -> Path:
    tmp_dir = AUTO_FULL_ROOT / f"Auto_full_{kk}" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def polyfit2d(alpha_vals: np.ndarray, delta_vals: np.ndarray, values: np.ndarray) -> np.ndarray:
    alpha_grid, delta_grid = np.meshgrid(alpha_vals, delta_vals)
    a = np.array([np.ones(alpha_grid.size), alpha_grid.flatten(), delta_grid.flatten()]).T
    coeff, _, _, _ = np.linalg.lstsq(a, values.flatten(), rcond=None)
    return coeff


def build_geom(row: pd.Series):
    f_geo = ac.input_lift_surface_data(
        float(row["f_aspect"]),
        float(row["f_sweep"]),
        float(row["f_taper"]),
        float(row["f_twist"]),
        0.0,
        0.0,
        float(row["S_ref"]) * (1.0 - float(row["a_S_rel"])),
    )
    a_geo = ac.input_lift_surface_data(
        float(row["a_aspect"]),
        float(row["a_sweep"]),
        float(row["a_taper"]),
        float(row["a_twist"]),
        0.0,
        0.0,
        float(row["S_ref"]) * float(row["a_S_rel"]),
        float(row["a_x_loc"]),
    )
    v_geo = ac.input_lift_surface_data(
        float(row["v_aspect"]),
        15.0,
        2.0,
        0.0,
        90.0,
        0.0,
        float(row["S_ref"]) * float(row["v_S_rel"]),
        float(row["a_x_loc"]),
    )
    # fixed simple fuselage for portable workflow
    fuse_geo = ac.input_body_data(2.0, 4.0, 2.0, 0.76, 0.76, -2.0 * 0.76)
    scheme_fuse = 1
    return np.asarray(f_geo), np.asarray(a_geo), np.asarray(v_geo), np.asarray(fuse_geo), scheme_fuse


def balance_one(row_dict: dict, kk: int, alpha_span: float):
    row = pd.Series(row_dict)
    f_geo_base, a_geo_base, v_geo, fuse_geo, scheme_fuse = build_geom(row)
    delta_vals = np.array([-5.0, 5.0], dtype=float)
    alpha_vals = np.array([-alpha_span, alpha_span], dtype=float)
    cy = np.zeros((2, 2))
    mz_ref = np.zeros((2, 2))
    aero_center = np.zeros((2, 1))
    mz0 = np.zeros((2, 1))
    mz_cg = np.zeros((2, 2))
    margin = float(row["margin"])

    cleanup_kk(kk)
    ensure_kk_tmp(kk)
    try:
        for i, delta in enumerate(delta_vals):
            f_geo = f_geo_base.copy()
            a_geo = a_geo_base.copy()
            if a_geo[6] <= f_geo[6]:
                a_geo[5] = delta
            else:
                f_geo[5] = delta
            for j, alpha in enumerate(alpha_vals):
                flight = ac.input_flight_cond(float(row["V"]), float(alpha), float(row["H"]))
                _, c_y, _, _, m_z = ac.aero_calc(f_geo, a_geo, v_geo, fuse_geo, scheme_fuse, flight, kk)
                cy[i, j] = float(c_y)
                mz_ref[i, j] = float(m_z)
            denom = cy[i, 1] - cy[i, 0]
            aero_center[i, 0] = -float((mz_ref[i, 1] - mz_ref[i, 0]) / denom) if abs(denom) > 1e-12 else 0.0
            order = np.argsort(cy[i])
            mz0[i, 0] = float(np.interp(0.0, cy[i, order], mz_ref[i, order]))

        for i in range(2):
            ac_i = float(aero_center[i, 0])
            if abs(ac_i) < 1e-12:
                mz_cg[i, :] = mz_ref[i, :]
            else:
                mz_cg[i, :] = (mz_ref[i, :] - mz0[i, 0]) * (-margin) / ac_i + mz0[i, 0]

        coeff_mz = polyfit2d(alpha_vals, delta_vals, mz_cg)
        coeff_cy = polyfit2d(alpha_vals, delta_vals, cy)
        hs = np.array([[coeff_mz[1], coeff_mz[2]], [coeff_cy[1], coeff_cy[2]]], dtype=float)
        cy_req = float(row["cy_req"])
        kq = np.array([-coeff_mz[0], cy_req - coeff_cy[0]], dtype=float)
        try:
            alpha_bal, delta_bal = np.linalg.solve(hs, kq)
        except np.linalg.LinAlgError:
            alpha_bal, delta_bal = np.linalg.lstsq(hs, kq, rcond=None)[0]

        center_mass = float(aero_center[0, 0] + margin)
        A, _ = ac.vol_coeff(f_geo_base, a_geo_base, center_mass)
        return {
            "alpha_bal": float(alpha_bal),
            "delta_bal": float(delta_bal),
            "center_mass": center_mass,
            "aero_center": float(aero_center[0, 0]),
            "A": float(A),
            "cy_trim_target": cy_req,
            "status": "ok",
            "fail_reason": "",
        }
    except Exception as exc:
        return {
            "alpha_bal": np.nan,
            "delta_bal": np.nan,
            "center_mass": np.nan,
            "aero_center": np.nan,
            "A": np.nan,
            "cy_trim_target": np.nan,
            "status": "fail",
            "fail_reason": repr(exc),
        }
    finally:
        cleanup_kk(kk)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", type=str, default="data/lhs_v2.csv")
    p.add_argument("--out", type=str, default="data/balance_labeled_v2.csv")
    p.add_argument("--workers", type=int, default=int(os.environ.get("BALANCE_WORKERS", "10")))
    p.add_argument("--alpha-span", type=float, default=float(os.environ.get("BALANCE_ALPHA_SPAN", "5")))
    p.add_argument("--kk-base", type=int, default=920000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    infile = Path(args.infile)
    if not infile.is_absolute():
        infile = PROJECT_ROOT / infile
    outfile = Path(args.out)
    if not outfile.is_absolute():
        outfile = PROJECT_ROOT / outfile

    df = pd.read_csv(infile)
    miss = [c for c in INPUT_COLS_V2 if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns: {miss}")

    rows = []
    if args.workers <= 1:
        for i, row in df.iterrows():
            rec = row.to_dict()
            rec.update(balance_one(rec, kk=args.kk_base + i, alpha_span=args.alpha_span))
            rows.append(rec)
    else:
        jobs = [(i, r.to_dict()) for i, r in df.iterrows()]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(balance_one, rec, args.kk_base + i, args.alpha_span): (i, rec)
                for i, rec in jobs
            }
            for fut in as_completed(futs):
                i, rec = futs[fut]
                rec.update(fut.result())
                rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.sort_values("case_id").reset_index(drop=True)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outfile, index=False)
    ok = int((out["status"] == "ok").sum())
    print(f"Wrote {outfile}: rows={len(out)} ok={ok} fail={len(out)-ok}")


if __name__ == "__main__":
    main()


