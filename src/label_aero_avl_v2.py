from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from problem_v2_spec import AERO_INPUT_COLS, AERO_OUTPUT_COLS
from branch_geometry_v2 import build_geom_from_row


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

def aero_one(task: tuple[dict, int]) -> dict:
    row_dict, kk = task
    row = pd.Series(row_dict)
    cleanup_kk(kk)
    ensure_kk_tmp(kk)
    try:
        f_geo, a_geo, v_geo, fuse_geo, scheme_fuse = build_geom_from_row(row, ac)
        alpha = float(row["alpha"])
        delta = float(row["delta"])
        if a_geo[6] <= f_geo[6]:
            a_geo[5] = delta
        else:
            f_geo[5] = delta

        flight = ac.input_flight_cond(float(row["V"]), float(alpha), float(row["H"]))
        cx, cy, _, _, mz = ac.aero_calc(f_geo, a_geo, v_geo, fuse_geo, scheme_fuse, flight, kk)

        alpha_step = float(os.environ.get("AERO_DMZ_DCY_ALPHA_STEP", "1.0"))
        flight_minus = ac.input_flight_cond(float(row["V"]), float(alpha - alpha_step), float(row["H"]))
        flight_plus = ac.input_flight_cond(float(row["V"]), float(alpha + alpha_step), float(row["H"]))
        _, cy_minus, _, _, mz_minus = ac.aero_calc(
            f_geo, a_geo, v_geo, fuse_geo, scheme_fuse, flight_minus, kk + 20
        )
        _, cy_plus, _, _, mz_plus = ac.aero_calc(
            f_geo, a_geo, v_geo, fuse_geo, scheme_fuse, flight_plus, kk + 21
        )
        k_val = np.nan if abs(float(cx)) < 1e-12 else float(cy) / float(cx)

        betas = [0.0, 5.0]
        mxs = []
        mys = []
        for idx, beta in enumerate(betas):
            flight_beta = ac.input_flight_cond(float(row["V"]), float(alpha), float(row["H"]), beta)
            _, _, mx_b, my_b, _ = ac.aero_calc(
                f_geo, a_geo, v_geo, fuse_geo, scheme_fuse, flight_beta, kk + 100 + idx
            )
            mxs.append(float(mx_b))
            mys.append(float(my_b))

        mx_beta = (mxs[1] - mxs[0]) / (betas[1] - betas[0])
        my_beta = (mys[1] - mys[0]) / (betas[1] - betas[0])

        rec = {k: row_dict[k] for k in row_dict.keys() if k in {"case_id", "branch", "sample_pool"} or k in AERO_INPUT_COLS}
        rec["cx"] = float(cx)
        rec["cy"] = float(cy)
        rec["mz_ref"] = float(mz)
        rec["mx_beta"] = float(mx_beta)
        rec["my_beta"] = float(my_beta)
        rec["K"] = float(k_val)
        rec["status"] = "ok"
        rec["fail_reason"] = ""
        return rec
    except Exception as exc:
        rec = {k: row_dict[k] for k in row_dict.keys() if k in {"case_id", "branch", "sample_pool"} or k in AERO_INPUT_COLS}
        rec["cx"] = np.nan
        rec["cy"] = np.nan
        rec["mz_ref"] = np.nan
        rec["mx_beta"] = np.nan
        rec["my_beta"] = np.nan
        rec["K"] = np.nan
        rec["status"] = "fail"
        rec["fail_reason"] = repr(exc)
        return rec
    finally:
        cleanup_kk(kk)


def parse_values(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", type=str, default="data/aero_lhs_v2.csv")
    p.add_argument("--out", type=str, default="data/aero_labeled_v2.csv")
    p.add_argument("--workers", type=int, default=int(os.environ.get("AERO_WORKERS", "10")))
    p.add_argument("--kk-base", type=int, default=930000)
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
    miss = [c for c in AERO_INPUT_COLS if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns: {miss}")

    tasks = []
    kk = args.kk_base
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        tasks.append((row_dict, kk))
        kk += 1000

    rows: list[dict] = []
    if args.workers <= 1:
        for task in tasks:
            rows.append(aero_one(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(aero_one, task): task for task in tasks}
            for fut in as_completed(futs):
                rows.append(fut.result())

    out = pd.DataFrame(rows)
    out = out.dropna(subset=AERO_INPUT_COLS + AERO_OUTPUT_COLS, how="any")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outfile, index=False)
    ok = int((out["status"] == "ok").sum()) if "status" in out.columns else len(out)
    print(f"Wrote {outfile}: rows={len(out)} ok={ok} fail={len(rows)-ok}")


if __name__ == "__main__":
    main()
