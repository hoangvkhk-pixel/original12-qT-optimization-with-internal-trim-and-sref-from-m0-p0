from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", type=str, default="analysis_best_cases_qmission_summary.csv")
    p.add_argument("--outdir", type=str, default="analysis_best_cases_qmission_avl_recheck")
    p.add_argument("--runs-dir", type=str, default="runs_best_cases_qmission_avl_recheck")
    p.add_argument("--kk-base", type=int, default=9100000)
    p.add_argument("--cleanup-runs", action="store_true")
    return p.parse_args()


def pct_diff(avl: float, mlp: float) -> float:
    denom = abs(mlp)
    if denom < 1e-12:
        return np.nan
    return 100.0 * (avl - mlp) / denom


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_absolute():
        summary_csv = PROJECT_ROOT / summary_csv
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = PROJECT_ROOT / runs_dir

    os.environ["AUTO_FULL_ROOT"] = str(runs_dir)
    os.environ["NEW20_FIXEDPOINT"] = "1"
    os.environ.setdefault("NEW20_FP_RELAX", "0.5")
    os.environ.setdefault("NEW20_FP_MAX_ITER", "10")
    os.environ.setdefault("NEW20_FP_TOL_ABS", "1.0")
    os.environ.setdefault("NEW20_FP_TOL_REL", "0.001")
    os.environ.setdefault("NEW20_MAX_CY", "0.6")

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from new20_sizing_eval import AvlBackend, INFO_COLS, INPUT_COLS_V2, evaluate_candidate

    outdir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    backend = AvlBackend()
    rows = []
    design_rows = []
    run_root = PROJECT_ROOT / "gen_mlp_fixedpoint_split300k_qmission3000_H500_cy06"

    for case_idx, row in enumerate(df.itertuples(index=False), start=0):
        cand = {name: float(getattr(row, name)) for name in INPUT_COLS_V2}
        info_path = run_root / str(row.branch) / f"info_aircraft_{int(row.generation)}.xlsx"
        info_df = pd.read_excel(info_path)
        for col in info_df.columns:
            info_df[col] = pd.to_numeric(info_df[col], errors="coerce")
        mlp_info = info_df.loc[int(row.row_index_0based)].to_dict()
        design_row = {
            "branch": str(row.branch),
            "generation": int(row.generation),
            "row_index_0based": int(row.row_index_0based),
        }
        design_row.update(cand)
        design_rows.append(design_row)

        print(
            f"[{case_idx + 1:02d}/{len(df):02d}] AVL recheck {row.branch} "
            f"gen={int(row.generation)} row={int(row.row_index_0based)} q_mlp={float(row.q_g_per_ton_km):.6f}"
        )
        avl_tuple = evaluate_candidate(cand, backend, args.kk_base + case_idx)
        avl_info = dict(zip(INFO_COLS, avl_tuple[: len(INFO_COLS)]))

        rec = {
            "branch": str(row.branch),
            "generation": int(row.generation),
            "row_index_0based": int(row.row_index_0based),
        }
        for name in INFO_COLS:
            m = float(mlp_info[name])
            a = float(avl_info[name])
            rec[f"mlp_{name}"] = m
            rec[f"avl_{name}"] = a
            rec[f"diff_{name}"] = a - m
            rec[f"diff_pct_{name}"] = pct_diff(a, m)
        rec["S_ref"] = float(row.S_ref)
        rec["V"] = float(row.V)
        rec["H"] = float(row.H)
        rec["cy_req"] = float(row.cy_req)
        rec["p0_mlp"] = float(row.p0)
        rec["p0_avl"] = float(avl_info["mtow_out"]) / max(float(row.S_ref), 1e-12)
        rec["diff_p0"] = rec["p0_avl"] - rec["p0_mlp"]
        rec["avl_feasible_cy06"] = bool(float(avl_info["cy"]) <= 0.6)
        rows.append(rec)

    result = pd.DataFrame(rows).sort_values("mlp_q_g_per_ton_km").reset_index(drop=True)
    design_df = pd.DataFrame(design_rows)
    result.to_csv(outdir / "best_qmission_12cases_avl_recheck.csv", index=False)
    result.to_excel(outdir / "best_qmission_12cases_avl_recheck.xlsx", index=False)
    design_df.to_csv(outdir / "best_qmission_12cases_design_vectors.csv", index=False)

    summary_cols = [
        "branch",
        "generation",
        "row_index_0based",
        "mlp_q_g_per_ton_km",
        "avl_q_g_per_ton_km",
        "diff_q_g_per_ton_km",
        "mlp_mtow_out",
        "avl_mtow_out",
        "diff_mtow_out",
        "mlp_cx",
        "avl_cx",
        "diff_cx",
        "mlp_cy",
        "avl_cy",
        "diff_cy",
        "mlp_K",
        "avl_K",
        "diff_K",
        "mlp_alpha_bal",
        "avl_alpha_bal",
        "diff_alpha_bal",
        "mlp_delta_bal",
        "avl_delta_bal",
        "diff_delta_bal",
        "p0_mlp",
        "p0_avl",
        "diff_p0",
        "avl_feasible_cy06",
    ]
    summary = result[summary_cols].copy()
    summary.to_csv(outdir / "best_qmission_12cases_avl_recheck_summary.csv", index=False)
    print("\nSUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote {outdir}")

    if args.cleanup_runs:
        resolved_root = runs_dir.resolve()
        if resolved_root.exists() and str(resolved_root).startswith(str(PROJECT_ROOT.resolve())):
            shutil.rmtree(resolved_root, ignore_errors=True)
            print(f"Deleted runs dir {resolved_root}")


if __name__ == "__main__":
    main()
