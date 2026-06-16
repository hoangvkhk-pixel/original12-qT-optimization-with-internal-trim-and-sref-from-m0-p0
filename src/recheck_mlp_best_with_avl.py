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
    p.add_argument("--mlp-dir", type=str, default="gen_mlp_new20_12branches")
    p.add_argument("--outdir", type=str, default="mlp_best_avl_recheck")
    p.add_argument("--runs-dir", type=str, default="runs_mlp_best_avl_recheck")
    p.add_argument("--kk-base", type=int, default=7100000)
    p.add_argument("--cleanup-runs", action="store_true")
    return p.parse_args()


def numeric_frame(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)
    df.columns = list(df.iloc[0])
    df = df.iloc[1:].reset_index(drop=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def last_generation_file(branch_dir: Path, prefix: str) -> Path:
    files = sorted(
        branch_dir.glob(f"{prefix}_*.xlsx"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not files:
        raise FileNotFoundError(f"No {prefix}_*.xlsx in {branch_dir}")
    return files[-1]


def pct_diff(avl: float, mlp: float) -> float:
    denom = abs(mlp)
    if denom < 1e-12:
        return np.nan
    return 100.0 * (avl - mlp) / denom


def main() -> None:
    args = parse_args()
    mlp_dir = Path(args.mlp_dir)
    if not mlp_dir.is_absolute():
        mlp_dir = PROJECT_ROOT / mlp_dir
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = PROJECT_ROOT / runs_dir

    os.environ["AUTO_FULL_ROOT"] = str(runs_dir)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from new20_sizing_eval import AvlBackend, INFO_COLS, INPUT_COLS_V2, evaluate_candidate

    outdir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    backend = AvlBackend()
    rows = []
    candidate_rows = []
    branch_dirs = sorted([p for p in mlp_dir.iterdir() if p.is_dir()])
    for case_idx, branch_dir in enumerate(branch_dirs):
        info_path = last_generation_file(branch_dir, "info_aircraft")
        gen = int(info_path.stem.rsplit("_", 1)[-1])
        px_path = branch_dir / f"Px_{gen}.xlsx"
        if not px_path.exists():
            raise FileNotFoundError(px_path)

        info_df = numeric_frame(info_path)
        px_df = pd.read_excel(px_path)
        for col in px_df.columns:
            px_df[col] = pd.to_numeric(px_df[col], errors="coerce")

        best_idx = info_df["mtow_out"].astype(float).idxmin()
        mlp_info = info_df.loc[best_idx].to_dict()
        cand = {name: float(px_df.loc[best_idx, name]) for name in INPUT_COLS_V2}
        cand_row = {"branch": branch_dir.name, "generation": gen, "row_index_0based": int(best_idx)}
        cand_row.update(cand)
        candidate_rows.append(cand_row)

        print(f"[{case_idx + 1:02d}/{len(branch_dirs):02d}] AVL recheck {branch_dir.name} gen={gen} row={best_idx}")
        avl_tuple = evaluate_candidate(cand, backend, args.kk_base + case_idx)
        avl_info = dict(zip(INFO_COLS, avl_tuple[: len(INFO_COLS)]))

        rec = {
            "branch": branch_dir.name,
            "generation": gen,
            "row_index_0based": int(best_idx),
        }
        for name in INFO_COLS:
            m = float(mlp_info[name])
            a = float(avl_info[name])
            rec[f"mlp_{name}"] = m
            rec[f"avl_{name}"] = a
            rec[f"diff_{name}"] = a - m
            rec[f"diff_pct_{name}"] = pct_diff(a, m)
        rows.append(rec)

    result = pd.DataFrame(rows)
    candidate_df = pd.DataFrame(candidate_rows)
    result = result.sort_values("mlp_mtow_out").reset_index(drop=True)
    result.to_csv(outdir / "mlp_best_12cases_avl_recheck.csv", index=False)
    result.to_excel(outdir / "mlp_best_12cases_avl_recheck.xlsx", index=False)
    candidate_df.to_csv(outdir / "mlp_best_12cases_design_vectors.csv", index=False)

    summary_cols = [
        "branch",
        "mlp_mtow_out",
        "avl_mtow_out",
        "diff_mtow_out",
        "diff_pct_mtow_out",
        "mlp_K",
        "avl_K",
        "diff_pct_K",
        "mlp_cy",
        "avl_cy",
        "diff_cy",
        "mlp_mz",
        "avl_mz",
        "diff_mz",
        "mlp_alpha_bal",
        "avl_alpha_bal",
        "diff_alpha_bal",
        "mlp_delta_bal",
        "avl_delta_bal",
        "diff_delta_bal",
        "mlp_A",
        "avl_A",
        "diff_A",
        "mlp_mx_beta",
        "avl_mx_beta",
        "mlp_my_beta",
        "avl_my_beta",
    ]
    summary = result[summary_cols].copy()
    summary.to_csv(outdir / "mlp_best_12cases_avl_recheck_summary.csv", index=False)
    print("\nSUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nWrote {outdir}")

    if args.cleanup_runs:
        resolved_root = runs_dir.resolve()
        if resolved_root.exists() and str(resolved_root).startswith(str(PROJECT_ROOT.resolve())):
            shutil.rmtree(resolved_root, ignore_errors=True)
            print(f"Deleted runs dir {resolved_root}")


if __name__ == "__main__":
    main()
