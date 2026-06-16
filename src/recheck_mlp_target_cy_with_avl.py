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
    p.add_argument("--outdir", type=str, default="mlp_target_cy_avl_recheck")
    p.add_argument("--runs-dir", type=str, default="runs_mlp_target_cy_avl_recheck")
    p.add_argument("--target-cy", type=float, default=0.585)
    p.add_argument("--kk-base", type=int, default=8100000)
    p.add_argument("--cleanup-runs", action="store_true")
    return p.parse_args()


def numeric_frame(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)
    df.columns = list(df.iloc[0])
    df = df.iloc[1:].reset_index(drop=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pct_diff(avl: float, mlp: float) -> float:
    denom = abs(mlp)
    if denom < 1e-12:
        return np.nan
    return 100.0 * (avl - mlp) / denom


def generation_number(info_path: Path) -> int:
    return int(info_path.stem.rsplit("_", 1)[-1])


def select_target_cy_case(branch_dir: Path, target_cy: float, input_cols: list[str]) -> tuple[int, int, dict, dict]:
    best = None
    for info_path in sorted(branch_dir.glob("info_aircraft_*.xlsx"), key=generation_number):
        gen = generation_number(info_path)
        px_path = branch_dir / f"Px_{gen}.xlsx"
        if not px_path.exists():
            continue
        info_df = numeric_frame(info_path)
        px_df = pd.read_excel(px_path)
        for col in px_df.columns:
            px_df[col] = pd.to_numeric(px_df[col], errors="coerce")

        required = ["mtow_out", "cy"]
        if any(col not in info_df.columns for col in required):
            continue
        local = info_df.copy()
        local["__cy_abs_err"] = (local["cy"].astype(float) - target_cy).abs()
        local["__mtow"] = local["mtow_out"].astype(float)
        local = local.replace([np.inf, -np.inf], np.nan).dropna(subset=["__cy_abs_err", "__mtow"])
        if local.empty:
            continue
        idx = local.sort_values(["__cy_abs_err", "__mtow"]).index[0]
        cand_score = (float(local.loc[idx, "__cy_abs_err"]), float(local.loc[idx, "__mtow"]))
        if best is None or cand_score < best[0]:
            row = {name: float(px_df.loc[idx, name]) for name in input_cols}
            best = (cand_score, gen, int(idx), row, info_df.loc[idx].to_dict())

    if best is None:
        raise FileNotFoundError(f"No selectable info/Px rows in {branch_dir}")
    _, gen, idx, row, info = best
    return gen, idx, row, info


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
        gen, row_idx, cand, mlp_info = select_target_cy_case(branch_dir, args.target_cy, INPUT_COLS_V2)
        candidate_row = {
            "branch": branch_dir.name,
            "generation": gen,
            "row_index_0based": row_idx,
            "target_cy": args.target_cy,
        }
        candidate_row.update(cand)
        candidate_rows.append(candidate_row)

        print(
            f"[{case_idx + 1:02d}/{len(branch_dirs):02d}] AVL recheck {branch_dir.name} "
            f"gen={gen} row={row_idx} mlp_cy={float(mlp_info['cy']):.6g}"
        )
        avl_tuple = evaluate_candidate(cand, backend, args.kk_base + case_idx)
        avl_info = dict(zip(INFO_COLS, avl_tuple[: len(INFO_COLS)]))

        rec = {
            "branch": branch_dir.name,
            "generation": gen,
            "row_index_0based": row_idx,
            "target_cy": args.target_cy,
            "mlp_cy_abs_err_to_target": abs(float(mlp_info["cy"]) - args.target_cy),
        }
        for name in INFO_COLS:
            m = float(mlp_info[name])
            a = float(avl_info[name])
            rec[f"mlp_{name}"] = m
            rec[f"avl_{name}"] = a
            rec[f"diff_{name}"] = a - m
            rec[f"diff_pct_{name}"] = pct_diff(a, m)
        rows.append(rec)

    result = pd.DataFrame(rows).sort_values("mlp_mtow_out").reset_index(drop=True)
    candidate_df = pd.DataFrame(candidate_rows)

    result["avl_feasible"] = (
        (result["avl_cy"] <= 0.6)
        & (result["avl_alpha_bal"].between(-10, 10))
        & (result["avl_delta_bal"].between(-5, 5))
        & (result["avl_A"].between(0.5, 1.2))
        & (result["avl_mx_beta"] <= 0)
        & (result["avl_my_beta"] <= 0)
    )

    result.to_csv(outdir / "mlp_target_cy_12cases_avl_recheck.csv", index=False)
    result.to_excel(outdir / "mlp_target_cy_12cases_avl_recheck.xlsx", index=False)
    candidate_df.to_csv(outdir / "mlp_target_cy_12cases_design_vectors.csv", index=False)

    summary_cols = [
        "branch",
        "generation",
        "target_cy",
        "mlp_cy_abs_err_to_target",
        "mlp_mtow_out",
        "avl_mtow_out",
        "diff_pct_mtow_out",
        "mlp_K",
        "avl_K",
        "diff_pct_K",
        "mlp_center_mass",
        "avl_center_mass",
        "diff_center_mass",
        "mlp_cy",
        "avl_cy",
        "diff_cy",
        "mlp_alpha_bal",
        "avl_alpha_bal",
        "mlp_delta_bal",
        "avl_delta_bal",
        "mlp_A",
        "avl_A",
        "mlp_mx_beta",
        "avl_mx_beta",
        "mlp_my_beta",
        "avl_my_beta",
        "avl_feasible",
    ]
    summary = result[summary_cols].copy()
    summary.to_csv(outdir / "mlp_target_cy_12cases_avl_recheck_summary.csv", index=False)
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
