from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from original12_branching import (
    BRANCHES,
    TRAIN_H_BROAD_RANGE,
    TRAIN_H_FOCUS_FRACTION,
    TRAIN_H_FOCUS_RANGE,
    BranchSpec,
)
from problem_v2_spec import AERO_BOUNDS_V2, AERO_INPUT_COLS


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def lhs_block(n: int, bounds: dict[str, tuple[float, float]], cols: list[str], seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=len(cols), seed=seed)
    u = sampler.random(n=n)
    lo = np.array([bounds[c][0] for c in cols], dtype=float)
    hi = np.array([bounds[c][1] for c in cols], dtype=float)
    return lo + u * (hi - lo)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000, help="Cases per branch")
    p.add_argument("--total", type=int, default=0, help="Optional total cases, split equally over 12 branches")
    p.add_argument("--full-fraction", type=float, default=0.70, help="Kept for CLI compatibility; unused")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/aero_lhs_v2.csv")
    return p.parse_args()


def branch_bounds(branch: BranchSpec, h_range: tuple[float, float]) -> dict[str, tuple[float, float]]:
    bounds = dict(AERO_BOUNDS_V2)
    bounds["a_S_rel"] = branch.a_s_range
    bounds["scheme_fuse"] = (branch.scheme_fuse, branch.scheme_fuse)
    bounds["scheme_vertical"] = (branch.scheme_vertical, branch.scheme_vertical)
    bounds["a_dihedral_mag"] = branch.a_dihedral_range
    bounds["H"] = h_range
    return bounds


def main() -> None:
    args = parse_args()
    n_per_branch = int(args.n)
    if args.total:
        n_per_branch = int(round(int(args.total) / len(BRANCHES)))

    frames = []
    for idx, branch in enumerate(BRANCHES):
        focus_n = int(round(n_per_branch * TRAIN_H_FOCUS_FRACTION))
        broad_n = n_per_branch - focus_n
        parts = []
        if broad_n:
            parts.append(lhs_block(broad_n, branch_bounds(branch, TRAIN_H_BROAD_RANGE), AERO_INPUT_COLS, args.seed + idx))
        if focus_n:
            parts.append(
                lhs_block(
                    focus_n,
                    branch_bounds(branch, TRAIN_H_FOCUS_RANGE),
                    AERO_INPUT_COLS,
                    args.seed + 1000 + idx,
                )
            )
        x_branch = np.vstack(parts)
        frame = pd.DataFrame(x_branch, columns=AERO_INPUT_COLS)
        frame.insert(0, "branch", branch.name)
        frame.insert(1, "sample_pool", "original12_branch")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(args.seed + 2)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df.insert(0, "case_id", np.arange(len(df), dtype=int))

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(
        f"Wrote {out} with {len(df)} rows ({n_per_branch} per branch; "
        f"H broad={TRAIN_H_BROAD_RANGE}, H focus={TRAIN_H_FOCUS_RANGE}, focus_fraction={TRAIN_H_FOCUS_FRACTION})"
    )


if __name__ == "__main__":
    main()
