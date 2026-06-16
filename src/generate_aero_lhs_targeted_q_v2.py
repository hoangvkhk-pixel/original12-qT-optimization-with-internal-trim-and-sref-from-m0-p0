from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from original12_branching import BRANCHES, BranchSpec
from problem_v2_spec import AERO_BOUNDS_V2, AERO_INPUT_COLS


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def lhs_block(n: int, bounds: dict[str, tuple[float, float]], cols: list[str], seed: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, len(cols)), dtype=float)
    sampler = qmc.LatinHypercube(d=len(cols), seed=seed)
    u = sampler.random(n=n)
    lo = np.array([bounds[c][0] for c in cols], dtype=float)
    hi = np.array([bounds[c][1] for c in cols], dtype=float)
    return lo + u * (hi - lo)


def base_branch_bounds(branch: BranchSpec) -> dict[str, tuple[float, float]]:
    bounds = dict(AERO_BOUNDS_V2)
    bounds["a_S_rel"] = branch.a_s_range
    bounds["scheme_fuse"] = (branch.scheme_fuse, branch.scheme_fuse)
    bounds["scheme_vertical"] = (branch.scheme_vertical, branch.scheme_vertical)
    bounds["a_dihedral_mag"] = branch.a_dihedral_range
    return bounds


def targeted_bounds(branch: BranchSpec, mode: str) -> dict[str, tuple[float, float]]:
    bounds = base_branch_bounds(branch)
    is_duck = branch.name.startswith("duck")

    if mode == "h500_near_best":
        bounds.update(
            {
                "f_aspect": (10.0, 15.0),
                "f_sweep": (0.0, 15.0),
                "a_aspect": (4.0, 8.5),
                "a_sweep": (-30.0, 45.0),
                "v_aspect": (2.0, 4.0),
                "v_S_rel": (0.05, 0.30),
                "S_ref": (35.0, 50.0),
                "V": (34.0, 45.0),
                "H": (450.0, 550.0),
                "alpha": (-10.0, 10.0),
                "delta": (-5.0, 5.0),
            }
        )
        if is_duck:
            bounds["a_S_rel"] = (0.65, 0.80)
            bounds["a_dihedral_mag"] = (0.0, min(3.0, branch.a_dihedral_range[1]))
            bounds["V"] = (34.0, 60.0)
            bounds["cy_req"] = (0.30, 0.65)
        else:
            bounds["a_S_rel"] = (0.20, min(0.34, branch.a_s_range[1]))
            bounds["cy_req"] = (0.54, 0.62)
        return bounds

    if mode == "h500_boundary":
        bounds.update(
            {
                "f_aspect": (12.0, 15.0),
                "a_aspect": (4.0, 7.0),
                "a_S_rel": (0.20, 0.28) if not is_duck else (0.72, 0.80),
                "S_ref": (45.0, 50.0),
                "V": (35.0, 43.0) if not is_duck else (45.0, 60.0),
                "H": (490.0, 510.0),
                "cy_req": (0.57, 0.615) if not is_duck else (0.30, 0.45),
                "alpha": (-8.0, 10.0),
                "delta": (-5.0, 5.0),
            }
        )
        if is_duck:
            bounds["a_dihedral_mag"] = (0.0, min(1.0, branch.a_dihedral_range[1]))
        return bounds

    if mode == "branch_general_h500":
        bounds.update(
            {
                "S_ref": (15.0, 50.0),
                "V": (30.0, 70.0),
                "H": (400.0, 650.0),
                "cy_req": (0.45, 0.70) if not is_duck else (0.25, 0.70),
                "alpha": (-15.0, 15.0),
                "delta": (-5.0, 5.0),
            }
        )
        return bounds

    if mode == "broad_keepout":
        bounds.update(
            {
                "H": (200.0, 5000.0),
                "S_ref": (1.0, 50.0),
                "V": (30.0, 90.0),
                "alpha": (-20.0, 20.0),
                "delta": (-5.0, 5.0),
            }
        )
        return bounds

    raise ValueError(f"Unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-branch", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--out", type=str, default="data/aero_lhs_targeted_q_5k_v2.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    modes = [
        ("h500_near_best", 0.45),
        ("h500_boundary", 0.25),
        ("branch_general_h500", 0.20),
        ("broad_keepout", 0.10),
    ]

    for bidx, branch in enumerate(BRANCHES):
        counts = [int(round(args.n_per_branch * frac)) for _, frac in modes]
        counts[-1] += args.n_per_branch - sum(counts)
        parts = []
        pools = []
        for midx, ((mode, _), count) in enumerate(zip(modes, counts)):
            block = lhs_block(count, targeted_bounds(branch, mode), AERO_INPUT_COLS, args.seed + 100 * bidx + midx)
            parts.append(block)
            pools.extend([mode] * count)
        x = np.vstack(parts)
        frame = pd.DataFrame(x, columns=AERO_INPUT_COLS)
        frame.insert(0, "branch", branch.name)
        frame.insert(1, "sample_pool", pools)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(args.seed + 999)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df.insert(0, "case_id", np.arange(len(df), dtype=int))

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows ({args.n_per_branch} per branch)")


if __name__ == "__main__":
    main()
