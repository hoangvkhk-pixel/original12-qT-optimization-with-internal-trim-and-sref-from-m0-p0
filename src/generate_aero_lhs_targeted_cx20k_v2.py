from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from original12_branching import BRANCHES, BranchSpec
from problem_v2_spec import AERO_BOUNDS_V2, AERO_INPUT_COLS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BRANCH_COUNTS = {
    "normal_2_1": 2200,
    "normal_2_2": 2200,
    "normal_2_3": 2200,
    "normal_3_1": 2200,
    "normal_3_3": 2200,
    "duck_1_x": 1800,
    "duck_2_x": 1800,
    "duck_3_x": 1800,
    "normal_1_1": 900,
    "normal_1_2": 900,
    "normal_1_3": 900,
    "normal_3_2": 900,
}


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


def branch_role(branch: BranchSpec) -> str:
    if branch.name.startswith("duck"):
        return "duck_fix"
    if branch.name in {"normal_2_1", "normal_2_2", "normal_2_3", "normal_3_1", "normal_3_3"}:
        return "normal_fix"
    return "control"


def targeted_bounds(branch: BranchSpec, mode: str) -> dict[str, tuple[float, float]]:
    bounds = base_branch_bounds(branch)
    is_duck = branch.name.startswith("duck")

    if mode == "cx_near_best":
        bounds.update(
            {
                "f_aspect": (12.0, 15.0),
                "f_sweep": (0.0, 18.0),
                "a_aspect": (4.0, 7.5),
                "a_sweep": (-30.0, 45.0) if not is_duck else (0.0, 35.0),
                "v_aspect": (2.0, 4.0),
                "v_S_rel": (0.05, 0.30),
                "S_ref": (38.0, 50.0) if not is_duck else (35.0, 50.0),
                "V": (35.0, 44.0) if not is_duck else (34.0, 60.0),
                "H": (480.0, 520.0),
                "alpha": (-10.0, 10.0),
                "delta": (-5.0, 5.0),
            }
        )
        if is_duck:
            bounds["a_S_rel"] = (0.72, 0.80)
            bounds["a_dihedral_mag"] = (0.0, min(2.0, branch.a_dihedral_range[1]))
            bounds["cy_req"] = (0.30, 0.55)
        else:
            bounds["a_S_rel"] = (0.20, min(0.28, branch.a_s_range[1]))
            bounds["cy_req"] = (0.54, 0.62)
        return bounds

    if mode == "cx_boundary":
        bounds.update(
            {
                "f_aspect": (13.0, 15.0),
                "f_sweep": (0.0, 12.0),
                "a_aspect": (4.0, 6.5),
                "S_ref": (46.0, 50.0) if not is_duck else (40.0, 50.0),
                "V": (36.0, 42.0) if not is_duck else (45.0, 60.0),
                "H": (495.0, 505.0),
                "alpha": (-8.0, 10.0),
                "delta": (-5.0, 5.0),
            }
        )
        if is_duck:
            bounds["a_S_rel"] = (0.76, 0.80)
            bounds["a_dihedral_mag"] = (0.0, min(1.0, branch.a_dihedral_range[1]))
            bounds["cy_req"] = (0.30, 0.45)
        else:
            bounds["a_S_rel"] = (0.20, 0.24)
            bounds["cy_req"] = (0.57, 0.615)
        return bounds

    if mode == "cx_branch_general_h500":
        bounds.update(
            {
                "S_ref": (20.0, 50.0),
                "V": (32.0, 65.0),
                "H": (430.0, 650.0),
                "cy_req": (0.48, 0.68) if not is_duck else (0.25, 0.65),
                "alpha": (-15.0, 15.0),
                "delta": (-5.0, 5.0),
            }
        )
        return bounds

    if mode == "control_h500":
        bounds.update(
            {
                "S_ref": (10.0, 50.0),
                "V": (30.0, 70.0),
                "H": (400.0, 700.0),
                "cy_req": (0.30, 0.70),
                "alpha": (-15.0, 15.0),
                "delta": (-5.0, 5.0),
            }
        )
        return bounds

    raise ValueError(f"Unknown mode: {mode}")


def mode_plan(branch: BranchSpec) -> list[tuple[str, float]]:
    role = branch_role(branch)
    if role == "normal_fix":
        return [
            ("cx_boundary", 0.50),
            ("cx_near_best", 0.30),
            ("cx_branch_general_h500", 0.15),
            ("control_h500", 0.05),
        ]
    if role == "duck_fix":
        return [
            ("cx_boundary", 0.45),
            ("cx_near_best", 0.35),
            ("cx_branch_general_h500", 0.15),
            ("control_h500", 0.05),
        ]
    return [
        ("cx_near_best", 0.30),
        ("cx_branch_general_h500", 0.45),
        ("control_h500", 0.25),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--out", type=str, default="data/aero_lhs_targeted_cx20k_v2.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames = []

    for bidx, branch in enumerate(BRANCHES):
        n_total = BRANCH_COUNTS[branch.name]
        plan = mode_plan(branch)
        counts = [int(round(n_total * frac)) for _, frac in plan]
        counts[-1] += n_total - sum(counts)

        parts = []
        pools = []
        roles = []
        for midx, ((mode, _), count) in enumerate(zip(plan, counts)):
            block = lhs_block(count, targeted_bounds(branch, mode), AERO_INPUT_COLS, args.seed + 100 * bidx + midx)
            parts.append(block)
            pools.extend([mode] * count)
            roles.extend([branch_role(branch)] * count)

        x = np.vstack(parts)
        frame = pd.DataFrame(x, columns=AERO_INPUT_COLS)
        frame.insert(0, "branch", branch.name)
        frame.insert(1, "focus_role", roles)
        frame.insert(2, "sample_pool", pools)
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
    print(f"Wrote {out} with {len(df)} rows")
    print("Branch counts:")
    print(df.groupby("branch").size().to_string())


if __name__ == "__main__":
    main()
