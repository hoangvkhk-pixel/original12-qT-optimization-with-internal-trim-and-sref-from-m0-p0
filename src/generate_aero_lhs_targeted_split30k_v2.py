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
    "duck_1_x": 6700,
    "duck_2_x": 6700,
    "duck_3_x": 6600,
    "normal_2_1": 1400,
    "normal_2_2": 1400,
    "normal_2_3": 1400,
    "normal_3_1": 1400,
    "normal_3_3": 1400,
    "normal_1_1": 750,
    "normal_1_2": 750,
    "normal_1_3": 750,
    "normal_3_2": 750,
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
    return "duck" if branch.name.startswith("duck") else "normal"


def targeted_bounds(branch: BranchSpec, mode: str) -> dict[str, tuple[float, float]]:
    bounds = base_branch_bounds(branch)
    if mode == "duck_low_sref_fake":
        bounds.update(
            {
                "f_aspect": (10.0, 15.0),
                "f_sweep": (0.0, 20.0),
                "a_aspect": (4.0, 8.0),
                "a_sweep": (-10.0, 30.0),
                "v_aspect": (2.0, 3.8),
                "v_S_rel": (0.05, 0.25),
                "S_ref": (1.0, 30.0),
                "V": (45.0, 90.0),
                "H": (490.0, 510.0),
                "alpha": (-16.0, 16.0),
                "delta": (-5.0, 5.0),
                "a_S_rel": (0.55, 0.80),
                "a_dihedral_mag": (0.0, branch.a_dihedral_range[1]),
                "cy_req": (0.18, 0.58),
            }
        )
        return bounds

    if mode == "duck_delta_low":
        bounds.update(
            {
                "f_aspect": (10.0, 15.0),
                "f_sweep": (0.0, 20.0),
                "a_aspect": (4.0, 8.0),
                "S_ref": (20.0, 45.0),
                "V": (35.0, 65.0),
                "H": (495.0, 505.0),
                "alpha": (4.0, 12.0),
                "delta": (-5.0, -3.5),
                "a_S_rel": (0.55, 0.80),
                "a_dihedral_mag": (0.0, branch.a_dihedral_range[1]),
                "cy_req": (0.30, 0.60),
            }
        )
        return bounds

    if mode == "duck_avl_good":
        bounds.update(
            {
                "S_ref": (35.0, 50.0),
                "V": (34.0, 55.0),
                "H": (470.0, 530.0),
                "alpha": (-12.0, 12.0),
                "delta": (-5.0, 5.0),
                "a_S_rel": (0.65, 0.78),
                "a_dihedral_mag": (0.0, min(3.0, branch.a_dihedral_range[1])),
                "cy_req": (0.35, 0.62),
            }
        )
        return bounds

    if mode == "normal_high_alpha":
        bounds.update(
            {
                "f_aspect": (10.0, 15.0),
                "f_sweep": (0.0, 20.0),
                "a_aspect": (4.0, 8.0),
                "a_sweep": (-30.0, 45.0),
                "v_aspect": (2.0, 4.0),
                "v_S_rel": (0.05, 0.30),
                "S_ref": (9.0, 50.0),
                "V": (34.0, 90.0),
                "H": (480.0, 520.0),
                "alpha": (-18.0, 18.0),
                "delta": (-5.0, 5.0),
                "a_S_rel": (max(0.40, branch.a_s_range[0]), min(0.50, branch.a_s_range[1])),
                "cy_req": (0.45, 0.64),
            }
        )
        return bounds

    if mode == "normal_delta_low":
        bounds.update(
            {
                "f_aspect": (10.0, 15.0),
                "f_sweep": (0.0, 20.0),
                "a_aspect": (4.0, 8.0),
                "S_ref": (20.0, 50.0),
                "V": (33.0, 55.0),
                "H": (495.0, 505.0),
                "alpha": (-6.0, 10.0),
                "delta": (-5.0, -3.5),
                "a_S_rel": (0.20, min(0.40, branch.a_s_range[1])),
                "cy_req": (0.50, 0.63),
            }
        )
        return bounds

    if mode == "normal_avl_top":
        bounds.update(
            {
                "S_ref": (30.0, 50.0),
                "V": (33.0, 46.0),
                "H": (470.0, 530.0),
                "alpha": (-12.0, 12.0),
                "delta": (-5.0, 5.0),
                "a_S_rel": (0.20, min(0.34, branch.a_s_range[1])),
                "cy_req": (0.50, 0.64),
            }
        )
        return bounds

    raise ValueError(f"Unknown mode: {mode}")


def mode_plan(branch: BranchSpec) -> list[tuple[str, float]]:
    if branch.name.startswith("duck"):
        return [
            ("duck_low_sref_fake", 0.50),
            ("duck_delta_low", 0.25),
            ("duck_avl_good", 0.25),
        ]
    return [
        ("normal_high_alpha", 0.60),
        ("normal_delta_low", 0.20),
        ("normal_avl_top", 0.20),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20260603)
    p.add_argument("--out", type=str, default="data/aero_lhs_targeted_split30k_v2.csv")
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
