from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from original12_branching import BRANCHES
from problem_v2_spec import BOUNDS_V2, INPUT_COLS_V2


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-branch", type=int, default=140)
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument(
        "--outdir",
        type=str,
        default="shared_branch_initial_populations_v2",
    )
    return p.parse_args()


def build_bounds(a_s_range: tuple[float, float], a_dihedral_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lo = []
    hi = []
    for col in INPUT_COLS_V2:
        a, b = BOUNDS_V2[col]
        if col == "a_S_rel":
            a, b = a_s_range
        elif col == "a_dihedral_mag":
            a, b = a_dihedral_range
        elif col == "H":
            fixed_h = os.environ.get("NEW20_FIXED_H", "").strip()
            if fixed_h:
                a = b = float(fixed_h)
        lo.append(a)
        hi.append(b)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows = ["branch,file,n_per_branch,a_s_min,a_s_max,scheme_fuse,scheme_vertical,a_dihedral_min,a_dihedral_max"]
    scheme_fuse_idx = INPUT_COLS_V2.index("scheme_fuse")
    scheme_vertical_idx = INPUT_COLS_V2.index("scheme_vertical")
    for idx, branch in enumerate(BRANCHES):
        lo, hi = build_bounds(branch.a_s_range, branch.a_dihedral_range)
        sampler = qmc.LatinHypercube(d=len(INPUT_COLS_V2), seed=args.seed + idx)
        u = sampler.random(n=args.n_per_branch)
        x = lo + u * (hi - lo)
        x[:, scheme_fuse_idx] = branch.scheme_fuse
        x[:, scheme_vertical_idx] = branch.scheme_vertical
        path = outdir / f"{branch.name}_initial_population_{args.n_per_branch}.npy"
        np.save(path, x.astype(np.float32))
        manifest_rows.append(
            f"{branch.name},{path.name},{args.n_per_branch},{branch.a_s_range[0]},{branch.a_s_range[1]},"
            f"{branch.scheme_fuse},{branch.scheme_vertical},{branch.a_dihedral_range[0]},{branch.a_dihedral_range[1]}"
        )

    (outdir / "manifest.csv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    print(f"Wrote branch initial populations to {outdir}")


if __name__ == "__main__":
    main()
