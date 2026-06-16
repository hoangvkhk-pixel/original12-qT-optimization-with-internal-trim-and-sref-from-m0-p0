from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from problem_v2_spec import BOUNDS_V2, INPUT_COLS_V2


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/lhs_v2.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = int(args.n)
    d = len(INPUT_COLS_V2)
    sampler = qmc.LatinHypercube(d=d, seed=args.seed)
    u = sampler.random(n=n)
    lo = np.array([BOUNDS_V2[c][0] for c in INPUT_COLS_V2], dtype=float)
    hi = np.array([BOUNDS_V2[c][1] for c in INPUT_COLS_V2], dtype=float)
    x = qmc.scale(u, lo, hi)
    df = pd.DataFrame(x, columns=INPUT_COLS_V2)
    df.insert(0, "case_id", np.arange(len(df), dtype=int))
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows")


if __name__ == "__main__":
    main()
