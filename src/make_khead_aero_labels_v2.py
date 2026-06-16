from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    df = pd.read_csv(in_path)
    missing = [c for c in ("cx", "cy") if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset {in_path} missing columns: {missing}")

    cx = df["cx"].astype(float).to_numpy()
    cy = df["cy"].astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        k_val = np.where(np.abs(cx) > 1e-12, cy / cx, np.nan)
    df["K"] = k_val
    if "mz_cy" in df.columns:
        df = df.drop(columns=["mz_cy"])
    if "status" in df.columns:
        bad = ~np.isfinite(k_val)
        if bad.any():
            df.loc[bad, "status"] = "fail"
            if "fail_reason" in df.columns:
                df.loc[bad, "fail_reason"] = (
                    df.loc[bad, "fail_reason"].fillna("").astype(str) + "|invalid_K"
                ).str.strip("|")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: rows={len(df)} valid_K={int(np.isfinite(k_val).sum())}")


if __name__ == "__main__":
    main()
