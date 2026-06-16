from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from problem_v2_spec import INPUT_COLS_V2, OUTPUT_BALANCE_COLS


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/balance_labeled_v2.csv")
    p.add_argument("--outdir", type=str, default="data/prepared_balance_v2")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir

    df = pd.read_csv(in_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    df = df.dropna(subset=INPUT_COLS_V2 + OUTPUT_BALANCE_COLS).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(
            f"Not enough valid rows to split dataset: rows={len(df)}. "
            "Check AVL labeling results in the input CSV."
        )
    x = df[INPUT_COLS_V2].to_numpy(np.float32)
    y = df[OUTPUT_BALANCE_COLS].to_numpy(np.float32)
    w = np.ones(len(df), dtype=np.float32)

    x_train, x_tmp, y_train, y_tmp, w_train, w_tmp = train_test_split(
        x, y, w, test_size=0.2, random_state=args.seed
    )
    x_val, x_test, y_val, y_test, w_val, w_test = train_test_split(
        x_tmp, y_tmp, w_tmp, test_size=0.5, random_state=args.seed
    )

    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "train_X.npy", x_train)
    np.save(outdir / "train_Y.npy", y_train)
    np.save(outdir / "train_sample_weight.npy", w_train)
    np.save(outdir / "val_X.npy", x_val)
    np.save(outdir / "val_Y.npy", y_val)
    np.save(outdir / "val_sample_weight.npy", w_val)
    np.save(outdir / "test_X.npy", x_test)
    np.save(outdir / "test_Y.npy", y_test)
    np.save(outdir / "test_sample_weight.npy", w_test)
    print(f"Prepared split saved in {outdir} (rows={len(df)})")


if __name__ == "__main__":
    main()
