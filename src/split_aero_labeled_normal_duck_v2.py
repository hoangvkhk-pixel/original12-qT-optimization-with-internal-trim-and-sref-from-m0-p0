from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/aero_labeled_v2_plus_targeted_q5k_plus_cx20k.csv")
    p.add_argument("--normal-out", type=str, default="data/aero_labeled_v2_normal_200k.csv")
    p.add_argument("--duck-out", type=str, default="data/aero_labeled_v2_duck_200k.csv")
    p.add_argument("--a-s-threshold", type=float, default=0.5)
    return p.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    args = parse_args()
    in_path = resolve_path(args.input)
    normal_out = resolve_path(args.normal_out)
    duck_out = resolve_path(args.duck_out)

    df = pd.read_csv(in_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if "a_S_rel" not in df.columns:
        raise KeyError("Input dataset is missing required column 'a_S_rel'")

    normal_df = df[df["a_S_rel"] <= args.a_s_threshold].copy()
    duck_df = df[df["a_S_rel"] > args.a_s_threshold].copy()

    if normal_df.empty or duck_df.empty:
        raise ValueError(
            f"Split produced empty dataset(s): normal={len(normal_df)}, duck={len(duck_df)}, "
            f"threshold={args.a_s_threshold}"
        )

    normal_out.parent.mkdir(parents=True, exist_ok=True)
    duck_out.parent.mkdir(parents=True, exist_ok=True)
    normal_df.to_csv(normal_out, index=False)
    duck_df.to_csv(duck_out, index=False)

    print(
        "Wrote split datasets: "
        f"normal={normal_out} rows={len(normal_df)}, "
        f"duck={duck_out} rows={len(duck_df)}, "
        f"threshold={args.a_s_threshold}"
    )


if __name__ == "__main__":
    main()
