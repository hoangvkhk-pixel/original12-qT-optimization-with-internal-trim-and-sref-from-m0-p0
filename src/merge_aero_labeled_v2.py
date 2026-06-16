from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="data/aero_labeled_v2.csv")
    p.add_argument("--extra", type=str, required=True)
    p.add_argument("--out", type=str, default="data/aero_labeled_v2_plus_targeted_q5k.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_path = resolve(args.base)
    extra_path = resolve(args.extra)
    out_path = resolve(args.out)

    base = pd.read_csv(base_path)
    extra = pd.read_csv(extra_path)
    missing = [c for c in base.columns if c not in extra.columns]
    if missing:
        for col in missing:
            extra[col] = pd.NA
        print(f"Warning: extra labeled file missing base columns {missing}; filled with NaN before merge")
    merged = pd.concat([base, extra[base.columns]], ignore_index=True)
    if "case_id" in merged.columns:
        merged["case_id"] = range(len(merged))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Merged {len(base)} base + {len(extra)} extra -> {len(merged)} rows")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
