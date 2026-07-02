from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path, help="Directory containing profile_*.jsonl files.")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path.")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.profile_dir.glob("profile_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    if not rows:
        raise SystemExit(f"No profile rows found in {args.profile_dir}")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("section", dropna=False)["seconds"]
        .agg(calls="count", total_s="sum", mean_s="mean", median_s="median", max_s="max")
        .reset_index()
        .sort_values("total_s", ascending=False)
    )
    total = float(summary["total_s"].sum())
    summary["share_of_recorded_time"] = summary["total_s"] / total if total > 0 else 0.0

    out = args.out or (args.profile_dir / "profile_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
