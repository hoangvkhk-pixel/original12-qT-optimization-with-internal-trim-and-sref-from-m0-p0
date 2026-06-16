from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJECT_ROOT / "data" / "benchmarks"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from new20_sizing_eval import INFO_COLS, INPUT_COLS_V2, AvlBackend, SplitMlpBackend, evaluate_candidate  # noqa: E402


def parse_case_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in INPUT_COLS_V2 if c not in df.columns]
    if missing:
        raise KeyError(f"Benchmark file {path} missing columns: {missing}")
    return df


def feasible(info: dict[str, float], max_cy: float) -> bool:
    return (
        info["mx_beta"] <= 0
        and info["my_beta"] <= 0
        and info["cy"] <= max_cy
        and -5.0 <= info["delta_bal"] <= 5.0
        and -10.0 <= info["alpha_bal"] <= 10.0
        and 0.5 <= info["A"] <= 1.2
    )


def rerun(df: pd.DataFrame, backend_split: SplitMlpBackend, backend_avl: AvlBackend, name: str) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    max_cy = float(os.environ.get("NEW20_MAX_CY", "0.6"))
    for i, row in df.iterrows():
        case = {col: float(row[col]) for col in INPUT_COLS_V2}
        split = dict(zip(INFO_COLS, evaluate_candidate(case, backend_split, 1300000 + i * 1000)))
        avl = dict(zip(INFO_COLS, evaluate_candidate(case, backend_avl, 1400000 + i * 1000)))
        rows.append(
            {
                "branch": row.get("branch", f"case_{i}"),
                "split_q": split["q_g_per_ton_km"],
                "avl_q": avl["q_g_per_ton_km"],
                "split_K": split["K"],
                "avl_K": avl["K"],
                "split_cx": split["cx"],
                "avl_cx": avl["cx"],
                "split_cy": split["cy"],
                "avl_cy": avl["cy"],
                "split_feasible": feasible(split, max_cy),
                "avl_feasible": feasible(avl, max_cy),
            }
        )
    out = pd.DataFrame(rows)
    summary = {
        "set": name,
        "n_cases": int(len(out)),
        "mean_abs_q_err": float((out["split_q"] - out["avl_q"]).abs().mean()),
        "max_abs_q_err": float((out["split_q"] - out["avl_q"]).abs().max()),
        "mean_abs_K_err": float((out["split_K"] - out["avl_K"]).abs().mean()),
        "mean_abs_cx_err": float((out["split_cx"] - out["avl_cx"]).abs().mean()),
        "split_feasible_count": int(out["split_feasible"].sum()),
        "avl_feasible_count": int(out["avl_feasible"].sum()),
    }
    return out, summary


def main() -> None:
    normal_dir = os.environ.get("NEW20_MODEL_DIR_NORMAL", "").strip()
    duck_dir = os.environ.get("NEW20_MODEL_DIR_DUCK", "").strip()
    if not normal_dir or not duck_dir:
        raise EnvironmentError("Set NEW20_MODEL_DIR_NORMAL and NEW20_MODEL_DIR_DUCK before running benchmark_split_models_v2.py")

    outdir = PROJECT_ROOT / "analysis_split_benchmark"
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ["TRIM_ALPHA_VALUES"] = os.environ.get("TRIM_ALPHA_VALUES", "-5,5")
    os.environ["TRIM_DELTA_VALUES"] = os.environ.get("TRIM_DELTA_VALUES", "-5,5")
    os.environ["NEW20_OBJECTIVE"] = os.environ.get("NEW20_OBJECTIVE", "q_g_per_ton_km")
    os.environ["NEW20_MISSION_L_KM"] = os.environ.get("NEW20_MISSION_L_KM", "3000")

    split_backend = SplitMlpBackend(normal_dir, duck_dir)
    avl_backend = AvlBackend()

    summaries = []
    for name, fname in [
        ("old_q20k_best12", "old_q20k_best12_cases.csv"),
        ("boundary10", "boundary10_cases.csv"),
        ("avl_cy06_full12", "avl_cy06_full12_cases.csv"),
    ]:
        df = parse_case_csv(BENCH_DIR / fname)
        out, summary = rerun(df, split_backend, avl_backend, name)
        out.to_csv(outdir / f"{name}.csv", index=False)
        summaries.append(summary)

    pd.DataFrame(summaries).to_csv(outdir / "summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
