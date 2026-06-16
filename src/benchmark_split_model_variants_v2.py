from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJECT_ROOT / "data" / "benchmarks"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from new20_sizing_eval import INFO_COLS, INPUT_COLS_V2, AvlBackend, SplitMlpBackend, evaluate_candidate  # noqa: E402


VARIANTS = [
    ("best_loss", "aero_mlp_v2_best_loss.keras"),
    ("best_cx", "aero_mlp_v2_best_cx.keras"),
    ("best_combo", "aero_mlp_v2_best_combo.keras"),
]


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


def activate_variant(model_dir: Path, filename: str) -> None:
    src = model_dir / filename
    if not src.exists():
        raise FileNotFoundError(f"Missing checkpoint {src}")
    shutil.copy2(src, model_dir / "aero_mlp_v2_best.keras")


def rerun_set(df: pd.DataFrame, backend_split: SplitMlpBackend, backend_avl: AvlBackend, max_cy: float) -> pd.DataFrame:
    rows = []
    for i, row in df.iterrows():
        case = {col: float(row[col]) for col in INPUT_COLS_V2}
        split = dict(zip(INFO_COLS, evaluate_candidate(case, backend_split, 1500000 + i * 1000)))
        avl = dict(zip(INFO_COLS, evaluate_candidate(case, backend_avl, 1600000 + i * 1000)))
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
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, variant: str, set_name: str) -> dict[str, float | str]:
    return {
        "variant": variant,
        "set": set_name,
        "n_cases": int(len(df)),
        "mean_abs_q_err": float((df["split_q"] - df["avl_q"]).abs().mean()),
        "max_abs_q_err": float((df["split_q"] - df["avl_q"]).abs().max()),
        "mean_abs_K_err": float((df["split_K"] - df["avl_K"]).abs().mean()),
        "mean_abs_cx_err": float((df["split_cx"] - df["avl_cx"]).abs().mean()),
        "mean_abs_cy_err": float((df["split_cy"] - df["avl_cy"]).abs().mean()),
        "split_feasible_count": int(df["split_feasible"].sum()),
        "avl_feasible_count": int(df["avl_feasible"].sum()),
    }


def main() -> None:
    normal_dir = Path(os.environ.get("NEW20_MODEL_DIR_NORMAL", "").strip())
    duck_dir = Path(os.environ.get("NEW20_MODEL_DIR_DUCK", "").strip())
    if not normal_dir or not duck_dir:
        raise EnvironmentError("Set NEW20_MODEL_DIR_NORMAL and NEW20_MODEL_DIR_DUCK before running benchmark_split_model_variants_v2.py")

    outdir = PROJECT_ROOT / "analysis_split_benchmark"
    outdir.mkdir(parents=True, exist_ok=True)
    max_cy = float(os.environ.get("NEW20_MAX_CY", "0.6"))
    os.environ["TRIM_ALPHA_VALUES"] = os.environ.get("TRIM_ALPHA_VALUES", "-5,5")
    os.environ["TRIM_DELTA_VALUES"] = os.environ.get("TRIM_DELTA_VALUES", "-5,5")
    os.environ["NEW20_OBJECTIVE"] = os.environ.get("NEW20_OBJECTIVE", "q_g_per_ton_km")
    os.environ["NEW20_MISSION_L_KM"] = os.environ.get("NEW20_MISSION_L_KM", "3000")
    os.environ["NEW20_USE_DIRECT_K"] = os.environ.get("NEW20_USE_DIRECT_K", "1")

    bench_sets = [
        ("old_q20k_best12", parse_case_csv(BENCH_DIR / "old_q20k_best12_cases.csv")),
        ("boundary10", parse_case_csv(BENCH_DIR / "boundary10_cases.csv")),
        ("avl_cy06_full12", parse_case_csv(BENCH_DIR / "avl_cy06_full12_cases.csv")),
    ]
    avl_backend = AvlBackend()
    summaries: list[dict[str, float | str]] = []

    for variant, filename in VARIANTS:
        activate_variant(normal_dir, filename)
        activate_variant(duck_dir, filename)
        backend_split = SplitMlpBackend(normal_dir, duck_dir)
        for set_name, df in bench_sets:
            out = rerun_set(df, backend_split, avl_backend, max_cy)
            out.to_csv(outdir / f"{variant}_{set_name}.csv", index=False)
            summaries.append(summarize(out, variant, set_name))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(outdir / "variant_summary.csv", index=False)

    pivot = summary_df.pivot(index="variant", columns="set", values="mean_abs_q_err")
    qavg = pivot.mean(axis=1).rename("avg_mean_abs_q_err")
    rank_df = pd.DataFrame(qavg)
    rank_df["avg_mean_abs_K_err"] = summary_df.groupby("variant")["mean_abs_K_err"].mean()
    rank_df["avg_mean_abs_cx_err"] = summary_df.groupby("variant")["mean_abs_cx_err"].mean()
    rank_df["avg_mean_abs_cy_err"] = summary_df.groupby("variant")["mean_abs_cy_err"].mean()
    rank_df["score"] = rank_df["avg_mean_abs_q_err"] + 0.5 * rank_df["avg_mean_abs_K_err"] + 20.0 * rank_df["avg_mean_abs_cx_err"] + 2.0 * rank_df["avg_mean_abs_cy_err"]
    rank_df = rank_df.sort_values("score").reset_index()
    rank_df.to_csv(outdir / "variant_ranking.csv", index=False)

    winner = rank_df.iloc[0]["variant"]
    winner_file = dict(VARIANTS)[winner]
    activate_variant(normal_dir, winner_file)
    activate_variant(duck_dir, winner_file)
    (outdir / "selected_variant.json").write_text(json.dumps({"winner": winner, "winner_file": winner_file}, indent=2), encoding="utf-8")
    print(rank_df.to_string(index=False))


if __name__ == "__main__":
    main()
