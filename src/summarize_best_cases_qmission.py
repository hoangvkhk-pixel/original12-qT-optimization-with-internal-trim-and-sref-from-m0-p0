from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = PROJECT_ROOT / "gen_mlp_fixedpoint_split300k_qmission3000_H500_cy06"
OUT_CSV = PROJECT_ROOT / "analysis_best_cases_qmission_summary.csv"
OUT_XLSX = PROJECT_ROOT / "analysis_best_cases_qmission_summary.xlsx"


def last_generation_file(branch_dir: Path, prefix: str) -> Path:
    files = sorted(
        branch_dir.glob(f"{prefix}_*.xlsx"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not files:
        raise FileNotFoundError(f"No {prefix}_*.xlsx in {branch_dir}")
    return files[-1]


def numeric_df(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    return df


def main() -> None:
    rows: list[dict[str, float | int | str]] = []
    branch_dirs = sorted([p for p in RUN_ROOT.iterdir() if p.is_dir()])
    for branch_dir in branch_dirs:
        info_path = last_generation_file(branch_dir, "info_aircraft")
        gen = int(info_path.stem.rsplit("_", 1)[-1])
        px_path = branch_dir / f"Px_{gen}.xlsx"
        if not px_path.exists():
            raise FileNotFoundError(px_path)

        info_df = numeric_df(info_path)
        px_df = numeric_df(px_path)

        info_df = info_df.replace([float("inf"), float("-inf")], pd.NA)
        info_df = info_df.dropna(subset=["q_g_per_ton_km"])
        best_idx = info_df["q_g_per_ton_km"].astype(float).idxmin()

        info = info_df.loc[best_idx]
        px = px_df.loc[best_idx]

        rows.append(
            {
                "branch": branch_dir.name,
                "generation": gen,
                "row_index_0based": int(best_idx),
                "q_g_per_ton_km": float(info["q_g_per_ton_km"]),
                "mtow_out": float(info["mtow_out"]),
                "cx": float(info["cx"]),
                "cy": float(info["cy"]),
                "K": float(info["K"]),
                "mz": float(info["mz"]),
                "alpha_bal": float(info["alpha_bal"]),
                "delta_bal": float(info["delta_bal"]),
                "A": float(info["A"]),
                "center_mass": float(info["center_mass"]),
                "S_ref": float(px["S_ref"]),
                "V": float(px["V"]),
                "H": float(px["H"]),
                "cy_req": float(px["cy_req"]),
                "f_aspect": float(px["f_aspect"]),
                "f_sweep": float(px["f_sweep"]),
                "f_taper": float(px["f_taper"]),
                "f_twist": float(px["f_twist"]),
                "a_aspect": float(px["a_aspect"]),
                "a_sweep": float(px["a_sweep"]),
                "a_taper": float(px["a_taper"]),
                "a_twist": float(px["a_twist"]),
                "a_x_loc": float(px["a_x_loc"]),
                "a_S_rel": float(px["a_S_rel"]),
                "v_aspect": float(px["v_aspect"]),
                "v_S_rel": float(px["v_S_rel"]),
                "scheme_fuse": float(px["scheme_fuse"]),
                "scheme_vertical": float(px["scheme_vertical"]),
                "a_dihedral_mag": float(px["a_dihedral_mag"]),
                "margin": float(px["margin"]),
                "p0": float(info["mtow_out"]) / max(float(px["S_ref"]), 1e-12),
            }
        )

    result = pd.DataFrame(rows).sort_values("q_g_per_ton_km").reset_index(drop=True)
    result.to_csv(OUT_CSV, index=False)
    result.to_excel(OUT_XLSX, index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
