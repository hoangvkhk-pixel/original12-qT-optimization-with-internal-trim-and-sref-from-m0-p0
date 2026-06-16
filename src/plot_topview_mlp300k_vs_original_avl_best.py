from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AVL_SRC = PROJECT_ROOT / "avl_optimize_portable" / "src_avl_full"
if str(AVL_SRC) not in sys.path:
    sys.path.insert(0, str(AVL_SRC))

import AeroCoeff_AVL as ac  # type: ignore  # noqa: E402

from branch_geometry_v2 import build_geom_from_row
from problem_v2_spec import INPUT_COLS_V2


BRANCH_ORDER = [
    "normal_1_1",
    "normal_1_2",
    "normal_1_3",
    "normal_2_1",
    "normal_2_2",
    "normal_2_3",
    "normal_3_1",
    "normal_3_2",
    "normal_3_3",
    "duck_1_x",
    "duck_2_x",
    "duck_3_x",
]


def half_surface_polygon(x_sec: list[float], y_sec: list[float], chords: list[float]) -> np.ndarray:
    lead = np.array([[x_sec[0], y_sec[0]], [x_sec[1], y_sec[1]]], dtype=float)
    trail = np.array([[x_sec[1] + chords[1], y_sec[1]], [x_sec[0] + chords[0], y_sec[0]]], dtype=float)
    return np.vstack([lead, trail])


def full_surface_polygons(x_sec: list[float], y_sec: list[float], chords: list[float]) -> list[np.ndarray]:
    half = half_surface_polygon(x_sec, y_sec, chords)
    mirrored = half.copy()
    mirrored[:, 1] *= -1.0
    return [half, mirrored]


def surface_sections(row: dict[str, float]) -> list[np.ndarray]:
    f_geo, a_geo, _, fuse_geo, scheme_fuse = build_geom_from_row(row, ac)
    polys: list[np.ndarray] = []

    if float(row["a_S_rel"]) <= 0.5:
        f_x, f_y, _, f_chord, _ = ac.lift_wing_sec(f_geo)
        a_x, a_y, _, a_chord, _ = ac.lift_h_tail_sec(a_geo, scheme_fuse)
    else:
        f_x, f_y, _, f_chord, _ = ac.lift_h_tail_sec(f_geo, scheme_fuse)
        a_x, a_y, _, a_chord, _ = ac.lift_wing_sec(a_geo)

    a_shift = float(row["a_x_loc"])
    polys.extend(full_surface_polygons(f_x, f_y, f_chord))
    polys.extend(full_surface_polygons([a_x[0] + a_shift, a_x[1] + a_shift], a_y, a_chord))

    body_d, body_l, _, _ = ac.body_def(fuse_geo)
    body_x0 = float(fuse_geo[5])
    body_poly = np.array(
        [
            [body_x0, -body_d / 2.0],
            [body_x0 + body_l, -body_d / 2.0],
            [body_x0 + body_l, body_d / 2.0],
            [body_x0, body_d / 2.0],
        ],
        dtype=float,
    )
    polys.append(body_poly)
    return polys


def draw_case(ax: plt.Axes, row: dict[str, float], color: str, alpha: float, linewidth: float) -> None:
    for poly in surface_sections(row):
        ax.fill(poly[:, 0], poly[:, 1], facecolor=color, edgecolor=color, alpha=alpha, linewidth=linewidth)


def render_group(
    branches: list[str],
    orig_map: dict[str, pd.Series],
    re_map: dict[str, pd.Series],
    baseline_df: pd.DataFrame,
    out_path: Path,
    title: str,
    nrows: int,
    ncols: int,
) -> None:
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 6.6 * nrows))
    axes_arr = np.atleast_1d(axes).flatten()
    fig.patch.set_facecolor("white")

    for ax, branch in zip(axes_arr, branches):
        if branch not in orig_map or branch not in re_map:
            ax.axis("off")
            continue

        orig_row = {name: float(orig_map[branch][name]) for name in INPUT_COLS_V2}
        mlp_row = {name: float(re_map[branch][name]) for name in INPUT_COLS_V2}

        draw_case(ax, orig_row, color="#1f77b4", alpha=0.25, linewidth=1.2)
        draw_case(ax, mlp_row, color="#d62728", alpha=0.25, linewidth=1.2)

        q_orig = float(baseline_df.loc[branch, "avl_q"])
        q_new = float(re_map[branch]["avl_q_g_per_ton_km"])
        feasible = bool(re_map[branch]["avl_feasible"])

        ax.set_title(f"{branch}\norig AVL q={q_orig:.2f} | new AVL q={q_new:.2f}", fontsize=12)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.text(
            0.02,
            0.02,
            f"S_ref {mlp_row['S_ref']:.2f}\nV {mlp_row['V']:.2f}\nfeasible {int(feasible)}",
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    for ax in axes_arr[len(branches) :]:
        ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=8, alpha=0.4, label="Original AVL best"),
        plt.Line2D([0], [0], color="#d62728", lw=8, alpha=0.4, label="New MLP-best AVL recheck"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(title, fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    orig_path = PROJECT_ROOT / "data" / "benchmarks" / "avl_cy06_full12_cases.csv"
    recheck_path = PROJECT_ROOT / "analysis_optimizer_300k_hard40k_avl_recheck.csv"
    baseline_path = PROJECT_ROOT / "analysis_split_benchmark" / "best_combo_avl_cy06_full12.csv"
    out_path = PROJECT_ROOT / "analysis_topview_mlp300k_vs_original_avl_best.png"
    out_normal = PROJECT_ROOT / "analysis_topview_mlp300k_vs_original_avl_best_normal.png"
    out_duck = PROJECT_ROOT / "analysis_topview_mlp300k_vs_original_avl_best_duck.png"

    orig_df = pd.read_csv(orig_path)
    re_df = pd.read_csv(recheck_path)
    baseline_df = pd.read_csv(baseline_path).set_index("branch")
    orig_map = {str(r["branch"]): r for _, r in orig_df.iterrows()}
    re_map = {str(r["branch"]): r for _, r in re_df.iterrows()}

    render_group(
        BRANCH_ORDER,
        orig_map,
        re_map,
        baseline_df,
        out_path,
        "Top View: New MLP Best vs Original AVL Best",
        nrows=4,
        ncols=3,
    )
    render_group(
        [b for b in BRANCH_ORDER if b.startswith("normal")],
        orig_map,
        re_map,
        baseline_df,
        out_normal,
        "Top View: Normal Branches",
        nrows=3,
        ncols=3,
    )
    render_group(
        [b for b in BRANCH_ORDER if b.startswith("duck")],
        orig_map,
        re_map,
        baseline_df,
        out_duck,
        "Top View: Duck Branches",
        nrows=1,
        ncols=3,
    )
    print(f"Wrote {out_path}")
    print(f"Wrote {out_normal}")
    print(f"Wrote {out_duck}")


if __name__ == "__main__":
    main()
